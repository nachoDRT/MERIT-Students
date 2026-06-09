"""
Interpret discriminative SAE features via max-activating examples.

For each target feature (the ones that distinguish subject_8 from subject_0
in conditions_layer{L}_{pair}.json — top-delta + exclusives), run a natural
text corpus through Qwen3-VL, capture layer-L residual activations at every
token position, project them onto the feature's encoder direction, and keep
the tokens/contexts that fire it hardest.

This grounds each feature in the concept the SAE actually learned, which is
far more reliable than logit lens for naming a feature. The output places the
max-activating examples side by side with the logit-lens tokens and the
discriminative stats, so each feature is self-contained for interpretation.

Speed: a forward hook at LAYER captures activations and raises _StopForward
to skip the upper layers and lm_head. Corpus is processed in padded batches.

Inputs (env vars):
  LAYER         decoder layer index            (default: 20)
  VECTOR_PAIR   pair id                        (default: subject_8_vs_subject_0)
  N_DOCS        corpus lines to scan           (default: 4000)
  MAX_TOKENS    max tokens per line            (default: 64)
  TOP_EXAMPLES  examples kept per feature      (default: 20)
  BATCH         batch size                     (default: 16)
  CORPUS        'wikitext' (neutral) | 'social' (bias/discrimination) (default: wikitext)
  FEATURES      comma-sep override of feature ids (default: from conditions json)

Outputs:
  outputs/results/maxact_layer{LAYER}_{VECTOR_PAIR}_{CORPUS}.json

The 'social' corpus probes whether features that look silent on neutral text
(wikitext) actually encode bias/demographic concepts: it draws from
measuring-hate-speech (natural discourse targeting demographic groups —
contains offensive language by design) and social-justice news categories.
Comparing corpus_fire_rate across corpora tells whether a near-silent feature
is genuinely vision-specific or just unprobed by neutral text.
"""

import heapq
import json
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

script_dir  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from sae_module import TopKSAE

DEVICE     = 'cuda'
MODEL_NAME = 'Qwen/Qwen3-VL-8B-Instruct'

LAYER        = int(os.environ.get('LAYER',        '20'))
VECTOR_PAIR  = os.environ.get('VECTOR_PAIR',      'subject_8_vs_subject_0')
N_DOCS       = int(os.environ.get('N_DOCS',       '4000'))
MAX_TOKENS   = int(os.environ.get('MAX_TOKENS',   '64'))
TOP_EXAMPLES = int(os.environ.get('TOP_EXAMPLES', '20'))
BATCH        = int(os.environ.get('BATCH',        '16'))
CORPUS       = os.environ.get('CORPUS',           'wikitext').strip()
FEATURES_ENV = os.environ.get('FEATURES',         '').strip()

CKPT_DIR    = os.path.join(script_dir, 'outputs', 'checkpoints')
RESULTS_DIR = os.path.join(script_dir, 'outputs', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


class _StopForward(Exception):
    pass


def get_decoder_layers(model):
    mm = model.model
    if hasattr(mm, 'language_model') and hasattr(mm.language_model, 'layers'):
        return mm.language_model.layers
    return mm.layers


def load_target_features():
    """Gather discriminative feature ids (+ their stats/logit-lens) from conditions json."""
    cond_path = os.path.join(RESULTS_DIR, f'conditions_layer{LAYER}_{VECTOR_PAIR}.json')
    if not os.path.exists(cond_path):
        raise FileNotFoundError(f'Run analyze_conditions first: {cond_path}')
    with open(cond_path) as f:
        cond = json.load(f)

    # group / stat each feature comes from (for the report)
    meta = {}
    for fid, val in cond['top_delta_pos']:
        meta.setdefault(fid, {})['delta'] = val
        meta[fid].setdefault('groups', []).append('delta_pos(subject_8)')
    for fid, val in cond['top_delta_neg']:
        meta.setdefault(fid, {})['delta'] = val
        meta[fid].setdefault('groups', []).append('delta_neg(subject_0)')
    for fid, mp, mn in cond['exclusive_pos']:
        meta.setdefault(fid, {}).setdefault('groups', []).append('exclusive_pos(subject_8)')
    for fid, mn, mp in cond['exclusive_neg']:
        meta.setdefault(fid, {}).setdefault('groups', []).append('exclusive_neg(subject_0)')

    if FEATURES_ENV:
        ids = [int(x) for x in FEATURES_ENV.split(',')]
    else:
        ids = sorted(meta.keys())

    # attach logit-lens promotes (from conditions) for cross-reference
    for fid in ids:
        info = cond['features'].get(str(fid), {})
        meta.setdefault(fid, {})['logit_lens'] = [t for t, _ in info.get('promotes', [])[:8]]
        meta[fid].setdefault('groups', [])
    return ids, meta


def _load_wikitext():
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    lines = []
    for row in ds:
        t = row['text'].strip()
        if len(t) < 30 or t.startswith('='):   # skip headers / very short
            continue
        lines.append(t)
        if len(lines) >= N_DOCS:
            break
    return lines


def _load_social():
    """Bias/discrimination-dense corpus to probe demographic-bias features."""
    lines, seen = [], set()
    # 1) Natural discourse targeting demographic groups (~60%).
    ds = load_dataset('ucberkeley-dlab/measuring-hate-speech', split='train', streaming=True)
    cap1 = int(N_DOCS * 0.6)
    for r in ds:
        t = (r.get('text') or '').strip().replace('\n', ' ')
        if len(t) < 30 or t in seen:
            continue
        seen.add(t); lines.append(t)
        if len(lines) >= cap1:
            break
    # 2) Social-justice / demographic news prose (non-toxic complement).
    SOCIAL_CATS = {'BLACK VOICES', 'QUEER VOICES', 'WOMEN', 'LATINO VOICES',
                   'RELIGION', 'CRIME', 'IMPACT', 'POLITICS', 'U.S. NEWS', 'WORLD NEWS'}
    ds2 = load_dataset('heegyu/news-category-dataset', split='train', streaming=True)
    for r in ds2:
        if r['category'] not in SOCIAL_CATS:
            continue
        t = f"{r['headline']}. {r['short_description']}".strip().replace('\n', ' ')
        if len(t) < 30 or t in seen:
            continue
        seen.add(t); lines.append(t)
        if len(lines) >= N_DOCS:
            break
    return lines


def load_corpus():
    lines = _load_social() if CORPUS == 'social' else _load_wikitext()
    print(f'Corpus [{CORPUS}]: {len(lines)} lines')
    return lines


def main():
    out_path = os.path.join(RESULTS_DIR, f'maxact_layer{LAYER}_{VECTOR_PAIR}_{CORPUS}.json')
    if os.path.exists(out_path):
        print(f'Already computed: {out_path}')
        return

    target_ids, feat_meta = load_target_features()
    print(f'Target features ({len(target_ids)}): {target_ids}')

    ckpt_path = os.path.join(CKPT_DIR, f'layer{LAYER}.sae.pt')
    sae = TopKSAE(ckpt_path, k=100, device=DEVICE)
    sae.eval()
    # Rank by the TRUE TopK-gated feature activation (a feature only counts
    # when it makes the top-k at that token), not the raw encoder pre-activation
    # — otherwise a large b_enc would make a feature look active everywhere.
    idx_t = torch.tensor(target_ids, device=DEVICE)
    n_target = len(target_ids)

    print(f'Loading model ({MODEL_NAME})...')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map='auto',
        attn_implementation='flash_attention_2',
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    layers = get_decoder_layers(model)

    corpus = load_corpus()

    # per-feature min-heaps of (score, tiebreak, example_dict)
    heaps = {fid: [] for fid in target_ids}
    fire_lines = {fid: 0 for fid in target_ids}   # lines where feature fired (TopK)
    n_lines_seen = 0
    counter = 0

    captured = {}
    def hook(module, inp, out):
        captured['h'] = (out if isinstance(out, torch.Tensor) else out[0]).detach().float()
        raise _StopForward()
    handle = layers[LAYER].register_forward_hook(hook)

    n_batches = (len(corpus) + BATCH - 1) // BATCH
    for bi in range(n_batches):
        batch = corpus[bi * BATCH:(bi + 1) * BATCH]
        enc = tokenizer(
            batch, return_tensors='pt', padding=True, truncation=True,
            max_length=MAX_TOKENS, add_special_tokens=False,
        ).to(DEVICE)

        with torch.no_grad():
            try:
                model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
            except _StopForward:
                pass
            h = captured['h']                                   # (B, seq, d)
            B_, S_, D_ = h.shape
            # full TopK-gated features, then keep target columns: (B, seq, n_target)
            feats = sae.encode(h.reshape(-1, D_), apply_topk=True)[:, idx_t]
            pre   = feats.reshape(B_, S_, n_target)

        mask = enc['attention_mask'].bool()                      # (B, seq)
        ids  = enc['input_ids']
        for b in range(len(batch)):
            seq_len = int(mask[b].sum())
            if seq_len == 0:
                continue
            n_lines_seen += 1
            tok_ids = ids[b, :seq_len].tolist()
            for ti, fid in enumerate(target_ids):
                col = pre[b, :seq_len, ti]
                mx = torch.argmax(col)
                score = float(col[mx])
                pos = int(mx)
                if score <= 0.0:          # feature not selected (TopK) anywhere in this line
                    continue
                fire_lines[fid] += 1
                h_ = heaps[fid]
                if len(h_) < TOP_EXAMPLES or score > h_[0][0]:
                    # build highlighted context
                    toks = [tokenizer.decode([t]) for t in tok_ids]
                    toks[pos] = f'«{toks[pos]}»'
                    ctx = ''.join(toks)
                    ex = {
                        'score':     round(score, 4),
                        'token':     tokenizer.decode([tok_ids[pos]]),
                        'context':   ctx[:300],
                    }
                    counter += 1
                    if len(h_) < TOP_EXAMPLES:
                        heapq.heappush(h_, (score, counter, ex))
                    else:
                        heapq.heappushpop(h_, (score, counter, ex))

        if (bi + 1) % 20 == 0:
            print(f'  batch {bi+1}/{n_batches}', flush=True)

    handle.remove()

    # assemble output sorted by score desc
    features_out = {}
    for fid in target_ids:
        examples = sorted(heaps[fid], key=lambda x: -x[0])
        features_out[str(fid)] = {
            'feature_id':  fid,
            'groups':      feat_meta[fid].get('groups', []),
            'delta':       feat_meta[fid].get('delta'),
            'logit_lens':  feat_meta[fid].get('logit_lens', []),
            'corpus_fire_rate': round(fire_lines[fid] / max(n_lines_seen, 1), 4),
            'max_activating': [ex for _, _, ex in examples],
        }

    results = {
        'layer':       LAYER,
        'vector_pair': VECTOR_PAIR,
        'n_docs':      len(corpus),
        'max_tokens':  MAX_TOKENS,
        'n_features':  n_target,
        'features':    features_out,
    }
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nSaved: {out_path}')

    # readable summary
    for fid in target_ids:
        fo = features_out[str(fid)]
        top_tokens = [ex['token'] for ex in fo['max_activating'][:8]]
        print(f"\nF{fid}  {fo['groups']}  fire_rate={fo['corpus_fire_rate']}")
        print(f"  logit_lens: {fo['logit_lens']}")
        print(f"  max-act tokens: {top_tokens}")
        if fo['max_activating']:
            print(f"  top ctx: {fo['max_activating'][0]['context'][:160]}")
        else:
            print(f"  (never fired in text corpus — likely vision-specific)")


if __name__ == '__main__':
    main()
