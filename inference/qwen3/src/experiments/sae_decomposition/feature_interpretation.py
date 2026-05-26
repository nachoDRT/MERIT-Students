"""
Interpret top SAE features via logit lens.

For each robust feature i, project its decoder direction W_dec[:, i]
through the model's final layer norm and unembedding matrix to get
the tokens it most promotes/suppresses.

Inputs (env vars):
  LAYER        decoder layer index         (default: 20)
  VECTOR_PAIR  pair id of steering vector  (default: subject_8_vs_subject_0)
  TOP_TOKENS   tokens to show per feature  (default: 20)

Outputs:
  outputs/results/interpret_layer{LAYER}_{vector_pair}.json
"""

import json
import os
import sys

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

script_dir  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from sae_module import TopKSAE

DEVICE     = 'cuda'
MODEL_NAME = 'Qwen/Qwen3-VL-8B-Instruct'

LAYER       = int(os.environ.get('LAYER',       '20'))
VECTOR_PAIR = os.environ.get('VECTOR_PAIR',     'subject_8_vs_subject_0')
TOP_TOKENS  = int(os.environ.get('TOP_TOKENS',  '20'))

CKPT_DIR    = os.path.join(script_dir, 'outputs', 'checkpoints')
RESULTS_DIR = os.path.join(script_dir, 'outputs', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    ckpt_path   = os.path.join(CKPT_DIR, f'layer{LAYER}.sae.pt')
    decomp_path = os.path.join(RESULTS_DIR, f'decompose_layer{LAYER}_{VECTOR_PAIR}.json')
    out_path    = os.path.join(RESULTS_DIR, f'interpret_layer{LAYER}_{VECTOR_PAIR}.json')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'SAE checkpoint not found. Run sae_download first.')
    if not os.path.exists(decomp_path):
        raise FileNotFoundError(f'Decomposition results not found. Run sae_decompose first.')

    if os.path.exists(out_path):
        print(f'Already computed: {out_path}')
        _print_results(out_path)
        return

    with open(decomp_path) as f:
        decomp = json.load(f)

    robust_features = decomp['robust_features']
    top_a_ids       = {idx for idx, _ in decomp['top_features_a'][:50]}
    top_b_ids       = {idx for idx, _ in decomp['top_features_b'][:50]}
    # Also interpret top-10 from each analysis individually
    interpret_ids   = sorted(set(robust_features)
                             | {idx for idx, _ in decomp['top_features_a'][:10]}
                             | {idx for idx, _ in decomp['top_features_b'][:10]})

    print(f'Loading SAE...')
    sae = TopKSAE(ckpt_path, k=100, device=DEVICE)
    sae.eval()

    print(f'Loading model ({MODEL_NAME}) for logit lens...')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map='auto',
        attn_implementation='flash_attention_2',
    )
    processor  = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer  = processor.tokenizer
    lm_head    = model.get_output_embeddings().weight.detach().float()  # (vocab, d)
    # Qwen3-VL nests the text decoder: model.model.language_model.norm
    mm = model.model
    final_norm = mm.language_model.norm if hasattr(mm, 'language_model') else mm.norm

    def feature_top_tokens(feature_idx: int):
        direction = sae.W_dec[:, feature_idx].float()  # (d,)
        with torch.no_grad():
            direction_normed = final_norm(direction.unsqueeze(0)).squeeze(0)
        logits = direction_normed.cpu() @ lm_head.cpu().T  # (vocab,)
        top_pos = logits.topk(TOP_TOKENS)
        top_neg = (-logits).topk(TOP_TOKENS)
        pos_tokens = [tokenizer.decode([i]) for i in top_pos.indices.tolist()]
        neg_tokens = [tokenizer.decode([i]) for i in top_neg.indices.tolist()]
        return {
            'promotes':    list(zip(pos_tokens, [round(v, 4) for v in top_pos.values.tolist()])),
            'suppresses':  list(zip(neg_tokens, [round(v, 4) for v in top_neg.values.tolist()])),
        }

    print(f'Interpreting {len(interpret_ids)} features...')
    interpretations = {}
    for fid in interpret_ids:
        in_a      = fid in top_a_ids
        in_b      = fid in top_b_ids
        is_robust = fid in set(robust_features)
        tokens    = feature_top_tokens(fid)
        interpretations[str(fid)] = {
            'feature_id': fid,
            'robust':     is_robust,
            'in_a':       in_a,
            'in_b':       in_b,
            'score_a':    next((s for i, s in decomp['top_features_a'] if i == fid), None),
            'score_b':    next((s for i, s in decomp['top_features_b'] if i == fid), None),
            **tokens,
        }
        top_p = [t for t, _ in tokens['promotes'][:5]]
        top_s = [t for t, _ in tokens['suppresses'][:5]]
        robust_tag = ' [ROBUST]' if is_robust else ''
        print(f'  Feature {fid}{robust_tag}: +{top_p}  −{top_s}')

    results = {
        'layer':           LAYER,
        'vector_pair':     VECTOR_PAIR,
        'n_robust':        len(robust_features),
        'robust_features': robust_features,
        'features':        interpretations,
    }

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'\nSaved: {out_path}')
    _print_results(out_path)


def _print_results(path):
    with open(path) as f:
        d = json.load(f)
    print(f'\nRobust features ({d["n_robust"]}):')
    for fid in d['robust_features']:
        info = d['features'].get(str(fid), {})
        top_p = [t for t, _ in info.get('promotes', [])[:5]]
        top_s = [t for t, _ in info.get('suppresses', [])[:5]]
        print(f'  {fid:6d}: +{top_p}  −{top_s}')


if __name__ == '__main__':
    main()
