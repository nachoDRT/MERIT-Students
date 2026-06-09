"""
Analyze SAE decomposition of per-condition activations.

Instead of decomposing v (off-manifold difference-of-means), decompose
the individual activations h_pos (subject_8) and h_neg (subject_0) and
compare features via delta = mean(features_pos) - mean(features_neg).

Two-phase execution to avoid OOM:
  Phase 1 — SAE per layer (no Qwen model): encode activations, compute
             delta / exclusive features, collect decoder directions.
  Phase 2 — Qwen model (no SAE): logit lens on collected directions.

Requires: cache_activations.py completed first.

Outputs (outputs/results/):
  conditions_layer{N}_{VECTOR_PAIR}.json  for each layer in TARGET_LAYERS
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

script_dir  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from sae_module import TopKSAE, explained_variance, reconstruction_fidelity

DEVICE      = 'cuda'
MODEL_NAME  = 'Qwen/Qwen3-VL-8B-Instruct'
VECTOR_PAIR = 'subject_8_vs_subject_0'

TOP_N            = 10
EXCLUSIVE_THRESH = 0.01
TOP_TOKENS       = 15

CKPT_DIR    = os.path.join(script_dir, 'outputs', 'checkpoints')
RESULTS_DIR = os.path.join(script_dir, 'outputs', 'results')
ACT_DIR     = os.path.join(script_dir, 'outputs', 'activations')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_activations():
    h_pos = np.load(os.path.join(ACT_DIR, 'h_pos.npy'))  # [N, L, d]
    h_neg = np.load(os.path.join(ACT_DIR, 'h_neg.npy'))  # [N, L, d]
    with open(os.path.join(ACT_DIR, 'meta.json')) as f:
        meta = json.load(f)
    return h_pos, h_neg, meta


def sae_phase(h_pos, h_neg, meta, layers_to_run):
    """
    For each layer: encode activations, compute delta, collect feature directions.
    Returns per-layer analysis dicts and {layer: {fid: cpu tensor}} for logit lens.
    """
    layer_list   = meta['target_layers']
    per_layer    = {}
    feature_dirs = {}

    for layer in layers_to_run:
        layer_idx   = layer_list.index(layer)
        ckpt_path   = os.path.join(CKPT_DIR, f'layer{layer}.sae.pt')
        decomp_path = os.path.join(RESULTS_DIR, f'decompose_layer{layer}_{VECTOR_PAIR}.json')

        print(f'\n[Layer {layer}] Loading SAE...')
        sae = TopKSAE(ckpt_path, k=100, device=DEVICE)
        sae.eval()

        h_p = torch.from_numpy(h_pos[:, layer_idx, :]).float().to(DEVICE)  # [N, d]
        h_n = torch.from_numpy(h_neg[:, layer_idx, :]).float().to(DEVICE)  # [N, d]

        with torch.no_grad():
            acts_all  = torch.cat([h_p, h_n], dim=0)
            recon_all = sae(acts_all)
            # Activations here are concentrated (same prompt, only photo differs),
            # so the centered R² collapses negative — recon_fidelity is the
            # metric to trust for reconstruction quality. See sae_module docs.
            var_explained = explained_variance(acts_all, recon_all)
            recon_fid     = reconstruction_fidelity(acts_all, recon_all)
            cos_sim       = F.cosine_similarity(acts_all, recon_all, dim=-1).mean().item()
            l0_efectivo   = (sae.encode(acts_all, apply_topk=True) > 0).float().sum(-1).mean().item()
        print(f'  SAE recon: fidelity={recon_fid:.4f}  cos_sim={cos_sim:.4f}  R²_centered={var_explained:.2f}  L0={l0_efectivo:.1f}')

        with torch.no_grad():
            feat_pos = sae.encode(h_p, apply_topk=True)  # [N, D]
            feat_neg = sae.encode(h_n, apply_topk=True)  # [N, D]

        mean_pos = feat_pos.mean(dim=0)  # [D]
        mean_neg = feat_neg.mean(dim=0)  # [D]
        delta    = mean_pos - mean_neg   # [D]

        top_delta_pos_idx = delta.topk(TOP_N).indices
        top_delta_neg_idx = (-delta).topk(TOP_N).indices
        top_delta_pos = [(int(i), float(delta[i])) for i in top_delta_pos_idx.tolist()]
        top_delta_neg = [(int(i), float(delta[i])) for i in top_delta_neg_idx.tolist()]

        TOP_MEAN = 20
        top_mean_pos_idx = mean_pos.topk(TOP_MEAN).indices
        top_mean_neg_idx = mean_neg.topk(TOP_MEAN).indices
        top_mean_pos = [(int(i), float(mean_pos[i])) for i in top_mean_pos_idx.tolist()]
        top_mean_neg = [(int(i), float(mean_neg[i])) for i in top_mean_neg_idx.tolist()]

        excl_pos_mask = (mean_pos > EXCLUSIVE_THRESH) & (mean_neg < EXCLUSIVE_THRESH)
        excl_neg_mask = (mean_neg > EXCLUSIVE_THRESH) & (mean_pos < EXCLUSIVE_THRESH)
        excl_pos_sorted = sorted(
            excl_pos_mask.nonzero(as_tuple=True)[0].tolist(), key=lambda i: -float(mean_pos[i])
        )
        excl_neg_sorted = sorted(
            excl_neg_mask.nonzero(as_tuple=True)[0].tolist(), key=lambda i: -float(mean_neg[i])
        )

        coherence = {}
        if os.path.exists(decomp_path):
            with open(decomp_path) as f:
                decomp = json.load(f)
            top_v_ids    = {idx for idx, _ in decomp['top_features_a'][:50]}
            our_pos_ids  = {idx for idx, _ in top_delta_pos}
            our_neg_ids  = {idx for idx, _ in top_delta_neg}
            coherence = {
                'overlap_v_delta_pos':    list(top_v_ids & our_pos_ids),
                'overlap_v_delta_neg':    list(top_v_ids & our_neg_ids),
                'robust_features_from_v': decomp.get('robust_features', []),
            }
            print(f'  Coherence: v∩delta_pos={len(coherence["overlap_v_delta_pos"])}  '
                  f'v∩delta_neg={len(coherence["overlap_v_delta_neg"])}')

        interpret_ids = set()
        interpret_ids.update(idx for idx, _ in top_delta_pos)
        interpret_ids.update(idx for idx, _ in top_delta_neg)
        interpret_ids.update(idx for idx, _ in top_mean_pos)
        interpret_ids.update(idx for idx, _ in top_mean_neg)
        interpret_ids.update(excl_neg_sorted[:10])
        interpret_ids.update(excl_pos_sorted[:10])
        if coherence:
            interpret_ids.update(coherence['robust_features_from_v'])

        feature_dirs[layer] = {
            fid: sae.W_dec[:, fid].cpu().float() for fid in interpret_ids
        }

        print(f'  top_delta_pos: {[i for i,_ in top_delta_pos[:5]]}')
        print(f'  top_delta_neg: {[i for i,_ in top_delta_neg[:5]]}')
        print(f'  excl_pos={len(excl_pos_sorted)}  excl_neg={len(excl_neg_sorted)}')
        print(f'  Features queued for logit lens: {len(interpret_ids)}')

        per_layer[layer] = {
            'layer':                layer,
            'vector_pair':          VECTOR_PAIR,
            'n_pairs':              int(h_pos.shape[0]),
            'sae_validation':       {
                'recon_fidelity': round(recon_fid, 4),
                'var_explained':  round(var_explained, 4),
                'cos_sim':        round(cos_sim, 4),
                'l0_efectivo':    round(l0_efectivo, 1),
            },
            'top_delta_pos':        top_delta_pos,
            'top_delta_neg':        top_delta_neg,
            'top_mean_pos':         top_mean_pos,
            'top_mean_neg':         top_mean_neg,
            'exclusive_pos':        [
                (int(i), float(mean_pos[i]), float(mean_neg[i])) for i in excl_pos_sorted[:20]
            ],
            'exclusive_neg':        [
                (int(i), float(mean_neg[i]), float(mean_pos[i])) for i in excl_neg_sorted[:20]
            ],
            'exclusive_pos_count':  len(excl_pos_sorted),
            'exclusive_neg_count':  len(excl_neg_sorted),
            'coherence':            coherence,
            'features':             {},
        }

        del sae, feat_pos, feat_neg, mean_pos, mean_neg, delta, h_p, h_n
        torch.cuda.empty_cache()

    return per_layer, feature_dirs


def logit_lens_phase(per_layer, feature_dirs):
    """Load Qwen model once, compute logit lens for all collected feature directions."""
    print('\nLoading Qwen model for logit lens...')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map='auto',
        attn_implementation='flash_attention_2',
    )
    processor  = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer  = processor.tokenizer
    lm_head    = model.get_output_embeddings().weight.detach().float().cpu()
    mm         = model.model
    final_norm = mm.language_model.norm if hasattr(mm, 'language_model') else mm.norm

    for layer, fids_dirs in feature_dirs.items():
        print(f'  [Layer {layer}] logit lens for {len(fids_dirs)} features...')
        for fid, direction in fids_dirs.items():
            direction = direction.to(DEVICE)
            with torch.no_grad():
                normed = final_norm(direction.unsqueeze(0)).squeeze(0)
            logits  = normed.cpu() @ lm_head.T
            top_pos = logits.topk(TOP_TOKENS)
            top_neg = (-logits).topk(TOP_TOKENS)
            per_layer[layer]['features'][str(fid)] = {
                'feature_id': fid,
                'promotes':  list(zip(
                    [tokenizer.decode([i]) for i in top_pos.indices.tolist()],
                    [round(v, 4) for v in top_pos.values.tolist()],
                )),
                'suppresses': list(zip(
                    [tokenizer.decode([i]) for i in top_neg.indices.tolist()],
                    [round(v, 4) for v in top_neg.values.tolist()],
                )),
            }

    del model
    torch.cuda.empty_cache()
    return per_layer


def print_summary(per_layer):
    for layer, result in per_layer.items():
        print(f'\n[Layer {layer}]')
        print(f'  Exclusive pos (subject_8): {result["exclusive_pos_count"]}')
        print(f'  Exclusive neg (subject_0): {result["exclusive_neg_count"]}')
        if result['coherence']:
            n_ov = len(result['coherence']['overlap_v_delta_pos'])
            print(f'  Overlap v∩delta_pos: {n_ov}')
        print('  Top delta_pos (subject_8 > subject_0):')
        for fid, val in result['top_delta_pos'][:5]:
            info  = result['features'].get(str(fid), {})
            top_p = [t for t, _ in info.get('promotes', [])[:4]]
            top_s = [t for t, _ in info.get('suppresses', [])[:4]]
            print(f'    F{fid:6d}  Δ={val:+.4f}  +{top_p}  −{top_s}')
        print('  Top delta_neg (subject_0 > subject_8):')
        for fid, val in result['top_delta_neg'][:5]:
            info  = result['features'].get(str(fid), {})
            top_p = [t for t, _ in info.get('promotes', [])[:4]]
            top_s = [t for t, _ in info.get('suppresses', [])[:4]]
            print(f'    F{fid:6d}  Δ={val:+.4f}  +{top_p}  −{top_s}')


def main():
    act_pos_path = os.path.join(ACT_DIR, 'h_pos.npy')
    act_neg_path = os.path.join(ACT_DIR, 'h_neg.npy')
    if not os.path.exists(act_pos_path) or not os.path.exists(act_neg_path):
        raise FileNotFoundError('Run cache_activations first: outputs/activations/ missing.')

    h_pos, h_neg, meta = load_activations()
    print(f'Loaded activations: h_pos {h_pos.shape}  h_neg {h_neg.shape}')

    target_layers = meta['target_layers']
    missing = [
        l for l in target_layers
        if not os.path.exists(
            os.path.join(RESULTS_DIR, f'conditions_layer{l}_{VECTOR_PAIR}.json')
        )
    ]

    if not missing:
        print('All layers already computed.')
        completed = {
            l: json.load(open(os.path.join(RESULTS_DIR, f'conditions_layer{l}_{VECTOR_PAIR}.json')))
            for l in target_layers
        }
        print_summary(completed)
        return

    print(f'Layers to process: {missing}')
    per_layer, feature_dirs = sae_phase(h_pos, h_neg, meta, layers_to_run=missing)
    per_layer = logit_lens_phase(per_layer, feature_dirs)

    for layer, result in per_layer.items():
        out_path = os.path.join(RESULTS_DIR, f'conditions_layer{layer}_{VECTOR_PAIR}.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f'Saved: {out_path}')

    print('\n' + '=' * 60)
    print_summary(per_layer)


if __name__ == '__main__':
    main()
