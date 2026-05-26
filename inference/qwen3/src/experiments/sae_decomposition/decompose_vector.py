"""
Decompose the steering vector via the Qwen-Scope SAE.

Analysis A — encoder + TopK:
  Treat v as if it were an activation; find which features fire.

Analysis B — cosine with decoder directions:
  Find which feature directions in W_dec are most aligned with v.
  Distribution-robust: doesn't depend on the encoder.

Robust features: intersection of top-50 from both analyses.

Inputs (env vars):
  LAYER        decoder layer index        (default: 20)
  VECTOR_PAIR  pair id of steering vector (default: subject_8_vs_subject_0)
  TOP_N        number of top features     (default: 50)

Outputs:
  outputs/results/decompose_layer{LAYER}_{vector_pair}.json
"""

import json
import os
import sys

import numpy as np
import torch

script_dir  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from sae_module import TopKSAE

DEVICE = 'cuda'

LAYER       = int(os.environ.get('LAYER',   '20'))
VECTOR_PAIR = os.environ.get('VECTOR_PAIR', 'subject_8_vs_subject_0')
TOP_N       = int(os.environ.get('TOP_N',   '50'))

CKPT_DIR    = os.path.join(script_dir, 'outputs', 'checkpoints')
RESULTS_DIR = os.path.join(script_dir, 'outputs', 'results')
VECTORS_DIR = os.path.join(script_dir, '..', 'visual', 'outputs', 'vectors', 'students')
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    ckpt_path   = os.path.join(CKPT_DIR, f'layer{LAYER}.sae.pt')
    vector_path = os.path.join(VECTORS_DIR, f'{VECTOR_PAIR}_vector.npy')
    out_path    = os.path.join(RESULTS_DIR, f'decompose_layer{LAYER}_{VECTOR_PAIR}.json')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'SAE checkpoint not found: {ckpt_path}')
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f'Steering vector not found: {vector_path}')

    if os.path.exists(out_path):
        print(f'Already computed: {out_path}')
        with open(out_path) as f:
            d = json.load(f)
        print(f"Robust features ({len(d['robust_features'])}): {d['robust_features'][:10]}...")
        return

    # Load steering vector for this layer
    all_layers = np.load(vector_path)          # (n_layers, d_model)
    v_np       = all_layers[LAYER]             # (d_model,)
    v          = torch.from_numpy(v_np).float().to(DEVICE)
    print(f'Steering vector layer {LAYER}: shape={v.shape}  norm={v.norm():.4f}')

    print(f'Loading SAE...')
    sae = TopKSAE(ckpt_path, k=100, device=DEVICE)
    sae.eval()

    # ---- Analysis A: encoder + TopK ----
    with torch.no_grad():
        features_topk = sae.encode(v.unsqueeze(0), apply_topk=True).squeeze(0)  # (D,)

    nonzero_idx = features_topk.nonzero(as_tuple=True)[0]
    top_a = sorted(
        [(int(i), float(features_topk[i])) for i in nonzero_idx],
        key=lambda x: -abs(x[1])
    )[:TOP_N]
    top_a_ids = {idx for idx, _ in top_a}

    print(f'Analysis A (encoder TopK): {len(top_a)} features active')

    # ---- Analysis B: cosine with W_dec columns ----
    # W_dec: (d_model, D) — columns are feature directions in model space
    W_dec     = sae.W_dec                                    # (d, D)
    col_norms = W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
    dec_dirs  = W_dec / col_norms                            # (d, D) normalised

    v_norm    = v / v.norm().clamp(min=1e-8)                 # (d,)
    cos_v_dec = (v_norm @ dec_dirs).cpu()                    # (D,)

    top_b_vals, top_b_idx = cos_v_dec.abs().topk(TOP_N)
    top_b = [(int(i), float(cos_v_dec[i])) for i in top_b_idx.tolist()]
    top_b_ids = {idx for idx, _ in top_b}

    print(f'Analysis B (decoder cosine): top cos_sim={top_b_vals[0]:.4f}')

    # ---- Robust: intersection ----
    robust = sorted(top_a_ids & top_b_ids)
    print(f'Robust features (in both top-{TOP_N}): {len(robust)}  → {robust[:20]}')

    # Reconstruction quality of v through SAE
    with torch.no_grad():
        v_recon     = sae(v.unsqueeze(0)).squeeze(0)
    recon_norm_ratio = float(v_recon.norm() / v.norm())
    recon_cos        = float(torch.nn.functional.cosine_similarity(
        v.unsqueeze(0), v_recon.unsqueeze(0)
    ).item())

    results = {
        'layer':               LAYER,
        'vector_pair':         VECTOR_PAIR,
        'v_norm':              float(v.norm()),
        'recon_norm_ratio':    round(recon_norm_ratio, 4),
        'recon_cos_sim':       round(recon_cos, 4),
        'top_n':               TOP_N,
        'top_features_a':      top_a,
        'top_features_b':      top_b,
        'robust_features':     robust,
        'n_robust':            len(robust),
    }

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nSaved: {out_path}')
    print(f'  recon_cos_sim={recon_cos:.4f}  recon_norm_ratio={recon_norm_ratio:.4f}')


if __name__ == '__main__':
    main()
