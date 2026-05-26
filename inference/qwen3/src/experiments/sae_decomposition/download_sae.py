"""
Download the Qwen-Scope SAE checkpoint for the target layer.

Inputs (env vars):
  LAYER     decoder layer index  (default: 20)
  SAE_REPO  HuggingFace repo id  (default: Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100)

Output:
  outputs/checkpoints/layer{LAYER}.sae.pt
"""

import os
from huggingface_hub import hf_hub_download

script_dir = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR   = os.path.join(script_dir, 'outputs', 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

LAYER    = int(os.environ.get('LAYER',    '20'))
SAE_REPO = os.environ.get('SAE_REPO', 'Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100')
FILENAME = f'layer{LAYER}.sae.pt'

out_path = os.path.join(CKPT_DIR, FILENAME)
if os.path.exists(out_path):
    print(f'Already downloaded: {out_path}')
else:
    print(f'Downloading {SAE_REPO}/{FILENAME}...')
    hf_hub_download(repo_id=SAE_REPO, filename=FILENAME, local_dir=CKPT_DIR)
    print(f'Saved: {out_path}')

import torch
ckpt = torch.load(out_path, map_location='cpu')
print(f'Keys: {list(ckpt.keys())}')
for k, v in ckpt.items():
    print(f'  {k}: {v.shape}  dtype={v.dtype}')
