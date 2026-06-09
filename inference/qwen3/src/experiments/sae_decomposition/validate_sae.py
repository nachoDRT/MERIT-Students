"""
Sanity check: how well does the Qwen3-8B-Base SAE reconstruct
activations from Qwen3-VL-8B-Instruct at the target layer?

Captures residual-stream activations at LAYER on a batch of diverse
text prompts (no images — keeps the distribution closer to what the
SAE saw during training), then computes reconstruction quality.

Inputs (env vars):
  LAYER       decoder layer index        (default: 20)
  N_PROMPTS   number of text prompts     (default: 50)

Outputs:
  outputs/results/validate_layer{LAYER}.json
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

DEVICE     = 'cuda'
MODEL_NAME = 'Qwen/Qwen3-VL-8B-Instruct'
CKPT_DIR   = os.path.join(script_dir, 'outputs', 'checkpoints')
RESULTS_DIR = os.path.join(script_dir, 'outputs', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

LAYER     = int(os.environ.get('LAYER',     '20'))
N_PROMPTS = int(os.environ.get('N_PROMPTS', '50'))

PROMPTS = [
    "The student applied to university.",
    "She graduated with honors from secondary school.",
    "His grades were below the passing threshold.",
    "The admissions committee reviewed the application.",
    "Academic performance is evaluated on a 0-10 scale.",
    "The candidate showed strong results in mathematics.",
    "Vocational training is an alternative to university.",
    "The teacher recommended him for the advanced program.",
    "Her transcript included failing marks in several subjects.",
    "University admission requires meeting minimum grade requirements.",
    "The interview assessed communication and analytical skills.",
    "He decided to defer his university enrollment by one year.",
    "Social background should not influence academic evaluation.",
    "The scholarship was awarded based on merit alone.",
    "Implicit bias can affect evaluation outcomes.",
    "The photograph on the application form was reviewed.",
    "Diversity in higher education benefits society.",
    "The evaluation was conducted blindly to reduce bias.",
    "Appearance should not factor into academic decisions.",
    "The committee was trained to recognize unconscious bias.",
] * (N_PROMPTS // 20 + 1)
PROMPTS = PROMPTS[:N_PROMPTS]


def get_decoder_layers(model):
    mm = model.model
    if hasattr(mm, 'language_model') and hasattr(mm.language_model, 'layers'):
        return mm.language_model.layers
    return mm.layers


def capture_activations(model, processor, prompts, layer_idx, device):
    acts = []
    layers = get_decoder_layers(model)

    for prompt in prompts:
        messages  = [{"role": "user", "content": prompt}]
        text      = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs    = processor(text=[text], return_tensors="pt").to(device)

        captured = {}

        def hook(module, inp, out):
            hs = out if isinstance(out, torch.Tensor) else out[0]
            captured['h'] = hs.detach().float()

        handle = layers[layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        # All token positions
        acts.append(captured['h'].squeeze(0))  # (seq_len, d_model)

    return torch.cat(acts, dim=0)  # (total_tokens, d_model)


def main():
    ckpt_path = os.path.join(CKPT_DIR, f'layer{LAYER}.sae.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'SAE checkpoint not found: {ckpt_path}\nRun sae_download first.')

    out_path = os.path.join(RESULTS_DIR, f'validate_layer{LAYER}.json')
    if os.path.exists(out_path):
        print(f'Already computed: {out_path}')
        with open(out_path) as f:
            print(json.dumps(json.load(f), indent=2))
        return

    print(f'Loading model ({MODEL_NAME})...')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map='auto',
        attn_implementation='flash_attention_2',
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print(f'Loading SAE (layer {LAYER})...')
    sae = TopKSAE(ckpt_path, k=100, device=DEVICE)
    sae.eval()

    print(f'Capturing activations at layer {LAYER} over {N_PROMPTS} prompts...')
    acts = capture_activations(model, processor, PROMPTS, LAYER, DEVICE)  # (N, 4096)
    print(f'  Activations shape: {acts.shape}')

    acts_gpu = acts.to(DEVICE)
    with torch.no_grad():
        recon = sae(acts_gpu).cpu()

    acts_cpu = acts.cpu()
    mse          = F.mse_loss(recon, acts_cpu).item()
    var_expl     = explained_variance(acts_cpu, recon)
    recon_fid    = reconstruction_fidelity(acts_cpu, recon)
    cos_sim      = F.cosine_similarity(acts_cpu, recon, dim=-1).mean().item()
    l0_effective = (sae.encode(acts_gpu) > 0).float().sum(-1).mean().item()

    results = {
        'layer':         LAYER,
        'n_prompts':     N_PROMPTS,
        'n_tokens':      int(acts.shape[0]),
        'mse':           round(mse, 6),
        'var_explained': round(var_expl, 4),
        'recon_fidelity': round(recon_fid, 4),
        'cos_sim':       round(cos_sim, 4),
        'l0_effective':  round(l0_effective, 2),
        'verdict':       (
            'RELIABLE'  if var_expl > 0.7 else
            'USABLE'    if var_expl > 0.5 else
            'UNRELIABLE — stop and reassess'
        ),
    }

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{"="*50}')
    print(json.dumps(results, indent=2))
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
