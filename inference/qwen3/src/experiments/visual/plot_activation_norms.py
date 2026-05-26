"""
Plot per-layer L2 norms of:
  1. Positive activations  (grades + subject_8 photo + eval prompt)
  2. Negative activations  (grades + subject_0 photo + eval prompt)
  3. Steering vector       (mean_diff saved from extract_student_vectors)

Runs K forward passes (default 5) to estimate activation norms — much faster
than rerunning all 50 pairs.

Inputs (env vars):
  K_SAMPLES    number of pairs to average over  (default: 5)
  VECTOR_PAIR  pair id  (default: "subject_8_vs_subject_0")

Output:
  outputs/plots/activation_norms_{vector_pair}.png
"""

import os
import sys
import random

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

script_dir  = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..', '..')
sys.path.append(project_dir)

from visual_utils import extract_representation_student, STUDENT_EVAL_PROMPT

DEVICE     = 'cuda'
MODEL_NAME = 'Qwen/Qwen3-VL-8B-Instruct'
DATA_DIR   = '/app/data'
SEED       = 42

VECTORS_DIR = os.path.join(script_dir, 'outputs', 'vectors', 'students')
PLOTS_DIR   = os.path.join(script_dir, 'outputs', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_images(subject, exts=('.jpg', '.jpeg', '.png')):
    path  = os.path.join(DATA_DIR, subject)
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(exts))
    return [Image.open(os.path.join(path, f)) for f in files]


def sample_pairs(pos_imgs, neg_imgs, k, seed):
    combos = [(i, j) for i in range(len(pos_imgs)) for j in range(len(neg_imgs))]
    random.seed(seed)
    random.shuffle(combos)
    selected = combos[:k]
    return [pos_imgs[i] for i, _ in selected], [neg_imgs[j] for _, j in selected]


def main():
    k           = int(os.environ.get('K_SAMPLES',   '5'))
    vector_pair = os.environ.get('VECTOR_PAIR', 'subject_8_vs_subject_0')

    pos_subject, neg_subject = vector_pair.split('_vs_')

    vec_path = os.path.join(VECTORS_DIR, f'{vector_pair}_vector.npy')
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f'Vector not found: {vec_path}')
    steering_vec = np.load(vec_path)          # [n_layers, d_model]
    vec_norms    = np.linalg.norm(steering_vec, axis=1)
    n_layers     = steering_vec.shape[0]

    print(f'Loading model ({MODEL_NAME})...')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map='auto',
        attn_implementation='flash_attention_2',
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    grades_img  = Image.open(os.path.join(DATA_DIR, 'grades', 'english_male_bad_grades.jpg'))
    pos_imgs    = load_images(pos_subject)
    neg_imgs    = load_images(neg_subject)
    pos_sampled, neg_sampled = sample_pairs(pos_imgs, neg_imgs, k, SEED)

    print(f'Running {k} pairs ({pos_subject} vs {neg_subject})...')
    pos_norms_all = []
    neg_norms_all = []

    for i, (pos_img, neg_img) in enumerate(zip(pos_sampled, neg_sampled)):
        print(f'  pair {i+1}/{k}...', flush=True)
        h_pos = extract_representation_student(model, processor, grades_img, pos_img, DEVICE)
        h_neg = extract_representation_student(model, processor, grades_img, neg_img, DEVICE)
        pos_norms_all.append(np.linalg.norm(h_pos, axis=1))   # [n_layers]
        neg_norms_all.append(np.linalg.norm(h_neg, axis=1))

    pos_norms = np.stack(pos_norms_all).mean(axis=0)   # [n_layers]
    neg_norms = np.stack(neg_norms_all).mean(axis=0)

    layers = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(layers, pos_norms, label=f'Activation norm — {pos_subject} (positive)',
            color='steelblue', linewidth=1.8)
    ax.plot(layers, neg_norms, label=f'Activation norm — {neg_subject} (negative)',
            color='tomato', linewidth=1.8)
    ax.plot(layers, vec_norms, label='Steering vector norm (mean diff)',
            color='forestgreen', linewidth=1.8, linestyle='--')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('L2 norm', fontsize=12)
    ax.set_title(
        f'Per-layer activation norms vs steering vector norm\n'
        f'{vector_pair}  |  K={k} pairs  |  {MODEL_NAME}',
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f'activation_norms_{vector_pair}.png')
    plt.savefig(out_path, dpi=150)
    print(f'\nSaved: {out_path}')

    print(f'\nPer-layer summary (mean over {k} pairs):')
    print(f"{'Layer':>5}  {'pos_norm':>10}  {'neg_norm':>10}  {'vec_norm':>10}  {'ratio pos':>10}")
    for l in range(n_layers):
        ratio = vec_norms[l] / pos_norms[l] if pos_norms[l] > 0 else 0
        print(f'{l:5d}  {pos_norms[l]:10.2f}  {neg_norms[l]:10.2f}  '
              f'{vec_norms[l]:10.2f}  {ratio:10.4f}')


if __name__ == '__main__':
    main()
