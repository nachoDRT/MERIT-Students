"""
Cache last-token activations per contrastive condition at target layers.

Runs Qwen3-VL over the same 50 pairs used for CAA vector extraction
(SEED=42, subject_8 vs subject_0) and saves h_pos / h_neg.

Outputs (outputs/activations/):
  h_pos.npy  — [N_pairs, len(TARGET_LAYERS), d_model]
  h_neg.npy  — [N_pairs, len(TARGET_LAYERS), d_model]
  meta.json
"""

import json
import os
import random
import sys

import numpy as np
from PIL import Image

script_dir  = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_dir)

from utils.model_utils import load_model_visual
from experiments.visual.visual_utils import extract_representation_student

TARGET_LAYERS = [16, 17, 19, 20]
DEVICE        = 'cuda'
MODEL_NAME    = 'qwen3-vl-8b-instruct'
MAX_PAIRS     = 50
SEED          = 42
DATA_DIR      = '/app/data'
POS_SUBJECT   = 'subject_8'
NEG_SUBJECT   = 'subject_0'
GRADE_FILE    = 'english_male_bad_grades.jpg'

ACT_DIR  = os.path.join(script_dir, 'outputs', 'activations')
os.makedirs(ACT_DIR, exist_ok=True)

OUT_POS  = os.path.join(ACT_DIR, 'h_pos.npy')
OUT_NEG  = os.path.join(ACT_DIR, 'h_neg.npy')
OUT_META = os.path.join(ACT_DIR, 'meta.json')


def load_subject_images(subject_dir):
    path  = os.path.join(DATA_DIR, subject_dir)
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    return [Image.open(os.path.join(path, f)) for f in files]


def sample_pairs(pos_imgs, neg_imgs, max_pairs, seed):
    combos = [(i, j) for i in range(len(pos_imgs)) for j in range(len(neg_imgs))]
    random.seed(seed)
    random.shuffle(combos)
    sel = combos[:max_pairs]
    return [pos_imgs[i] for i, _ in sel], [neg_imgs[j] for _, j in sel]


def main():
    if os.path.exists(OUT_POS) and os.path.exists(OUT_NEG):
        print(f'Already cached: {OUT_POS}')
        h_pos = np.load(OUT_POS)
        h_neg = np.load(OUT_NEG)
        print(f'  h_pos {h_pos.shape}  h_neg {h_neg.shape}')
        return

    print('Loading model...')
    model, processor = load_model_visual(MODEL_NAME, device=DEVICE)
    model.to(DEVICE)

    grades_img = Image.open(os.path.join(DATA_DIR, 'grades', GRADE_FILE))
    pos_imgs   = load_subject_images(POS_SUBJECT)
    neg_imgs   = load_subject_images(NEG_SUBJECT)

    pos_sampled, neg_sampled = sample_pairs(pos_imgs, neg_imgs, MAX_PAIRS, SEED)
    n_pairs = len(pos_sampled)
    print(f'Pairs: {n_pairs}  target_layers={TARGET_LAYERS}')

    h_pos_list, h_neg_list = [], []

    for i, (pos_img, neg_img) in enumerate(zip(pos_sampled, neg_sampled)):
        print(f'  pair {i+1}/{n_pairs}...', flush=True)

        # extract_representation_student returns [n_layers, d_model]; index target layers
        h_pos_all = extract_representation_student(model, processor, grades_img, pos_img, DEVICE)
        h_pos_list.append(np.stack([h_pos_all[l] for l in TARGET_LAYERS]))

        h_neg_all = extract_representation_student(model, processor, grades_img, neg_img, DEVICE)
        h_neg_list.append(np.stack([h_neg_all[l] for l in TARGET_LAYERS]))

    h_pos = np.stack(h_pos_list)  # [N, L, d]
    h_neg = np.stack(h_neg_list)  # [N, L, d]

    np.save(OUT_POS, h_pos)
    np.save(OUT_NEG, h_neg)

    meta = {
        'n_pairs':       n_pairs,
        'target_layers': TARGET_LAYERS,
        'd_model':       int(h_pos.shape[-1]),
        'pos_subject':   POS_SUBJECT,
        'neg_subject':   NEG_SUBJECT,
        'grade_file':    GRADE_FILE,
        'seed':          SEED,
        'shape_pos':     list(h_pos.shape),
        'shape_neg':     list(h_neg.shape),
    }
    with open(OUT_META, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\nSaved h_pos {h_pos.shape} -> {OUT_POS}')
    print(f'Saved h_neg {h_neg.shape} -> {OUT_NEG}')


if __name__ == '__main__':
    main()
