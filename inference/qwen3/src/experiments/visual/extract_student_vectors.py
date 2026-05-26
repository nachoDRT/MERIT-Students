"""
Phase 1 (students): Extract the steering vector capturing demographic bias
in student university admission evaluation.

Contrastive pair:
  positive x+: english_male_bad_grades.jpg + subject_8 photo + eval prompt
  negative x-: english_male_bad_grades.jpg + subject_0 photo + eval prompt
  (same grade document, same name, different person photo)

Pair sampling (up to MAX_PAIRS=50):
  All unique (subject_8_i, subject_0_j) index combinations are generated,
  shuffled with a fixed seed, and truncated to MAX_PAIRS. This ensures:
    - No duplicate pairs
    - Each photo appears roughly proportionally (no image dominates)

Data (local):
  /app/data/grades/english_male_bad_grades.jpg
  /app/data/subject_0/  — student photos (positive class)
  /app/data/subject_8/  — student photos (negative class)

Outputs (experiments/visual/outputs/vectors/students/):
  subject_8_vs_subject_0_vector.npy  — [n_layers, d_model]
  student_vectors_meta.json

Usage (inside Docker):
  PIPELINE_STEP=visual_extract_student_vectors bash src/start.sh
"""

import json
import os
import random
import sys

import numpy as np
from PIL import Image

script_dir  = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..', '..')
sys.path.append(project_dir)

from utils.model_utils import load_model_visual
from visual_utils import compute_student_bias_vector

DEVICE     = 'cuda'
MODEL_NAME = 'qwen3-vl-8b-instruct'
MAX_PAIRS  = 50
SEED       = 42

DATA_DIR    = '/app/data'
GRADES_DIR  = os.path.join(DATA_DIR, 'grades')
VECTORS_DIR = os.path.join(script_dir, 'outputs', 'vectors', 'students')
os.makedirs(VECTORS_DIR, exist_ok=True)

GRADE_FILE   = 'english_male_bad_grades.jpg'
POS_SUBJECT  = 'subject_8'
NEG_SUBJECT  = 'subject_0'


def load_subject_images(subject_dir):
    path  = os.path.join(DATA_DIR, subject_dir)
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    imgs  = [Image.open(os.path.join(path, f)) for f in files]
    print(f'  {subject_dir}: {len(imgs)} image(s) — {files}')
    return imgs


def sample_pairs(pos_images, neg_images, max_pairs, seed):
    """
    Generate up to max_pairs unique (i, j) index combinations from
    pos_images × neg_images, shuffled for balanced coverage.
    Returns two lists of PIL images: (positives, negatives).
    """
    all_combos = [(i, j) for i in range(len(pos_images)) for j in range(len(neg_images))]
    random.seed(seed)
    random.shuffle(all_combos)
    selected = all_combos[:max_pairs]
    pos = [pos_images[i] for i, _ in selected]
    neg = [neg_images[j] for _, j in selected]
    return pos, neg


def main():
    print(f'Loading model ({MODEL_NAME})...')
    model, processor = load_model_visual(MODEL_NAME, device=DEVICE)
    model.to(DEVICE)

    pair_id  = f'{POS_SUBJECT}_vs_{NEG_SUBJECT}'
    out_path = os.path.join(VECTORS_DIR, f'{pair_id}_vector.npy')

    if os.path.exists(out_path):
        print(f'[{pair_id}] Already computed, skipping.')
        vec   = np.load(out_path)
        norms = np.linalg.norm(vec, axis=1)
        print(f'  shape={vec.shape}  norm_mean={norms.mean():.4f}')
        return

    print(f'\n{"="*60}')
    print(f'[{pair_id}]  grade: {GRADE_FILE}')

    grades_img = Image.open(os.path.join(GRADES_DIR, GRADE_FILE))
    pos_images = load_subject_images(POS_SUBJECT)
    neg_images = load_subject_images(NEG_SUBJECT)

    pos_sampled, neg_sampled = sample_pairs(pos_images, neg_images, MAX_PAIRS, SEED)
    n_pairs = len(pos_sampled)
    grades_repeated = [grades_img] * n_pairs

    print(f'  Contrastive pairs: {n_pairs} (max={MAX_PAIRS}, seed={SEED})')
    print(f'  x+: {POS_SUBJECT} | x-: {NEG_SUBJECT}')

    vector, activation_norms = compute_student_bias_vector(
        model, processor,
        grades_repeated, pos_sampled, neg_sampled,
        DEVICE,
    )

    np.save(out_path, vector)
    norms = np.linalg.norm(vector, axis=1)

    meta = {
        pair_id: {
            'shape':        list(vector.shape),
            'norm_mean':    float(norms.mean()),
            'norm_min':     float(norms.min()),
            'norm_max':     float(norms.max()),
            'n_pairs':      n_pairs,
            'pos_subject':  POS_SUBJECT,
            'neg_subject':  NEG_SUBJECT,
            'grade_file':   GRADE_FILE,
            'seed':         SEED,
        }
    }

    meta_path = os.path.join(VECTORS_DIR, 'student_vectors_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\nSaved {out_path}')
    print(f'  shape={vector.shape}  norm_mean={norms.mean():.4f}')
    print(f'Metadata: {meta_path}')


if __name__ == '__main__':
    main()
