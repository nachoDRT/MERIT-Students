"""
Verdict-filtered contrastive steering vectors: non-successful subject -> subject_8.

Same methodology as extract_student_vectors.py (contrastive pairs, per-pair
activation difference, averaged over up to MAX_PAIRS), but with two changes:

  * images are filtered by the bias-sweep verdict, so the vector is "pure":
      positive pool : POS_SUBJECT (subject_8) images judged proceed=True
      negative pool : subject X images judged proceed=False
  * one vector per target subject X (negative class is X, not a fixed subject_0)

Pair sampling (mirrors the original):
  All unique (pos_i, neg_j) index combinations from the filtered pools are
  generated, shuffled with a fixed seed, and truncated to MAX_PAIRS. If fewer
  combinations than MAX_PAIRS exist, all of them are used. For each pair the
  difference h_pos - h_neg is computed [n_layers, d_model]; the steering vector
  is the mean over the pairs. The number of pairs actually used is recorded
  per subject in the metadata.

Verdict labels are read from the bias sweep results (bias_<subject>.json under
BIAS_RESULTS_DIR).

Env:
  POS_SUBJECT        default subject_8
  SUBJECTS           target subjects, space/comma list (ids or subject_<id>)
                     default: 2 3 7 11 12
  GRADE_FILE         default english_male_bad_grades.jpg
  MAX_PAIRS          default 50
  SEED               default 42
  BIAS_RESULTS_DIR   default /app/src/outputs/students_bias

Outputs (experiments/visual/outputs/vectors/students/):
  {subject}_neg_to_{POS_SUBJECT}_pos_vector.npy
  filtered_vectors_meta.json

Usage (inside Docker):
  PIPELINE_STEP=extract_filtered_vectors bash src/start.sh
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
MAX_PAIRS  = int(os.environ.get('MAX_PAIRS', '50'))
SEED       = int(os.environ.get('SEED', '42'))

DATA_DIR         = '/app/data'
GRADES_DIR       = os.path.join(DATA_DIR, 'grades')
VECTORS_DIR      = os.path.join(script_dir, 'outputs', 'vectors', 'students')
BIAS_RESULTS_DIR = os.environ.get('BIAS_RESULTS_DIR', '/app/src/outputs/students_bias')
os.makedirs(VECTORS_DIR, exist_ok=True)


def normalize(subject):
    subject = str(subject).strip()
    return subject if subject.startswith('subject_') else f'subject_{subject}'


def verdict_images(subject, want_proceed):
    """Sorted image filenames for `subject` whose bias-sweep verdict == want_proceed."""
    bias_path = os.path.join(BIAS_RESULTS_DIR, f'bias_{subject}.json')
    per = json.load(open(bias_path))['per_image']
    return sorted(p['image'] for p in per if p.get('proceed') is want_proceed)


def load_images(subject, names):
    subj_dir = os.path.join(DATA_DIR, subject)
    return [Image.open(os.path.join(subj_dir, n)) for n in names]


# TODO(cleanup): sample_pairs is duplicated verbatim across extract_student_vectors.py,
# plot_activation_norms.py and cache_activations.py. Consolidate into visual_utils.py
# in a future refactor (kept local for now to match the repo's standalone-script idiom).
def sample_pairs(pos_images, neg_images, max_pairs, seed):
    """Up to max_pairs unique (i, j) index combos, shuffled for balanced coverage."""
    all_combos = [(i, j) for i in range(len(pos_images)) for j in range(len(neg_images))]
    random.seed(seed)
    random.shuffle(all_combos)
    selected = all_combos[:max_pairs]
    pos = [pos_images[i] for i, _ in selected]
    neg = [neg_images[j] for _, j in selected]
    return pos, neg


def main():
    pos_subject = normalize(os.environ.get('POS_SUBJECT', 'subject_8'))
    grade_file  = os.environ.get('GRADE_FILE', 'english_male_bad_grades.jpg')
    targets_env = os.environ.get('SUBJECTS', '2 3 7 11 12').replace(',', ' ').split()
    targets     = [normalize(t) for t in targets_env]

    print(f'Loading model ({MODEL_NAME})...')
    model, processor = load_model_visual(MODEL_NAME, device=DEVICE)
    model.to(DEVICE)

    grades_img = Image.open(os.path.join(GRADES_DIR, grade_file))

    pos_names  = verdict_images(pos_subject, True)      # subject_8 University images
    pos_images = load_images(pos_subject, pos_names)
    print(f'[positive pool] {pos_subject}: {len(pos_images)} proceed=True image(s)')

    meta = {}
    for subject in targets:
        neg_names  = verdict_images(subject, False)     # subject X no-University images
        neg_images = load_images(subject, neg_names)

        pos_sampled, neg_sampled = sample_pairs(pos_images, neg_images, MAX_PAIRS, SEED)
        n_pairs = len(pos_sampled)
        n_possible = len(pos_images) * len(neg_images)
        print(f'\n[{subject}] neg pool={len(neg_images)}  '
              f'possible pairs={n_possible}  using {n_pairs} (max={MAX_PAIRS})')

        grades_repeated = [grades_img] * n_pairs
        vector, _ = compute_student_bias_vector(
            model, processor, grades_repeated, pos_sampled, neg_sampled, DEVICE,
        )

        pair_id  = f'{subject}_neg_to_{pos_subject}_pos'
        out_path = os.path.join(VECTORS_DIR, f'{pair_id}_vector.npy')
        np.save(out_path, vector)

        norms = np.linalg.norm(vector, axis=1)
        meta[pair_id] = {
            'shape':           list(vector.shape),
            'norm_mean':       float(norms.mean()),
            'norm_min':        float(norms.min()),
            'norm_max':        float(norms.max()),
            'pos_subject':     pos_subject,
            'neg_subject':     subject,
            'n_pos_pool':      len(pos_images),
            'n_neg_pool':      len(neg_images),
            'n_pairs':         n_pairs,
            'n_possible_pairs': n_possible,
            'max_pairs':       MAX_PAIRS,
            'seed':            SEED,
            'grade_file':      grade_file,
            'method':          'verdict_filtered_contrastive_pairs',
        }
        print(f'[{subject}] saved {out_path}  shape={vector.shape}  '
              f'n_pairs={n_pairs}  norm_mean={norms.mean():.4f}')

    meta_path = os.path.join(VECTORS_DIR, 'filtered_vectors_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print('\n' + '=' * 60)
    print('Pairs used per subject:')
    for pid, m in meta.items():
        print(f"  {m['neg_subject']:<12} n_pairs={m['n_pairs']:<3} "
              f"(pos_pool={m['n_pos_pool']}, neg_pool={m['n_neg_pool']})")
    print(f'Metadata: {meta_path}')


if __name__ == '__main__':
    main()
