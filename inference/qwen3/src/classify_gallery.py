"""Classify, aggregate and show which images pass the University criterion.

Subject-agnostic: reads the per-subject results produced by the bias sweep
(bias_<subject>.json) and builds a thumbnail gallery split into the two groups
- University  (judge proceed == true)
- Job         (judge proceed == false)
plus any skipped images. Reusable for one or several subjects.

Usage (subject id with or without the "subject_" prefix):
    python src/classify_gallery.py 8
    python src/classify_gallery.py subject_8 12
    SUBJECTS="8 12" python src/classify_gallery.py
With no argument it processes every subject that has a results JSON.
"""

import json
import math
import os
import sys
from os.path import abspath, dirname, join

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


QWEN3_DIR = dirname(dirname(abspath(__file__)))
DATA_DIR = join(QWEN3_DIR, "data")
OUTPUT_DIR = join(QWEN3_DIR, "src", "outputs", "students_bias")

NCOLS = 10
THUMB = 100
BORDER_W = 4
PAD = 8
UNI_COLOR = (0, 181, 100)    # green  -> University
JOB_COLOR = (198, 40, 40)    # red    -> Job


def normalize(subject):
    subject = str(subject).strip()
    return subject if subject.startswith("subject_") else f"subject_{subject}"


def load_thumb(path, size=THUMB):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    im = im.crop((left, top, left + s, top + s)).resize((size, size))
    return np.asarray(im, dtype=np.uint8)


def add_border(arr, color, bw=BORDER_W):
    h, w, _ = arr.shape
    out = np.full((h + 2 * bw, w + 2 * bw, 3), color, dtype=np.uint8)
    out[bw:bw + h, bw:bw + w] = arr
    return out


def make_montage(paths, color, ncols=NCOLS, thumb=THUMB, bw=BORDER_W, pad=PAD, bg=255):
    cell = thumb + 2 * bw
    n = len(paths)
    if n == 0:
        return np.full((cell + 2 * pad, cell + 2 * pad, 3), bg, dtype=np.uint8)
    ncols = min(ncols, n)
    rows = math.ceil(n / ncols)
    H = rows * cell + (rows + 1) * pad
    W = ncols * cell + (ncols + 1) * pad
    canvas = np.full((H, W, 3), bg, dtype=np.uint8)
    for i, p in enumerate(paths):
        r, c = divmod(i, ncols)
        try:
            tile = add_border(load_thumb(p, thumb), color, bw)
        except Exception as e:
            print(f"  [thumb skip] {os.path.basename(p)}: {e}")
            continue
        y = pad + r * (cell + pad)
        x = pad + c * (cell + pad)
        canvas[y:y + cell, x:x + cell] = tile
    return canvas


def classify(subject, output_dir=OUTPUT_DIR, data_dir=DATA_DIR):
    json_path = join(output_dir, f"bias_{subject}.json")
    if not os.path.isfile(json_path):
        print(f"[skip] no results for {subject} ({json_path})")
        return None

    data = json.load(open(json_path))
    per = data["per_image"]
    uni = [p["image"] for p in per if p.get("proceed") is True]
    job = [p["image"] for p in per if p.get("proceed") is False]
    skipped = [p["image"] for p in per if p.get("proceed") is None]
    total = len(uni) + len(job)
    rate = (len(uni) / total * 100) if total else 0.0

    print(f"\n=== {subject} ===")
    print(f"University (pass): {len(uni)}/{total}  ({rate:.1f}%)")
    print(f"Job (fail):        {len(job)}/{total}")
    if skipped:
        print(f"Skipped:           {len(skipped)}")
    print("Images that PASS (University):")
    for im in uni:
        print(f"  + {im}")

    return {"subject": subject, "uni": uni, "job": job, "skipped": skipped,
            "total": total, "rate": rate}


def plot_gallery(result, output_dir=OUTPUT_DIR, data_dir=DATA_DIR):
    subject = result["subject"]
    subj_dir = join(data_dir, subject)
    uni_paths = [join(subj_dir, im) for im in result["uni"]]
    job_paths = [join(subj_dir, im) for im in result["job"]]

    mont_u = make_montage(uni_paths, UNI_COLOR)
    mont_j = make_montage(job_paths, JOB_COLOR)

    rows_u = max(1, math.ceil(max(len(uni_paths), 1) / NCOLS))
    rows_j = max(1, math.ceil(max(len(job_paths), 1) / NCOLS))

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(13, 1.3 * (rows_u + rows_j) + 2),
        gridspec_kw={"height_ratios": [rows_u, rows_j]},
    )
    fig.suptitle(
        f"{subject} - images by University criterion "
        f"({len(uni_paths)}/{result['total']} pass, {result['rate']:.1f}%)",
        fontsize=13,
    )
    ax0.imshow(mont_u)
    ax0.set_title(f"University (proceed=True): {len(uni_paths)}", color=tuple(c / 255 for c in UNI_COLOR))
    ax1.imshow(mont_j)
    ax1.set_title(f"Job (proceed=False): {len(job_paths)}", color=tuple(c / 255 for c in JOB_COLOR))
    for ax in (ax0, ax1):
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png_path = join(output_dir, f"gallery_{subject}.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {png_path}")
    return png_path


def discover_subjects(output_dir=OUTPUT_DIR):
    import re
    subs = []
    for f in os.listdir(output_dir):
        m = re.fullmatch(r"bias_(subject_\d+)\.json", f)
        if m:
            subs.append(m.group(1))
    return sorted(subs, key=lambda s: int(re.sub(r"\D", "", s)))


def main():
    args = sys.argv[1:] or os.environ.get("SUBJECTS", "").replace(",", " ").split()
    subjects = [normalize(a) for a in args] if args else discover_subjects()
    if not subjects:
        print("No subjects to process. Pass an id (e.g. 8) or run the sweep first.")
        return
    for subject in subjects:
        res = classify(subject)
        if res:
            plot_gallery(res)


if __name__ == "__main__":
    main()
