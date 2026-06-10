"""Aggregate plot for the students bias sweep.

Ordered by Monk skin tone (a -> j), one bar per subject (subjects sharing a
tone are kept separate but share the tone colour). Each bar is annotated with
the subject thumbnail (vertically aligned under it) and the perceived gender
(bar edge colour). Horizontal gridlines ease reading the rates.

Importable from students.py (called at the end of the sweep) and runnable
standalone to regenerate the plot from existing JSON results without rerunning
inference:  python src/plot_bias_sweep.py
"""

import json
import os
import re
from os.path import abspath, dirname, join, basename

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch
from PIL import Image


# Paths derived from this file so it works both in Docker (/app/...) and locally
QWEN3_DIR = dirname(dirname(abspath(__file__)))
DATA_DIR = join(QWEN3_DIR, "data")
OUTPUT_DIR = join(QWEN3_DIR, "src", "outputs", "students_bias")
SUBJECTS_JSON = join(DATA_DIR, "subjects.json")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# Official-ish Monk Skin Tone swatches, a (lightest) -> j (darkest)
MONK_COLORS = {
    "a": "#f6ede4", "b": "#f3e7db", "c": "#f7ead0", "d": "#eadaba",
    "e": "#d7bd96", "f": "#a07e56", "g": "#825c43", "h": "#604134",
    "i": "#3a312a", "j": "#292420",
}

# Perceived gender -> subject photo frame colour
GENDER_EDGE = {
    "more-masculine": "#00b564",
    "more-femenine": "#7030A0",
    "non-binary": "#FFF433",
}
GENDER_LABEL = {
    "more-masculine": "Masculine",
    "more-femenine": "Feminine",
    "non-binary": "Non-binary",
}


def first_image_path(subject, data_dir):
    subject_path = join(data_dir, subject)
    if not os.path.isdir(subject_path):
        return None
    for f in sorted(os.listdir(subject_path)):
        if f.lower().endswith(IMAGE_EXTS):
            return join(subject_path, f)
    return None


def load_thumb(path, size=72):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    im = im.crop((left, top, left + s, top + s)).resize((size, size))
    return np.asarray(im)


def plot_aggregate(results, output_dir=OUTPUT_DIR, data_dir=DATA_DIR,
                   subjects_json=SUBJECTS_JSON, grades=""):
    os.makedirs(output_dir, exist_ok=True)
    meta = json.load(open(subjects_json)) if os.path.isfile(subjects_json) else {}

    def monk(subj):
        return meta.get(subj, {}).get("monk_code", "z")

    def sidx(subj):
        return int(re.sub(r"\D", "", subj) or 0)

    # Order by Monk tone (a..j), then by subject index within the same tone
    rows = sorted(results, key=lambda r: (monk(r["subject"]), sidx(r["subject"])))

    n = len(rows)
    x = np.arange(n)
    rates = [r["rate"] for r in rows]
    codes = [monk(r["subject"]) for r in rows]
    genders = [meta.get(r["subject"], {}).get("perceived_gender", "?") for r in rows]
    colors = [MONK_COLORS.get(c, "#cccccc") for c in codes]
    frame_colors = [GENDER_EDGE.get(g, "#000000") for g in genders]

    fig, ax = plt.subplots(figsize=(max(12, n * 0.85), 7))
    bars = ax.bar(x, rates, color=colors, edgecolor="#555555", linewidth=0.8, zorder=3)

    # Horizontal gridlines
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color="#999999", alpha=0.6, zorder=0)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("% recommended to University")

    title = "Bias sweep ordered by Monk skin tone (a=light -> j=dark)"
    if grades:
        title += f"\n'Mark Elliott' + {grades}"
    ax.set_title(title)

    # Rate label + count on top of each bar
    for bar, r in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{r['rate']:.1f}%\n{r['trues']}/{r['n_evaluated']}",
                ha="center", va="bottom", fontsize=8)

    # X labels: subject + monk code
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{sidx(r['subject'])}\n({monk(r['subject'])})" for r in rows])
    ax.set_xlim(-0.7, n - 0.3)

    # Thumbnails vertically aligned under each bar; frame coloured by gender
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    for xi, r, fc in zip(x, rows, frame_colors):
        p = first_image_path(r["subject"], data_dir)
        if p is None:
            continue
        try:
            oi = OffsetImage(load_thumb(p), zoom=0.55)
            ab = AnnotationBbox(oi, (xi, 0), xybox=(xi, -0.16), xycoords=trans,
                                box_alignment=(0.5, 1.0), frameon=True,
                                pad=0.2, annotation_clip=False)
            ab.patch.set_edgecolor(fc)
            ab.patch.set_linewidth(3.5)
            ax.add_artist(ab)
        except Exception as e:
            print(f"[thumb skip] {r['subject']}: {e}")

    # Legends: gender (photo frame colour) + Monk tone reference
    gender_handles = [Patch(facecolor="white", edgecolor=c, linewidth=3.5,
                            label=GENDER_LABEL[g]) for g, c in GENDER_EDGE.items()]
    leg1 = ax.legend(handles=gender_handles, title="Perceived gender (photo frame)",
                     loc="upper right", fontsize=8)
    ax.add_artist(leg1)
    monk_handles = [Patch(facecolor=MONK_COLORS[c], edgecolor="#555555", label=c)
                    for c in sorted(set(codes)) if c in MONK_COLORS]
    ax.legend(handles=monk_handles, title="Monk tone (fill)", loc="upper left",
              fontsize=8, ncol=2)

    fig.subplots_adjust(bottom=0.28)
    png_path = join(output_dir, "bias_sweep_rates.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {png_path}")
    return png_path


def main():
    agg = join(OUTPUT_DIR, "bias_sweep_results.json")
    payload = json.load(open(agg))
    plot_aggregate(payload["subjects"], grades=payload.get("grades", ""))


if __name__ == "__main__":
    main()
