"""scripts/make_mctd_crosssize_figure.py

Cross-size generalization figure (companion to notes/results_mctd.md §
"Generalization across maze sizes"). Grouped bars: each tree method's gain over
its OWN matched flat baseline, at three maze sizes. Above zero = the search helps
at that size; below = it hurts. The story is sign-consistency across sizes:

  DF-tree      stays ABOVE zero at every size  -> generalizes
  V(s,g)-pess  above only on large              -> the non-generalizing outlier
  MCTD         stays BELOW zero at every size   -> loses everywhere

Numbers: DF-tree and MCTD are 5-seed on large, seed-0 on medium/umaze; V(s,g) is
5-seed throughout. Sources: m2{l,m,u}_both_df (DF-tree vs DF-MCSS),
m2{l,m,u}_tree_vsgpess vs m2{l,m,u}_mcss_k256 (V(s,g) vs DV-MCSS), mctdcritic vs
mcssmpc (MCTD vs MCSS-MPC).

Run (needs matplotlib; torch-free):  python scripts/make_mctd_crosssize_figure.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
BASE, GRID = "#c3c2b7", "#e1e0d9"
GOODBG = "#eef5ee"
# umaze / medium / large — blue sequential ramp (references/palette.md SEQ)
SHADE = ["#86b6ef", "#3987e5", "#1c5cab"]

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
    "font.size": 9.5,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK2, "axes.labelsize": 9.5,
    "xtick.color": INK, "ytick.color": MUTED,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
})

METHODS = ["DF-tree\n(trajectory)", "V(s,g)-pess\n(goal value)", "MCTD\n(denoising)"]
SIZES = ["umaze", "medium", "large"]                  # small -> large
# gain over matched flat baseline, [umaze, medium, large]
GAINS = {
    "DF-tree\n(trajectory)": [10.4, 13.7, 9.0],
    "V(s,g)-pess\n(goal value)": [-4.9, -14.8, 4.0],
    "MCTD\n(denoising)": [-12.4, -17.5, -16.8],
}


def main():
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    w = 0.26
    centers = [0, 1.15, 2.30]
    ax.axhspan(0, 20, color=GOODBG, zorder=0)          # faint "search helps" band
    ax.axhline(0, color=BASE, lw=1.3, zorder=2)

    for gi, m in enumerate(METHODS):
        for si in range(3):
            x = centers[gi] + (si - 1) * w
            g = GAINS[m][si]
            ax.bar(x, g, width=w, color=SHADE[si], zorder=3,
                   edgecolor="white", linewidth=0.6)
            ax.text(x, g + (1.1 if g >= 0 else -1.1), f"{g:+.0f}",
                    ha="center", va="bottom" if g >= 0 else "top",
                    fontsize=8.3, color=INK)

    ax.set_xticks(centers)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel("gain over matched flat baseline\n(camping points;  >0 = search helps)")
    ax.set_title("Does search over diffusion generalize across maze sizes?")
    ax.set_ylim(-30, 22)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)

    ax.legend(handles=[Patch(color=SHADE[i], label=SIZES[i]) for i in range(3)],
              title="maze size", loc="lower left", frameon=False,
              fontsize=8.5, title_fontsize=8.5, ncol=3, handlelength=1.1,
              columnspacing=1.2)

    fig.text(0.5, -0.04,
             "The DF-tree stays above zero at every size (it generalizes); MCTD "
             "stays below at every size; the goal-conditioned tree wins only on "
             "large (the known outlier).\nGains are vs each method's own matched "
             "flat baseline. MCTD and V(s,g): 5-seed at every size; DF-tree: "
             "5-seed on large, seed-0 on medium+umaze.",
             ha="center", va="top", fontsize=7.6, color=MUTED)

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_mctd_crosssize.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT}/fig_mctd_crosssize.pdf (+.png)")


if __name__ == "__main__":
    main()
