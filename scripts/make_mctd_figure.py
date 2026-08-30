"""scripts/make_mctd_figure.py

The MCTD-addendum figure (companion to notes/results_mctd.md): a single
harness-controlled comparison of what search structures actually help.

Why "gain over the no-search baseline" and not raw scores: the flat best-of-K
baseline itself scores ~201.6 per-step vs ~238.8 under MPC execution, so raw
camping numbers are not comparable across methods run in different driving
setups. Each method's gain over ITS OWN flat baseline (same grader, same driving
setup) IS comparable — it isolates the one thing that varies, the search
structure — and that is what this figure plots.

Numbers:
  * DF-tree (this project's trajectory-axis tree): the confirmed dissertation
    gains over compute-matched MCSS [DOC: runbook §1 / results_chapter — maze2d
    +9.04 t=3.90; kitchen +10.5 t=5.47].
  * MCTD family (denoising-axis): from this project's paired runs vs the
    execution-matched MCSS-MPC control (scripts/collate_mctd.py, maze2d-large,
    5 seeds): MCTD faithful -83.33 t=-14.56; MCTD-critic (4c) -16.79 t=-6.10;
    guided-BoN (4b) +0.14 t=+0.19.

Run (needs matplotlib; torch-free):  python scripts/make_mctd_figure.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# palette (references/palette.md, matching scripts/make_figures.py)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
BASE, GRID = "#c3c2b7", "#e1e0d9"
GOOD, TIE, CRIT = "#006300", "#898781", "#d03b3b"

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
    "font.size": 9.5,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK2, "axes.labelsize": 9.5,
    "xtick.color": MUTED, "ytick.color": INK,
    "xtick.labelsize": 9, "ytick.labelsize": 9.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
})

# (label, env, gain over matched flat baseline, seed-level t)
ROWS = [
    ("DF-tree",            "kitchen", +10.50,  +5.47),
    ("DF-tree",            "maze2d-large", +9.04, +3.90),
    ("Guided-BoN (4b)",    "maze2d-large",   +0.14,  +0.19),
    ("MCTD-critic (4c)",   "maze2d-large",  -16.79,  -6.10),
    ("MCTD (faithful)",    "maze2d-large",  -83.33, -14.56),
]


def verdict_color(t):
    if t >= 2.0:
        return GOOD
    if t <= -2.0:
        return CRIT
    return TIE


def main():
    # plot best (largest gain) at the TOP
    rows = sorted(ROWS, key=lambda r: r[2])       # ascending -> top of a barh is last
    labels = [f"{lab}\n({env})" for lab, env, _, _ in rows]
    gains = [g for _, _, g, _ in rows]
    ts = [t for _, _, _, t in rows]
    ses = [abs(g / t) if t else 0.0 for g, t in zip(gains, ts)]
    colors = [verdict_color(t) for t in ts]
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.axvline(0, color=BASE, lw=1.2, zorder=1)
    ax.barh(list(y), gains, color=colors, height=0.62, zorder=3,
            xerr=ses, error_kw=dict(ecolor=MUTED, elinewidth=1.0, capsize=3))

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("gain over flat best-of-K, same grader & driving setup "
                  "(camping-return points; 0 = no benefit)")
    ax.set_title("What search actually helps a diffusion planner")
    ax.set_xlim(-118, 26)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    for yi, (g, t) in enumerate(zip(gains, ts)):
        off = 2.2 if g >= 0 else -2.2
        ha = "left" if g >= 0 else "right"
        # place the label outside the bar + error bar
        x = g + (ses[yi] + off if g >= 0 else -(ses[yi]) + off)
        ax.text(x, yi, f"{g:+.1f}  (t={t:+.1f})", va="center", ha=ha,
                fontsize=8.6, color=INK)

    # verdict legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GOOD, label="search HELPS (t≥2)"),
                       Patch(color=TIE, label="no effect (|t|<2)"),
                       Patch(color=CRIT, label="search HURTS (t≤−2)")],
              loc="lower right", frameon=False, fontsize=8.3, handlelength=1.1)

    fig.text(0.5, -0.06,
             "Trajectory-axis search (DF-tree) is the only structure that beats "
             "flat selection; denoising-axis search (MCTD) loses even with the "
             "learned grader.\nEach bar is a method vs its own no-search baseline "
             "in the same harness. MCTD family: maze2d-large-v1, 5 seeds; DF-tree: "
             "confirmed dissertation gains.",
             ha="center", va="top", fontsize=7.6, color=MUTED)

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_mctd_what_works.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT}/fig_mctd_what_works.pdf (+.png)")


if __name__ == "__main__":
    main()
