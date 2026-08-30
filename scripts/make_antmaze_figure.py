"""scripts/make_antmaze_figure.py

Figure 4-1: the antmaze ceiling is a control failure, not a planning failure.

Built from the instrumented rollouts in results/instr/instr_mcss_critic_s{0,1,2}.npz,
which log, per environment step, the BFS geodesic distance to the goal, the torso
uprightness, the torso height and the executed xy for every failed episode.

Left column, one representative episode (seed 0, episode 10): the ant closes from
14 cells to 3, then topples at step 433 and the distance never moves again. The two
measures sit on separate stacked axes sharing the x-axis, never on twin y-scales.

Right, every failed episode across the three seeds: cells closed per 100 steps in
the 200 steps before the topple against the 200 steps after. Not one episode makes
any progress after falling.

Palette and rcParams follow scripts/make_figures.py so the figure sits in the same
visual system as the other six; type is scaled for the 4.3 in placement width the
builder uses. Palette validated with the dataviz skill's checker (blue/red, CVD
deltaE 21.6 protan, all six checks pass).

Run (needs matplotlib + numpy; torch-free):
    python scripts/make_antmaze_figure.py
"""
import glob
import os
import re
import sys

sys.path.insert(0, ".")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# palette shared with make_figures.py (references/palette.md, light surface)
BLUE, RED = "#2a78d6", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"

OUT = "figures"
INSTR = "results/instr"
WINDOW = 200                       # steps either side of the topple
EPISODE = ("results/instr/instr_mcss_critic_s0.npz", 10)

os.makedirs(OUT, exist_ok=True)

# the builder places figures 4.3 in wide, so this is typeset at 7.6 in with
# proportionally larger type: 11 pt here lands near 6 pt on the page
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
    "font.size": 11,
    "axes.edgecolor": BASE, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK2, "axes.labelsize": 11,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 220,
})


def episodes(path):
    z = np.load(path)
    return z, sorted({int(re.match(r"e(\d+)_", k).group(1)) for k in z.files})


def topple_step(upright):
    """First step at which the torso passes onto its side and stays there."""
    neg = np.where(upright < 0)[0]
    return int(neg[0]) if len(neg) else None


def progress_rates():
    """Cells closed per 100 env steps, before and after the topple, per episode."""
    pre, post = [], []
    for path in sorted(glob.glob(f"{INSTR}/instr_mcss_critic_s*.npz")):
        z, eps = episodes(path)
        for e in eps:
            up, dist = z[f"e{e}_upright"], z[f"e{e}_dist"]
            t = topple_step(up)
            if t is None:
                continue
            a, b = dist[max(0, t - WINDOW):t], dist[t:t + WINDOW]
            if len(a) < 20 or len(b) < 20:
                continue                       # topple too early to have a window
            pre.append(-(a[-1] - a[0]) / len(a) * 100)
            post.append(-(b[-1] - b[0]) / len(b) * 100)
    return np.array(pre), np.array(post)


def main():
    z, _ = episodes(EPISODE[0])
    e = EPISODE[1]
    dist, up = z[f"e{e}_dist"], z[f"e{e}_upright"]
    t = topple_step(up)
    step = np.arange(len(dist))

    fig = plt.figure(figsize=(7.6, 3.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1], height_ratios=[1, 1],
                          wspace=0.40, hspace=0.26)
    ax_d = fig.add_subplot(gs[0, 0])
    ax_u = fig.add_subplot(gs[1, 0], sharex=ax_d)
    ax_s = fig.add_subplot(gs[:, 1])

    for ax in (ax_d, ax_u):
        ax.axvspan(t, len(dist), color=GRID, alpha=0.55, lw=0, zorder=0)
        ax.axvline(t, color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=3)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)

    # ── left, top: distance to goal ──────────────────────────────────────
    ax_d.plot(step, dist, color=BLUE, lw=2.2, solid_capstyle="round", zorder=4)
    ax_d.set_ylabel("distance\n(BFS cells)", labelpad=2)
    ax_d.set_ylim(-1.0, 16.5)
    ax_d.set_yticks([0, 5, 10, 15])
    ax_d.set_title("One failed episode", loc="left", pad=6)
    ax_d.tick_params(labelbottom=False)
    ax_d.annotate("closes 14 to 3", xy=(330, 6.6), xytext=(28, 0.4),
                  color=INK2, fontsize=10.5,
                  arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax_d.text(len(dist) - 25, 12.4, "frozen, 567 steps",
              color=INK2, fontsize=10.5, ha="right", va="top")

    # ── left, bottom: uprightness ────────────────────────────────────────
    ax_u.axhline(0, color=BASE, lw=0.9, zorder=1)
    ax_u.plot(step, up, color=RED, lw=2.2, solid_capstyle="round", zorder=4)
    ax_u.set_ylabel("uprightness", labelpad=2)
    ax_u.set_xlabel("environment step", labelpad=2)
    ax_u.set_ylim(-1.45, 1.45)
    ax_u.set_yticks([-1, 0, 1])
    ax_u.set_xlim(0, len(dist))
    ax_u.text(t + 30, 0.62, f"topple, step {t}", color=INK2, fontsize=10.5,
              ha="left", va="center")

    # ── right: every failed episode, before against after ────────────────
    pre, post = progress_rates()
    x = [0, 1]
    for a, b in zip(pre, post):
        ax_s.plot(x, [a, b], color=MUTED, lw=1.0, alpha=0.45,
                  marker="o", ms=3.2, mfc="white", mew=0.9, zorder=2)
    ax_s.plot(x, [np.median(pre), np.median(post)], color=RED, lw=2.4,
              marker="o", ms=7, mfc="white", mew=2.0, zorder=5)
    ax_s.axhline(0, color=BASE, lw=0.9, zorder=1)
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(["before\ntopple", "after\ntopple"])
    ax_s.set_xlim(-0.46, 1.46)
    ax_s.set_ylim(-1.15, 5.55)
    ax_s.set_ylabel("cells closed per 100 steps", labelpad=2)
    ax_s.set_title(f"All {len(pre)} failed episodes", loc="left", pad=6)
    ax_s.grid(axis="y", color=GRID, lw=0.7)
    ax_s.set_axisbelow(True)
    # medians direct-labelled outboard of their own marker
    ax_s.text(-0.10, np.median(pre), f"median\n{np.median(pre):.2f}", va="center",
              ha="right", color=INK, fontsize=10.5, fontweight="bold")
    ax_s.text(1.10, np.median(post), f"median\n{np.median(post):.2f}", va="center",
              ha="left", color=INK, fontsize=10.5, fontweight="bold")

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig4_3_antmaze_topple.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT}/fig4_3_antmaze_topple.pdf (+.png)")
    print(f"  episode {e} of {os.path.basename(EPISODE[0])}: topple at step {t}, "
          f"distance {dist[0]:.0f} -> {dist.min():.0f} cells, "
          f"{len(dist) - t} steps motionless")
    print(f"  cells per 100 steps: before median {np.median(pre):.2f}, "
          f"after median {np.median(post):.2f}, "
          f"{(post > 0.001).sum()} of {len(post)} episodes progressing after the topple")


if __name__ == "__main__":
    main()
