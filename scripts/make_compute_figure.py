"""scripts/make_compute_figure.py

Test-time-compute scaling on maze2d-large (per-step harness, camping return).

Addresses the "no score-versus-compute plot" gap. Three backbones are each swept
over the flat best-of-K width K in {16, 50, 256}, so every line is ONE method
scaled over its own sampling budget (connecting lines are therefore meaningful,
unlike a scatter of unrelated points). The look-ahead tree is a single operating
point at its own budget (290 draws).

All values are dissertation-authoritative:
    DV-MCSS   (base system, full-seq): 196.3 / 199.4 / 201.2   [K16 m2l_mcss_k16; K50 R1; K256 R3]
    DF-MCSS   (causal):                155.7 / 183.4 / 191.6   [K16/K256 widthsweep_df_*; K50 R6]
    shortcut-MCSS:                     152.3 / 148.3 / 159.1   [K16/K256 widthsweep_short_*; K50 R7]
    DF-tree   (this work):             192.4 at 290 draws       [Tables 4.1/4.3]

Seed counts vary by point (3-10); the trend, not any single value, is the message.

Message: within every backbone, more draws give sharply diminishing returns, and the
backbones never cross. The base system at only 16 draws already beats the causal
backbone and the tree at 256-290 draws, so backbone and critic quality set the level,
not the compute spent at decision time.

Run (needs matplotlib; torch-free):  python scripts/make_compute_figure.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── palette (matches scripts/make_figures.py) ──────────────────────────────
BLUE, AQUA, RED, ORANGE = "#2a78d6", "#1baf7a", "#e34948", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
    "font.size": 9.5,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK2, "axes.labelsize": 9.5,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
})

K = [16, 50, 256]
CURVES = [
    ("DV-MCSS (base system)", [196.3, 199.4, 201.2], BLUE, "o", 6.5),
    ("DF-MCSS (causal)",       [155.7, 183.4, 191.6], AQUA, "s", 6.0),
    ("shortcut-MCSS",          [152.3, 148.3, 159.1], ORANGE, "^", 7.0),
]
# DF-tree budget scaling, k_mcts=16, 1 seed. Leftmost point uses a lean 16-root
# (the only way below 50 draws); the rest use the matched 50-root (draws = 50 + 16*budget).
K_TREE = [32, 66, 178, 258]
Y_TREE = [180.9, 186.1, 191.8, 194.4]

fig, ax = plt.subplots(figsize=(6.8, 4.4))

for lab, ys, col, mk, ms in CURVES:
    ax.plot(K, ys, "-", color=col, lw=1.6, zorder=3)
    ax.plot(K, ys, mk, color=col, ms=ms, zorder=4,
            markeredgecolor="white", markeredgewidth=0.8)

# DF-tree: dashed (different search axis: budget, not width), diamonds.
ax.plot(K_TREE, Y_TREE, "--", color=RED, lw=1.5, zorder=4)
ax.plot(K_TREE, Y_TREE, "D", color=RED, ms=7.5, zorder=5,
        markeredgecolor="white", markeredgewidth=0.9)

# direct labels at each curve's right end
ax.annotate("DV-MCSS (base system)", (256, 201.2), textcoords="offset points",
            xytext=(6, 4), ha="left", va="bottom", color=BLUE, fontsize=8.5, fontweight="bold")
ax.annotate("DF-MCSS (causal)", (50, 183.4), textcoords="offset points",
            xytext=(0, 12), ha="center", va="bottom", color=AQUA, fontsize=8.5, fontweight="bold")
ax.annotate("shortcut-MCSS", (256, 159.1), textcoords="offset points",
            xytext=(8, 0), ha="left", va="center", color=ORANGE, fontsize=8.5, fontweight="bold")
ax.annotate("DF-tree (search budget)", (258, 194.4), textcoords="offset points",
            xytext=(8, 2), ha="left", va="center", color=RED, fontsize=8.5, fontweight="bold")

# endpoint value labels (K=16 and K=256) to anchor the reading
for lab, ys, col, mk, ms in CURVES:
    ax.annotate(f"{ys[0]:.0f}", (16, ys[0]), textcoords="offset points",
                xytext=(-8, 0), ha="right", va="center", color=col, fontsize=8)

# the base-system-at-16-draws reference: everything else sits below it
ax.axhline(196.3, color=BLUE, lw=0.8, ls=":", alpha=0.5, zorder=1)
ax.text(58, 169, "Even at 258 draws the tree (194.4) stays below\n"
        "the base system at just 16 draws (196.3).",
        ha="left", va="center", color=INK2, fontsize=8)

ax.set_xscale("log")
ax.set_xticks([16, 50, 100, 256])
ax.set_xticklabels(["16", "50", "100", "256"])
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlim(12, 430)
ax.set_ylim(140, 206)
ax.set_xlabel("planner draws per decision  (log scale)")
ax.set_ylabel("camping return (maze2d-large)")
ax.set_title("More draws saturate within a backbone; backbones never cross")
ax.grid(True, which="major", axis="y", color=GRID, lw=0.6, zorder=0)

for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_compute_frontier.{ext}")
plt.close(fig)
print(f"  wrote {OUT}/fig_compute_frontier.pdf (+.png)")
