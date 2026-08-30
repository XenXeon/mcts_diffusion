"""scripts/make_mcts_diagram.py

The four phases of Monte Carlo Tree Search, as a conceptual diagram for Chapter 2
(beside the "four phases" text). Selection -> Expansion -> Simulation -> Backup.

Style matches scripts/make_figures.py (dataviz palette, same fonts). Torch-free.
Run:  python scripts/make_mcts_diagram.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

BLUE, AQUA, RED, ORANGE = "#2a78d6", "#1baf7a", "#e34948", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
BASE, GRID = "#c3c2b7", "#e1e0d9"

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
})

# ── tree geometry (shared by all four panels) ──────────────────────────────
N = {
    "root": (0.50, 0.92),
    "A": (0.27, 0.66), "B": (0.50, 0.66), "C": (0.73, 0.66),
    "A1": (0.19, 0.40), "A2": (0.37, 0.40), "B1": (0.50, 0.40),
}
EDGES = [("root", "A"), ("root", "B"), ("root", "C"),
         ("A", "A1"), ("A", "A2"), ("B", "B1")]
PATH = ["root", "A", "A2"]              # the UCB-selected descent
PATH_EDGES = [("root", "A"), ("A", "A2")]
NEW = (0.37, 0.15)                      # expansion node (child of A2)
TERM = (0.37, 0.02)                     # rollout terminus
R = 0.052


def node(ax, xy, color, filled, label="", fs=8):
    ax.add_patch(Circle(xy, R, facecolor=(color if filled else "white"),
                        edgecolor=color, lw=1.7, zorder=4))
    if label:
        ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=fs,
                color=("white" if filled else color), zorder=5)


def edge(ax, a, b, color=BASE, lw=1.3, ls="-", z=1):
    (x0, y0), (x1, y1) = a, b
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, zorder=z)


def arrow(ax, a, b, color, lw=1.8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, zorder=6,
                                 shrinkA=9, shrinkB=9))


def base_tree(ax, dim=False):
    c = GRID if dim else BASE
    for a, b in EDGES:
        edge(ax, N[a], N[b], color=c)
    for k, xy in N.items():
        node(ax, xy, (MUTED if dim else INK2), False)


def frame(ax, title, subtitle):
    ax.set_xlim(0, 1); ax.set_ylim(-0.06, 1.06); ax.axis("off")
    ax.text(0.5, 1.14, title, ha="center", va="top", fontsize=10.5,
            fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.5, -0.02, subtitle, ha="center", va="top", fontsize=8.2,
            color=INK2, transform=ax.transAxes)


fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.3))

# 1 — Selection: descend the tree by UCB to a leaf.
ax = axes[0]; base_tree(ax, dim=True)
for a, b in PATH_EDGES:
    edge(ax, N[a], N[b], color=BLUE, lw=2.4, z=2)
for k in PATH:
    node(ax, N[k], BLUE, True)
arrow(ax, N["root"], N["A"], BLUE); arrow(ax, N["A"], N["A2"], BLUE)
frame(ax, "1. Selection", "descend by UCB to a leaf")

# 2 — Expansion: add a child to the selected leaf.
ax = axes[1]; base_tree(ax, dim=True)
node(ax, N["A2"], BLUE, True)
edge(ax, N["A2"], NEW, color=AQUA, lw=2.4, z=2)
node(ax, NEW, AQUA, True, label="+")
frame(ax, "2. Expansion", "add a child from the planner")

# 3 — Simulation: estimate the new node's value.
ax = axes[2]; base_tree(ax, dim=True)
node(ax, N["A2"], BLUE, True); edge(ax, N["A2"], NEW, color=AQUA, lw=2.0, z=2)
node(ax, NEW, AQUA, True)
edge(ax, NEW, (TERM[0], TERM[1] + 0.02), color=ORANGE, lw=1.8, ls=(0, (3, 2)), z=2)
ax.add_patch(Rectangle((TERM[0] - 0.045, TERM[1] - 0.028), 0.09, 0.056,
             facecolor="white", edgecolor=ORANGE, lw=1.7, zorder=4))
ax.text(TERM[0], TERM[1], "$v$", ha="center", va="center", fontsize=9,
        color=ORANGE, zorder=5)
frame(ax, "3. Evaluation", "estimate the leaf's value")

# 4 — Backpropagation: push the value up the path.
ax = axes[3]; base_tree(ax, dim=True)
node(ax, NEW, AQUA, True)
seq = [NEW, N["A2"], N["A"], N["root"]]
for a, b in zip(seq[:-1], seq[1:]):
    edge(ax, a, b, color=RED, lw=2.2, z=2)
    arrow(ax, a, b, RED)
for k in PATH:
    node(ax, N[k], RED, True)
frame(ax, "4. Backup", "update values up the path")

fig.subplots_adjust(wspace=0.08, top=0.82, bottom=0.10)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_mcts_phases.{ext}")
plt.close(fig)
print(f"  wrote {OUT}/fig_mcts_phases.pdf (+.png)")
