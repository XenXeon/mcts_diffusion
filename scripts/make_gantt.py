"""scripts/make_gantt.py  --  DRAFT plan-vs-actual Gantt for section 6.2.

Planned bars are the student's original plan; ACTUAL bars are inferred from the
project narrative and to be verified by the author. Timeline ends at the 1 Sep
submission deadline. Not wired into the report.

Run:  python scripts/make_gantt.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

BLUE, AQUA, RED, ORANGE = "#2a78d6", "#1baf7a", "#e34948", "#eb6834"
PLAN = "#c9d9f2"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
BASE, GRID = "#c3c2b7", "#e1e0d9"

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"], "font.size": 9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
})

# week 0 = 1 March; ~4.33 weeks / month; deadline = 1 Sep = week 26
MONTHS = [("Mar", 0), ("Apr", 4.33), ("May", 8.67), ("Jun", 13.0),
          ("Jul", 17.33), ("Aug", 21.67)]
DEADLINE = 26.0

# (label, plan[start,end] or None, actual[start,end], actual_type)
TASKS = [
    ("Literature review",                           (0, 11),   (0, 11),   "onplan"),
    ("Setup & baselines",                            (0, 2),    (0, 2),    "onplan"),
    ("MCTS development",                             (2, 11),   (2, 11),   "onplan"),
    ("Causal DF backbone",                           None,      (15, 19),  "added"),
    ("Experimentation & testing (maze2d, kitchen)",  (14.5, 18.5), (15, 24), "extended"),
    ("MCTD port + guidance",                         None,      (20, 24),  "added"),
    ("Analysis & report writing",                    (18.5, 22.5), (20, 26), "extended"),
]
ACT_COLOR = {"onplan": BLUE, "extended": ORANGE, "added": AQUA}
EXAM = (11.5, 14.5)

fig, ax = plt.subplots(figsize=(11.5, 4.2))
n = len(TASKS)
for i, (label, plan, act, atype) in enumerate(TASKS):
    y = n - 1 - i
    if plan:
        ax.add_patch(Rectangle((plan[0], y + 0.52), plan[1]-plan[0], 0.34,
                     facecolor=PLAN, edgecolor="#8fa9d6", lw=0.8, zorder=3))
    ax.add_patch(Rectangle((act[0], y + 0.12), act[1]-act[0], 0.34,
                 facecolor=ACT_COLOR[atype], edgecolor="white", lw=0.8, zorder=3))
    ax.text(-0.4, y + 0.5, label, ha="right", va="center", fontsize=8.5, color=INK)

ax.axvspan(EXAM[0], EXAM[1], color="#c8b6a6", alpha=0.18, zorder=0)
ax.text((EXAM[0]+EXAM[1])/2, n + 0.05, "exam break", ha="center", va="bottom",
        fontsize=8, color=INK2)

# 1 Sep submission deadline
ax.axvline(DEADLINE, color=RED, lw=1.6, zorder=2)
ax.text(DEADLINE + 0.15, n + 0.05, "1 Sep\ndeadline", ha="left", va="bottom",
        fontsize=8, color=RED, fontweight="bold")

for name, wk in MONTHS:
    ax.axvline(wk, color=GRID, lw=0.9, zorder=1)
    ax.text(wk + 0.1, -0.55, name, ha="left", va="center", fontsize=8.5, color=MUTED)

ax.set_xlim(-0.2, 28.2); ax.set_ylim(-0.95, n + 0.7)
ax.set_yticks([]); ax.set_xticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Project Gantt Chart (Planned vs Actual)",
             fontsize=11, fontweight="bold", color=INK, pad=10)

leg = [Patch(fc=PLAN, ec="#8fa9d6", label="Planned"),
       Patch(fc=BLUE, label="Actual (on plan)"),
       Patch(fc=ORANGE, label="Actual (extended/slipped)"),
       Patch(fc=AQUA, label="Added (unplanned)")]
ax.legend(handles=leg, ncol=4, frameon=False, fontsize=8,
          loc="lower center", bbox_to_anchor=(0.5, -0.17))

for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_gantt.{ext}")
plt.close(fig)
print(f"  wrote {OUT}/fig_gantt.pdf (+.png)")
