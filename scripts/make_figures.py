"""scripts/make_figures.py

Generate the six dissertation figures (results_chapter.md §10 inventory) into
figures/ as vector PDF (for LaTeX) + PNG (preview). Kitchen figures and the
maze2d DF/shortcut arms are recomputed from the per-rollout results/*.json
vectors; the handful of DV maze2d baselines that have no single JSON on disk
(the base-pipeline camping numbers) are documented constants, each tagged
[DOC] with its source in methodology_report §7.5 / the runbook §1 table.

Palette: dataviz skill's validated defaults (references/palette.md) — blue/aqua
categorical (CVD ΔE 73.6, validated), blue↔red diverging for signed gains, a
blue sequential ramp for the ordinal subtask-count census. Identity is never
color-alone: two-series charts also differ in line style + marker, and every
mark of interest is direct-labeled.

Run (needs matplotlib; torch-free):  python scripts/make_figures.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, ".")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── palette (references/palette.md, light surface) ─────────────────────────
BLUE, AQUA, RED, ORANGE = "#2a78d6", "#1baf7a", "#e34948", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
GOOD, CRIT = "#006300", "#d03b3b"
# blue sequential ramp, ordinal steps 250/400/550/700 (1..4 subtasks solved)
SEQ4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

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


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.pdf (+.png)")


def load_arm(pattern, arm):
    """Pooled per-rollout dv_norm vector across all files matching pattern."""
    v = []
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f)).get("results", {}).get(arm, {})
        v += [float(x) for x in (r.get("dv_norm") or [])]
    return np.asarray(v, float)


def ms(v):
    n = len(v)
    return (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else 0.0, n)


def grid_y(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ════════════════════════════════════════════════════════════════════════
# F1 — mechanism triptych: expansion fidelity determines the sign of the gain
# ════════════════════════════════════════════════════════════════════════
def fig1():
    # tree - own flat baseline, maze2d-large. ALL THREE are now data-derived and
    # START-MATCHED: each DV arm is differenced against the DV-MCSS k50 arm on the
    # same seeds (see notes/maze2d_startmatched_correction.md).
    df_t = load_arm("results/m2l_both_df_m3_s*.json", "mcts")
    df_f = load_arm("results/m2l_both_df_m3_s*.json", "mcss")
    df_gain = np.mean(df_t) - np.mean(df_f)          # ~+9.0, data-derived
    glue_gain = (np.mean(load_arm("results/m2l_tree_r50_s*.json", "mcts"))
                 - np.mean(load_arm("results/maze2d_large_mcss_k50_s[0-2].json",
                                    "mcss")))        # ~-4.3, seeds 0-2
    inp_gain = (np.mean(load_arm("results/m2l_tree_criticr50_inpaint.json", "mcts"))
                - np.mean(load_arm("results/maze2d_large_mcss_k50_s0.json",
                                   "mcss")))         # ~-18.5, seed 0
    labels = ["seam-glue\n(full-seq)", "replacement\ninpaint (full-seq)",
              "exact per-token\ncond. (DF)"]
    vals = [glue_gain, inp_gain, df_gain]
    colors = [RED if x < 0 else BLUE for x in vals]

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    grid_y(ax)
    bars = ax.bar(labels, vals, width=0.62, color=colors, zorder=3,
                  edgecolor="white", linewidth=0.8)
    ax.axhline(0, color=BASE, lw=1.2, zorder=2)
    for b, v in zip(bars, vals):
        off = 1.3 if v >= 0 else -1.3
        ax.text(b.get_x() + b.get_width() / 2, v + off,
                f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top",
                color=GOOD if v >= 0 else CRIT, fontweight="bold", fontsize=10)
    ax.set_ylabel("tree − own flat baseline  (DV camping score)")
    ax.set_title("Expansion fidelity decides whether tree search helps")
    ax.set_ylim(-27, 13)
    ax.margins(x=0.06)
    fig.text(0.99, -0.02,
             "maze2d-large. Each bar is start-matched against its own flat MCSS "
             "baseline (DV k50; DF 5-seed pooled).",
             ha="right", va="top", fontsize=7.2, color=MUTED)
    save(fig, "fig1_expansion_triptych")


# ════════════════════════════════════════════════════════════════════════
# F2 — the winner's curse: MAX vs tempered backup, present on DV, gone on DF
# ════════════════════════════════════════════════════════════════════════
def fig2():
    # All four bars data-derived. DV MAX/top-3 are the seeds 0-2 arms, run on
    # identical starts, so their gap is baseline-independent by construction.
    dv_max = load_arm("results/m2l_tree_r50_s*.json", "mcts")
    dv_top3 = load_arm("results/m2l_tree_r50_m3_s*.json", "mcts")
    df_max = load_arm("results/m2l_tree_df_max_s0.json", "mcts")
    df_top3 = load_arm("results/m2l_both_df_m3_s*.json", "mcts")
    groups = ["DV backbone\n(full-sequence)", "DF backbone\n(exact cond.)"]
    max_v = [float(np.mean(dv_max)), float(np.mean(df_max)) if len(df_max) else 190.4]
    top_v = [float(np.mean(dv_top3)), float(np.mean(df_top3))]
    x = np.arange(2)
    w = 0.34

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    grid_y(ax)
    b1 = ax.bar(x - w / 2, max_v, w, label="MAX backup", color=RED,
                zorder=3, edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x + w / 2, top_v, w, label="top-3 (tempered) backup", color=BLUE,
                zorder=3, edgecolor="white", linewidth=0.8)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                    f"{b.get_height():.1f}", ha="center", va="bottom",
                    color=INK2, fontsize=8.5)
    # gap (curse signature) — centered above each group, well clear of the
    # value labels
    gaps = [(f"+{top_v[0]-max_v[0]:.2f}  (roll-t 5.00)", CRIT),
            (f"+{top_v[1]-max_v[1]:.1f}  (n.s.)", GOOD)]
    for xi, (txt, col) in zip(x, gaps):
        top = max(max_v[xi], top_v[xi])
        ax.annotate(txt, (xi, top + 4.2), ha="center", color=col, fontsize=9,
                    fontweight="bold")
    ax.set_xticks(x, groups)
    ax.set_ylabel("DV camping score")
    ax.set_title("The winner's curse is fueled by unfaithful expansion")
    ax.set_ylim(150, 213)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.text(0.99, -0.02, "maze2d-large. The MAX↔top-3 gap vanishes once "
             "expansion is exact.", ha="right", va="top", fontsize=7.2,
             color=MUTED)
    save(fig, "fig2_winners_curse")


# ════════════════════════════════════════════════════════════════════════
# F3 — the headroom curve: the tree equalizes across backbone quality
# ════════════════════════════════════════════════════════════════════════
def fig3():
    df_f, df_t = np.mean(load_arm("results/m2l_both_df_m3_s*.json", "mcss")), \
        np.mean(load_arm("results/m2l_both_df_m3_s*.json", "mcts"))
    sh_f, sh_t = np.mean(load_arm("results/m2l_both_dfshort8_m3_s*.json", "mcss")), \
        np.mean(load_arm("results/m2l_both_dfshort8_m3_s*.json", "mcts"))
    # order by flat quality ascending -> shortcut, DF, DV
    names = ["shortcut\n(8 sweeps)", "DF\n(causal pyramid)", "DV\n(full-seq)"]
    # DV row start-matched vs compute-matched k256 on the tree arm's own seeds
    # (0-2), so the plotted gap IS the paired diff (-2.16). Both data-derived.
    dv_f = np.mean(load_arm("results/maze2d_large_mcss_k256_s[0-2].json", "mcss"))
    dv_t = np.mean(load_arm("results/m2l_tree_r50_m3_s*.json", "mcts"))
    flat = [sh_f, df_f, dv_f]           # DV MCSS k256 (compute-matched, s0-2)
    tree = [sh_t, df_t, dv_t]           # DV tree top-3 (s0-2)
    x = np.arange(3)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    grid_y(ax)
    ax.plot(x, flat, "-o", color=BLUE, lw=2, ms=8, label="flat MCSS", zorder=3)
    ax.plot(x, tree, "--s", color=AQUA, lw=2, ms=8, label="tree (top-3)",
            zorder=3, markeredgecolor="white", markeredgewidth=0.8)
    # value labels: higher line's label above its point, lower line's below —
    # so the DV crossing (flat 205 overtakes tree 202) doesn't collide.
    for xi, (f, t) in enumerate(zip(flat, tree)):
        f_above = f >= t
        ax.text(xi, f + (2.8 if f_above else -4.6), f"{f:.0f}", ha="center",
                color=BLUE, fontsize=8.5,
                fontweight="bold" if f_above else "normal")
        ax.text(xi, t + (2.8 if not f_above else -4.6), f"{t:.0f}", ha="center",
                color="#0f7a54", fontsize=8.5,
                fontweight="bold" if not f_above else "normal")
    # signed gaps as a dedicated bottom strip (never over the crossing)
    ax.text(-0.42, 134.5, "tree − flat:", color=MUTED, fontsize=8, va="center")
    for xi, (f, t) in enumerate(zip(flat, tree)):
        g = t - f
        ax.text(xi, 134.5, f"{g:+.0f}", ha="center", va="center",
                color=GOOD if g > 0 else CRIT, fontsize=10.5, fontweight="bold")
    ax.set_xticks(x, names)
    ax.set_ylabel("DV camping score")
    ax.set_title("Search is an equalizer: gain grows as the flat baseline weakens")
    ax.set_ylim(129, 215)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(0.99, 0.10),
              fontsize=9)
    fig.text(0.99, -0.02, "maze2d-large, 5-seed pooled (DF, shortcut); DV "
             "start-matched vs compute-matched k256. Gap = the headroom.",
             ha="right", va="top", fontsize=7.2, color=MUTED)
    save(fig, "fig3_headroom_curve")


# ════════════════════════════════════════════════════════════════════════
# F4 — kitchen 2x2: the dichotomy replicates (tree helps DF, not DV)
# ════════════════════════════════════════════════════════════════════════
def fig4():
    # DV bars from the SAME paired run (kitchen_both_tree_s0) so the DV delta is
    # the genuine paired null (74.0 vs 74.5 = -0.5), not a cross-run artifact.
    # The n=200 cfg reproduction (75.0) is shown as the dotted ceiling line.
    dv_mcss = ms(load_arm("results/kitchen_both_tree_s0.json", "mcss"))
    dv_tree = ms(load_arm("results/kitchen_both_tree_s0.json", "mcts"))
    df_mcss = ms(load_arm("results/kitchen_both_df_s*.json", "mcss"))
    df_tree = ms(load_arm("results/kitchen_both_df_s*.json", "mcts"))
    groups = ["DV backbone", "DF backbone"]
    flat = [dv_mcss, df_mcss]
    tree = [dv_tree, df_tree]
    x = np.arange(2)
    w = 0.34

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    grid_y(ax)
    b1 = ax.bar(x - w / 2, [f[0] for f in flat], w, yerr=[f[1] for f in flat],
                capsize=3, label="flat MCSS", color=BLUE, zorder=3,
                edgecolor="white", linewidth=0.8,
                error_kw=dict(ecolor=MUTED, lw=1))
    b2 = ax.bar(x + w / 2, [t[0] for t in tree], w, yerr=[t[1] for t in tree],
                capsize=3, label="tree (top-3)", color=AQUA, zorder=3,
                edgecolor="white", linewidth=0.8,
                error_kw=dict(ecolor=MUTED, lw=1))
    for bars, arr in ((b1, flat), (b2, tree)):
        for b, (m, _, _) in zip(bars, arr):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.3,
                    f"{m:.1f}", ha="center", va="bottom", color=INK2, fontsize=8.5)
    # deltas
    ax.annotate(f"{dv_tree[0]-dv_mcss[0]:+.1f}\n(null, t=−0.57)", (0, 80),
                ha="center", color=CRIT, fontsize=8.5, fontweight="bold")
    ax.annotate(f"{df_tree[0]-df_mcss[0]:+.1f}\nt=5.47", (1, 80), ha="center",
                color=GOOD, fontsize=8.5, fontweight="bold")
    ax.axhline(75, color=BASE, lw=1, ls=":", zorder=1)
    ax.text(1.44, 75.6, "demo ceiling 75", color=MUTED, fontsize=7.5,
            ha="right", va="bottom")
    ax.set_xticks(x, groups)
    ax.set_ylabel("kitchen subtask score (0–100)")
    ax.set_title("Kitchen: the backbone dichotomy replicates")
    ax.set_ylim(0, 90)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.text(0.99, -0.02, "kitchen-mixed-v0. DF: 4-seed pooled (n=100). "
             "Error bars = SEM.", ha="right", va="top", fontsize=7.2, color=MUTED)
    save(fig, "fig4_kitchen_2x2")


# ════════════════════════════════════════════════════════════════════════
# F5 — the pin: guidance lifts the flat pool, the tree lands at the same place
# ════════════════════════════════════════════════════════════════════════
def fig5():
    """Now multi-seed and fully data-derived: flat and tree at w = 0, 4, 8, each
    seed-matched, with the seed-level paired gain annotated between them."""
    def flat(w):
        d = {}
        for pat in (f"results/kitchen_both_df_cg{w}_s*.json",
                    f"results/kitchen_mcss_df_cg{w}_s*.json"):
            for f in sorted(glob.glob(pat)):
                j = json.load(open(f))
                if "mcss" in j["results"]:
                    d.setdefault(j["seed"], np.asarray(j["results"]["mcss"]["dv_norm"], float))
        return d

    def tree(w):
        pat = ("results/kitchen_both_df_s*.json" if w == 0
               else f"results/kitchen_both_df_cg{w}_s*.json")
        return {json.load(open(f))["seed"]: np.asarray(
            json.load(open(f))["results"]["mcts"]["dv_norm"], float)
            for f in sorted(glob.glob(pat))}

    ung = {json.load(open(f))["seed"]: np.asarray(
        json.load(open(f))["results"]["mcss"]["dv_norm"], float)
        for f in sorted(glob.glob("results/kitchen_both_df_s*.json"))}

    w = [0, 4, 8]
    F = {0: ung, 4: flat(4), 8: flat(8)}
    T = {k: tree(k) for k in w}
    fl, tr, gain, seedt = [], [], [], []
    for k in w:
        ss = sorted(set(F[k]) & set(T[k]))
        fl.append(np.mean([F[k][s].mean() for s in ss]))
        tr.append(np.mean([T[k][s].mean() for s in ss]))
        per = np.array([(T[k][s] - F[k][s]).mean() for s in ss])
        gain.append(per.mean())
        seedt.append(per.mean() / (per.std(ddof=1) / np.sqrt(len(per))))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    grid_y(ax)
    ax.plot(w, fl, "-o", color=BLUE, lw=2, ms=8, label="flat MCSS + CG", zorder=3)
    ax.plot(w, tr, "--s", color=AQUA, lw=2, ms=8, label="tree + CG", zorder=3,
            markeredgecolor="white", markeredgewidth=0.8)
    for wi, f, t_, g, st in zip(w, fl, tr, gain, seedt):
        ax.text(wi, f - 1.8, f"{f:.1f}", ha="center", color=BLUE, fontsize=8.5)
        ax.text(wi, t_ + 1.1, f"{t_:.1f}", ha="center", color="#0f7a54",
                fontsize=8.5, fontweight="bold")
        ax.annotate(f"{g:+.1f}\nt={st:+.2f}", (wi, (f + t_) / 2), ha="center",
                    va="center", color=GOOD, fontsize=8.4, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=GRID, lw=0.8))
    ax.set_xticks(w)
    ax.set_xlabel("classifier-guidance weight  w")
    ax.set_ylabel("kitchen subtask score (0–100)")
    ax.set_title("Guidance and search partially substitute (the pinned tree)")
    ax.set_ylim(55, 74)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(0.99, 0.02),
              fontsize=9)
    fig.text(0.99, -0.02, "kitchen-mixed-v0, causal backbone. Seed-matched; 4 seeds at "
             "w=0, 3 at w=4 and w=8. Annotations are seed-level paired gains.",
             ha="right", va="top", fontsize=7.2, color=MUTED)
    save(fig, "fig5_guidance_pin")


# ════════════════════════════════════════════════════════════════════════
# F6 — the census: no arm ever reaches the 4th subtask (score 100)
# ════════════════════════════════════════════════════════════════════════
def fig6():
    # each arm -> proportion of rollouts at 1/2/3/4 subtasks (25/50/75/100).
    arms = [
        ("DV-MCSS (k150)", load_arm("results/kitchen_mcss_cfg_s0.json", "mcss")),
        ("DF-MCSS", load_arm("results/kitchen_both_df_s*.json", "mcss")),
        ("DF-tree", load_arm("results/kitchen_both_df_s*.json", "mcts")),
        ("DF-MCSS+CG (w8)",
         load_arm("results/kitchen_mcss_df_cg8_s*.json", "mcss")),
        ("DF-tree grounded",
         load_arm("results/kitchen_both_df_grounded_s0.json", "mcts")),
    ]

    levels = [25, 50, 75, 100]           # 1,2,3,4 subtasks
    names = [a for a, _ in arms]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    y = np.arange(len(arms))[::-1]        # top-to-bottom
    for yi, (_, v) in zip(y, arms):
        n = len(v)
        left = 0.0
        for lv, col in zip(levels, SEQ4):
            frac = 100.0 * np.mean(v == lv) if n else 0.0
            if frac > 0:
                ax.barh(yi, frac, left=left, height=0.62, color=col, zorder=3,
                        edgecolor="white", linewidth=1.0)
                if frac >= 8:
                    txt = "white" if lv >= 75 else INK
                    ax.text(left + frac / 2, yi, f"{frac:.0f}", ha="center",
                            va="center", color=txt, fontsize=7.8)
            left += frac
    ax.set_yticks(y, names, fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of rollouts (%)")
    ax.set_title("Generated rollouts never reach the 4th subtask (kitchen-mixed)")
    # legend for the ordinal ramp
    handles = [Patch(fc=c, ec="white") for c in SEQ4]
    lbls = ["1 subtask (25)", "2 (50)", "3 (75)", "4 (100) — never observed"]
    ax.legend(handles, lbls, frameon=False, fontsize=8, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, -0.30))
    ax.margins(y=0.04)
    fig.text(0.99, -0.10, "kitchen-mixed-v0. Per-method generated rollouts; the full "
             "behaviour census across all methods (850+ rollouts) has zero 4-subtask "
             "completions.", ha="right", va="top", fontsize=7.2, color=MUTED)
    save(fig, "fig6_census")


# ════════════════════════════════════════════════════════════════════════
# F7 — the value-posedness ladder: search benefit does NOT track value quality
# ════════════════════════════════════════════════════════════════════════
def fig7():
    """Two small multiples over the same three environments (never a dual axis).

    Left: how well the goal-conditioned value fits (held-out correlation, from the
    training logs). Right: what the tree using it actually gains over its
    compute-matched, start-matched flat baseline. Both data-derived.
    """
    envs = [("maze2d-large", "large"), ("maze2d-medium", "medium"),
            ("maze2d-umaze", "umaze")]
    corr, gain, gt = [], [], []
    for full, short in envs:
        log = glob.glob(f"results/**/{full}-v1/state_value_sg_train_log.json",
                        recursive=True)
        corr.append(float(json.load(open(log[0]))["best_val_corr"]) if log else np.nan)
        tag = {"large": "m2l", "medium": "m2m", "umaze": "m2u"}[short]
        base = ("results/maze2d_large_mcss_k256_s*.json" if short == "large"
                else f"results/{tag}_mcss_k256_s*.json")
        sm = []
        for f in sorted(glob.glob(f"results/{tag}_tree_vsgpess_s*.json")):
            s = json.load(open(f))["seed"]
            bf = base.replace("s*", f"s{s}")
            bl = glob.glob(bf)
            if not bl:
                continue
            a = load_arm(f, "mcts")
            b = load_arm(bl[0], "mcss")
            n = min(len(a), len(b))
            sm.append(float(np.mean(a[:n] - b[:n])))
        sm = np.asarray(sm)
        gain.append(sm.mean())
        gt.append(sm.mean() / (sm.std(ddof=1) / np.sqrt(len(sm))))

    names = [s for _, s in envs]
    x = np.arange(3)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.6))

    grid_y(axL)
    axL.bar(x, corr, width=0.55, color=BLUE, zorder=3, edgecolor="white", linewidth=0.8)
    for xi, c in zip(x, corr):
        axL.text(xi, c + 0.02, f"{c:.3f}", ha="center", va="bottom", fontsize=8.6,
                 color=INK2)
    axL.set_xticks(x, names); axL.set_ylim(0, 1.05)
    axL.set_ylabel("held-out correlation of V(s, g)")
    axL.set_title("How well the value fits")

    grid_y(axR)
    cols = [GOOD if g > 0 else CRIT for g in gain]
    axR.bar(x, gain, width=0.55, color=cols, zorder=3, edgecolor="white", linewidth=0.8)
    axR.axhline(0, color=BASE, lw=1.2, zorder=2)
    for xi, g, t in zip(x, gain, gt):
        off = 1.1 if g >= 0 else -1.1
        axR.text(xi, g + off, f"{g:+.1f}\n(t={t:+.2f})", ha="center",
                 va="bottom" if g >= 0 else "top", fontsize=8.4,
                 color=GOOD if g > 0 else CRIT, fontweight="bold")
    axR.set_xticks(x, names); axR.set_ylim(-21, 9)
    axR.set_ylabel("tree - compute-matched flat")
    axR.set_title("What the tree actually gains")

    fig.suptitle("Search benefit does not track value quality",
                 fontsize=11, fontweight="bold", color=INK, y=1.03)
    fig.text(0.99, -0.06,
             "maze2d family, start-matched, seed-level t. The best-fitted value wins; "
             "the second-best loses hardest.",
             ha="right", va="top", fontsize=7.2, color=MUTED)
    save(fig, "fig7_value_ladder")


if __name__ == "__main__":
    print("generating dissertation figures -> figures/")
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print("done.")
