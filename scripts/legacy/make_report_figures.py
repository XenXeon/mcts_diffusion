"""scripts/make_report_figures.py — presentation/report figures from the results on disk.

Generates the static PNGs for the dissertation progress report (no GPU; reads the result
JSONs in results/ and the failure-trace npz in results/instr/). Figures:

  fig1_matched_compute   n=500 headline bars: MCSS k50 / MCSS k272 / MCTS b16 (binomial SEM),
                         annotating the matched-compute pair and the flat-scaling backfire.
  fig2_compute_scaling   the divergence: flat best-of-N slopes DOWN with compute, the tree
                         does not (n=150 full sweep — the only N with all five cells).
  fig3_ceiling_cluster   every sampler AND a perfect value cluster at ~76-83%; the cap is
                         locomotion (100% of failures are topples), not selection.
  fig4_topple_anatomy    one failed episode: BFS-distance, uprightness, torso height, speed
                         vs time — the ant approaches, then tips and lies motionless.
  fig5_fall_geometry     walls refuted (clearance at topple ≈ normal) + sharp-turn is a
                         symptom (near-reversal entering the stall, but capping it doesn't help).

    python scripts/make_report_figures.py --out-dir results/figs

Rule-1: the oracle/orc arms in fig3 are privileged ceiling probes — labelled DIAGNOSTIC and
never presented as achievable samplers. Needs matplotlib + numpy.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

# ---- shared style ---------------------------------------------------------------------
C_MCSS, C_MCTS, C_ORACLE, C_BAD = "#c44e52", "#4c72b0", "#8172b3", "#937860"
BAND = "#dddddd"


def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False})


def _pool_reach(pattern):
    """Pool the per-rollout success vectors across all seed files -> (reach%, sem%, n)."""
    succ = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        arm = d["results"][list(d["results"].keys())[0]]
        succ += [int(x) for x in arm["success"]]
    if not succ:
        return None
    a = np.asarray(succ, float)
    p = a.mean()
    return 100 * p, 100 * np.sqrt(p * (1 - p) / a.size), a.size


# ---- fig 1: matched-compute headline (n=500) ------------------------------------------
def fig1(out_dir):
    import matplotlib.pyplot as plt
    arms = [("MCSS\nk=50\n(50 cand)", "results/s10_mcss_k50_s*.json", C_MCSS),
            ("MCSS\nk=272\n(272 cand)", "results/s10_mcss_k272_s*.json", C_MCSS),
            ("MCTS\nb=16\n(272 cand)", "results/s10_mcts_b16_s*.json", C_MCTS)]
    vals = [(lbl, _pool_reach(pat), c) for lbl, pat, c in arms]
    vals = [(lbl, v, c) for lbl, v, c in vals if v]
    if len(vals) < 3:
        print("  fig1: missing s10 files, skip"); return
    labels = [v[0] for v in vals]
    reach = [v[1][0] for v in vals]
    sem = [v[1][1] for v in vals]
    n = vals[0][1][2]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    x = np.arange(3)
    bars = ax.bar(x, reach, yerr=sem, capsize=6, color=[v[2] for v in vals],
                  edgecolor="black", linewidth=0.8, width=0.62, alpha=0.92)
    for xi, r, s in zip(x, reach, sem):
        ax.text(xi, r + s + 0.6, f"{r:.1f}%", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("closed-loop reach %"); ax.set_ylim(60, 92)
    ax.set_title(f"Matched-compute comparison  (antmaze-large-diverse-v2, n={n})", pad=12)
    # matched-compute bracket k272 <-> b16 (both 272 cand/step)
    ax.annotate("", xy=(2, 86.3), xytext=(1, 86.3),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(1.5, 86.8, "same compute (272 cand/step)\ntree +4.2 pp, p = 0.12 (n.s.)",
            ha="center", va="bottom", fontsize=9)
    # backfire bracket k50 -> k272
    ax.annotate("", xy=(1, 82.5), xytext=(0, 82.5),
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.8))
    ax.text(0.5, 83.1, "flat best-of-N backfires (-4 pp)", ha="center", va="bottom",
            fontsize=9, color=C_BAD,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    fig.tight_layout()
    p = os.path.join(out_dir, "fig1_matched_compute.png"); fig.savefig(p); plt.close(fig)
    print(f"  wrote {p}  (k50 {reach[0]:.1f} / k272 {reach[1]:.1f} / b16 {reach[2]:.1f})")


# ---- fig 2: compute-scaling divergence (n=150 full sweep) -----------------------------
def fig2(out_dir):
    import matplotlib.pyplot as plt
    # n=150 (3-seed) pooled grid — writeup §5.3 (the only N with all 5 cells).
    mcss = [(50, 79.3), (272, 72.0)]
    mcts = [(80, 78.0), (144, 78.7), (272, 83.3)]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    mx, my = zip(*mcss); tx, ty = zip(*mcts)
    ax.plot(mx, my, "o-", color=C_MCSS, lw=2.2, ms=9, label="MCSS (flat best-of-N)")
    ax.plot(tx, ty, "s-", color=C_MCTS, lw=2.2, ms=9, label="MCTS (value-guided tree)")
    for xx, yy in mcss:
        ax.annotate(f"{yy:.1f}", (xx, yy), textcoords="offset points", xytext=(6, -14), color=C_MCSS)
    for xx, yy in mcts:
        ax.annotate(f"{yy:.1f}", (xx, yy), textcoords="offset points", xytext=(6, 8), color=C_MCTS)
    ax.set_xscale("log"); ax.set_xticks([50, 80, 144, 272])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("candidates / step (inference compute)")
    ax.set_ylabel("closed-loop reach %  (n=150, 3 seeds)")
    ax.set_ylim(68, 86)
    ax.set_title("Same compute, opposite slope: flat scaling backfires, the tree does not")
    ax.text(95, 73.2, "optimizer's curse:\nargmax over more critic\nscores picks the more\nover-estimated plan",
            fontsize=8.5, color=C_MCSS)
    ax.legend(loc="lower left")
    ax.text(0.99, 0.02, "at n=500 the gap narrows to +4.2 pp (n.s.); the sign is robust",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, style="italic", color="#555")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig2_compute_scaling.png"); fig.savefig(p); plt.close(fig)
    print(f"  wrote {p}")


# ---- fig 3: the ceiling cluster -------------------------------------------------------
def fig3(out_dir):
    import matplotlib.pyplot as plt
    # n=150 pooled, where every arm exists (writeup §5-§7). (*)=Rule-1 ceiling probe.
    rows = [("MCSS k=272 (flat-scaled)", 72.0, C_MCSS, False),
            ("MCTS b16 + V(s,g)", 76.7, C_MCTS, False),
            ("oracle flat re-rank (*)", 78.7, C_ORACLE, True),
            ("MCSS k=50 (cheap baseline)", 79.3, C_MCSS, False),
            ("oracle in the tree (*)", 82.0, C_ORACLE, True),
            ("MCTS b16 + V(s)", 83.3, C_MCTS, False)]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.axvspan(72, 83.3, color=BAND, alpha=0.7, zorder=0)
    y = np.arange(len(rows))
    for yi, (lbl, v, c, probe) in zip(y, rows):
        ax.barh(yi, v, color=c, edgecolor="black", linewidth=0.7, height=0.6,
                hatch="///" if probe else None, alpha=0.92)
        ax.text(v + 0.3, yi, f"{v:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(60, 90); ax.set_xlabel("closed-loop reach %  (n=150)")
    ax.set_title("Every sampler — and a perfect value — saturates at ~76-83%", pad=10)
    ax.text(80.6, 0.95,
            "The cap is not selection\nor value. 100% of the ~20%\nfailures are physical TOPPLES\n— the Ant's locomotion policy,\nbelow the sampler.",
            fontsize=9, va="center", ha="left",
            bbox=dict(boxstyle="round", fc="#fff3d6", ec="#caa84a"))
    fig.text(0.985, 0.015,
             "(*) Rule-1 DIAGNOSTIC ceiling probe (privileged geodesic) — not an achievable sampler",
             ha="right", fontsize=8, style="italic", color="#555")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    p = os.path.join(out_dir, "fig3_ceiling_cluster.png"); fig.savefig(p); plt.close(fig)
    print(f"  wrote {p}")


# ---- fig 4: anatomy of a topple -------------------------------------------------------
def _pick_topple(npz, scen):
    """Pick a clean MID-ROUTE topple: started upright (>0.8), ended flipped (<-0.5), got
    within a few cells of the goal but NOT to the goal cell (so 'reached then failed' is
    unambiguously a fall, not the goal-radius artifact), and tipped before the episode end."""
    cands = []
    for s in scen:
        i = s["env_idx"]
        if s["success"] or f"e{i}_upright" not in npz:
            continue
        up = np.asarray(npz[f"e{i}_upright"], float)
        if not (up[:max(1, len(up)//2)].max() > 0.8 and up.min() < -0.5):
            continue
        dist = np.asarray(npz[f"e{i}_dist"], float)
        mind = float(np.nanmin(np.where(np.isfinite(dist), dist, np.nan)))
        onset = int(np.argmax(up < 0.0)) if (up < 0.0).any() else len(up) - 1
        cands.append((s, mind, onset, len(up)))
    if not cands:
        return None
    enr = [c for c in cands if 2.0 <= c[1] <= 10.0 and c[2] < 0.85 * c[3]]
    pool = enr if enr else cands
    pool.sort(key=lambda c: (c[1], c[2]))         # closest approach, then earliest fall
    return pool[0][0]


def fig4(out_dir, tag="instr_mcss_critic", seed=0):
    import matplotlib.pyplot as plt
    idxp = f"results/instr/{tag}_s{seed}_index.json"
    if not os.path.exists(idxp):
        print(f"  fig4: {idxp} missing, skip"); return None
    idx = json.load(open(idxp))
    npz = np.load(f"results/instr/{idx['npz']}", allow_pickle=False)
    scen = _pick_topple(npz, idx["scenarios"])
    if scen is None:
        print("  fig4: no clean topple found, skip"); return None
    i = scen["env_idx"]
    up = np.asarray(npz[f"e{i}_upright"], float)
    h = np.asarray(npz[f"e{i}_height"], float)
    sp = np.asarray(npz[f"e{i}_speed"], float)
    dist = np.asarray(npz[f"e{i}_dist"], float)
    T = len(up); t = np.arange(T)
    # topple onset = first step uprightness drops below 0
    onset = int(np.argmax(up < 0.0)) if (up < 0.0).any() else T - 1
    fig, axs = plt.subplots(4, 1, figsize=(8.6, 7.8), sharex=True)
    axs[0].plot(t, np.where(np.isfinite(dist), dist, np.nan), color="#333"); axs[0].set_ylabel("BFS cells\nto goal")
    axs[0].set_title(f"Anatomy of a topple — {tag} seed {seed}, env {i} "
                     f"(reached cell {np.nanmin(dist):.0f}, then fell)")
    axs[1].plot(t, up, color=C_MCTS); axs[1].axhline(0.8, ls=":", c="green", lw=1)
    axs[1].axhline(-0.5, ls=":", c=C_MCSS, lw=1); axs[1].set_ylabel("uprightness\n(1=up,-1=flipped)")
    axs[1].set_ylim(-1.1, 1.1)
    axs[2].plot(t, h, color="#55a868"); axs[2].set_ylabel("torso\nheight")
    axs[3].plot(t, sp, color=C_BAD); axs[3].set_ylabel("planar\nspeed"); axs[3].set_xlabel("timestep")
    for ax in axs:
        ax.axvline(onset, color=C_MCSS, ls="--", lw=1.2, alpha=0.8)
    axs[1].annotate("topples", (onset, 0.0), textcoords="offset points", xytext=(8, 4),
                    color=C_MCSS, fontsize=9, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig4_topple_anatomy.png"); fig.savefig(p); plt.close(fig)
    print(f"  wrote {p}  (env {i}; onset step {onset}; end upright {up[-1]:.2f}, height {h[-1]:.2f})")
    return i


# ---- fig 5: fall geometry (walls refuted, turn = symptom) -----------------------------
def fig5(out_dir, seed=0):
    import matplotlib.pyplot as plt
    from diag_fall_geometry import analyse_tag
    tags = ["flatlog_k50gnt0", "flatlog_k50gnt30", "flatlog_k50smt20", "flatlog_k50fsf2m1"]
    short = {"flatlog_k50gnt0": "gnt0", "flatlog_k50gnt30": "gnt30",
             "flatlog_k50smt20": "smt20", "flatlog_k50fsf2m1": "fsf"}
    rows = [analyse_tag("results/instr", t, seed, 0.02, 25, 6) for t in tags]
    rows = [(short[r["tag"]], r) for r in rows if r]
    if not rows:
        print("  fig5: no flatlog tags, skip"); return
    labels = [r[0] for r in rows]
    x = np.arange(len(rows))
    med = lambda L: (np.median(L) if len(L) else np.nan)
    ratio = [med(r["onset_clear"]) / med(r["base_clear"]) if med(r["base_clear"]) else np.nan
             for _, r in rows]
    pre = [med(r["pre_turn"]) for _, r in rows]
    base = [med(r["base_turn"]) for _, r in rows]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.8))
    # panel A: wall clearance ratio
    axA.bar(x, ratio, color=C_ORACLE, edgecolor="black", width=0.6, alpha=0.92)
    axA.axhline(1.0, color="black", ls="--", lw=1.2)
    axA.set_xticks(x); axA.set_xticklabels(labels); axA.set_ylim(0, 1.4)
    axA.set_ylabel("clearance at topple ÷ moving baseline")
    for xi, v in zip(x, ratio):
        axA.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    # panel B: pre-turn vs base-turn
    w = 0.38
    axB.bar(x - w/2, pre, w, label="sharpest turn entering stall", color=C_MCSS, edgecolor="black")
    axB.bar(x + w/2, base, w, label="moving baseline", color=C_MCTS, edgecolor="black")
    axB.set_xticks(x); axB.set_xticklabels(labels); axB.set_ylabel("commanded turn (deg)")
    axB.set_title("H-turn is a SYMPTOM: a near-reversal precedes the\nstall, but capping it doesn't cut topples", fontsize=10.5)
    axA.set_title("H-wall REFUTED: topple clearance ≈ normal\n(ratio ≈ 1 → falls don't cluster at walls)", fontsize=10.5)
    axB.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig5_fall_geometry.png"); fig.savefig(p); plt.close(fig)
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/figs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    try:
        import matplotlib  # noqa
    except Exception as e:
        sys.exit(f"matplotlib needed ({e!r}); pip install matplotlib")
    _style()
    os.makedirs(args.out_dir, exist_ok=True)
    fig1(args.out_dir)
    fig2(args.out_dir)
    fig3(args.out_dir)
    env_idx = fig4(args.out_dir, seed=args.seed)
    fig5(args.out_dir, seed=args.seed)
    if env_idx is not None:
        print(f"\n  topple env for the GIF: --tag instr_mcss_critic --seed {args.seed} --env-idx {env_idx}")
    print(f"\ndone -> {args.out_dir}")


if __name__ == "__main__":
    main()
