"""scripts/plot_candidates.py — Tier-3: the 50-candidate dispersion / mis-ranking view.

Reads the Tier-0 CRITIC dumps (results/instr/instr_mcss_critic_s*.{npz,index.json}) —
no torch, no re-run. It shows, concretely, WHY the DV critic loses the ~15 pp the
oracle recovers (§7 of the writeup): at a decision step the planner's 50 candidate
ENDPOINTS spread across the maze, several are genuinely goalward, and the critic's
argmax pick is NOT the geodesically-closest one.

Two kinds of output:
  * per-(episode, step) panels  -> {out}/cand_s{seed}e{idx}_t{step}.png
      left : 50 candidate endpoints on the wall map, coloured by TRUE BFS-geodesic to
             goal; the critic's pick (argmax score) and the oracle's pick (argmin
             geodesic) marked, plus current state + goal.
      right: scatter of DV-critic score vs true geodesic for the 50 — if the critic
             ranked by goal-distance this would slope down; the gap between the two
             marked picks is the per-step ranking loss.
  * an AGGREGATE of PER-DECISION stats + a rho histogram:
      per-decision Spearman rho(score, geodesic) — within-decision ranking quality (the
                                       thing selection depends on; NOT pooled across
                                       steps, which would be dominated by the between-
                                       state level trend);
      mis-rank rate                  — % of decisions where a candidate >=2 cells closer
                                       than the critic's pick existed (the headroom);
      mean (chosen - best) geodesic  — average cells left on the table per step.
    Rates are FAILURE-CONDITIONED (the dumps keep only failed episodes); run
    scripts/run_instrumentation.py --keep-success-frac 0.2 to also get the
    success-episode contrast.

    python scripts/plot_candidates.py --in-dir results/instr --out-dir results/instr/figs

matplotlib-guarded. Everything comes from the dump (cand_xy / cand_dist / cand_scores
/ chosen_idx per step), so this is DIAGNOSTIC-ONLY like the rest of the instrumentation.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from mcts.failure_modes import progress_features
from mcts.instrument import maze_xy_to_colrow

GOOD_MARGIN = 2.0   # a candidate >= this many cells closer than the critic's pick == "clearly better"


def _rankdata(a):
    """Average ranks (ties shared) — for Spearman without scipy."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _iter_traced(in_dir, tag):
    """Yield (idx, npz, scen) for every env that has a candidate trace (any outcome).
    scen['success'] gives the group; the dumps keep all FAILED envs and, if the run
    used --keep-success-frac, a sample of successful ones too."""
    for ipath in sorted(glob.glob(os.path.join(in_dir, f"{tag}_s*_index.json"))):
        with open(ipath) as fh:
            idx = json.load(fh)
        npz = np.load(os.path.join(in_dir, idx["npz"]), allow_pickle=False)
        for scen in idx["scenarios"]:
            if f"e{scen['env_idx']}_cand_dist" in npz:
                yield idx, npz, scen


def _decision_stats(in_dir, tag):
    """Per-DECISION ranking stats, split by episode outcome.

    Selection is a WITHIN-decision ranking of the 50 candidates, so the honest measure
    is Spearman of (critic score, geodesic) computed PER STEP over that step's <=50
    finite candidates — NOT pooled across steps (a pooled rho is dominated by the
    between-state level trend: far states score low and are far, which can look like
    good ranking even when the within-decision order is poor). 50 candidates is tiny,
    so no subsampling is needed. Returns {outcome: dict(...)} for outcome in
    {'failed','success'} that actually has traces.
    """
    acc = {False: dict(rhos=[], n=0, mis=0, gapsum=0.0),
           True: dict(rhos=[], n=0, mis=0, gapsum=0.0)}
    for idx, npz, scen in _iter_traced(in_dir, tag):
        i, succ = scen["env_idx"], bool(scen["success"])
        cd = npz[f"e{i}_cand_dist"]                       # (T,K) geodesic cells (inf=unreachable)
        cs = npz[f"e{i}_cand_scores"]                     # (T,K) DV critic scores
        ch = npz[f"e{i}_chosen_idx"]                      # (T,)
        a = acc[succ]
        for t in range(cd.shape[0]):
            d = cd[t]
            finite = np.isfinite(d)
            if finite.sum() < 2:
                continue
            a["n"] += 1
            best = d[finite].min()
            chosen_d = d[ch[t]] if np.isfinite(d[ch[t]]) else d[finite].max()
            gap = chosen_d - best
            a["gapsum"] += gap
            if gap >= GOOD_MARGIN:
                a["mis"] += 1
            if finite.sum() >= 3:                         # within-decision ranking quality
                rho = spearman(cs[t][finite], d[finite])
                if not math.isnan(rho):
                    a["rhos"].append(rho)
    out = {}
    for succ, name in ((False, "failed"), (True, "success")):
        a = acc[succ]
        if a["n"] == 0:
            continue
        rhos = np.array(a["rhos"]) if a["rhos"] else np.array([np.nan])
        out[name] = dict(decisions=a["n"],
                         rho_mean=float(np.nanmean(rhos)),
                         rho_median=float(np.nanmedian(rhos)),
                         frac_poorly_ranked=float(np.nanmean(rhos > -0.4)),
                         misrank_pct=100.0 * a["mis"] / a["n"],
                         mean_gap=a["gapsum"] / a["n"],
                         _rhos=rhos)
    return out


def aggregate(in_dir, tag, plt, out_dir):
    """Per-decision ranking stats (the selection-relevant measure) + a rho histogram."""
    stats = _decision_stats(in_dir, tag)
    if not stats:
        print("  no candidate data found")
        return None
    print("\n" + "=" * 70)
    print("CANDIDATE RANKING — DV critic vs true geodesic  (PER-DECISION over the 50)")
    has_succ = "success" in stats
    if not has_succ:
        print("  NOTE: failure-conditioned — dumps keep only failed episodes (re-run with")
        print("        --keep-success-frac for the success-episode contrast).")
    print(f"  {'group':<9} {'decisions':>9} {'rho_mean':>9} {'rho_med':>8} "
          f"{'%poorly<-.4':>11} {'misrank%':>9} {'meanGap':>8}")
    for name in ("failed", "success"):
        if name not in stats:
            continue
        s = stats[name]
        print(f"  {name:<9} {s['decisions']:>9} {s['rho_mean']:>+9.3f} "
              f"{s['rho_median']:>+8.3f} {100*s['frac_poorly_ranked']:>10.1f}% "
              f"{s['misrank_pct']:>8.1f}% {s['mean_gap']:>8.2f}")
    lead = stats["failed"]
    print(f"  => within-decision ranking is "
          f"{'WEAK' if lead['rho_median'] > -0.4 else 'reasonable'} on failures "
          f"(median rho {lead['rho_median']:+.2f}); a clearly-closer candidate "
          f"(>= {GOOD_MARGIN:.0f} cells) went unpicked in {lead['misrank_pct']:.0f}% of decisions.")
    print("=" * 70)
    # figure: distribution of per-decision rho (the selection-relevant quantity)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins = np.linspace(-1, 1, 41)
    for name, color in (("failed", "tab:red"), ("success", "tab:green")):
        if name in stats:
            r = stats[name]["_rhos"]
            r = r[~np.isnan(r)]
            if r.size:
                ax.hist(r, bins=bins, alpha=0.55, color=color, density=True,
                        label=f"{name} (median {np.median(r):+.2f})")
    ax.axvline(-0.4, color="k", ls="--", lw=1, label="rho = -0.4 (ranks goal-distance)")
    ax.set_xlabel("per-decision Spearman rho(critic score, true geodesic)")
    ax.set_ylabel("density")
    ax.set_title("Within-decision ranking quality\n(left of the dashed line = the critic "
                 "orders the 50 by goal-distance)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "cand_per_decision_rho.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"wrote {p}")
    return {k: {kk: vv for kk, vv in v.items() if kk != "_rhos"}
            for k, v in stats.items()}


def plot_one(plt, npz, scen, maze, i, step, out_path):
    cand_xy = npz[f"e{i}_cand_xy"][step]                  # (K,2)
    cand_dist = npz[f"e{i}_cand_dist"][step]              # (K,)
    cand_scores = npz[f"e{i}_cand_scores"][step]          # (K,)
    chosen = int(npz[f"e{i}_chosen_idx"][step])
    cur_xy = npz[f"e{i}_xy"][step]
    finite = np.isfinite(cand_dist)
    safe = np.where(finite, cand_dist, np.inf)
    oracle_pick = int(safe.argmin())
    wall = np.asarray(maze["wall"], dtype=float)

    fig, (axm, axs) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ── left: candidate endpoint cloud on the wall map ──────────────────────────
    axm.imshow(wall, cmap="Greys", origin="upper", interpolation="nearest", alpha=0.85)
    cc, cr = maze_xy_to_colrow(cand_xy, maze)
    sc = axm.scatter(cc[finite], cr[finite], c=cand_dist[finite], cmap="viridis_r",
                     s=26, edgecolor="k", linewidths=0.3, zorder=4)
    if (~finite).any():
        axm.scatter(cc[~finite], cr[~finite], marker="x", c="gray", s=24, zorder=4,
                    label="unreachable endpoint")
    ucur = maze_xy_to_colrow(cur_xy, maze)
    ug = maze_xy_to_colrow(scen["goal"], maze)
    axm.scatter([ucur[0]], [ucur[1]], facecolors="none", edgecolors="blue", s=140,
                linewidths=2, zorder=6, label="current state")
    axm.scatter([ug[0]], [ug[1]], c="red", s=160, marker="*", edgecolor="k",
                zorder=6, label="goal")
    axm.scatter([cc[chosen]], [cr[chosen]], marker="X", c="magenta", s=110,
                edgecolor="k", zorder=7, label=f"critic pick (geo={_g(cand_dist[chosen])})")
    axm.scatter([cc[oracle_pick]], [cr[oracle_pick]], marker="P", c="lime", s=110,
                edgecolor="k", zorder=7, label=f"oracle pick (geo={_g(cand_dist[oracle_pick])})")
    fig.colorbar(sc, ax=axm, fraction=0.046, pad=0.04, label="BFS cells to goal")
    axm.set_title(f"s{scen['seed']} e{i}  step {step} — 50 candidate endpoints")
    axm.legend(loc="upper right", fontsize=7, framealpha=0.9)

    # ── right: critic score vs geodesic (the mis-ranking) ───────────────────────
    axs.scatter(cand_dist[finite], cand_scores[finite], s=30, color="tab:blue",
                edgecolor="k", linewidths=0.3)
    axs.scatter([cand_dist[chosen]], [cand_scores[chosen]], marker="X", c="magenta",
                s=120, edgecolor="k", zorder=5, label="critic pick (max score)")
    if finite[oracle_pick]:
        axs.scatter([cand_dist[oracle_pick]], [cand_scores[oracle_pick]], marker="P",
                    c="lime", s=120, edgecolor="k", zorder=5, label="oracle pick (min geo)")
    axs.set_xlabel("true BFS geodesic to goal (cells)")
    axs.set_ylabel("DV critic score")
    axs.set_title("critic score vs goal-distance\n(critic picks max-y; oracle picks min-x)")
    axs.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _g(x):
    return "inf" if not np.isfinite(x) else f"{x:.0f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=str, default="results/instr")
    p.add_argument("--tag", type=str, default="instr_mcss_critic")
    p.add_argument("--out-dir", type=str, default="results/instr/figs")
    p.add_argument("--max-episodes", type=int, default=6,
                   help="how many failed episodes to draw cloud panels for")
    p.add_argument("--steps", type=int, nargs="+", default=None,
                   help="explicit step indices (default: start, junction, mid)")
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        sys.exit(f"matplotlib needed ({e!r}); pip install matplotlib")

    os.makedirs(args.out_dir, exist_ok=True)
    # aggregate stats + scatter first (the quantitative headline of the mis-ranking)
    agg = aggregate(args.in_dir, args.tag, plt, args.out_dir)

    n = 0
    for idx, npz, scen in _iter_traced(args.in_dir, args.tag):
        if n >= args.max_episodes:
            break
        if scen["success"]:           # cloud panels illustrate the FAILURES we're explaining
            continue
        i = scen["env_idx"]
        T = npz[f"e{i}_dist"].shape[0]
        if args.steps is not None:
            steps = [s for s in args.steps if 0 <= s < T]
        else:
            junction = progress_features([float(x) for x in npz[f"e{i}_dist"]]).argmin_step
            steps = sorted(set(s for s in (0, junction if junction >= 0 else T // 2, T // 2)
                               if 0 <= s < T))
        for step in steps:
            out = os.path.join(args.out_dir, f"cand_s{scen['seed']}e{i}_t{step}.png")
            plot_one(plt, npz, scen, idx["maze"], i, step, out)
        n += 1
    print(f"wrote cloud panels for {n} episodes -> {args.out_dir}")
    if agg:
        with open(os.path.join(args.out_dir, "cand_ranking_stats.json"), "w") as f:
            json.dump(agg, f, indent=2)


if __name__ == "__main__":
    main()
