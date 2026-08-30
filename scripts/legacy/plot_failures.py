"""scripts/plot_failures.py

Tier-1 figures: one panel per failed rollout —
  (left)  the executed (x, y) path overlaid on the maze wall map, with the start,
          the goal, and the candidate-endpoint cloud at the closest-approach
          (junction) step (the chosen endpoint highlighted);
  (right) the oracle progress curve (BFS cell distance to goal vs step) with the
          junction marked, and uprightness on a twin axis to spot a pose collapse.
Each panel is titled with the Tier-1 mode the classifier assigned, so the visual
and the tally are checked against each other by eye.

    python scripts/plot_failures.py --in-dir results/instr --out-dir results/instr/figs

Needs matplotlib (guarded — prints an install hint if absent). Reads only the npz
+ index.json (the maze geometry is stored in the index), so no oracle/env import:
these are DIAGNOSTIC-ONLY figures.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from mcts.failure_modes import classify_failure, progress_features
from mcts.instrument import record_from_npz


def _xy_to_colrow(xy, maze):
    """World xy -> fractional (col, row) for plotting on the imshow wall grid."""
    col = (np.asarray(xy)[..., 0] + maze["init_x"]) / maze["scaling"]
    row = (np.asarray(xy)[..., 1] + maze["init_y"]) / maze["scaling"]
    return col, row


def _plot_one(plt, npz, scen, maze, mode, out_path):
    i = scen["env_idx"]
    wall = np.asarray(maze["wall"], dtype=float)
    xy = npz[f"e{i}_xy"]
    dist = npz[f"e{i}_dist"]
    upright = npz[f"e{i}_upright"]
    f = progress_features([float(x) for x in dist])

    fig, (axm, axp) = plt.subplots(1, 2, figsize=(13, 5))

    # ── left: path on the wall map ───────────────────────────────────────────────
    axm.imshow(wall, cmap="Greys", origin="upper", interpolation="nearest", alpha=0.85)
    col, row = _xy_to_colrow(xy, maze)
    axm.plot(col, row, "-", color="tab:blue", lw=1.5, alpha=0.8, label="executed path")
    sc, sr = _xy_to_colrow(scen["start"], maze)
    gc, gr = _xy_to_colrow(scen["goal"], maze)
    axm.scatter([sc], [sr], c="lime", s=90, marker="o", edgecolor="k",
                zorder=5, label="start")
    axm.scatter([gc], [gr], c="red", s=140, marker="*", edgecolor="k",
                zorder=5, label="goal")
    # candidate cloud at the junction step (closest approach)
    if f.argmin_step >= 0 and f"e{i}_cand_xy" in npz:
        cand_xy = npz[f"e{i}_cand_xy"][f.argmin_step]          # (K,2)
        cc, cr = _xy_to_colrow(cand_xy, maze)
        axm.scatter(cc, cr, c="orange", s=14, alpha=0.7, zorder=4,
                    label="candidates @ junction")
        chosen_j = int(npz[f"e{i}_chosen_idx"][f.argmin_step])
        axm.scatter([cc[chosen_j]], [cr[chosen_j]], c="magenta", s=70, marker="X",
                    edgecolor="k", zorder=6, label="chosen")
        jc, jr = _xy_to_colrow(xy[f.argmin_step], maze)
        axm.scatter([jc], [jr], facecolors="none", edgecolors="blue", s=120,
                    zorder=6, label="junction state")
    axm.set_title(f"seed {scen['seed']} env {i} — path")
    axm.set_xlabel("maze col"); axm.set_ylabel("maze row")
    axm.legend(loc="upper right", fontsize=7, framealpha=0.9)

    # ── right: progress curve + uprightness ──────────────────────────────────────
    steps = np.arange(len(dist))
    dplot = np.where(np.isfinite(dist), dist, np.nan)
    axp.plot(steps, dplot, "-", color="tab:blue", label="BFS dist to goal (cells)")
    if f.argmin_step >= 0:
        axp.axvline(f.argmin_step, color="orange", ls="--", lw=1,
                    label=f"junction (min={f.min_dist:.0f})")
    offgraph = ~np.isfinite(np.asarray(dist, dtype=float))
    if offgraph.any():
        axp.scatter(steps[offgraph], np.full(offgraph.sum(), 0.0), c="red", s=8,
                    label="off-graph (in-wall/unreach)")
    axp.set_xlabel("step"); axp.set_ylabel("BFS cells to goal", color="tab:blue")
    axt = axp.twinx()
    axt.plot(steps, upright, color="tab:green", lw=1, alpha=0.6)
    axt.axhline(0.5, color="tab:green", ls=":", lw=1)
    axt.set_ylabel("uprightness", color="tab:green")
    axt.set_ylim(-1.1, 1.2)
    axp.set_title(f"mode: {mode}")
    axp.legend(loc="upper left", fontsize=7, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=str, default="results/instr")
    p.add_argument("--tag", type=str, default="instr_mcss_critic")
    p.add_argument("--out-dir", type=str, default="results/instr/figs")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="restrict to these seeds (default: all found)")
    p.add_argument("--max-per-seed", type=int, default=None,
                   help="cap figures per seed (default: all failures)")
    p.add_argument("--far-quantile", type=float, default=80.0,
                   help="start-distance percentile above which a failure is 'far'; "
                        "MUST match scripts/analyze_failures.py so the UNREACHABLE_FAR/"
                        "TIMEOUT_ON_TRACK label in a filename agrees with the tally")
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        sys.exit(f"matplotlib needed for plotting ({e!r}); pip install matplotlib")

    indices = []
    for ipath in sorted(glob.glob(os.path.join(args.in_dir, f"{args.tag}_s*_index.json"))):
        with open(ipath) as fh:
            indices.append(json.load(fh))
    if not indices:
        sys.exit(f"no runs found: {args.in_dir}/{args.tag}_s*_index.json")

    # far threshold over ALL loaded scenarios (mirrors analyze_failures._far_threshold;
    # deliberately NOT filtered by --seeds, so is_far — and thus the UNREACHABLE_FAR vs
    # TIMEOUT_ON_TRACK label baked into each filename — matches the analyzer's tally).
    all_starts = [s["start_geo_cells"] for idx in indices for s in idx["scenarios"]
                  if s.get("start_geo_cells") is not None]
    far_thr = float(np.percentile(all_starts, args.far_quantile)) if all_starts else math.inf

    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for idx in indices:
        if args.seeds is not None and idx["seed"] not in args.seeds:
            continue
        npz = np.load(os.path.join(args.in_dir, idx["npz"]), allow_pickle=False)
        maze = idx["maze"]
        made = 0
        for scen in idx["scenarios"]:
            if scen["success"]:
                continue
            i = scen["env_idx"]
            if f"e{i}_dist" not in npz:
                continue
            far = (scen.get("start_geo_cells") is not None
                   and scen["start_geo_cells"] >= far_thr)
            rec = record_from_npz(npz, i, reach_step=scen.get("reach_step"),
                                  is_far=far, goal=scen.get("goal"))
            mode, _ = classify_failure(rec)
            out_path = os.path.join(args.out_dir,
                                    f"s{idx['seed']}_e{i}_{mode}.png")
            _plot_one(plt, npz, scen, maze, mode, out_path)
            n += 1
            made += 1
            if args.max_per_seed and made >= args.max_per_seed:
                break
    print(f"wrote {n} figures -> {args.out_dir}  (far-threshold {far_thr:.1f} cells)")


if __name__ == "__main__":
    main()
