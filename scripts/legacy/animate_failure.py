"""scripts/animate_failure.py — watch a failed episode unfold, step by step.

Numbers say "it failed"; this shows WHY. For a failed rollout it renders a GIF where,
at each step, you see: the ant's path so far on the wall map, the 50 candidate plan
endpoints (coloured by true geodesic-to-goal), the one the selector PICKED, the goal,
and — on a second panel — the BFS-distance-to-goal curve filling in. You can literally
watch it march into a dead end, ping-pong at a junction, stall just short, or crawl too
slowly to beat the horizon.

Consumes the per-failed-episode npz produced by either
  scripts/run_instrumentation.py            (tag instr_mcss_critic / instr_mcss_oracle), or
  scripts/diag_oracle_flat.py --log         (tag flatlog_k50fsf2m1, etc.)
both of which store e{i}_xy / e{i}_dist / e{i}_cand_xy / e{i}_cand_dist / e{i}_chosen_idx.

    python scripts/animate_failure.py --in-dir results/instr --tag instr_mcss_critic --seed 0 --max 3
    python scripts/animate_failure.py --in-dir results/instr --tag flatlog_k50fsf2m1 --seed 0 --env-idx 10

Needs matplotlib + pillow (GIF). DIAGNOSTIC-ONLY (the geodesic colouring is privileged).
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from mcts.instrument import maze_xy_to_colrow


def _animate_one(plt, animation, npz, scen, maze, stride, fps, out_path):
    i = scen["env_idx"]
    xy = np.asarray(npz[f"e{i}_xy"], dtype=float)              # (T,2) executed path
    dist = np.asarray(npz[f"e{i}_dist"], dtype=float)          # (T,) BFS cells to goal
    cand_xy = np.asarray(npz[f"e{i}_cand_xy"], dtype=float)    # (T,K,2) endpoints
    cand_dist = np.asarray(npz[f"e{i}_cand_dist"], dtype=float)  # (T,K)
    chosen = np.asarray(npz[f"e{i}_chosen_idx"], dtype=int)    # (T,)
    # chosen FIRST waypoint (the executed-toward point; the cause of the next step) — only
    # dumped by diag_oracle_flat --log, so optional.
    cfw = np.asarray(npz[f"e{i}_chosen_fw"], dtype=float) if f"e{i}_chosen_fw" in npz else None
    T = xy.shape[0]
    wall = np.asarray(maze["wall"], dtype=float)
    gcol, grow = maze_xy_to_colrow(scen["goal"], maze)
    scol, srow = maze_xy_to_colrow(scen["start"], maze)
    dplot = np.where(np.isfinite(dist), dist, np.nan)
    dmax = float(np.nanmax(dplot)) if np.isfinite(dplot).any() else 1.0
    cmax = float(np.nanmax(np.where(np.isfinite(cand_dist), cand_dist, np.nan)))
    cmax = cmax if math.isfinite(cmax) else dmax
    frames = list(range(0, T, max(1, stride)))

    fig, (axm, axp) = plt.subplots(1, 2, figsize=(13, 5.4))

    def draw(t):
        axm.clear(); axp.clear()
        # ── left: maze, path so far, candidate cloud at step t, chosen ──────────
        axm.imshow(wall, cmap="Greys", origin="upper", interpolation="nearest", alpha=0.85)
        pcol, prow = maze_xy_to_colrow(xy[:t + 1], maze)
        axm.plot(pcol, prow, "-", color="tab:blue", lw=1.4, alpha=0.7)
        cx, cy = maze_xy_to_colrow(cand_xy[t], maze)
        fin = np.isfinite(cand_dist[t])
        if fin.any():
            axm.scatter(cx[fin], cy[fin], c=cand_dist[t][fin], cmap="viridis_r",
                        s=18, vmin=0, vmax=cmax, edgecolor="k", linewidths=0.2, zorder=4)
        if (~fin).any():
            axm.scatter(cx[~fin], cy[~fin], marker="x", c="gray", s=14, zorder=4)
        j = int(chosen[t])
        axm.scatter([cx[j]], [cy[j]], marker="X", c="magenta", s=120, edgecolor="k",
                    zorder=6, label="picked plan endpoint")
        # the CAUSE: where the policy is steering this step (chosen first waypoint), with an
        # arrow from the ant — an aggressive/clipping first step shows up here directly.
        if cfw is not None:
            fcol, frow = maze_xy_to_colrow(cfw[t], maze)
            axm.annotate("", xy=(fcol, frow), xytext=(pcol[-1], prow[-1]),
                         arrowprops=dict(arrowstyle="->", color="cyan", lw=1.4), zorder=6)
            axm.scatter([fcol], [frow], marker="D", c="cyan", s=55, edgecolor="k",
                        zorder=7, label="picked first waypoint (steering toward)")
        axm.scatter([pcol[-1]], [prow[-1]], c="tab:blue", s=70, edgecolor="k",
                    zorder=6, label="ant now")
        axm.scatter([scol], [srow], c="lime", s=70, marker="o", edgecolor="k", zorder=5)
        axm.scatter([gcol], [grow], c="red", s=170, marker="*", edgecolor="k", zorder=5,
                    label="goal")
        axm.set_title(f"step {t}/{T}   dist-to-goal = "
                      f"{('%.0f' % dist[t]) if math.isfinite(dist[t]) else 'off-graph'} cells")
        axm.set_xlabel("maze col"); axm.set_ylabel("maze row")
        axm.legend(loc="upper right", fontsize=7, framealpha=0.9)
        # ── right: BFS-distance progress curve filling in ───────────────────────
        axp.plot(np.arange(t + 1), dplot[:t + 1], "-", color="tab:blue")
        if math.isfinite(dplot[t]):
            axp.scatter([t], [dplot[t]], c="magenta", s=40, zorder=5)
        axp.axhline(0, color="red", ls=":", lw=1, label="goal")
        axp.set_xlim(0, T); axp.set_ylim(-0.5, dmax * 1.05 + 0.5)
        axp.set_xlabel("step"); axp.set_ylabel("BFS cells to goal")
        axp.set_title("progress (down = closer; flat/up = stuck or wrong turn)")
        axp.legend(loc="upper right", fontsize=7)
        fig.suptitle(f"seed {scen['seed']}  env {i}  —  FAILED "
                     f"(min {np.nanmin(dplot):.0f} cells, ended {dplot[-1]:.0f})", fontsize=11)

    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="results/instr")
    p.add_argument("--tag", default="instr_mcss_critic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-idx", type=int, default=None,
                   help="animate this scenario only (default: first --max failures)")
    p.add_argument("--max", type=int, default=3, help="how many failed episodes to render")
    p.add_argument("--stride", type=int, default=8, help="render every Nth step (smaller=smoother/bigger)")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--out-dir", default="results/instr/anim")
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except Exception as e:
        sys.exit(f"matplotlib+pillow needed ({e!r}); pip install matplotlib pillow")

    ipath = os.path.join(args.in_dir, f"{args.tag}_s{args.seed}_index.json")
    matches = glob.glob(ipath)
    if not matches:
        sys.exit(f"no index found: {ipath}")
    with open(matches[0]) as fh:
        idx = json.load(fh)
    npz = np.load(os.path.join(args.in_dir, idx["npz"]), allow_pickle=False)
    maze = idx["maze"]

    os.makedirs(args.out_dir, exist_ok=True)
    failed = [s for s in idx["scenarios"]
              if not s["success"] and f"e{s['env_idx']}_xy" in npz]
    if args.env_idx is not None:
        failed = [s for s in failed if s["env_idx"] == args.env_idx]
        if not failed:
            sys.exit(f"env {args.env_idx} not a logged failure in {args.tag}_s{args.seed}")
    n = 0
    for scen in failed[:args.max]:
        out = os.path.join(args.out_dir, f"anim_{args.tag}_s{args.seed}e{scen['env_idx']}.gif")
        _animate_one(plt, animation, npz, scen, maze, args.stride, args.fps, out)
        print(f"wrote {out}")
        n += 1
    print(f"done: {n} animation(s) -> {args.out_dir}")


if __name__ == "__main__":
    main()
