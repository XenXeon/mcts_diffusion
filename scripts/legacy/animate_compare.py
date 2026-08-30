"""scripts/animate_compare.py — MCSS vs MCTS, same scenario, side by side (the money shot).

Consumes the bundle from scripts/run_compare_trace.py (cmp_s{seed}.npz + _index.json) and
renders, for one scenario, the MCSS executed path next to the MCTS executed path on the maze,
plus a shared BFS-distance-to-goal curve. Pick a FIX scenario (MCSS fails, MCTS reaches) and
you see look-ahead routing through a corridor where greedy MPC committed wrong / tipped.

Static PNG by default (report-usable); --gif also writes a growing-path animation (slides).

    python scripts/animate_compare.py --seed 0 --env-idx 7            # static PNG
    python scripts/animate_compare.py --seed 0 --env-idx 7 --gif      # + animated GIF

Torch-free (numpy + matplotlib[+pillow for --gif]). The geodesic colouring is Rule-1 dev-only.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from mcts.instrument import maze_xy_to_colrow

C_MCSS, C_MCTS = "#c44e52", "#4c72b0"


def _load(in_dir, seed):
    ip = os.path.join(in_dir, f"cmp_s{seed}_index.json")
    if not os.path.exists(ip):
        sys.exit(f"no bundle: {ip} (run scripts/run_compare_trace.py first)")
    idx = json.load(open(ip))
    npz = np.load(os.path.join(in_dir, idx["npz"]), allow_pickle=False)
    return idx, npz


def _valid(xy):
    return xy[np.isfinite(xy).all(axis=1)]


def _draw_path(ax, maze, xy, color, label):
    wall = np.asarray(maze["wall"], float)
    ax.imshow(wall, cmap="Greys", origin="upper", interpolation="nearest", alpha=0.85)
    v = _valid(xy)
    if len(v):
        col, row = maze_xy_to_colrow(v, maze)
        ax.plot(col, row, "-", color=color, lw=1.8, alpha=0.85)
        ax.scatter([col[-1]], [row[-1]], c=color, s=70, edgecolor="k", zorder=6)
    ax.set_title(label, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def _mark(ax, maze, start, goal):
    sc, sr = maze_xy_to_colrow(np.asarray(start), maze)
    gc, gr = maze_xy_to_colrow(np.asarray(goal), maze)
    ax.scatter([sc], [sr], c="lime", s=80, marker="o", edgecolor="k", zorder=5, label="start")
    ax.scatter([gc], [gr], c="red", s=190, marker="*", edgecolor="k", zorder=5, label="goal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results/instr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-idx", type=int, required=True)
    ap.add_argument("--out-dir", default="results/figs/anim")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        sys.exit(f"matplotlib needed ({e!r})")

    idx, npz = _load(args.in_dir, args.seed)
    maze = idx["maze"]
    i = args.env_idx
    scen = next((s for s in idx["scenarios"] if s["env_idx"] == i), None)
    if scen is None or f"mcss_e{i}_xy" not in npz:
        sys.exit(f"env {i} not in bundle")
    mcss_xy = np.asarray(npz[f"mcss_e{i}_xy"], float)
    mcts_xy = np.asarray(npz[f"mcts_e{i}_xy"], float)
    mcss_d = np.asarray(npz[f"mcss_e{i}_dist"], float)
    mcts_d = np.asarray(npz[f"mcts_e{i}_dist"], float)
    sres = "REACH" if scen["mcss_success"] else "FAIL"
    tres = "REACH" if scen["mcts_success"] else "FAIL"
    os.makedirs(args.out_dir, exist_ok=True)

    fig, (axm, axt, axp) = plt.subplots(1, 3, figsize=(15.5, 5.2),
                                        gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    _draw_path(axm, maze, mcss_xy, C_MCSS, f"MCSS (k50)  —  {sres}")
    _mark(axm, maze, scen["start"], scen["goal"]); axm.legend(loc="upper right", fontsize=8)
    _draw_path(axt, maze, mcts_xy, C_MCTS, f"MCTS (b16)  —  {tres}")
    _mark(axt, maze, scen["start"], scen["goal"])
    axp.plot(np.where(np.isfinite(mcss_d), mcss_d, np.nan), color=C_MCSS, label="MCSS")
    axp.plot(np.where(np.isfinite(mcts_d), mcts_d, np.nan), color=C_MCTS, label="MCTS")
    axp.axhline(0, color="red", ls=":", lw=1)
    axp.set_xlabel("timestep"); axp.set_ylabel("BFS cells to goal")
    axp.set_title("progress to goal (down = closer)"); axp.legend(loc="upper right")
    axp.grid(alpha=0.25)
    fig.suptitle(f"seed {args.seed}  env {i}  —  same start/goal: MCSS {sres}, MCTS {tres}",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = os.path.join(args.out_dir, f"cmp_s{args.seed}e{i}.png")
    fig.savefig(png, dpi=150); print(f"wrote {png}")

    if args.gif:
        from matplotlib import animation
        T = len(mcss_xy)
        frames = list(range(0, T, max(1, args.stride)))

        def draw(t):
            for ax in (axm, axt):
                ax.clear()
            _draw_path(axm, maze, mcss_xy[:t + 1], C_MCSS, f"MCSS (k50)  —  {sres}")
            _mark(axm, maze, scen["start"], scen["goal"])
            _draw_path(axt, maze, mcts_xy[:t + 1], C_MCTS, f"MCTS (b16)  —  {tres}")
            _mark(axt, maze, scen["start"], scen["goal"])

        anim = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / args.fps)
        gif = os.path.join(args.out_dir, f"cmp_s{args.seed}e{i}.gif")
        anim.save(gif, writer=animation.PillowWriter(fps=args.fps)); print(f"wrote {gif}")
    plt.close(fig)


if __name__ == "__main__":
    main()
