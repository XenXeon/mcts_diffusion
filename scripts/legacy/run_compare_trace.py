"""scripts/run_compare_trace.py — the "money shot": trace MCSS vs MCTS on the SAME scenarios.

Runs the real DV-critic MCSS (k50) and the V(s) MCTS (b16) closed-loop with per-step
executed-path logging (run_episodes trace=True) on one seed, then dumps an
animate_compare-compatible bundle. Use it to render a scenario where MCTS REACHES a goal that
MCSS FAILS — the visual proof that look-ahead rescues a wrong early commitment (the brief).

This is the ONE presentation asset that needs a GPU run (torch/d4rl). ~0.6 h k50 + ~0.6 h b16
at n=50. Everything else (the 5 static report figures + the topple GIF) is already produced
locally by scripts/make_report_figures.py + scripts/animate_failure.py.

    python scripts/run_compare_trace.py --env antmaze-large-diverse-v2 --seed 0 --n-envs 50
    # it prints the FIX env indices (MCSS fail, MCTS reach); then, locally:
    python scripts/animate_compare.py --seed 0 --env-idx <a FIX idx>

NB the executed paths use the real samplers; the geodesic distance overlaid by the animator is
Rule-1 dev-only colouring (a measurement aid), not part of the sampler.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, ".")

import numpy as np

from mcts.maze_oracle import AntMazeOracle
from mcts.mcts_loop import Sampler, load_models, run_episodes


def _geo(oracle, grid, xy):
    r, c = oracle.cell(xy)
    return float(grid[r][c])


def _dist_track(oracle, goal, xy_T2):
    """Per-step BFS-geodesic distance (cells) along an executed path; NaN where the path is
    already finished (NaN xy) or off-graph."""
    grid = oracle.dist_grid_from(np.asarray(goal, dtype=np.float64))
    out = np.full(len(xy_T2), np.nan, dtype=np.float32)
    for t, xy in enumerate(xy_T2):
        if np.isfinite(xy).all():
            out[t] = _geo(oracle, grid, xy)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="antmaze-large-diverse-v2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=50)
    p.add_argument("--k", type=int, default=50, help="MCSS candidates/step")
    p.add_argument("--budget", type=int, default=16)
    p.add_argument("--k-mcts", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out-dir", default="results/instr")
    p.add_argument("--device", default=None)
    p.add_argument("--critic-step", type=int, default=1000000)
    args = p.parse_args()

    models = load_models(args.env, critic_step=args.critic_step, device=args.device)
    oracle = AntMazeOracle(models["env_single"])
    os.makedirs(args.out_dir, exist_ok=True)

    runs = {}
    print("=== MCSS k%d (DV critic) ===" % args.k)
    smp = Sampler(models, k_mcss=args.k, value_mode="v_s")
    runs["mcss"] = run_episodes(smp, "mcss", args.n_envs, 1, seed=args.seed,
                                max_steps=args.max_steps, trace=True, verbose=True)
    print("=== MCTS b%d k%d (V(s)) ===" % (args.budget, args.k_mcts))
    smp = Sampler(models, k_mcts=args.k_mcts, budget=args.budget, value_mode="v_s")
    runs["mcts"] = run_episodes(smp, "mcts", args.n_envs, 1, seed=args.seed,
                                max_steps=args.max_steps, trace=True, verbose=True)

    goals = runs["mcss"]["goals"]
    starts = runs["mcss"]["starts"]
    arrays = {}
    for tag in ("mcss", "mcts"):
        tr = np.asarray(runs[tag]["trace_xy"], dtype=np.float32)     # (T, n_envs, 2)
        for i in range(args.n_envs):
            if goals[i] is None:
                continue
            xy = tr[:, i, :]
            arrays[f"{tag}_e{i}_xy"] = xy
            arrays[f"{tag}_e{i}_dist"] = _dist_track(oracle, goals[i], xy)
    npz = os.path.join(args.out_dir, f"cmp_s{args.seed}.npz")
    np.savez_compressed(npz, **arrays)

    scen = [dict(env_idx=i, goal=[float(g[0]), float(g[1])] if goals[i] else None,
                 start=[float(starts[i][0]), float(starts[i][1])],
                 mcss_success=bool(runs["mcss"]["success"][i]),
                 mcts_success=bool(runs["mcts"]["success"][i]))
            for i in range(args.n_envs)]
    index = dict(DIAGNOSTIC_ONLY=True, env=args.env, seed=int(args.seed),
                 npz=os.path.basename(npz),
                 maze=dict(wall=[[1 if w else 0 for w in row] for row in oracle.wall],
                           scaling=oracle.scaling, init_x=oracle.init_x,
                           init_y=oracle.init_y, n_rows=oracle.n_rows, n_cols=oracle.n_cols),
                 mcss_reach=runs["mcss"]["reach_pct"], mcts_reach=runs["mcts"]["reach_pct"],
                 scenarios=scen)
    with open(os.path.join(args.out_dir, f"cmp_s{args.seed}_index.json"), "w") as f:
        json.dump(index, f, indent=2)

    fix = [s["env_idx"] for s in scen if s["goal"] and not s["mcss_success"] and s["mcts_success"]]
    brk = [s["env_idx"] for s in scen if s["goal"] and s["mcss_success"] and not s["mcts_success"]]
    print(f"\nMCSS {runs['mcss']['reach_pct']:.1f}%  MCTS {runs['mcts']['reach_pct']:.1f}%  "
          f"-> {npz}")
    print(f"FIX  (MCSS fail, MCTS reach) — the money shot: {fix}")
    print(f"BREAK(MCSS reach, MCTS fail):                  {brk}")
    print(f"render:  python scripts/animate_compare.py --seed {args.seed} "
          f"--env-idx {fix[0] if fix else '<idx>'}")


if __name__ == "__main__":
    main()
