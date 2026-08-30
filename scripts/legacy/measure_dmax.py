"""scripts/measure_dmax.py

Stage-0 prerequisite (plan v5.1 §3b): measure d_max on D4RL antmaze and apply the
discount rule γ ≥ 1 − 0.7/d_max. Dev-only oracle (Rule 1) — this number informs
the IQL-u branch's γ and the D1 band design; it never enters a trained component.

Run on the GPU box (needs gym + d4rl):
    python scripts/measure_dmax.py --env antmaze-large-diverse-v2 --diagnose
    python scripts/measure_dmax.py --env antmaze-large-diverse-v2
"""
import argparse
import sys

sys.path.insert(0, ".")

import numpy as np

from mcts.maze_oracle import AntMazeOracle, calibrate_steps_per_cell
from mcts.relabel import terminus_indices_from_tml
from mcts.specs import get_goal, make_dataset, spec_for


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="antmaze-large-diverse-v2")
    p.add_argument("--diagnose", action="store_true",
                   help="print the maze + start/goal marks to validate the "
                        "xy->cell transform, then exit (run this FIRST)")
    args = p.parse_args()

    env, ds = make_dataset(args.env)
    oracle = AntMazeOracle(env)
    env.reset()
    start_xy = (0.0, 0.0)                       # antmaze resets at the origin cell
    goal_xy = tuple(map(float, get_goal(env)))

    if args.diagnose:
        print(f"{args.env}: {oracle.n_rows}x{oracle.n_cols} cells, "
              f"scaling={oracle.scaling}, init=({oracle.init_x}, {oracle.init_y})")
        print(oracle.ascii_map(marks={"S": start_xy, "G": goal_xy}))
        print("\nIf S sits on the reset cell and G in the goal corner on free "
              "cells, the transform is correct. Then run without --diagnose.")
        return

    # Raw xy per path (seq_obs is normalised; invert with the state normaliser).
    normalizer = ds.get_normalizer()
    seq_obs = np.asarray(ds.seq_obs)
    terminus = terminus_indices_from_tml(np.asarray(ds.seq_tml))
    path_xys = []
    for p_idx in range(seq_obs.shape[0]):
        raw = normalizer.unnormalize(seq_obs[p_idx, :terminus[p_idx] + 1])
        path_xys.append(raw[:, :2])

    spc = calibrate_steps_per_cell(oracle, path_xys, terminus)
    dmax_cells = oracle.dmax_cells(start_xy)
    goal_cells = oracle.geodesic_cells(start_xy, goal_xy)
    d_max = dmax_cells * spc
    gamma = 1.0 - 0.7 / d_max

    print(f"steps-per-cell (median, data-calibrated): {spc:.1f}")
    print(f"d_max from start: {dmax_cells:.0f} cells = {d_max:.0f} dense steps")
    print(f"start->eval-goal geodesic: {goal_cells:.0f} cells = "
          f"{goal_cells * spc:.0f} dense steps")
    print(f"discount rule: gamma >= 1 - 0.7/d_max = {gamma:.5f}")
    print("(only consumed by the IQL-u branch, §3b — the MC critic at discount "
          "1.0 needs no gamma)")


if __name__ == "__main__":
    main()
