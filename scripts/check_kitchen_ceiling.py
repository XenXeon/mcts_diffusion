"""scripts/check_kitchen_ceiling.py

The raw-dataset demonstration-ceiling verification (runbook §2.5.7e,
methodology_report §8.6, results_chapter §7). Prints, for a kitchen dataset,
the distribution of per-trajectory completed-subtask totals in the RAW data.

Why this matters: across all recorded kitchen-mixed rollouts (750 at the time
of writing) no method — DV-MCSS included — ever completed the 4th subtask.
The hypothesized mechanism is that kitchen-mixed's demonstrations never solve
all four tasks, so every LEARNED component (planner, policy, critic, noise
critic) carries labels/targets capped at the 3-task ceiling, and a learned
value cannot prefer a plan outside its label range. This script verifies the
data-side premise directly.

READ: if MAX < 4, the SUPERVISION ceiling is confirmed — no demonstration
solves all four tasks. Combined with the rollout census this supports the
claim "no learned-value method can prefer, and none was observed to produce,
a 4-task plan on kitchen-mixed." It does NOT prove 4 tasks are impossible in
the environment (kitchen-partial's demos do solve all 4, and DV scores 94
there) — the claim is about what this data can teach, not what the env allows.

REWARD SEMANTICS (corrected 2026-07-11 — the first version of this script got
it wrong): the DATASET's reward field is DENSE — reward at step i = the COUNT
of goal subtasks currently solved at that step (0..4). Evidence: per-trajectory
sums land in the hundreds (impossible for sparse +1 events, which cap at 4),
and the dataset loader's "max discounted return: 401.6" printout matches a
0.997-discounted dense 0-4 signal. The ENVIRONMENT at eval time pays sparse
completion events (hence the DV pipeline's [0,4] cumsum clip works closed-
loop) — the two reward forms differ. The demonstration-ceiling statistic is
therefore the per-trajectory MAX of the reward field: the most goal subtasks
ever simultaneously solved in that demonstration.

Run (GPU box / any box with d4rl):
    python scripts/check_kitchen_ceiling.py                          # mixed
    python scripts/check_kitchen_ceiling.py --env kitchen-partial-v0 # control
"""
import argparse
import sys

sys.path.insert(0, ".")

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="kitchen-mixed-v0")
    args = p.parse_args()

    import gym
    import d4rl  # noqa: F401  (registers the envs)

    env = gym.make(args.env)
    d = env.get_dataset()
    r = d["rewards"].astype(float).reshape(-1)
    done = (d["terminals"].astype(bool).reshape(-1)
            | d["timeouts"].astype(bool).reshape(-1))

    maxes, start = [], 0
    for i in np.where(done)[0]:
        maxes.append(r[start:i + 1].max())
        start = i + 1
    if start < len(r):                     # trailing partial trajectory
        maxes.append(r[start:].max())
    maxes = np.asarray(maxes)

    u, c = np.unique(np.round(maxes).astype(int), return_counts=True)
    print(f"[{args.env}] {len(maxes)} trajectories")
    print(f"per-trajectory MAX simultaneously-solved goal subtasks: "
          f"{dict(zip(u.tolist(), c.tolist()))}")
    print(f"MAX = {maxes.max():.1f}")
    if maxes.max() < 4:
        print("=> SUPERVISION CEILING CONFIRMED: no demonstration ever has "
              "all 4 goal subtasks solved — the planner never saw such "
              "states, the policy never saw such transitions, and both "
              "critics' return targets top out below the 4-task return. "
              "(Cite this distribution in the boundary argument.)")
    else:
        frac = float((maxes >= 4).mean())
        print(f"=> demonstrations DO reach all-4-solved in {frac:.1%} of "
              f"trajectories — the hard-wall interpretation does NOT hold "
              f"for this split; revisit the boundary argument.")


if __name__ == "__main__":
    main()
