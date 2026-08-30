"""mcts/mctd_verify.py

MCTD's value function = a NON-LEARNED geometric check on the (jumpy-denoised)
clean plan itself. Faithful port of
mctd-main/algorithms/diffusion_forcing/df_planning.py::calculate_values.

For each plan, walk it token by token:
  * if a single token-to-token position jump exceeds `warp_threshold`, the plan
    is dynamically infeasible ("Warp") -> value 0 and stop scoring it;
  * else, the first token that lands within `goal_radius` of the goal marks the
    plan "Achieved" at time t, with value (T - t) / T (reach sooner -> higher);
  * a plan that never warps and never reaches is "NotReached" -> value 0.

Everything is measured in RAW WORLD units (unnormalize the plan first), matching
the reference — so the thresholds are physical distances, not normalized ones.

MULTI-ENV NOTE. This geometric verifier is Way-1 faithful MCTD and applies only
where the goal is a POSITION in observation space: maze2d and antmaze (goal =
obs[:2]). Kitchen has no such position (its "goal" is a set of object-joint
subtasks in a ~30-D space), so the reference never defined a geometric value for
it and neither do we — kitchen would need a grounded subtask verifier (a Way-4c
variant reusing mcts/grounded.py), which is intentionally NOT this module. See
MCTD_ENV below: kitchen is absent on purpose.

TOKEN-SPACING CAVEAT. In the reference the plan is dense (consecutive timesteps),
so warp_threshold is a per-step displacement (~1.0 world unit). In THIS repo a
plan token is a stride-spaced waypoint (maze2d stride=15, antmaze stride=25 dense
steps apart), so a plausible token-to-token move is much larger. warp_threshold
is therefore a per-family, data-scaled value (see MCTD_ENV), NOT the reference's
1.0. Pass warp_threshold=None to disable the warp gate entirely.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Per-family geometric-verifier config. pos_dims selects the position channels
# of the observation; goal_radius / warp_threshold are in raw world units.
# These defaults are starting points to be tuned per env from data — the unit
# tests pass thresholds explicitly, so they do not depend on these numbers.
MCTD_ENV: Dict[str, Dict[str, Any]] = {
    "maze2d": dict(pos_dims=(0, 1), goal_radius=1.0, warp_threshold=6.0),
    "antmaze": dict(pos_dims=(0, 1), goal_radius=2.0, warp_threshold=12.0),
    # "kitchen": intentionally absent — no positional goal (see module docstring)
}


def geometric_values(plan_pos: np.ndarray, start_pos: np.ndarray,
                     goal_pos: np.ndarray, goal_radius: float,
                     warp_threshold: Optional[float] = None
                     ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Score a batch of plans by MCTD's geometric reachability heuristic.

    plan_pos:  (N, T, P) raw-world positions of each plan's T tokens.
    start_pos: (N, P) or (P,) raw-world start position (for the t=0 warp check).
    goal_pos:  (N, P) or (P,) raw-world goal position.
    goal_radius:    reach radius (world units).
    warp_threshold: max plausible token-to-token move; None disables the gate.

    Returns (values (N,), infos list[str] len N, achieved_t (N,) int, -1 if not).
    First event wins: a Warp before any reach yields value 0; a reach before any
    warp yields (T - t)/T. Ties broken by earliest token (loop order).
    """
    plan_pos = np.asarray(plan_pos, dtype=np.float64)
    if plan_pos.ndim != 3:
        raise ValueError(f"plan_pos must be (N, T, P); got {plan_pos.shape}")
    N, T, P = plan_pos.shape
    start_pos = np.broadcast_to(np.asarray(start_pos, dtype=np.float64).reshape(-1, P), (N, P))
    goal_pos = np.broadcast_to(np.asarray(goal_pos, dtype=np.float64).reshape(-1, P), (N, P))

    values = np.zeros(N, dtype=np.float64)
    infos = ["NotReached"] * N
    achieved_t = np.full(N, -1, dtype=np.int64)

    for t in range(T):
        cur = plan_pos[:, t, :]                                  # (N, P)
        prev = start_pos if t == 0 else plan_pos[:, t - 1, :]
        # unresolved = still "NotReached" (a plan resolves at its first event)
        unresolved = np.array([info == "NotReached" for info in infos])

        if warp_threshold is not None:
            step = np.linalg.norm(cur - prev, axis=-1)           # (N,)
            warp = (step > warp_threshold) & unresolved
            for i in np.nonzero(warp)[0]:
                infos[i] = "Warp"
                values[i] = 0.0
            unresolved = unresolved & ~warp

        d_goal = np.linalg.norm(cur - goal_pos, axis=-1)         # (N,)
        reach = (d_goal < goal_radius) & unresolved
        for i in np.nonzero(reach)[0]:
            infos[i] = "Achieved"
            values[i] = (T - t) / T
            achieved_t[i] = t

    return values, infos, achieved_t


def is_degenerate(plan_pos: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """True where a plan never moves (all token-to-token position steps < eps)
    — MCTD resamples such plans (df_planning.py num_tries_for_bad_plans). Works
    on raw-world positions. plan_pos: (N, T, P) -> (N,) bool."""
    plan_pos = np.asarray(plan_pos, dtype=np.float64)
    diffs = np.linalg.norm(plan_pos[:, 1:] - plan_pos[:, :-1], axis=-1)   # (N, T-1)
    return np.all(diffs < eps, axis=1)
