"""mcts/relabel.py

Within-trajectory goal relabeling for the MC V(s, g) critic (plan v5.1 §3a).

For a state at time t in path p (terminus index T_p), a relabelled sample is
(s = seq_obs[p, t], g = xy of seq_obs[p, t'], target = scale.val(t' - t)) with the
goal index t' drawn from the pre-registered mixture:

    70%  future-state goal: t' = t + 1 + Geometric(mean = geo_mean), capped at T_p
    20%  terminus goal:     t' = T_p
    10%  current-state goal: t' = t      (target exactly 1.0 — pins the zero point)

Targets use the IDENTICAL pipeline affine (mcts/value_scale.StepScale) — see R5.7a.

The scalar core (`draw_goal_index`) is pure stdlib so the mixture/capping/target
logic is unit-tested on the torch-free local box; `sample_batch` is the vectorised
wrapper the trainer uses on the GPU box (imports numpy lazily).
"""
from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

from mcts.value_scale import StepScale

MIX_FUTURE, MIX_TERMINUS, MIX_CURRENT = 0.70, 0.20, 0.10


def terminus_indices_from_tml(seq_tml) -> List[int]:
    """First index where tml == 1 per path. Works on any (P, T, 1)/(P, T) nested
    sequence (numpy array or lists). For learn_policy=False antmaze data every
    stored path reaches a terminus, so a path with no tml==1 is a data error."""
    out: List[int] = []
    for p, row in enumerate(seq_tml):
        idx = -1
        for i, v in enumerate(row):
            x = v[0] if hasattr(v, "__len__") else v
            if x >= 0.5:
                idx = i
                break
        if idx < 0:
            raise ValueError(f"path {p} has no terminus (tml never 1) — "
                             f"unexpected for learn_policy=False antmaze data")
        out.append(idx)
    return out


def draw_goal_index(t: int, terminus: int, geo_mean: float,
                    rng: random.Random) -> int:
    """Draw t' >= t per the 70/20/10 mixture. Pure stdlib (locally tested)."""
    if not (0 <= t <= terminus):
        raise ValueError(f"t={t} outside [0, terminus={terminus}]")
    u = rng.random()
    if u < MIX_CURRENT:                      # 10% current-state: t' = t
        return t
    if u < MIX_CURRENT + MIX_TERMINUS:       # 20% terminus
        return terminus
    # 70% future-state: geometric offset with the given mean, >= 1, capped.
    # Geometric on {1, 2, ...} with success prob q has mean 1/q.
    q = 1.0 / max(geo_mean, 1.0)
    offset = 1 + int(math.log(max(rng.random(), 1e-12)) / math.log(1.0 - q)) \
        if q < 1.0 else 1
    return min(t + offset, terminus)


def make_sample(t: int, terminus: int, geo_mean: float, scale: StepScale,
                rng: random.Random) -> Tuple[int, float]:
    """(goal index t', target value) for one state. target = scale.val(t' - t)."""
    tp = draw_goal_index(t, terminus, geo_mean, rng)
    return tp, scale.val(tp - t)


def sample_batch(seq_obs, terminus: Sequence[int], scale: StepScale,
                 batch_size: int, geo_mean: float, rng,
                 paths: Sequence[int] = None):
    """Vectorised relabelled batch for the trainer (GPU box; numpy required).

    Args:
        seq_obs:   (P, T, obs_dim) normalised states (the DV dataset's seq_obs —
                   its xy dims ARE the goal representation, already normalised
                   with the state normaliser's [0:2] statistics).
        terminus:  per-path terminus indices (terminus_indices_from_tml).
        scale:     the shared StepScale affine.
        batch_size, geo_mean: as in draw_goal_index.
        rng:       numpy Generator.
        paths:     optional restriction (e.g. the train split's path ids).

    Returns (states (B, obs_dim), goals (B, 2), targets (B, 1)) as numpy arrays.
    """
    import numpy as np
    pool = np.asarray(paths if paths is not None else range(len(terminus)),
                      dtype=np.int64)
    term = np.asarray(terminus, dtype=np.int64)
    p = pool[rng.integers(0, len(pool), size=batch_size)]
    T = term[p]
    t = (rng.random(batch_size) * (T + 1)).astype(np.int64)   # uniform [0, T]

    u = rng.random(batch_size)
    tp = np.empty(batch_size, dtype=np.int64)
    cur = u < MIX_CURRENT
    ter = (~cur) & (u < MIX_CURRENT + MIX_TERMINUS)
    fut = ~(cur | ter)
    tp[cur] = t[cur]
    tp[ter] = T[ter]
    q = 1.0 / max(geo_mean, 1.0)
    n_fut = int(fut.sum())
    if q < 1.0:                                              # matches the scalar guard (B3)
        geo = 1 + np.floor(np.log(np.clip(rng.random(n_fut), 1e-12, None))
                           / np.log(1.0 - q)).astype(np.int64)
    else:                                                    # geo_mean <= 1 ⇒ offset 1
        geo = np.ones(n_fut, dtype=np.int64)
    tp[fut] = np.minimum(t[fut] + geo, T[fut])

    states = seq_obs[p, t]                                    # (B, obs_dim)
    goals = seq_obs[p, tp, :2]                                # (B, 2) normalised xy
    d = (tp - t).astype(np.float64)
    targets = scale.val_array(d)[:, None]                     # shared affine (C2)
    return states.astype(np.float32), goals.astype(np.float32), \
        targets.astype(np.float32)
