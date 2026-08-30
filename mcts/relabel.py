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


def _first_terminus(row) -> int:
    """First index where tml == 1 in one path's tml row, else -1."""
    for i, v in enumerate(row):
        x = v[0] if hasattr(v, "__len__") else v
        if x >= 0.5:
            return i
    return -1


def terminus_indices_from_tml(seq_tml) -> List[int]:
    """First index where tml == 1 per path. Works on any (P, T, 1)/(P, T) nested
    sequence (numpy array or lists). For learn_policy=False antmaze data every
    stored path reaches a terminus, so a path with no tml==1 is a data error."""
    out: List[int] = []
    for p, row in enumerate(seq_tml):
        idx = _first_terminus(row)
        if idx < 0:
            raise ValueError(f"path {p} has no terminus (tml never 1) — "
                             f"unexpected for learn_policy=False antmaze data")
        out.append(idx)
    return out


def path_end_indices(seq_obs, seq_tml, tol: float = 1e-8) -> List[Tuple[int, bool]]:
    """Per-path (end_index, is_terminus) — the goal-cap for relabeling that works
    for BOTH terminus-reaching and timeout paths (full-data mode, plan v5.1).

    * Terminus path (tml has a 1): end = first tml==1. The terminus state is then
      repeat-padded after it, so last-non-zero would be wrong — tml is authoritative.
    * Timeout path (no tml==1, zero-padded after the real data): end = last
      non-zero state row. This is a valid future-goal cap (a real state), just not
      a goal-reach; the relabel target −(t'−t) is a correct distance label either
      way, so the 70/20/10 mixture is unchanged.

    Dispatches to a vectorised numpy path when given numpy arrays (the GPU side —
    ~25M scalar ops on full antmaze would be slow in pure Python), and to the
    pure-stdlib loop for nested lists (the local torch-free tests). Both verified
    identical on the test fixtures.
    """
    if hasattr(seq_obs, "shape"):
        return _path_end_indices_np(seq_obs, seq_tml, tol)
    out: List[Tuple[int, bool]] = []
    for p in range(len(seq_tml)):
        term = _first_terminus(seq_tml[p])
        if term >= 0:
            out.append((term, True))
            continue
        obs = seq_obs[p]
        last = 0
        for i in range(len(obs) - 1, -1, -1):
            if any(abs(float(x)) > tol for x in obs[i]):
                last = i
                break
        out.append((last, False))
    return out


def _path_end_indices_np(seq_obs, seq_tml, tol):
    """Vectorised path_end_indices for numpy arrays (GPU side)."""
    import numpy as np
    tml = np.asarray(seq_tml)
    if tml.ndim == 3:
        tml = tml[..., 0]
    is_term = tml >= 0.5                              # (P, T)
    has_term = is_term.any(axis=1)                    # (P,)
    term_idx = is_term.argmax(axis=1)                 # first True (0 if none)
    obs = np.asarray(seq_obs)
    # Identical predicate to the pure-Python branch (R-A): a row is real iff its
    # max-abs exceeds tol — NOT sum-abs, so the shipped numpy path and the tested
    # stdlib path are the same function by construction, not just on this data.
    real = (np.abs(obs) > tol).any(axis=2)            # (P, T) real-state mask
    has_real = real.any(axis=1)
    last_real = obs.shape[1] - 1 - real[:, ::-1].argmax(axis=1)  # last real index
    last_real = np.where(has_real, last_real, 0)
    end = np.where(has_term, term_idx, last_real)
    return [(int(e), bool(h)) for e, h in zip(end, has_term)]


def build_relabel_inputs(ds):
    """THE single relabel derivation shared by the trainer, D1, and D4 (R5.7a +
    audit D1-1/D1-2): given a built DV dataset, return
        (seq_obs[np], ends[list], term_only[list], scale[StepScale]).

    The D-anchor is `term_only` (terminus-reaching paths only) so the scale stays
    867 across V(s), V(s,g)-terminus, and V(s,g)-full — identical by construction.
    `ends` (all goal-caps, incl. timeout) drives the sampler/coverage. Routing all
    three scripts through this prevents the terminus-only-vs-full divergence that
    would silently over-credit a full-data critic on the stitched gate.
    """
    import numpy as np
    seq_obs = np.asarray(ds.seq_obs)
    ends_info = path_end_indices(seq_obs, np.asarray(ds.seq_tml))
    ends = [e for e, _ in ends_info]
    term_only = [e for e, is_t in ends_info if is_t]
    if not term_only:
        raise ValueError("no terminus-reaching paths — cannot anchor the D scale")
    return seq_obs, ends, term_only, StepScale.from_terminus_indices(term_only)


def path_val_split(num_paths: int, val_frac: float, seed: int):
    """Path-level holdout shared by the trainer and D4 so 'held-out' means the
    SAME paths in both (audit D1-2): same seed + same num_paths => same split.
    Returns (val_paths, train_paths)."""
    import numpy as np
    perm = np.random.default_rng(seed).permutation(num_paths)
    n_val = max(1, int(val_frac * num_paths))
    return perm[:n_val], perm[n_val:]


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
        terminus:  per-path goal-cap indices — terminus_indices_from_tml in
                   terminus-only mode, or the end-index of path_end_indices in
                   full-data mode (timeout paths capped at their last real state).
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
