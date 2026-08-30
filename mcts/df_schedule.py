"""mcts/df_schedule.py

Torch-free noise-schedule helpers for the Diffusion Forcing planner
(mcts/df_model.py). Kept numpy-only so the local (torch-free) test box can
verify the scheduling matrices that drive DF sampling.

Diffusion Forcing (Chen et al., NeurIPS 2024, arXiv 2407.01392) trains a
causal model to denoise sequences where EVERY token carries its own noise
level k in {0..K} (0 = clean, K = pure noise). Sampling walks a 2D scheduling
matrix: rows are denoising sweeps, columns are sequence positions, entry =
the noise level position t must have after that sweep. Because the model is
trained on independent per-token levels, ANY matrix is in-distribution — the
pyramid matrix keeps the far future noisier than the near future ("causal
uncertainty"), which is the property full-sequence diffusion cannot express.

Level convention: alpha_bar[0] = 1 exactly (level 0 is the clean token /
"unmasked" case — DF's noising-as-masking view), alpha_bar[K] ~ 0.
"""
from __future__ import annotations

import numpy as np


def alpha_bar_cosine(K: int, s: float = 0.008) -> np.ndarray:
    """Nichol–Dhariwal cosine alpha-bar over levels 0..K, alpha_bar[0] == 1."""
    k = np.arange(K + 1, dtype=np.float64)
    f = np.cos((k / K + s) / (1.0 + s) * np.pi / 2.0) ** 2
    ab = f / f[0]
    ab[0] = 1.0                       # level 0 = clean, exactly
    return np.clip(ab, 1e-8, 1.0).astype(np.float32)


def pyramid_matrix(K: int, T: int, slope: int = 1, row_stride: int = 1) -> np.ndarray:
    """DF pyramid scheduling matrix, COLUMN-anchored: (M+1, T) int levels.

    Row 0 is the start state (everything at level K where not clean), the last
    row is all-zeros (fully denoised). Entry [m, t] = clip(m_level - slope *
    (T - 1 - t), 0, K) walked from the top: later columns (the far future)
    stay at higher noise while early columns denoise first — sampling then
    conditions each token on a MORE-resolved past, DF's causal-uncertainty
    ("zig-zag") scheme.

    Column-anchored means the level of column t does not depend on how many
    history tokens precede it — so ONE matrix serves a whole batch of nodes
    with different history lengths (history columns are simply forced to
    level 0 by the sampler). row_stride subsamples sweeps (like DDIM step
    subsampling): the first and last rows are always kept.
    """
    if K < 1 or T < 1 or slope < 1 or row_stride < 1:
        raise ValueError(f"K={K}, T={T}, slope={slope}, row_stride={row_stride} "
                         f"must all be >= 1")
    m_top = K + slope * (T - 1)             # level index at which every column = K
    rows = list(range(m_top, -1, -row_stride))
    if rows[-1] != 0:
        rows.append(0)                       # always end fully denoised
    t = np.arange(T)
    out = np.stack([np.clip(m - slope * (T - 1 - t), 0, K) for m in rows])
    return out.astype(np.int64)


def fullseq_matrix(K: int, T: int, row_stride: int = 1) -> np.ndarray:
    """All columns share one level per row — full-sequence-diffusion ablation."""
    if K < 1 or T < 1 or row_stride < 1:
        raise ValueError(f"K={K}, T={T}, row_stride={row_stride} must all be >= 1")
    rows = list(range(K, -1, -row_stride))
    if rows[-1] != 0:
        rows.append(0)
    return np.repeat(np.asarray(rows, dtype=np.int64)[:, None], T, axis=1)


def sample_training_levels(K: int, T: int, n: int, rng: np.random.Generator,
                           p_sched: float = 0.5, p_hist: float = 0.5,
                           slope: int = 1) -> np.ndarray:
    """(n, T) int64 per-token noise-level patterns for training the noise-
    aware critic (mcts/noise_critic.py), mixing TWO distributions:

    * uniform coverage (prob 1 - p_sched): each token's level drawn i.i.d. in
      {0..K} — the same distribution DFPlanner.loss trains the eps-net on.
      Needed so the critic has a well-defined V(x, k) everywhere the tree's
      dynamics-level noise perturbations (relabeling, rollout scoring) might
      probe it, not just the levels the sampler happens to visit.
    * pyramid + clean-prefix (prob p_sched): a row of pyramid_matrix(K, T,
      slope) — the EXACT per-token level pattern DFPlanner.sample walks
      through at inference — optionally (prob p_hist) with a clean history
      prefix of random length h forced to level 0, mirroring how the tree
      conditions expansion on h clean history tokens (MCSS uses h=1). This is
      the query distribution classifier guidance is actually evaluated on
      during sampling: eps <- eps - w*sqrt(1-alpha_bar[k])*grad_x V(x,k) is
      called at pyramid rows with a clamped clean prefix, never at a random
      i.i.d. pattern. Training on uniform alone would leave the critic
      under-trained exactly where guidance queries it.

    rng: numpy Generator (caller controls seeding, matching the rest of this
    module's torch-free / explicit-rng convention).
    """
    if K < 1 or T < 1 or n < 1:
        raise ValueError(f"K={K}, T={T}, n={n} must all be >= 1")
    if not (0.0 <= p_sched <= 1.0) or not (0.0 <= p_hist <= 1.0):
        raise ValueError(f"p_sched={p_sched}, p_hist={p_hist} must be in [0, 1]")
    out = rng.integers(0, K + 1, size=(n, T), dtype=np.int64)   # uniform default
    sched_mask = rng.random(n) < p_sched
    n_sched = int(sched_mask.sum())
    if n_sched > 0:
        mat = pyramid_matrix(K, T, slope)
        rows = mat[rng.integers(0, mat.shape[0], size=n_sched)]
        hist_mask = rng.random(n_sched) < p_hist
        if T > 1 and hist_mask.any():
            h = rng.integers(1, T, size=n_sched)          # {1..T-1}
            cols = np.arange(T)[None, :]
            clean = cols < h[:, None]
            rows = np.where(hist_mask[:, None] & clean, 0, rows)
        out[sched_mask] = rows
    return out
