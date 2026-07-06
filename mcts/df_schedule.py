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
