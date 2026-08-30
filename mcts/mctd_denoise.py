"""mcts/mctd_denoise.py

The one new piece of diffusion code MCTD needs on top of this repo's DF planner:
the ability to denoise a plan over an EXPLICIT slice of the scheduling matrix,
starting from a partially-denoised plan (not just fresh noise). MCTD uses this
twice per expansion (mctd-main/.../df_planning.py):

  * EXPANSION  — advance a node's plan one denoising BLOCK deeper (a contiguous
                 run of matrix rows), under a chosen guidance scale. The result
                 is the child node's stored partial plan.
  * SIMULATION — a fast JUMPY denoise of that child all the way to clean (matrix
                 rows subsampled by skip_level_steps), used only to compute the
                 node's value; it is thrown away afterward.

`denoise_rows` is the primitive for both. It is deliberately the SAME per-step
math as DFPlanner.sample's inner loop (DDIM x0-prediction, per-token levels,
clamped clean history, optional guidance eps-shift) — just driven by an explicit
row list instead of the full pyramid. tests/test_mctd_denoise.py pins this: over
the full matrix from fresh noise it reproduces DFPlanner.sample bit-for-bit, and
a block followed by its continuation equals the whole schedule. That equivalence
is what guarantees the MCTD arm is denoising correctly and not fabricating plans.

DFPlanner internals used (all public attributes): net_ema/net, sqrt_ab,
sqrt_1mab, K, dev, x0_clip. df_model.py is left untouched (this is purely
additive) so every existing DF/DV arm stays bit-identical.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch


def fresh_plan(planner, x_hist: torch.Tensor, hist_len: Union[int, torch.Tensor],
               temperature: float = 1.0) -> torch.Tensor:
    """Root init, IDENTICAL to DFPlanner.sample's first two lines: Gaussian
    noise with the clean history columns clamped in. x_hist: (n, T, D) whose
    rows [0:hist_len] are the history (rest ignored). Returns (n, T, D)."""
    n, T, D = x_hist.shape
    hist_mask = _hist_mask(hist_len, n, T, planner.dev)
    x = torch.randn(n, T, D, device=planner.dev) * temperature
    return torch.where(hist_mask.unsqueeze(-1), x_hist, x)


def _hist_mask(hist_len: Union[int, torch.Tensor], n: int, T: int,
               dev) -> torch.Tensor:
    cols = torch.arange(T, device=dev)
    if isinstance(hist_len, int):
        hl = torch.full((n,), hist_len, device=dev, dtype=torch.long)
    else:
        hl = hist_len.to(dev).long()
    return cols[None, :] < hl[:, None]                          # (n, T) bool


@torch.no_grad()
def denoise_rows(planner, x: torch.Tensor, rows: np.ndarray,
                 hist_len: Union[int, torch.Tensor] = 1,
                 x_hist: Optional[torch.Tensor] = None,
                 guide=None, w: float = 0.0, use_ema: bool = True
                 ) -> torch.Tensor:
    """Walk `x` from level rows[0] to level rows[-1] along the given matrix rows.

    x:    (n, T, D) plan currently AT per-token levels rows[0] (history columns
          excepted — they are held at level 0 throughout).
    rows: (R, T) int noise levels (numpy or tensor); rows[0] is where x sits now,
          rows[-1] is the target. A single row (R=1) is a no-op (used for a jumpy
          rollout that starts already clean).
    hist_len: clean history length per sample (int or (n,)); row 0 = start.
    x_hist:   the clean history to clamp; defaults to x at entry (its history
              columns are assumed already correct).
    guide/w:  optional GoalGuide (mcts/mctd_guidance.py) and guidance scale.

    Returns the denoised (n, T, D).
    """
    net = planner.net_ema if use_ema else planner.net
    n, T, D = x.shape
    rows_t = torch.as_tensor(np.asarray(rows), device=planner.dev, dtype=torch.long)
    if rows_t.ndim != 2 or rows_t.shape[1] != T:
        raise ValueError(f"rows must be (R, T={T}); got {tuple(rows_t.shape)}")
    hist_mask = _hist_mask(hist_len, n, T, planner.dev)         # (n, T)
    if x_hist is None:
        x_hist = x.clone()

    def levels(row_idx: int) -> torch.Tensor:
        k = rows_t[row_idx][None].expand(n, T).clone()
        return torch.where(hist_mask, torch.zeros_like(k), k)   # history clamped

    k_prev = levels(0)
    for m in range(1, rows_t.shape[0]):
        k_new = levels(m)
        sa_p = planner.sqrt_ab[k_prev].unsqueeze(-1)
        s1_p = planner.sqrt_1mab[k_prev].unsqueeze(-1)
        eps = net(x, k_prev)
        if guide is not None and w:
            with torch.enable_grad():
                xg = x.detach().requires_grad_(True)
                g = torch.autograd.grad(guide.value(xg, k_prev).sum(), xg)[0]
            g = torch.where(hist_mask.unsqueeze(-1), torch.zeros_like(g), g)
            eps = eps - w * s1_p * g
        upd = (k_new < k_prev).unsqueeze(-1)
        x0 = ((x - s1_p * eps) / sa_p.clamp_min(1e-4)).clamp(
            -planner.x0_clip, planner.x0_clip)
        x_next = (planner.sqrt_ab[k_new].unsqueeze(-1) * x0
                  + planner.sqrt_1mab[k_new].unsqueeze(-1) * eps)   # DDIM
        x = torch.where(upd, x_next, x)
        x = torch.where(hist_mask.unsqueeze(-1), x_hist, x)     # exact history
        k_prev = k_new
    return x


# ── matrix-slicing helpers (the tree-depth <-> denoising-block mapping) ───────

def block_rows(mat: np.ndarray, depth: int, block: int) -> np.ndarray:
    """Rows to denoise for a depth-`depth` child (1-indexed): the block
    [(depth-1)*block : depth*block], i.e. matrix rows
    [(depth-1)*block ... depth*block] inclusive (block+1 rows = block DDIM
    steps). Padded with the final (all-clean) row if the slice runs past the end
    — so the deepest block always finishes fully denoised, matching the
    reference's noise-level zero-padding."""
    if depth < 1:
        raise ValueError(f"depth must be >= 1; got {depth}")
    lo = (depth - 1) * block
    hi = depth * block + 1
    sl = mat[lo:hi]
    if sl.shape[0] < 2:                       # need >= 2 rows to take a step
        sl = np.concatenate([sl, mat[-1:]], axis=0) if sl.shape[0] == 1 else mat[-2:]
    if hi > mat.shape[0]:                     # pad the terminal block to clean
        pad = np.repeat(mat[-1:], hi - mat.shape[0], axis=0)
        sl = np.concatenate([sl, pad], axis=0)
    return sl


def jumpy_rows(mat: np.ndarray, depth: int, block: int, skip: int) -> np.ndarray:
    """Fast rollout schedule for a depth-`depth` node: from where its block ends
    (row depth*block) to clean, subsampled by `skip` (few big steps), always
    ending on the final all-clean row. If the node is already clean (block ended
    at/after the last row), returns a single clean row (a no-op rollout)."""
    start = depth * block
    if start >= mat.shape[0] - 1:
        return mat[-1:]                       # already clean -> no-op
    rows = list(range(start, mat.shape[0], max(1, skip)))
    if rows[-1] != mat.shape[0] - 1:
        rows.append(mat.shape[0] - 1)         # always finish clean
    return mat[rows]
