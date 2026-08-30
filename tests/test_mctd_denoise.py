"""tests/test_mctd_denoise.py

Tests for the MCTD denoising primitive (mcts/mctd_denoise.py). The load-bearing
one is test_full_schedule_matches_dfplanner_sample: walking the whole pyramid
matrix with denoise_rows must reproduce DFPlanner.sample BIT-FOR-BIT (same random
init, same net). That is what certifies the MCTD arm denoises correctly rather
than fabricating plans — every other MCTD guarantee rides on the denoiser being
the trusted sampler, just sliced.

Uses a small RANDOM-WEIGHT DFPlanner (no checkpoint needed), so it runs anywhere
torch + cleandiffuser import.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

torch = pytest.importorskip("torch")
try:
    from mcts.df_model import DFPlanner
except Exception as exc:                       # cleandiffuser/torch missing
    pytest.skip(f"DFPlanner unavailable: {exc!r}", allow_module_level=True)

from mcts.df_schedule import pyramid_matrix
from mcts.mctd_denoise import (block_rows, denoise_rows, fresh_plan, jumpy_rows)

D, K, T = 4, 8, 6
N = 3


def _planner():
    torch.manual_seed(123)
    return DFPlanner(in_dim=D, K=K, d_model=32, n_heads=2, depth=1, emb_dim=16,
                     device="cpu")


def _x_hist(n=N):
    xh = torch.zeros(n, T, D)
    xh[:, 0] = torch.linspace(-1, 1, D)[None].repeat(n, 1)   # a nontrivial start
    return xh


def test_full_schedule_matches_dfplanner_sample():
    p = _planner()
    xh = _x_hist()
    hist = torch.ones(N, dtype=torch.long)
    torch.manual_seed(7)
    ref = p.sample(xh, hist, T, schedule="pyramid", slope=1, row_stride=1,
                   temperature=1.0)
    torch.manual_seed(7)                          # identical noise draw
    x0 = fresh_plan(p, xh, 1, temperature=1.0)
    mat = pyramid_matrix(K, T, slope=1, row_stride=1)
    mine = denoise_rows(p, x0, mat, hist_len=1, x_hist=xh)
    assert torch.allclose(ref, mine, atol=1e-5, rtol=1e-4)


def test_history_tokens_stay_clamped_exactly():
    p = _planner()
    xh = _x_hist()
    x0 = fresh_plan(p, xh, 1)
    mat = pyramid_matrix(K, T)
    out = denoise_rows(p, x0, mat, hist_len=1, x_hist=xh)
    assert torch.allclose(out[:, 0], xh[:, 0], atol=0)     # row 0 == start, exact


def test_block_then_continue_equals_full_walk():
    p = _planner()
    xh = _x_hist()
    torch.manual_seed(3)
    x0 = fresh_plan(p, xh, 1)
    mat = pyramid_matrix(K, T)
    full = denoise_rows(p, x0.clone(), mat, hist_len=1, x_hist=xh)
    m = mat.shape[0] // 2
    mid = denoise_rows(p, x0.clone(), mat[:m], hist_len=1, x_hist=xh)      # rows 0..m-1
    cont = denoise_rows(p, mid, mat[m - 1:], hist_len=1, x_hist=xh)        # rows m-1..end
    assert torch.allclose(full, cont, atol=1e-6)


def test_single_row_is_noop():
    p = _planner()
    xh = _x_hist()
    x0 = fresh_plan(p, xh, 1)
    mat = pyramid_matrix(K, T)
    out = denoise_rows(p, x0.clone(), mat[-1:], hist_len=1, x_hist=xh)
    assert torch.allclose(out, x0, atol=0)                 # no rows to step -> unchanged


def test_zero_weight_guidance_equals_unguided():
    p = _planner()
    xh = _x_hist()
    x0 = fresh_plan(p, xh, 1)
    mat = pyramid_matrix(K, T)

    class _Guide:                       # would change eps if w != 0
        def value(self, x, k):
            return x[..., :2].sum(dim=(1, 2))
    a = denoise_rows(p, x0.clone(), mat, hist_len=1, x_hist=xh, guide=None, w=0.0)
    b = denoise_rows(p, x0.clone(), mat, hist_len=1, x_hist=xh,
                     guide=_Guide(), w=0.0)                # w=0 -> branch skipped
    assert torch.allclose(a, b, atol=0)


# ── matrix-slicing helpers (numpy) ───────────────────────────────────────────

def test_block_rows_partition_and_pad():
    mat = pyramid_matrix(K, T)
    R = mat.shape[0]
    block = 5
    b1 = block_rows(mat, 1, block)
    assert np.array_equal(b1[0], mat[0])                   # depth 1 starts at top
    assert b1.shape[0] >= 2                                # at least one DDIM step
    # a deep block is padded so it finishes on the clean (all-zero) row
    deep = block_rows(mat, 3, block)
    assert np.all(deep[-1] == 0)


def test_jumpy_rows_end_clean_and_subsample():
    mat = pyramid_matrix(K, T)
    j = jumpy_rows(mat, depth=1, block=5, skip=3)
    assert np.all(j[-1] == 0)                              # rollout ends clean
    assert j.shape[0] < mat.shape[0]                       # fewer (jumpy) steps
    # a node already at/after the last row -> single clean row (no-op rollout)
    noop = jumpy_rows(mat, depth=100, block=5, skip=3)
    assert noop.shape[0] == 1 and np.all(noop[0] == 0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
