"""Torch-free tests for mcts/df_schedule.py (Diffusion Forcing schedules).

These matrices DRIVE the DF sampler: a wrong matrix silently produces windows
denoised in the wrong order (or never fully denoised), which would corrupt
every DF arm while still "working". The properties below are the contract
mcts/df_model.py::DFPlanner.sample relies on.
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import pytest

from mcts.df_schedule import alpha_bar_cosine, fullseq_matrix, pyramid_matrix


def test_alpha_bar_clean_at_zero_and_monotone():
    ab = alpha_bar_cosine(20)
    assert ab.shape == (21,)
    assert ab[0] == 1.0                      # level 0 = exactly clean
    assert np.all(np.diff(ab) < 0)           # strictly more noise per level
    assert ab[-1] < 0.05                     # level K ~ pure noise


def test_pyramid_starts_at_K_ends_at_zero():
    K, T = 10, 6
    m = pyramid_matrix(K, T)
    assert m.shape == (K + (T - 1) + 1, T)
    assert np.all(m[0] == K)                 # row 0: everything pure noise
    assert np.all(m[-1] == 0)                # last row: fully denoised


def test_pyramid_levels_decrease_down_rows_and_increase_with_t():
    m = pyramid_matrix(8, 5)
    assert np.all(np.diff(m.astype(int), axis=0) <= 0)   # denoising only
    assert np.all(np.diff(m.astype(int), axis=1) >= 0)   # far future noisier


def test_pyramid_early_tokens_reach_clean_first():
    m = pyramid_matrix(8, 5, slope=2)
    first_clean = (m == 0).argmax(axis=0)     # row index where col hits 0
    assert np.all(np.diff(first_clean) >= 0)  # col t clean no later than t+1


def test_pyramid_row_stride_keeps_endpoints():
    K, T = 10, 4
    full = pyramid_matrix(K, T, row_stride=1)
    sub = pyramid_matrix(K, T, row_stride=3)
    assert np.array_equal(sub[0], full[0])
    assert np.all(sub[-1] == 0)
    assert sub.shape[0] < full.shape[0]
    assert np.all(np.diff(sub.astype(int), axis=0) <= 0)


def test_fullseq_uniform_rows():
    m = fullseq_matrix(6, 3)
    assert m.shape == (7, 3)
    assert np.all(m == m[:, :1])              # every column identical
    assert np.all(m[0] == 6) and np.all(m[-1] == 0)


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        pyramid_matrix(0, 5)
    with pytest.raises(ValueError):
        pyramid_matrix(5, 5, slope=0)
    with pytest.raises(ValueError):
        fullseq_matrix(5, 5, row_stride=0)
