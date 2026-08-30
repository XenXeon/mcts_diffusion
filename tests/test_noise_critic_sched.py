"""Torch-free tests for mcts.df_schedule.sample_training_levels — the
per-token noise-level mixture the noise-aware critic (mcts/noise_critic.py)
trains on. Wrong mixing here silently under-trains the critic exactly where
classifier guidance queries it (pyramid rows with a clean history prefix),
while still "training fine" on the uniform-coverage majority — the same
"silently produces a corrupted-but-working arm" failure mode
tests/test_df_schedule.py documents for the sampling matrices.
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import pytest

from mcts.df_schedule import pyramid_matrix, sample_training_levels


def _pyramid_family_match(levels: np.ndarray, pyramid: np.ndarray,
                          min_h: int = 0) -> np.ndarray:
    """Boolean (n,): row i equals some pyramid row with columns [0:h) forced
    to zero, for some h in {min_h,...,T-1} (h=0 = exact match, no zeroing —
    pyramid rows can already have their own leading zeros, so exact matches
    are a degenerate case of "prefix zeroed" and are handled by the same
    lo/hi logic below rather than as a special case).
    """
    n, T = levels.shape
    eq = (levels[:, None, :] == pyramid[None, :, :])            # (n, R, T)
    # suf[i, r, h] = True iff eq[i, r, h:] are ALL True (suffix match at h)
    suf = np.minimum.accumulate(eq[:, :, ::-1].astype(np.int8), axis=2)[:, :, ::-1]
    has_any = suf[:, :, -1] == 1                                 # eq at T-1
    h0 = suf.argmax(axis=2)                # first h with a full suffix match
    nz = levels != 0
    first_nonzero = np.where(nz.any(axis=1), nz.argmax(axis=1), T)
    lo = np.maximum(min_h, h0)
    hi = np.minimum(first_nonzero[:, None], T - 1)
    ok = has_any & (lo <= hi)
    return ok.any(axis=1)


def _exact_pyramid_match(levels: np.ndarray, pyramid: np.ndarray) -> np.ndarray:
    """Boolean (n,): row i equals some pyramid row EXACTLY (no zeroing at all)."""
    return np.all(levels[:, None, :] == pyramid[None, :, :], axis=2).any(axis=1)


def test_shape_dtype_and_range():
    rng = np.random.default_rng(0)
    K, T, n = 10, 8, 200
    lv = sample_training_levels(K, T, n, rng)
    assert lv.shape == (n, T)
    assert lv.dtype == np.int64
    assert lv.min() >= 0 and lv.max() <= K


def test_determinism_same_seed():
    K, T, n = 10, 8, 200
    lv1 = sample_training_levels(K, T, n, np.random.default_rng(42))
    lv2 = sample_training_levels(K, T, n, np.random.default_rng(42))
    assert np.array_equal(lv1, lv2)


def test_p_sched_zero_is_mostly_uniform_not_pyramid():
    K, T, n = 20, 32, 4000
    rng = np.random.default_rng(1)
    lv = sample_training_levels(K, T, n, rng, p_sched=0.0)
    mat = pyramid_matrix(K, T, slope=1)
    frac = _exact_pyramid_match(lv, mat).mean()
    assert frac < 0.05                 # uniform coincidentally hitting a
                                        # pyramid row exactly should be rare


def test_p_sched_one_p_hist_zero_is_exact_pyramid_rows():
    K, T, n = 20, 32, 500
    rng = np.random.default_rng(2)
    lv = sample_training_levels(K, T, n, rng, p_sched=1.0, p_hist=0.0)
    mat = pyramid_matrix(K, T, slope=1)
    assert np.all(_exact_pyramid_match(lv, mat))


def test_p_sched_one_p_hist_one_is_pyramid_with_clean_prefix():
    K, T, n = 20, 32, 500
    rng = np.random.default_rng(3)
    lv = sample_training_levels(K, T, n, rng, p_sched=1.0, p_hist=1.0)
    mat = pyramid_matrix(K, T, slope=1)
    # every row must equal SOME pyramid row with SOME h in {1..T-1} zeroed
    assert np.all(_pyramid_family_match(lv, mat, min_h=1))


def test_mix_fraction_near_p_sched():
    K, T, n = 20, 32, 4000
    rng = np.random.default_rng(4)
    lv = sample_training_levels(K, T, n, rng, p_sched=0.5, p_hist=0.5)
    mat = pyramid_matrix(K, T, slope=1)
    frac = _pyramid_family_match(lv, mat, min_h=0).mean()
    assert 0.35 < frac < 0.65


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        sample_training_levels(20, 32, 0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        sample_training_levels(20, 32, 10, np.random.default_rng(0), p_sched=1.5)
    with pytest.raises(ValueError):
        sample_training_levels(0, 32, 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        sample_training_levels(20, 0, 10, np.random.default_rng(0))
