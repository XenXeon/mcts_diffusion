"""Tests for mcts/window.py::build_inpaint_prior (DF-inspired inpaint expansion).

Torch-free: the builder is pure numpy; the Sampler only converts its output to
tensors and swaps planner.fix_mask around the sample call.

The contract under test (what expand_fn relies on):
  * block i*k:(i+1)*k belongs to node i, all k rows identical;
  * prior rows [0:d_i] carry the prefix, row d_i the node state, rest zeros;
  * mask is 1.0 exactly on rows [0:d_i+1] (all feature dims), 0.0 elsewhere —
    with the denoiser's  xt = xt*(1-mask) + prior*mask  this clamps the whole
    search path plus the node state at every diffusion step;
  * root nodes (prefix None) reduce to the planner's ordinary row-0 conditioning;
  * a prefix that leaves no free rows raises (mirrors compose_window's guard).
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import pytest

from mcts.window import build_inpaint_prior

H, D, K = 8, 3, 4


def test_root_matches_row0_conditioning():
    states = np.arange(2 * D, dtype=np.float64).reshape(2, D)
    prior, mask, d_lens = build_inpaint_prior([None, None], states, H, K)
    assert prior.shape == mask.shape == (2 * K, H, D)
    assert d_lens.tolist() == [0, 0]
    # row 0 = state, all other rows zero; mask fixes only row 0
    for i in range(2):
        blk = prior[i * K:(i + 1) * K]
        assert np.allclose(blk[:, 0], states[i])
        assert np.all(blk[:, 1:] == 0.0)
        mblk = mask[i * K:(i + 1) * K]
        assert np.all(mblk[:, 0] == 1.0) and np.all(mblk[:, 1:] == 0.0)


def test_prefix_rows_state_row_and_mask_extent():
    rng = np.random.default_rng(0)
    pref = rng.normal(size=(3, D))
    state = rng.normal(size=(1, D))
    prior, mask, d_lens = build_inpaint_prior([pref], state, H, K)
    assert d_lens.tolist() == [3]
    assert np.allclose(prior[0, :3], pref.astype(np.float32))
    assert np.allclose(prior[0, 3], state[0].astype(np.float32))
    assert np.all(prior[0, 4:] == 0.0)
    # mask clamps rows [0:4] — prefix plus node state — and nothing beyond
    assert np.all(mask[0, :4] == 1.0) and np.all(mask[0, 4:] == 0.0)


def test_all_k_rows_in_a_block_identical():
    rng = np.random.default_rng(1)
    pref = rng.normal(size=(2, D))
    state = rng.normal(size=(1, D))
    prior, mask, _ = build_inpaint_prior([pref], state, H, K)
    for j in range(1, K):
        assert np.array_equal(prior[j], prior[0])
        assert np.array_equal(mask[j], mask[0])


def test_mixed_batch_keeps_per_node_depths():
    rng = np.random.default_rng(2)
    prefs = [None, rng.normal(size=(1, D)), rng.normal(size=(5, D))]
    states = rng.normal(size=(3, D))
    prior, mask, d_lens = build_inpaint_prior(prefs, states, H, K)
    assert d_lens.tolist() == [0, 1, 5]
    # per-node clamped-row count = d_i + 1
    for i, d in enumerate(d_lens):
        rows = mask[i * K].sum(axis=-1) > 0
        assert rows.sum() == d + 1
        assert np.allclose(prior[i * K, d], states[i].astype(np.float32))


def test_prefix_filling_window_raises():
    rng = np.random.default_rng(3)
    pref = rng.normal(size=(H - 1, D))       # d+1 == H -> zero free rows
    state = rng.normal(size=(1, D))
    with pytest.raises(ValueError, match="free rows"):
        build_inpaint_prior([pref], state, H, K)


def test_output_dtype_float32():
    pref = np.zeros((2, D), dtype=np.float64)
    state = np.ones((1, D), dtype=np.float64)
    prior, mask, _ = build_inpaint_prior([pref], state, H, K)
    assert prior.dtype == np.float32 and mask.dtype == np.float32
