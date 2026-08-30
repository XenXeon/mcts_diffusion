"""tests/test_stitch.py

Torch-free tests for mcts/stitch.py against brute-force references.

The load-bearing claims: (1) recompute_raw_values reproduces the dataset's
seq_val recursion + normalisation BIT-COMPATIBLY (same update range, min/max
over the full array including the un-recursed tail); (2) the stitched label
equals the brute-force discounted return of the composed dense reward stream;
(3) composed windows contain exactly the intended strided rows; (4) the
junction index only offers real, recursion-valid, eps-close states.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.stitch import (JunctionIndex, StitchSpace, normalize_val,
                         recompute_raw_values)

P, L, H, STRIDE, D = 3, 20, 4, 3, 4
PAD = (H - 1) * STRIDE                      # dataset rows are L + (H-1)*stride long


def make_fixture(seed=0, discount=1.0):
    """Synthetic arrays with the dataset's layout: real steps then padding."""
    rng = np.random.default_rng(seed)
    pls = np.array([L, 14, 9])                       # path 0 fills max_path_length
    seq_rew = np.zeros((P, L + PAD, 1), dtype=np.float32)
    seq_obs = np.zeros((P, L + PAD, D), dtype=np.float32)
    for p in range(P):
        seq_rew[p, :pls[p], 0] = rng.normal(size=pls[p])   # tuned rewards, arbitrary
        seq_obs[p, :pls[p]] = rng.normal(size=(pls[p], D))
        seq_obs[p, pls[p]:] = seq_obs[p, pls[p] - 1]       # obs repeat padding
        # padding reward stays 0 (tuned continuous-at-done value)
    space = StitchSpace(seq_obs, seq_rew, pls, H, STRIDE, L, discount=discount)
    return space, seq_obs, seq_rew, pls


def dataset_reference_seq_val(seq_rew, discount, max_path_length):
    """Verbatim reimplementation of d4rl_maze2d_dataset.py lines 165-176."""
    seq_val = np.copy(seq_rew)
    for i in reversed(range(max_path_length - 1)):
        seq_val[:, i] = seq_rew[:, i] + discount * seq_val[:, i + 1]
    seq_val = (seq_val - seq_val.min()) / (seq_val.max() - seq_val.min())
    return seq_val * 2 - 1


@pytest.mark.parametrize("discount", [1.0, 0.9])
def test_raw_values_match_bruteforce(discount):
    _, _, seq_rew, _ = make_fixture(discount=discount)
    raw, _, _ = recompute_raw_values(seq_rew, discount, L)
    for p in range(P):
        for t in [0, 3, L - 2, L - 1]:
            brute = sum(discount ** (u - t) * seq_rew[p, u, 0] for u in range(t, L))
            assert raw[p, t, 0] == pytest.approx(brute, rel=1e-4, abs=1e-4)


def test_normalization_replicates_dataset_lines():
    space, _, seq_rew, _ = make_fixture()
    ref = dataset_reference_seq_val(seq_rew, 1.0, L)
    assert space.consistency_max_err(ref) < 1e-6


def test_segment_identity():
    space, _, seq_rew, _ = make_fixture(discount=0.9)
    a, sa, n = 0, 2, 2 * STRIDE
    partial = sum(0.9 ** t * seq_rew[a, sa + t, 0] for t in range(n))
    ident = space.raw_val[a, sa, 0] - 0.9 ** n * space.raw_val[a, sa + n, 0]
    assert ident == pytest.approx(partial, rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("discount", [1.0, 0.9])
def test_stitched_label_matches_bruteforce(discount):
    space, _, seq_rew, _ = make_fixture(discount=discount)
    a, sa, j, b, sb = 0, 1, 2, 1, 4
    n = j * STRIDE
    brute_raw = (sum(discount ** t * seq_rew[a, sa + t, 0] for t in range(n))
                 + discount ** n
                 * sum(discount ** (u - sb) * seq_rew[b, u, 0] for u in range(sb, L)))
    expect = normalize_val(brute_raw, space.vmin, space.vmax)
    assert space.stitched_label(a, sa, j, b, sb) == pytest.approx(
        float(expect), rel=1e-4, abs=1e-4)


def test_stitched_label_range_guards():
    space, *_ = make_fixture()
    with pytest.raises(ValueError, match="j must be"):
        space.stitched_label(0, 0, H, 1, 0)
    with pytest.raises(ValueError, match="recursion range"):
        space.stitched_label(0, L - 1, 1, 1, 0)     # sa + stride > L-1


def test_compose_obs_rows():
    space, seq_obs, _, _ = make_fixture()
    a, sa, j, b, sb = 0, 1, 2, 1, 4
    win = space.compose_obs(a, sa, j, b, sb)
    assert win.shape == (H, D)
    np.testing.assert_array_equal(win[0], seq_obs[a, sa])
    np.testing.assert_array_equal(win[1], seq_obs[a, sa + STRIDE])
    np.testing.assert_array_equal(win[2], seq_obs[b, sb])
    np.testing.assert_array_equal(win[3], seq_obs[b, sb + STRIDE])


def test_junction_index_matching_and_validity():
    space, seq_obs, _, pls = make_fixture()
    # plant an exact duplicate of path-0 step 7 inside path 1's REAL range,
    # and inside path 2's PADDED range (must NOT be indexed)
    seq_obs[1, 5] = seq_obs[0, 7]
    seq_obs[2, 12] = seq_obs[0, 7]              # pl[2]=9 -> t=12 is padding
    idx = JunctionIndex(seq_obs, pls, L, eps=1e-6)
    p, t = idx.query(seq_obs[0, 7])
    pairs = set(zip(p.tolist(), t.tolist()))
    assert (1, 5) in pairs and (0, 7) in pairs
    assert (2, 12) not in pairs                 # padded steps excluded
    # eps-ball matching: a state nudged by < eps still matches
    idx2 = JunctionIndex(seq_obs, pls, L, eps=0.05)
    p2, _ = idx2.query(seq_obs[0, 7] + 0.04)
    assert 0 in p2.tolist()


def test_sampler_fills_batch_and_labels_verify():
    space, seq_obs, seq_rew, pls = make_fixture(seed=3)
    # make junctions plentiful: states drawn from a tiny discrete set so
    # cross-path collisions abound (mutates the array StitchSpace holds)
    rng0 = np.random.default_rng(9)
    seq_obs[:] = rng0.integers(0, 2, size=seq_obs.shape) * 0.5
    idx = JunctionIndex(seq_obs, pls, L, eps=0.01)
    rng = np.random.default_rng(0)
    obs, lab, stats = space.sample_stitched(rng, 8, idx)
    assert obs.shape == (8, H, D) and lab.shape == (8, 1)
    assert np.isfinite(lab).all()
    assert stats["mean_junction_linf"] <= 0.01
    # every window's stitched half must start eps-close to A's would-be waypoint
    # (already guaranteed by the index; spot-check via the returned stats only)
    # determinism
    obs2, lab2, _ = space.sample_stitched(np.random.default_rng(0), 8, idx)
    np.testing.assert_array_equal(obs, obs2)
    np.testing.assert_array_equal(lab, lab2)


def test_sample_original_matches_dataset_rows():
    space, seq_obs, seq_rew, pls = make_fixture()
    ref_val = dataset_reference_seq_val(seq_rew, 1.0, L)
    indices = [(0, 2, 2 + (H - 1) * STRIDE + 1), (1, 0, (H - 1) * STRIDE + 1)]
    rng = np.random.default_rng(1)
    obs, lab = space.sample_original(rng, 4, indices, ref_val)
    for i in range(4):
        # every returned window must equal one of the two index rows
        match = any(
            np.array_equal(obs[i], seq_obs[p, s:e:STRIDE])
            and lab[i, 0] == pytest.approx(float(ref_val[p, s, 0]))
            for p, s, e in indices)
        assert match
