"""tests/test_relabel_scale_coverage.py

Pure-stdlib tests for the week-1 v5.1 modules: the shared value-scale affine
(R5.7a), the 70/20/10 relabeling mixture (§3a), and the trajectory two-point
coverage stratum (R5.1). No torch/numpy — runs on the local box.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.coverage import TrajectoryCoverage
from mcts.relabel import (draw_goal_index, make_sample, path_end_indices,
                          terminus_indices_from_tml)
from mcts.value_scale import StepScale


# ── StepScale ────────────────────────────────────────────────────────────────

def test_scale_endpoints_and_inverse():
    s = StepScale(867)
    assert s.val(0) == 1.0                      # the pinned zero point / terminal value
    assert s.val(867) == -1.0
    assert abs(s.steps(s.val(123)) - 123) < 1e-9
    assert s.val(2000) == -1.0                  # beyond-range offsets clip at -1


def test_scale_from_terminus_and_consistency():
    s = StepScale.from_terminus_indices([100, 867, 50])
    assert s.D == 867
    s.assert_consistent_with_dataset(seq_val_start=s.val(100), terminus_index=100)
    try:
        s.assert_consistent_with_dataset(seq_val_start=0.5, terminus_index=100)
    except AssertionError:
        return
    raise AssertionError("expected scale-mismatch assertion to fire")


# ── Relabeling ───────────────────────────────────────────────────────────────

def test_terminus_from_tml_nested_and_flat():
    seq_tml = [[[0.0], [0.0], [1.0], [1.0]],     # terminus at 2 (nested (T,1))
               [[1.0], [1.0], [1.0], [1.0]]]     # terminus at 0
    assert terminus_indices_from_tml(seq_tml) == [2, 0]
    assert terminus_indices_from_tml([[0, 0, 0, 1]]) == [3]
    try:
        terminus_indices_from_tml([[0, 0, 0, 0]])
    except ValueError:
        return
    raise AssertionError("expected ValueError for a path with no terminus")


def test_path_end_indices_terminus_and_timeout():
    # path 0: terminus at index 2 (tml==1), then repeat-padding (terminus state
    #   repeated) — last-non-zero would be wrong, so tml must win.
    # path 1: timeout, no tml==1, real states [0:3] then zero-padding — end = 2.
    seq_tml = [[[0.0], [0.0], [1.0], [1.0], [1.0]],     # terminus path
               [[0.0], [0.0], [0.0], [0.0], [0.0]]]     # timeout path
    seq_obs = [[[1, 1], [1, 1], [9, 9], [9, 9], [9, 9]],   # repeat-pad of (9,9)
               [[2, 2], [3, 3], [4, 4], [0, 0], [0, 0]]]   # zero-pad after idx 2
    out = path_end_indices(seq_obs, seq_tml)
    assert out == [(2, True), (2, False)]               # (end_index, is_terminus)
    # terminus-only mode: matches terminus_indices_from_tml on a tml-complete set
    tml_full = [[[0.0], [1.0], [0.0]], [[1.0], [0.0], [0.0]]]
    obs_any = [[[1, 1], [2, 2], [2, 2]], [[5, 5], [5, 5], [5, 5]]]
    ends = [e for e, _ in path_end_indices(obs_any, tml_full)]
    assert ends == terminus_indices_from_tml(tml_full)


def test_path_end_np_matches_py():
    # The numpy branch is what SHIPS on the GPU box but the list fixtures only
    # exercise the pure-Python branch (R-A). Assert the two are identical on the
    # same fixtures; skips when numpy is absent (local box), runs on the GPU box.
    try:
        import numpy as np
    except ImportError:
        print("  (skip: numpy absent — numpy branch covered on the GPU box)")
        return
    from mcts.relabel import _path_end_indices_np
    seq_tml = [[[0.0], [0.0], [1.0], [1.0], [1.0]],
               [[0.0], [0.0], [0.0], [0.0], [0.0]],
               [[1.0], [0.0], [0.0], [0.0], [0.0]]]            # terminus at 0
    seq_obs = [[[1, 1], [1, 1], [9, 9], [9, 9], [9, 9]],
               [[2, 2], [3, 3], [4, 4], [0, 0], [0, 0]],
               [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]]
    py = path_end_indices(seq_obs, seq_tml)                    # list -> pure-Python
    npr = _path_end_indices_np(np.asarray(seq_obs, dtype=float),
                               np.asarray(seq_tml, dtype=float), 1e-8)
    assert py == npr == [(2, True), (2, False), (0, True)]


def test_mixture_proportions_and_bounds():
    rng = random.Random(0)
    t, terminus = 100, 800
    n = 20000
    cur = ter = fut = 0
    for _ in range(n):
        tp = draw_goal_index(t, terminus, geo_mean=200.0, rng=rng)
        assert t <= tp <= terminus               # never behind t, never past terminus
        if tp == t:
            cur += 1
        elif tp == terminus:
            ter += 1
        else:
            fut += 1
    # 70/20/10 within sampling tolerance; terminus also absorbs capped futures,
    # so ter is slightly ABOVE 0.20 and fut slightly below 0.70 — bound loosely.
    assert abs(cur / n - 0.10) < 0.01
    assert 0.18 < ter / n < 0.35
    assert 0.55 < fut / n < 0.72


def test_current_state_target_is_exactly_one():
    rng = random.Random(1)
    scale = StepScale(867)
    saw_current = False
    for _ in range(200):
        tp, target = make_sample(50, 400, 200.0, scale, rng)
        if tp == 50:
            assert target == 1.0                 # exact, by construction (R5.7a)
            saw_current = True
        else:
            assert target == scale.val(tp - 50)
    assert saw_current


def test_capping_at_terminus():
    rng = random.Random(2)
    # t one step before terminus: every future draw must cap at the terminus
    for _ in range(100):
        tp = draw_goal_index(99, 100, geo_mean=200.0, rng=rng)
        assert tp in (99, 100)


def test_geo_mean_one_edge_no_crash():
    # geo_mean <= 1 must not divide by zero (B3): scalar guard returns offset 1,
    # so futures land exactly one step ahead (or cap at terminus). The vectorized
    # sample_batch mirrors this guard (numpy path, exercised on the GPU box).
    rng = random.Random(3)
    seen_future = False
    for _ in range(300):
        tp = draw_goal_index(50, 400, geo_mean=1.0, rng=rng)
        assert tp in (50, 51, 400)                   # current, +1 future, or terminus
        if tp == 51:
            seen_future = True
    assert seen_future


# ── Coverage / connectivity stratum ──────────────────────────────────────────

def _line(x0, y0, x1, y1, n=50):
    return [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n + 1)]


def test_coverable_within_one_trajectory():
    cov = TrajectoryCoverage(eps=0.5)
    cov.add_path(0, _line(0, 0, 10, 0))          # horizontal corridor
    cov.add_path(1, _line(0, 5, 10, 5))          # disjoint corridor
    assert cov.stratum((1, 0), (9, 0)) == "coverable"      # same path 0
    assert cov.stratum((1, 5), (9, 5)) == "coverable"      # same path 1
    assert cov.stratum((1, 0), (9, 5)) == "stitched"       # needs both paths
    assert cov.stratum((1, 0), (5, 2.5)) == "stitched"     # g not near any path


def test_eps_tolerance_and_overlap():
    cov = TrajectoryCoverage(eps=0.5)
    cov.add_path(0, _line(0, 0, 10, 0))
    # query points within eps of the path still count as near
    assert cov.coverable((0.0, 0.4), (10.0, -0.4))
    # two paths that overlap mid-way still do NOT make a cross-path pair coverable
    cov.add_path(1, _line(5, -5, 5, 5))          # crosses path 0 at (5, 0)
    assert cov.stratum((0, 0), (5, 4)) == "stitched"
    assert cov.stratum((5, -4), (5, 4)) == "coverable"     # within path 1 alone


def test_stats():
    cov = TrajectoryCoverage(eps=0.5)
    cov.add_path(0, _line(0, 0, 2, 0, n=10))
    st = cov.stats()
    assert st["n_paths"] == 1 and st["n_cells"] >= 4 and st["eps"] == 0.5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
