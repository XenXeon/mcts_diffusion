"""tests/test_compose_window.py

Torch-free tests for critic-mode window composition (mcts/window.py).

Every tree node must be scored on the SAME [s0, s0+H) critic window, so a
composed window must (a) start with the search-chosen prefix, (b) continue with
the planner continuation, (c) always be exactly H waypoints long, and (d) refuse
prefixes that leave no room for a continuation. extend_prefix must accumulate
the waypoint path so that composed windows at ANY depth begin at s0.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.window import compose_window, extend_prefix

K, H, D = 4, 8, 3


def _trajs(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(K, H, D)).astype(np.float32)


def test_root_passthrough():
    t = _trajs()
    assert compose_window(None, t) is t                      # no copy at the root
    np.testing.assert_array_equal(compose_window(np.zeros((0, D), np.float32), t), t)


def test_prefix_prepended_and_window_truncated():
    t, d = _trajs(), 3
    prefix = np.arange(d * D, dtype=np.float32).reshape(d, D)
    out = compose_window(prefix, t)
    assert out.shape == (K, H, D)
    for k in range(K):
        np.testing.assert_array_equal(out[k, :d], prefix)    # same prefix for every child
    np.testing.assert_array_equal(out[:, d:], t[:, :H - d])  # continuation, truncated to H


def test_prefix_filling_window_raises():
    t = _trajs()
    for d in (H, H + 1):
        with pytest.raises(ValueError, match="prefix length"):
            compose_window(np.zeros((d, D), np.float32), t)
    # d = H-1 is the deepest legal node: one continuation waypoint survives
    out = compose_window(np.zeros((H - 1, D), np.float32), t)
    np.testing.assert_array_equal(out[:, H - 1], t[:, 0])


def test_extend_from_root_copies():
    traj = _trajs()[0]
    p = extend_prefix(None, traj, 1)
    np.testing.assert_array_equal(p, traj[:1])
    p[0, 0] = 999.0                                          # own storage, not a view
    assert traj[0, 0] != 999.0


def test_extend_chain_accumulates_path():
    tA, tB = _trajs(1)[0], _trajs(2)[0]
    p1 = extend_prefix(None, tA, 2)                          # child_index L=2: two rows
    p2 = extend_prefix(p1, tB, 2)
    assert p2.shape == (4, D)
    np.testing.assert_array_equal(p2, np.stack([tA[0], tA[1], tB[0], tB[1]]))


def test_composed_window_always_starts_at_s0():
    # Simulate three levels of the tree with child_index=1: whatever the depth,
    # the scored window's first row must be the REAL state s0 (= level-0 traj[0],
    # inpainted by the planner at every expansion from the root).
    lvl0, lvl1, lvl2 = _trajs(1), _trajs(2), _trajs(3)
    s0 = lvl0[0, 0]
    p1 = extend_prefix(None, lvl0[0], 1)                     # path to a depth-1 node
    p2 = extend_prefix(p1, lvl1[0], 1)                       # path to a depth-2 node
    for prefix, cont in ((p1, lvl1), (p2, lvl2)):
        out = compose_window(prefix, cont)
        for k in range(K):
            np.testing.assert_array_equal(out[k, 0], s0)


def test_dtype_preserved():
    t = _trajs()
    prefix = extend_prefix(None, t[0], 1)
    assert prefix.dtype == np.float32
    assert compose_window(prefix, t).dtype == np.float32
