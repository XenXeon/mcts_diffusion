"""Torch-free tests for mcts.grounded.cumulative_solved_count — the grounded
(non-learned) subtask-completion checker's numpy core. mcts/grounded.py keeps
this function importable without torch (a try/except guard around the torch
wrapper class, mcts.grounded.KitchenGroundedChecker) specifically so this
suite can run on the local torch-free Windows dev box; see mcts/grounded.py's
module docstring.
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import pytest

from mcts.grounded import cumulative_solved_count

# Synthetic 2-element task: e0 spans obs dims [0, 1] with goal (1, 1);
# e1 is obs dim [4] alone with goal (0,). D=6 leaves dims 2, 3, 5 unused (the
# way kitchen's other task elements would sit alongside these two).
D = 6
IDX = [np.array([0, 1]), np.array([4])]
GOAL = [np.array([1.0, 1.0]), np.array([0.0])]
THRESH = 0.3


def _window(T, e0_solved_at=None, e1_solved_at=None):
    """Build one (T, D) window; optionally place e0/e1 exactly at their goal
    (distance 0, well inside THRESH) at the given step(s) — everywhere else
    is set far away (distance >> THRESH from every goal)."""
    w = np.full((T, D), 10.0, dtype=np.float64)
    if e0_solved_at is not None:
        for t in np.atleast_1d(e0_solved_at):
            w[t, [0, 1]] = GOAL[0]
    if e1_solved_at is not None:
        for t in np.atleast_1d(e1_solved_at):
            w[t, 4] = GOAL[1][0]
    return w


def test_union_mid_step_not_final_still_counts():
    # e0 solved at t=2 only (not the final row t=4) -> still counts (union).
    w = _window(5, e0_solved_at=2)
    out = cumulative_solved_count(w[None], IDX, GOAL, THRESH)
    assert out.shape == (1,)
    assert out[0] == 1.0


def test_row0_already_done_counts():
    # already solved at the START of the window (row 0) -> still counts.
    w = _window(4, e0_solved_at=0)
    out = cumulative_solved_count(w[None], IDX, GOAL, THRESH)
    assert out[0] == 1.0


def test_never_solved_is_zero():
    w = _window(4)
    out = cumulative_solved_count(w[None], IDX, GOAL, THRESH)
    assert out[0] == 0.0


def test_both_solved_is_two():
    w = _window(6, e0_solved_at=1, e1_solved_at=4)
    out = cumulative_solved_count(w[None], IDX, GOAL, THRESH)
    assert out[0] == 2.0


def test_threshold_boundary_is_strict():
    # distance exactly == thresh must NOT count (strict < comparison).
    w = _window(3)
    w[1, 4] = THRESH                     # |THRESH - 0| == THRESH exactly
    out = cumulative_solved_count(w[None], IDX, GOAL, THRESH)
    assert out[0] == 0.0
    # just under thresh DOES count, confirming the boundary sits at THRESH.
    w2 = _window(3)
    w2[1, 4] = THRESH - 1e-6
    out2 = cumulative_solved_count(w2[None], IDX, GOAL, THRESH)
    assert out2[0] == 1.0


def test_shape_and_vectorization_over_n():
    rows = [
        _window(4),                                    # 0 solved
        _window(4, e0_solved_at=0),                     # 1 solved
        _window(4, e0_solved_at=1, e1_solved_at=3),      # 2 solved
    ]
    windows = np.stack(rows, axis=0)                    # (3, 4, D)
    out = cumulative_solved_count(windows, IDX, GOAL, THRESH)
    assert out.shape == (3,)
    assert out.dtype == np.float64
    assert list(out) == [0.0, 1.0, 2.0]


def test_bad_windows_shape_raises():
    with pytest.raises(ValueError):
        cumulative_solved_count(np.zeros((4, D)), IDX, GOAL, THRESH)   # missing T dim


def test_mismatched_index_goal_lengths_raises():
    bad_goal = [np.array([1.0]), np.array([0.0])]   # e0 idx has 2 dims, goal has 1
    w = _window(3)
    with pytest.raises(ValueError):
        cumulative_solved_count(w[None], IDX, bad_goal, THRESH)


def test_nonpositive_thresh_raises():
    w = _window(3)
    with pytest.raises(ValueError):
        cumulative_solved_count(w[None], IDX, GOAL, 0.0)
    with pytest.raises(ValueError):
        cumulative_solved_count(w[None], IDX, GOAL, -0.1)


def test_import_without_torch():
    """mcts.grounded must import cleanly on a torch-free box (this IS that
    box) — the module guards its torch import in a try/except specifically so
    the numpy core stays usable here."""
    import importlib
    mod = importlib.import_module("mcts.grounded")
    assert hasattr(mod, "cumulative_solved_count")
    assert hasattr(mod, "KitchenGroundedChecker")


def test_from_env_finds_module_level_constants():
    """Regression: d4rl/cleandiffuser kitchen envs keep TASK_ELEMENTS on the
    CLASS but OBS_ELEMENT_INDICES / OBS_ELEMENT_GOALS / BONUS_THRESH at MODULE
    level — the first deployment crashed on exactly this
    (KitchenMicrowaveKettleBottomBurnerLightV0). from_env must search the
    defining module of the env's classes, not just the instance.

    Torch-free: from_env + __init__ are pure numpy; only count()/score() need
    torch.
    """
    import types

    from mcts.grounded import KitchenGroundedChecker

    mod = types.ModuleType("_fake_kitchen_envs_for_test")
    mod.OBS_ELEMENT_INDICES = {"e0": np.array([0, 1]), "e1": np.array([4])}
    mod.OBS_ELEMENT_GOALS = {"e0": np.array([1.0, 1.0]), "e1": np.array([0.0])}
    mod.BONUS_THRESH = 0.3
    sys.modules[mod.__name__] = mod
    try:
        class FakeKitchenEnv:
            TASK_ELEMENTS = ["e0", "e1"]      # class attr, like the real envs
        FakeKitchenEnv.__module__ = mod.__name__

        class FakeNormalizer:
            mean = np.zeros(D, dtype=np.float32)
            std = np.ones(D, dtype=np.float32)

        chk = KitchenGroundedChecker.from_env(FakeKitchenEnv(), FakeNormalizer())
        assert chk.thresh == 0.3
        assert len(chk.elem_indices) == 2
        assert np.array_equal(chk.elem_indices[0], np.array([0, 1]))

        # and a class with NO reachable constants anywhere must still refuse
        class NotAKitchenEnv:
            pass
        with pytest.raises(ValueError):
            KitchenGroundedChecker.from_env(NotAKitchenEnv(), FakeNormalizer())
    finally:
        del sys.modules[mod.__name__]
