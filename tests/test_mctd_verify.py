"""tests/test_mctd_verify.py

Torch-free tests for the MCTD geometric verifier (mcts/mctd_verify.py) — the
non-learned value function. Synthetic plans with known geometry pin the three
outcomes (Achieved / Warp / NotReached), the value formula (T - t) / T, and the
first-event-wins ordering. A wrong verifier would silently hand the tree garbage
values, so these are the guard against that.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mcts.mctd_verify import MCTD_ENV, geometric_values, is_degenerate


def test_straight_line_reaches_goal_with_correct_value():
    # T=10 tokens marching from (0,0) toward goal (9,0); reaches at t=9.
    T = 10
    plan = np.stack([np.linspace(0, 9, T), np.zeros(T)], axis=-1)[None]   # (1,T,2)
    vals, infos, ach = geometric_values(plan, start_pos=[0, 0], goal_pos=[9, 0],
                                        goal_radius=0.5, warp_threshold=None)
    assert infos[0] == "Achieved"
    assert ach[0] == 9
    assert abs(vals[0] - (T - 9) / T) < 1e-9        # = 0.1


def test_earlier_reach_scores_higher():
    T = 10
    # jumps straight onto the goal at t=2 (steps of 3 -> no warp gate here)
    pos = np.array([[0, 0], [3, 0], [6, 0]] + [[6, 0]] * (T - 3), dtype=float)
    vals, infos, ach = geometric_values(pos[None], [0, 0], [6, 0],
                                        goal_radius=0.5, warp_threshold=None)
    assert infos[0] == "Achieved" and ach[0] == 2
    assert abs(vals[0] - (T - 2) / T) < 1e-9        # 0.8 > the 0.1 above


def test_teleport_is_warp_value_zero():
    T = 5
    # a 100-unit jump between t0 and t1 -> warp before it could reach the goal
    pos = np.array([[0, 0], [100, 0], [100, 0], [100, 0], [100, 0]], dtype=float)
    vals, infos, ach = geometric_values(pos[None], [0, 0], [100, 0],
                                        goal_radius=0.5, warp_threshold=5.0)
    assert infos[0] == "Warp"
    assert vals[0] == 0.0
    assert ach[0] == -1


def test_never_reaches_is_not_reached_zero():
    T = 6
    pos = np.tile(np.array([1.0, 1.0]), (T, 1))     # sits far from goal, no move
    vals, infos, ach = geometric_values(pos[None], [1, 1], [50, 50],
                                        goal_radius=0.5, warp_threshold=100.0)
    assert infos[0] == "NotReached"
    assert vals[0] == 0.0 and ach[0] == -1


def test_first_event_wins_reach_before_warp():
    # token 0 (=start) is 0.6 from goal (outside radius), token 1 is 0.2 (inside)
    # -> reaches at t=1; then teleports at t=2. The reach must win over the warp.
    pos = np.array([[0, 0], [0.4, 0], [100, 0], [100, 0]], dtype=float)
    vals, infos, ach = geometric_values(pos[None], [0, 0], [0.6, 0],
                                        goal_radius=0.5, warp_threshold=5.0)
    assert infos[0] == "Achieved" and ach[0] == 1


def test_batched_independent_outcomes():
    T = 4
    reach = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    warp = np.array([[0, 0], [50, 0], [50, 0], [50, 0]], dtype=float)
    miss = np.array([[0, 0], [0, 0.1], [0, 0.2], [0, 0.3]], dtype=float)
    plans = np.stack([reach, warp, miss])           # (3, T, 2)
    vals, infos, ach = geometric_values(
        plans, start_pos=[[0, 0]] * 3, goal_pos=[[3, 0], [50, 0], [99, 99]],
        goal_radius=0.5, warp_threshold=5.0)
    assert infos == ["Achieved", "Warp", "NotReached"]
    assert ach[0] == 3 and ach[1] == -1 and ach[2] == -1


def test_degenerate_detects_nonmoving_plan():
    still = np.tile(np.array([2.0, 2.0]), (8, 1))[None]         # never moves
    moving = np.stack([np.arange(8.0), np.zeros(8)], axis=-1)[None]
    assert bool(is_degenerate(still, eps=0.1)[0]) is True
    assert bool(is_degenerate(moving, eps=0.1)[0]) is False


def test_env_config_present_for_maze_families_not_kitchen():
    assert "maze2d" in MCTD_ENV and "antmaze" in MCTD_ENV
    assert MCTD_ENV["maze2d"]["pos_dims"] == (0, 1)
    assert "kitchen" not in MCTD_ENV        # geometric verify N/A for kitchen


def test_bad_shape_raises():
    try:
        geometric_values(np.zeros((5, 2)), [0, 0], [1, 1], 0.5)   # missing T axis
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-(N,T,P) plan_pos")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
