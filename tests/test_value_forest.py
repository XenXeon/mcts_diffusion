"""tests/test_value_forest.py

Pure-Python tests for the value-MCTS search logic (no torch/numpy needed).

A fake 1-D world: a node's `state` is a float position; the goal is at 10.0; value is
-(distance to goal) so higher is better. Each expansion offers three moves: +3, +1, -1.
This lets us check selection, max-backup, and look-ahead extraction deterministically.

`import mcts` is torch-free: the package eagerly exports only the value-forest engine
and specs; the legacy torch-importing engine is lazy (PEP 562). So a plain import works
on machines without torch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts import ForestConfig, SearchNode, ValueForest, backprop, select_leaf

GOAL = 10.0
MOVES = (3.0, 1.0, -1.0)   # K = 3 candidate steps


def fake_expand(states):
    """One planner+value call, faked: children = state+move, value = -(dist to goal)."""
    results = []
    for s in states:
        child_states = [s + m for m in MOVES]
        first_wps = [("wp", cs) for cs in child_states]   # opaque waypoint marker
        vvals = [-abs(GOAL - cs) for cs in child_states]
        results.append((child_states, first_wps, vvals))
    return results


def test_root_expands_into_k_children():
    f = ValueForest([0.0], fake_expand, ForestConfig(k=3, budget=0))
    root = f.roots[0]
    assert len(root.children) == 3
    assert root.visit == 1                       # root backprop'd once at init
    # best child by prior is the +3 move (closest to goal)
    assert max(root.children, key=lambda c: c.value).state == 3.0


def test_lookahead_picks_progress_move():
    # Two independent trees (forest) advanced together.
    f = ValueForest([0.0, 5.0], fake_expand, ForestConfig(k=3, budget=12, c_ucb=1.0))
    f.run()
    wps = f.best_first_waypoints()
    # Root 0: best first step is +3 -> position 3.0 ; Root 5: +3 -> 8.0
    assert wps[0] == ("wp", 3.0)
    assert wps[1] == ("wp", 8.0)


def test_max_backup_raises_root_value_toward_goal():
    # With enough budget, the root's best value should climb above the 1-ply prior (-7)
    # because look-ahead discovers the +3,+3,... path approaching the goal.
    f = ValueForest([0.0], fake_expand, ForestConfig(k=3, budget=20, c_ucb=0.5))
    f.run()
    root = f.roots[0]
    one_ply_best = -abs(GOAL - 3.0)              # = -7.0, value of the best single step
    best_backed_up = max(c.value for c in root.children)
    assert best_backed_up > one_ply_best         # look-ahead improved the estimate


def test_backprop_takes_max_not_mean():
    # a has two children; its value must track the BETTER child, not the average.
    a = SearchNode("s", None, 0.0)
    b = SearchNode("b", None, -5.0, parent=a)
    c = SearchNode("c", None, -3.0, parent=a)
    a.children = [b, c]
    backprop(b)
    assert a.value == -3.0          # max(-5, -3), not mean (-4)
    assert a.visit == 1
    # b is expanded and its best continuation is -1; a must follow it up to -1.
    b.children = [SearchNode("b1", None, -1.0, parent=b)]
    backprop(b)
    assert b.value == -1.0
    assert a.value == -1.0          # max(b=-1, c=-3)


def test_select_descends_to_unexpanded_leaf():
    f = ValueForest([0.0], fake_expand, ForestConfig(k=3, budget=0, c_ucb=1.0))
    leaf = select_leaf(f.roots[0], 1.0)
    assert leaf.is_leaf and leaf.parent is f.roots[0]


def test_k_must_be_positive():
    try:
        ValueForest([0.0], fake_expand, ForestConfig(k=0, budget=1))
    except ValueError:
        return
    raise AssertionError("expected ValueError for k=0")


def test_expand_fn_wrong_count_raises():
    def bad_expand(states):
        return []                    # wrong: one root expected, zero results returned
    try:
        ValueForest([0.0], bad_expand, ForestConfig(k=3, budget=0))
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError for mismatched expand_fn result count")


def test_stats_reports_depth_and_nodes():
    f = ValueForest([0.0], fake_expand, ForestConfig(k=3, budget=5, c_ucb=1.0))
    f.run()
    st = f.stats()[0]
    assert st["n_nodes"] >= 1 + 3            # root + its K children at least
    assert st["max_depth"] >= 1
    assert isinstance(st["root_best_value"], float)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
