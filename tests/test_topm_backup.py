"""tests/test_topm_backup.py

Torch-free tests for the top-m backup (mcts/value_forest.py).

Max-backup (top_m=1) over K NOISY child values picks the most overrated child —
the winner's curse. top_m > 1 replaces the node value with the mean of its m
best children: same best-first semantics, tempered optimism. These tests pin the
aggregation itself and its propagation to the root.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.value_forest import ForestConfig, ValueForest

CHILD_VALUES = (5.0, 3.0, 1.0)   # K = 3, deliberately unsorted-agnostic


def fixed_expand(states):
    """Every expansion yields the same three child values (state = depth marker)."""
    return [([s + 1 for _ in CHILD_VALUES],
             [("wp", s) for _ in CHILD_VALUES],
             list(CHILD_VALUES)) for s in states]


def test_top1_is_classic_max_backup():
    f = ValueForest([0], fixed_expand, ForestConfig(k=3, budget=0, top_m=1))
    assert f.roots[0].value == 5.0


def test_top2_averages_two_best():
    f = ValueForest([0], fixed_expand, ForestConfig(k=3, budget=0, top_m=2))
    assert f.roots[0].value == pytest.approx((5.0 + 3.0) / 2)


def test_top_m_larger_than_k_averages_all():
    f = ValueForest([0], fixed_expand, ForestConfig(k=3, budget=0, top_m=10))
    assert f.roots[0].value == pytest.approx(sum(CHILD_VALUES) / 3)


def test_topm_propagates_to_root():
    # budget=1: the best root child (prior 5.0) is expanded; its value becomes
    # mean(top-2 grandchildren) = 4.0, and the root re-aggregates its children
    # as mean(top-2 of [4.0, 3.0, 1.0]) = 3.5.
    f = ValueForest([0], fixed_expand, ForestConfig(k=3, budget=1, top_m=2))
    f.run()
    root = f.roots[0]
    expanded = max(root.children, key=lambda ch: len(ch.children))
    assert len(expanded.children) == 3
    assert expanded.value == pytest.approx(4.0)
    assert root.value == pytest.approx(3.5)


def test_invalid_top_m_rejected():
    with pytest.raises(ValueError, match="top_m"):
        ValueForest([0], fixed_expand, ForestConfig(k=3, budget=0, top_m=0))
