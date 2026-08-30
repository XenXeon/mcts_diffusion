"""tests/test_mctd_tree.py

Torch-free tests for the MCTD search logic (mcts/mctd_tree.py) — the denoising-
axis tree. Same idiom as tests/test_value_forest.py: a deterministic fake
`expand_eval` stands in for the real denoise+verify callback, so selection,
max-backup, terminal handling, early-stopping and output collection are all
checked without torch or a diffusion model.

Fake world: a node's quality is the SUM of guidance scales chosen on the path to
it (a proxy for "how hard the plan was pushed toward the goal"). Value = sum /
(terminal_depth * max_scale) in [0, 1]; a plan is "Achieved" once that value
crosses a threshold. This makes the optimum (all-max-scale path) predictable.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mcts.mctd_tree import (ExpandResult, MCTDSearchConfig, MCTDTreeNode,
                            run_mctd_search)

MENU = [0.0, 1.0, 2.0]
MAX_SCALE = 2.0


def make_eval(terminal_depth, achieve_value=0.75):
    """expand_eval whose value = path guidance-sum / (terminal_depth*MAX_SCALE).
    The node payload carries the running sum so children extend their parent."""
    denom = terminal_depth * MAX_SCALE

    def expand_eval(node, cand):
        parent_sum = node.payload if node.payload is not None else 0.0
        s = parent_sum + float(cand["guidance_scale"])
        val = s / denom
        info = "Achieved" if val >= achieve_value else "NotReached"
        return ExpandResult(value=val, info=info, child_partial=s,
                            clean_plan=f"sum={s}",
                            achieved_t=(cand["depth"] if info == "Achieved" else None))
    return expand_eval


# ── node-level unit tests ────────────────────────────────────────────────────

def test_predicates_terminal_expandable_selectable():
    root = MCTDTreeNode("0", 0, terminal_depth=2, guidance_scales=MENU)
    assert not root.is_terminal()
    assert root.is_expandable()          # depth<terminal, all slots empty
    assert not root.is_selectable()      # no children yet
    # fill all slots
    for i in range(len(MENU)):
        root.add_child(i, value=0.1 * i, payload=None)
    assert root.is_selectable()
    assert not root.is_expandable()      # full -> not expandable
    leaf = root.children[0]["node"]
    assert leaf.depth == 1 and not leaf.is_terminal()
    # a depth-2 node is terminal (== terminal_depth) -> never expandable
    deep = leaf.add_child(0, 0.5, None)
    assert deep.is_terminal() and not deep.is_expandable()


def test_add_child_records_guidance_and_name():
    root = MCTDTreeNode("0", 0, 2, MENU)
    c = root.add_child(2, value=0.9, payload="p")
    assert c.name == "0-2"
    assert c.guidance_scale == 2.0       # slot 2 -> scale 2.0
    assert c.payload == "p"
    assert c.depth == 1


def test_backprop_is_max_not_mean_and_counts_visits():
    root = MCTDTreeNode("0", 0, 2, MENU)
    a = root.add_child(0, value=0.3, payload=None)
    root.backpropagate()
    assert root.visit == 1 and root.value == 0.3
    root.add_child(1, value=0.7, payload=None)
    root.backpropagate()
    assert root.visit == 2
    assert root.value == 0.7             # max(0.3, 0.7), not mean 0.5
    # deepen the weaker branch; root must still track the max over its children
    a.add_child(0, value=0.9, payload=None)
    a.backpropagate()                    # a.value <- 0.9 ; propagates to root
    assert a.value == 0.9
    assert root.value == 0.9             # max(a=0.9, other=0.7)


def test_uct_prefers_unvisited_then_value():
    root = MCTDTreeNode("0", 0, 2, MENU)
    for i in range(len(MENU)):
        root.add_child(i, value=0.1 * i, payload=None)
    # all children visit 0 -> unvisited-first -> first slot (index 0)
    assert root.uct_select(math.sqrt(2)) is root.children[0]["node"]
    # give every child a visit; now value should decide (highest value wins)
    for c in root.children:
        c["node"].visit = 1
    root.children[0]["node"].value = 0.0
    root.children[2]["node"].value = 0.9
    assert root.uct_select(0.0) is root.children[2]["node"]   # c=0 -> pure value


# ── full-search tests (fake evaluator) ───────────────────────────────────────

def test_search_grows_tree_and_backs_up_optimum():
    root = MCTDTreeNode("0", 0, terminal_depth=2, guidance_scales=MENU)
    root.value = 0.0
    cfg = MCTDSearchConfig(guidance_scales=MENU, terminal_depth=2,
                           max_search_num=40, early_stopping=None)
    res = run_mctd_search(root, make_eval(2), cfg, np.random.default_rng(0))
    assert res.n_nodes >= len(MENU)            # at least the root's children
    assert res.max_depth == 2                  # reached terminal depth
    # the best achievable path is (2.0, 2.0) -> value 1.0
    assert res.solved
    best = max(v for _, v, _ in res.achieved)
    assert abs(best - 1.0) < 1e-9
    # root value is the max-backup over everything expanded == the best leaf
    assert abs(root.value - 1.0) < 1e-9


def test_search_values_never_exceed_one_and_no_nans():
    root = MCTDTreeNode("0", 0, 3, MENU)
    root.value = 0.0
    cfg = MCTDSearchConfig(guidance_scales=MENU, terminal_depth=3,
                           max_search_num=60, early_stopping=None)
    res = run_mctd_search(root, make_eval(3), cfg, np.random.default_rng(1))
    # walk the whole tree: every value is a finite number in [0, 1]
    stack = [root]
    seen = 0
    while stack:
        nd = stack.pop()
        seen += 1
        assert nd.value is None or (0.0 <= nd.value <= 1.0 + 1e-9)
        assert nd.value is None or math.isfinite(nd.value)
        stack.extend(c["node"] for c in nd.children if c["node"] is not None)
    assert seen >= 1 + len(MENU)


def test_early_stopping_halts_at_first_achieved():
    root = MCTDTreeNode("0", 0, 2, MENU)
    root.value = 0.0
    # low threshold: any sum>=1 achieves, so a goal-reaching plan appears fast
    cfg = MCTDSearchConfig(guidance_scales=MENU, terminal_depth=2,
                           max_search_num=40, early_stopping="achieved")
    res = run_mctd_search(root, make_eval(2, achieve_value=0.25), cfg,
                          np.random.default_rng(0))
    assert res.solved
    assert len(res.achieved) == 1              # stopped at the first one
    assert res.n_search < cfg.max_search_num   # did not exhaust the budget


def test_no_achievable_goal_returns_not_reached():
    root = MCTDTreeNode("0", 0, 2, MENU)
    root.value = 0.0
    # threshold above the max possible value (1.0) -> never achieved
    cfg = MCTDSearchConfig(guidance_scales=MENU, terminal_depth=2,
                           max_search_num=30, early_stopping="achieved")
    res = run_mctd_search(root, make_eval(2, achieve_value=2.0), cfg,
                          np.random.default_rng(2))
    assert not res.solved
    assert len(res.achieved) == 0
    assert len(res.not_reached) >= 1           # collected the misses instead


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
