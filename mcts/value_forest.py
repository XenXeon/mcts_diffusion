"""mcts/value_forest.py

Pure-Python (torch-free) batched value-MCTS for the closed-loop sampler.

A "forest" of M independent search trees (one per parallel env) is grown in LOCKSTEP, so
that every expansion round batches all M trees' candidate states into ONE evaluation call.
The caller maps that single call to one batched planner.sample + value pass on the GPU —
this is the parallelism: instead of (budget x M) sequential planner calls, the search does
~budget batched calls, each covering all M trees.

Search semantics — state-value MCTS with MAX (look-ahead) backup
---------------------------------------------------------------
* Each node holds a normalised state, a value prior `v_prior = V(state)`, and the first
  waypoint of the segment that produced it (used only at the root, to extract the action).
* Selection descends by UCB:  Q(child) + c * sqrt(ln(N_parent + 1) / (N_child + 1)),
  where Q = `value_max` (the value net's prior until the node is expanded, then the best
  backed-up continuation).
* Expansion attaches K children (candidate continuations from the planner).
* Backup is a MAX over continuations, propagated to the root. This is the "look ahead
  before committing": a child whose first step looks mediocre one-ply but leads to a
  high-value region is rewarded — exactly what a mean/average backup washes out, and the
  thing the project brief cares about (a poor early step shouldn't sink the plan).
* The chosen action is the first waypoint of the root child with the highest Q.

The bookkeeping is intentionally free of torch/numpy: states/waypoints are opaque objects
the forest never inspects (it only passes them to `expand_fn` and stores what comes back),
and values are plain floats. That makes the whole search loop unit-testable with a
deterministic fake evaluator, with all GPU work isolated behind the `expand_fn` callback.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

# An ExpandResult is, per node, (child_states, child_first_waypoints, child_values).
ExpandResult = Tuple[Sequence[Any], Sequence[Any], Sequence[float]]
# expand_fn maps a list of states (one per node to expand) -> list of ExpandResult.
ExpandFn = Callable[[List[Any]], List[ExpandResult]]


class SearchNode:
    """One node in a search tree. `state`/`first_wp` are opaque to the forest.

    `value` is the node's current estimate of the best achievable return-to-go through it:
        - unexpanded node:  `v_prior` (the value net's estimate of its state)
        - expanded node:    max over its children's `value`  (the best continuation)
    The expanded value OVERRIDES the prior (look-ahead can revise an over-optimistic prior
    down as well as up) — that is the whole point of searching rather than trusting V(s).
    """

    __slots__ = ("state", "first_wp", "v_prior", "visit", "value",
                 "children", "parent", "depth")

    def __init__(self, state: Any, first_wp: Any, v_prior: float,
                 parent: Optional["SearchNode"] = None, depth: int = 0) -> None:
        self.state = state
        self.first_wp = first_wp
        self.v_prior = float(v_prior)
        self.visit = 0
        self.value = float(v_prior)       # == max(child.value) once expanded
        self.children: List["SearchNode"] = []
        self.parent = parent
        self.depth = depth

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def ucb(self, c: float) -> float:
        """Exploitation (best continuation value) + exploration (less-visited preferred).

        Uses (N+1) smoothing instead of the classic +inf-for-unvisited so the value net's
        prior steers which child to try first (best-first), rather than forcing every
        sibling to be expanded before any value is trusted — far more sample-efficient when
        V(state) is informative.
        """
        n_parent = self.parent.visit if self.parent is not None else self.visit
        return self.value + c * math.sqrt(math.log(n_parent + 1) / (self.visit + 1))


def select_leaf(root: SearchNode, c: float) -> SearchNode:
    """Descend from root by max-UCB until an unexpanded node (a leaf) is reached."""
    node = root
    while not node.is_leaf:
        node = max(node.children, key=lambda ch: ch.ucb(c))
    return node


def backprop(node: SearchNode, top_m: int = 1) -> None:
    """Walk to the root: bump visit counts and recompute value = mean of the top_m
    child values (top_m=1 is the classic MAX backup).

    top_m > 1 tempers the winner's curse: a max over K NOISY child values (e.g.
    critic scores of stitched composite plans) systematically selects the most
    OVERRATED child, so backed-up values inflate with every level. Averaging the
    best m keeps the best-first semantics while shrinking that optimism bias.

    Call AFTER attaching the node's new children. Bottom-up order guarantees each node's
    children are already current when it is recomputed.
    """
    cur: Optional[SearchNode] = node
    while cur is not None:
        cur.visit += 1
        if cur.children:
            vals = sorted((ch.value for ch in cur.children), reverse=True)
            m = min(top_m, len(vals))
            cur.value = sum(vals[:m]) / m
        cur = cur.parent


@dataclass
class ForestConfig:
    k: int                    # candidate continuations per expansion
    budget: int               # expansion rounds AFTER the root expansion (per tree)
    c_ucb: float = 1.4142136  # UCB exploration constant (sqrt 2)
    top_m: int = 1            # backup = mean of the m best children (1 = MAX backup)


class ValueForest:
    """M lockstep search trees sharing one batched `expand_fn`.

    Args:
        root_states: list of M starting states (one per env), opaque to the forest.
        expand_fn:   batched evaluator. Given a list of N states (one per node being
                     expanded this round, N == number of live trees), returns a list of N
                     ExpandResult tuples (child_states, child_first_wps, child_values),
                     each of length K. Internally this is ONE planner.sample + value pass.
        config:      ForestConfig.
    """

    def __init__(self, root_states: List[Any], expand_fn: ExpandFn,
                 config: ForestConfig) -> None:
        if config.k < 1:
            raise ValueError(f"k must be >= 1, got {config.k}")
        if config.top_m < 1:
            raise ValueError(f"top_m must be >= 1, got {config.top_m}")
        self.expand_fn = expand_fn
        self.cfg = config
        # Root v_prior is irrelevant (root value never competes), so 0.0. first_wp None.
        self.roots: List[SearchNode] = [SearchNode(s, None, 0.0) for s in root_states]
        self._expand_nodes(self.roots)   # root expansion (round 0)

    def _expand_nodes(self, nodes: List[SearchNode]) -> None:
        """Batch-expand `nodes` (one per tree): attach K children each, then backprop."""
        results = self.expand_fn([n.state for n in nodes])
        if len(results) != len(nodes):
            raise RuntimeError(
                f"expand_fn returned {len(results)} results for {len(nodes)} nodes")
        for node, (child_states, first_wps, vvals) in zip(nodes, results):
            for cs, fw, v in zip(child_states, first_wps, vvals):
                node.children.append(
                    SearchNode(cs, fw, v, parent=node, depth=node.depth + 1))
            backprop(node, self.cfg.top_m)   # value = mean(top-m children) up to root

    def run(self) -> None:
        """Grow every tree by `budget` more expansions, one promising leaf per tree/round."""
        for _ in range(self.cfg.budget):
            leaves = [select_leaf(root, self.cfg.c_ucb) for root in self.roots]
            self._expand_nodes(leaves)

    def best_first_waypoints(self) -> List[Any]:
        """For each tree, the first waypoint of the root child with the highest Q."""
        out: List[Any] = []
        for root in self.roots:
            if root.children:
                best = max(root.children, key=lambda ch: ch.value)
                out.append(best.first_wp)
            else:
                out.append(None)
        return out

    def best_leaf_states(self) -> List[Any]:
        """Per tree: the state of the DEEPEST node along the max-value branch —
        the tree's committed best plan. In critic/grounded mode a state is
        (vec, prefix), so prefix ++ vec is the full stitched waypoint path from
        s0; used to drive the DF-tree at an arbitrary replan cadence in the MPC
        harness (mcts/mctd_loop.py DFTreeMPCPlanner). Root-only trees (no
        expansion) return the root state."""
        out: List[Any] = []
        for root in self.roots:
            node = root
            while node.children:
                node = max(node.children, key=lambda ch: ch.value)
            out.append(node.state)
        return out

    # ── Introspection (for analysis / logging) ──────────────────────────────────
    def stats(self) -> List[dict]:
        """Per-tree summary: max depth reached, node count, root best value."""
        out = []
        for root in self.roots:
            n_nodes = 0
            max_depth = 0
            stack = [root]
            while stack:
                nd = stack.pop()
                n_nodes += 1
                max_depth = max(max_depth, nd.depth)
                stack.extend(nd.children)
            best_q = max((ch.value for ch in root.children), default=0.0)
            out.append(dict(n_nodes=n_nodes, max_depth=max_depth, root_best_value=best_q))
        return out
