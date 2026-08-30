"""mcts/mctd_tree.py

Monte Carlo Tree Diffusion (MCTD) search over the DENOISING axis — a faithful,
torch-free port of the tree in the official implementation
(mctd-main/algorithms/diffusion_forcing/tree_node.py + the p_mctd_plan control
flow), adapted to run on THIS repo's D4RL Diffusion-Forcing planner.

What makes MCTD different from this repo's OTHER tree (mcts/value_forest.py):

    value_forest  — a node is a physical STATE reached along a stitched path;
                    depth = look-ahead in the world; edge = which continuation.
    MCTD (here)   — a node is a PARTIALLY-DENOISED whole-horizon plan;
                    depth = denoising progress (one block of denoising steps);
                    edge  = which GUIDANCE SCALE to apply for the next block.

So the two search orthogonal axes. This module is the MCTD axis. It is kept
free of torch/numpy-heavy work by the same trick value_forest uses: the search
is CALLBACK-DRIVEN. `run_mctd_search` walks the tree (selection / expansion /
backprop / early-stop) and calls an injected `expand_eval(node, cand)` to do
the actual denoising + value estimation. Tests drive it with a deterministic
fake callback (see tests/test_mctd_tree.py); mcts/mctd_planner.py supplies the
real, torch-backed one. The node itself only ever holds an OPAQUE `payload`
(the partially-denoised plan tensor), never inspecting it — so this file
imports nothing heavier than `math`.

Faithfulness notes (MCTD, Yoon et al., ICML 2025, arXiv:2502.07202):
  * children are indexed by a fixed menu of guidance scales (the meta-actions);
  * selection is UCT (exploration weight sqrt(2) by default);
  * backup is MAX over children (not mean) — an optimistic search, matching the
    reference (tree_node.py backpropagate);
  * a node is TERMINAL at terminal_depth (= the plan is fully denoised);
  * the reference's default config is SEQUENTIAL (parallel_search_num=1,
    leaf_parallelization=False) — i.e. plain MCTD, not Fast-MCTD — which is what
    this port implements. The parallel virtual-visit machinery of Fast-MCTD is
    deliberately omitted; hooks (visit accounting) are structured so it could be
    added without reworking the node.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class MCTDTreeNode:
    """One partially-denoised plan. Children = the guidance scales that could be
    applied to advance it one denoising block deeper.

    payload: opaque handle to this node's partially-denoised plan (a tensor in
             the real run; anything in tests). The tree never looks inside it.
    guidance_scale: the meta-action that PRODUCED this node (None at the root).
    """

    def __init__(self, name: str, depth: int, terminal_depth: int,
                 guidance_scales: List[float], parent: Optional["MCTDTreeNode"] = None,
                 guidance_scale: Optional[float] = None, payload: Any = None):
        self.name = name
        self.depth = depth
        self.terminal_depth = terminal_depth
        self.guidance_scales = list(guidance_scales)
        self.parent = parent
        self.guidance_scale = guidance_scale
        self.payload = payload
        self.value: Optional[float] = None
        self.visit = 0
        # one child slot per guidance scale; slot filled lazily on expansion
        self.children: List[Dict[str, Any]] = [
            {"guidance_scale": g, "node": None} for g in self.guidance_scales
        ]

    # ── structural predicates (mirror tree_node.py) ─────────────────────────
    def is_terminal(self) -> bool:
        return self.depth == self.terminal_depth

    def is_expandable(self) -> bool:
        """A non-terminal node with at least one empty child slot."""
        if self.depth == self.terminal_depth:
            return False
        return any(c["node"] is None for c in self.children)

    def is_selectable(self) -> bool:
        """Every child slot filled — safe to descend by UCT."""
        return all(c["node"] is not None for c in self.children)

    def empty_slots(self) -> List[int]:
        return [i for i, c in enumerate(self.children) if c["node"] is None]

    # ── UCT selection (upper confidence bound on trees) ─────────────────────
    def uct_select(self, c_ucb: float) -> "MCTDTreeNode":
        """Argmax over filled children of value + c*sqrt(ln(N)/n). Only valid
        when selectable (all children present) — the caller guarantees that.

        Never-visited children take priority (bonus = +inf), the standard UCT
        "explore each child once first" rule. The reference computes
        sqrt(log(1e-6+total)/(1e-6+visit)), which for an unvisited child is
        sqrt(negative) = nan and relies on np.argmax(nan...) returning index 0;
        the explicit rule below is well-defined (math.sqrt would raise on a
        negative) and equivalent in effect — among all-unvisited children both
        pick the first in child-slot (guidance-scale) order.
        """
        total = sum(c["node"].visit for c in self.children)
        best, best_u = None, -math.inf
        for c in self.children:
            nd = c["node"]
            if nd.visit == 0:
                u = math.inf
            else:
                u = nd.value + c_ucb * math.sqrt(math.log(1 + total) / nd.visit)
            if u > best_u:
                best_u, best = u, nd
        return best

    # ── growth + backup ─────────────────────────────────────────────────────
    def add_child(self, index: int, value: float, payload: Any) -> "MCTDTreeNode":
        g = self.children[index]["guidance_scale"]
        child = MCTDTreeNode(f"{self.name}-{index}", self.depth + 1,
                             self.terminal_depth, self.guidance_scales,
                             parent=self, guidance_scale=g, payload=payload)
        child.value = float(value)
        self.children[index]["node"] = child
        return child

    def backpropagate(self) -> None:
        """Increment visit and set value = MAX over present children, then
        recurse to the parent. Called on the node that was just EXPANDED (which
        now has the new leaf among its children), exactly as the reference does
        — so a freshly created leaf keeps its verifier value until it is itself
        expanded later (its subtree still empty)."""
        self.visit += 1
        filled = [c["node"] for c in self.children if c["node"] is not None]
        if filled:
            self.value = max(nd.value for nd in filled)
        if self.parent is not None:
            self.parent.backpropagate()


# ── the search driver ────────────────────────────────────────────────────────

@dataclass
class MCTDSearchConfig:
    guidance_scales: List[float]
    terminal_depth: int
    max_search_num: int = 64
    c_ucb: float = math.sqrt(2.0)
    # None       -> always spend the whole budget, then return the best plan
    # "achieved" -> stop as soon as any expansion produces a goal-reaching plan
    early_stopping: Optional[str] = "achieved"


@dataclass
class ExpandResult:
    """What expand_eval returns for one expansion. `child_partial` becomes the
    new node's payload (its partially-denoised plan, to grow further); the
    `clean_plan` / `info` / `value` come from the jumpy-denoised value preview."""
    value: float
    info: str                       # "Achieved" | "NotReached" | "Warp"
    child_partial: Any
    clean_plan: Any = None
    achieved_t: Optional[int] = None


@dataclass
class MCTDSearchResult:
    solved: bool
    n_search: int
    n_nodes: int
    max_depth: int
    root_value: Optional[float]
    # (clean_plan, value, achieved_t) for every expansion that reached the goal
    achieved: List[tuple] = field(default_factory=list)
    # (clean_plan, value) for expansions that neither reached nor warped
    not_reached: List[tuple] = field(default_factory=list)


def run_mctd_search(root: MCTDTreeNode,
                    expand_eval: Callable[[MCTDTreeNode, Dict[str, Any]], ExpandResult],
                    cfg: MCTDSearchConfig, rng) -> MCTDSearchResult:
    """Sequential MCTD (faithful to p_mctd_plan with parallel_search_num=1).

    root: an MCTDTreeNode at depth 0 (value pre-set, usually 0.0).
    expand_eval(node, cand): does the denoising block + jumpy value preview for
        candidate child `cand` of `node`, returning an ExpandResult. `cand` is a
        dict: {index, depth, guidance_scale, parent}.
    rng: a numpy Generator (only used to pick which empty child slot to expand,
        matching the reference's np.random.choice — pass a seeded one in tests).
    """
    solved = False
    n_search = 0
    n_nodes = 0
    max_depth = 0
    achieved: List[tuple] = []
    not_reached: List[tuple] = []

    while n_search < cfg.max_search_num:
        # ── Selection: descend by UCT through fully-expanded internal nodes
        #    until we reach one with an empty slot (expandable) or a terminal.
        node = root
        while (not node.is_expandable()) and (not node.is_terminal()) \
                and node.is_selectable():
            node = node.uct_select(cfg.c_ucb)
        # Dead end: a terminal leaf, or a node that is neither expandable nor
        # selectable (can only happen if the whole tree is exhausted).
        if node.is_terminal() or (not node.is_expandable()
                                  and not node.is_selectable()):
            break

        # ── Expansion: pick an empty guidance slot, run one denoising block.
        slots = node.empty_slots()
        index = int(rng.choice(slots))
        cand = dict(index=index, depth=node.depth + 1,
                    guidance_scale=node.children[index]["guidance_scale"],
                    parent=node)
        res = expand_eval(node, cand)

        child = node.add_child(index, res.value, res.child_partial)
        n_nodes += 1
        max_depth = max(max_depth, child.depth)
        if res.info == "Achieved":
            achieved.append((res.clean_plan, res.value, res.achieved_t))
            solved = True
        elif res.info == "NotReached":
            not_reached.append((res.clean_plan, res.value))
        # "Warp" plans are discarded (value 0, dynamically infeasible) — the
        # child still exists in the tree with value 0, matching the reference.

        # ── Backprop: from the expanded PARENT up (max-backup).
        node.backpropagate()
        n_search += 1

        if cfg.early_stopping == "achieved" and solved:
            break

    return MCTDSearchResult(solved=solved, n_search=n_search, n_nodes=n_nodes,
                            max_depth=max_depth, root_value=root.value,
                            achieved=achieved, not_reached=not_reached)
