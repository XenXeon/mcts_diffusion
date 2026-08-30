"""mcts/tree.py

MCTS tree: selection → expansion → backpropagation loop.

Supports all three storage modes from TreeConfig so that the node-storage
design choice is measured as a Phase 3/4 ablation rather than assumed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch

from mcts.node import MCTSEdge, MCTSNode, TreeConfig


@dataclass
class StepRecord:
    """Per-expansion metrics for the ablation comparison."""
    step: int
    wall_time: float          # cumulative seconds since run() started
    expand_time: float        # seconds for this expand() call alone
    n_nodes: int              # total nodes in tree after this step
    tree_depth: int           # maximum depth in the tree after this step
    leaf_best_score: float    # max critic score among the K new children
    leaf_mean_score: float    # mean critic score among the K new children
    cumulative_best: float    # running max of leaf_best_score across all steps
    selected_depth: int       # depth of the node that was expanded this step


class MCTSTree:
    """MCTS tree over the DV planner expansion primitive.

    Each iteration of run():
        1. Selection   — traverse from root to a leaf using UCB1.
        2. Expansion   — call expand() from the leaf state; add K children.
        3. Backprop    — walk to root updating visit_count and value_sum
                         with the mean critic score from the expansion.

    All three storage modes (TreeConfig.storage_mode) share this logic;
    the mode only changes what data is materialised on nodes/edges.

    Args:
        root_state: (obs_dim,) normalised start state.
        expansion:  PlannerExpansion — must already be in eval mode.
        config:     TreeConfig.
    """

    def __init__(
        self,
        root_state: torch.Tensor,
        expansion: Any,
        config: TreeConfig,
    ) -> None:
        self.config = config
        self.expansion = expansion
        self.root = MCTSNode(
            s_norm=root_state,
            config=config,
            parent_edge=None,
        )
        self._all_nodes: List[MCTSNode] = [self.root]
        self._cumulative_best: float = -float("inf")
        self._records: List[StepRecord] = []
        self._max_depth_cached: int = 0

    # ── Selection ──────────────────────────────────────────────────────────────

    def _select(self) -> MCTSNode:
        """Traverse from root to a leaf using UCB1.

        Unvisited children always return +inf so they are selected before
        re-visiting already-expanded nodes.
        """
        node = self.root
        while not node.is_leaf:
            node = node.best_child(self.config.ucb_c)
        return node

    # ── Node depth ─────────────────────────────────────────────────────────────

    def _node_depth(self, node: MCTSNode) -> int:
        depth = 0
        current = node
        while current.parent_edge is not None:
            depth += 1
            current = current.parent_edge.parent
        return depth

    # ── Expansion ──────────────────────────────────────────────────────────────

    def _process_expansion_result(self, leaf: MCTSNode, result: Any) -> Tuple[float, float]:
        """Add K children from result to leaf. Returns (best, mean) critic score."""
        mode = self.config.storage_mode
        idx = self.config.child_state_index

        leaf_depth = self._node_depth(leaf)
        child_depth = leaf_depth + 1
        if child_depth > self._max_depth_cached:
            self._max_depth_cached = child_depth

        for k in range(result.trajs.shape[0]):
            traj_k: torch.Tensor = result.trajs[k].cpu()
            score_k: float = result.scores[k].item()
            next_s: torch.Tensor = traj_k[idx, : self.config.obs_dim]

            child = MCTSNode(
                s_norm=next_s,
                config=self.config,
                parent_edge=None,
                traj=traj_k if mode == "trajectory_node" else None,
            )
            edge = MCTSEdge(
                parent=leaf,
                child=child,
                traj=traj_k if mode == "state_edge_trajectory" else None,
                score=score_k if mode == "state_edge_trajectory" else None,
            )
            child.parent_edge = edge
            leaf.add_child(child, edge)
            self._all_nodes.append(child)

        best = result.scores[0].item()
        mean = result.scores.mean().item()
        return best, mean

    def _expand(self, leaf: MCTSNode) -> Tuple[float, float]:
        """Call expand() from leaf.s_norm; add K children; return (best, mean) score."""
        result = self.expansion.expand(leaf.s_norm.to(self.config.device))
        return self._process_expansion_result(leaf, result)

    # ── Virtual loss for batched selection ─────────────────────────────────────

    def _select_with_virtual_loss(self) -> Tuple[MCTSNode, List[MCTSNode], int]:
        """Select a leaf via UCB, applying virtual loss along the path.

        Virtual loss temporarily increments visit_count on each traversed node
        so that subsequent selections in the same batch are steered away from
        this path.  Call _undo_virtual_loss(path) before real backpropagation.

        Returns:
            (leaf, path, depth) where path contains every non-root node from
            root to leaf (inclusive), and depth = len(path).
        """
        path: List[MCTSNode] = []
        node = self.root
        while not node.is_leaf:
            node = node.best_child(self.config.ucb_c)
            path.append(node)
        for n in path:
            n.visit_count += 1
        return node, path, len(path)

    def _undo_virtual_loss(self, path: List[MCTSNode]) -> None:
        for n in path:
            n.visit_count -= 1

    # ── Backpropagation ────────────────────────────────────────────────────────

    def _backprop(self, node: MCTSNode, value: float) -> None:
        """Walk from node to root, updating visit_count and value_sum.

        NOTE: this is MEAN backup (value() = value_sum / visit_count), the Phase 3/4
        trajectory-critic search. The Phase C state-value search uses MAX (look-ahead)
        backup — see mcts/value_forest.py. The two engines intentionally co-exist.
        """
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            current = (
                current.parent_edge.parent if current.parent_edge is not None else None
            )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _max_depth(self) -> int:
        return self._max_depth_cached

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self) -> List[StepRecord]:
        """Run the search for config.max_expansions steps.

        Returns one StepRecord per expansion (always max_expansions records total).
        When leaf_batch_size > 1, expansions within a batch share one GPU call.
        """
        t_start = time.perf_counter()
        step = 0

        while step < self.config.max_expansions:
            B = min(self.config.leaf_batch_size, self.config.max_expansions - step)

            if B == 1:
                selected = self._select()
                selected_d = self._node_depth(selected)
                t_expand = time.perf_counter()
                best_score, mean_score = self._expand(selected)
                t_expand_end = time.perf_counter()
                if best_score > self._cumulative_best:
                    self._cumulative_best = best_score
                self._backprop(selected, mean_score)
                self._records.append(StepRecord(
                    step=step,
                    wall_time=time.perf_counter() - t_start,
                    expand_time=t_expand_end - t_expand,
                    n_nodes=len(self._all_nodes),
                    tree_depth=self._max_depth(),
                    leaf_best_score=best_score,
                    leaf_mean_score=mean_score,
                    cumulative_best=self._cumulative_best,
                    selected_depth=selected_d,
                ))
                step += 1
            else:
                leaves, paths, depths = [], [], []
                seen: set = set()
                for _ in range(B):
                    leaf, path, depth = self._select_with_virtual_loss()
                    if id(leaf) in seen:
                        # duplicate — undo its virtual loss and skip
                        self._undo_virtual_loss(path)
                        continue
                    seen.add(id(leaf))
                    leaves.append(leaf)
                    paths.append(path)
                    depths.append(depth)

                states = torch.stack([l.s_norm for l in leaves]).to(self.config.device)
                t_expand = time.perf_counter()
                results = self.expansion.expand_batch(states)
                t_expand_end = time.perf_counter()
                expand_time_each = (t_expand_end - t_expand) / B

                for i, (leaf, result, path, depth) in enumerate(
                    zip(leaves, results, paths, depths)
                ):
                    self._undo_virtual_loss(path)
                    best_score, mean_score = self._process_expansion_result(leaf, result)
                    if best_score > self._cumulative_best:
                        self._cumulative_best = best_score
                    self._backprop(leaf, mean_score)
                    self._records.append(StepRecord(
                        step=step + i,
                        wall_time=time.perf_counter() - t_start,
                        expand_time=expand_time_each,
                        n_nodes=len(self._all_nodes),
                        tree_depth=self._max_depth(),
                        leaf_best_score=best_score,
                        leaf_mean_score=mean_score,
                        cumulative_best=self._cumulative_best,
                        selected_depth=depth,
                    ))
                step += len(leaves)

        return self._records

    # ── Query ──────────────────────────────────────────────────────────────────

    def best_path(self) -> List[MCTSNode]:
        """Greedy highest-value path from root to a leaf.

        At each node, picks the child with the highest value() (average critic
        score from backpropagation).  Ties broken by Python's max() — first child
        with the maximum value, which is children[0] (the highest critic-scored
        candidate from expand()).  This is intentionally independent of
        ucb_tie_breaking: UCB governs exploration; value-greedy governs extraction.

        Depth note: with K candidates per expansion, step 0 expands the root
        (1 budget slot) and steps 1..K expand its K unvisited children (UCB=∞).
        Depth 3 requires budget ≥ K + 2; for K=10 that means budget ≥ 12.
        """
        path = [self.root]
        node = self.root
        while not node.is_leaf:
            node = max(node.children, key=lambda c: c.value())
            path.append(node)
        return path

    @staticmethod
    def theoretical_floats(config: TreeConfig, n_nodes: int) -> int:
        """Compute theoretical fp32 storage cost for the given tree size.

        Returns the number of floats stored for trajectory data only
        (s_norm is always stored and is excluded as it is the same across modes).
        """
        n_edges = max(0, n_nodes - 1)
        if config.storage_mode == "state_only":
            return 0
        elif config.storage_mode == "trajectory_node":
            return n_nodes * config.horizon * config.obs_dim
        else:  # state_edge_trajectory
            return n_edges * config.horizon * config.obs_dim
