"""mcts/node.py

MCTS node, edge, and tree configuration for the Phase 3/4 ablation.

Three storage modes (set via TreeConfig.storage_mode):
    "state_only"             Ablation A — node holds s_norm only
    "trajectory_node"        Ablation B — node holds s_norm + trajectory that produced it
    "state_edge_trajectory"  Ablation C — node holds s_norm; edge holds trajectory + score

Theoretical storage cost per node (fp32 floats, obs_dim=4, H=32):
    A:  4   floats / node
    B: 132   floats / node  (4 + 32*4)
    C:  4   floats / node + 128 floats / edge (32*4); n_edges = n_nodes - 1
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass(frozen=True)
class TreeConfig:
    """Immutable configuration for one MCTS run.

    Args:
        obs_dim:            Observation dimension (4 for maze2d-umaze-v1).
        horizon:            H — trajectory length from planner (32).
        child_state_index:  Index into the trajectory tensor for the child node's
                            start state.  Use 1 to advance one planning step
                            (the pipeline targets traj[:,1,:] — state M=15 dense
                            env steps ahead).  Do NOT confuse with M=15 (the dense
                            stride between waypoints); that is a dataset constant,
                            not a trajectory index.
        K:                  Candidates per expand() call (50).
        ucb_c:              UCB1 exploration constant (sqrt(2) ≈ 1.414 is standard).
        storage_mode:       One of the three ablation modes.
        max_expansions:     Total number of expand() calls (tree-search budget).
        device:             Torch device for expansion ("cpu" or "cuda:0").
    """
    obs_dim: int
    horizon: int
    child_state_index: int
    K: int
    ucb_c: float
    storage_mode: str
    max_expansions: int
    device: str
    leaf_batch_size: int = 1
    ucb_tie_breaking: str = "random"

    def __post_init__(self) -> None:
        valid = {"state_only", "trajectory_node", "state_edge_trajectory"}
        if self.storage_mode not in valid:
            raise ValueError(
                f"storage_mode must be one of {valid!r}, got {self.storage_mode!r}"
            )
        if self.child_state_index <= 0 or self.child_state_index >= self.horizon:
            raise ValueError(
                f"child_state_index ({self.child_state_index}) must be in [1, horizon-1]; "
                f"0 would make child == parent (fix_mask clamps traj[0] to s_norm), "
                f"negative values invoke PyTorch reverse indexing"
            )
        if self.leaf_batch_size < 1:
            raise ValueError(
                f"leaf_batch_size must be >= 1, got {self.leaf_batch_size}"
            )
        if self.ucb_tie_breaking not in {"random", "greedy"}:
            raise ValueError(
                f"ucb_tie_breaking must be 'random' or 'greedy', "
                f"got {self.ucb_tie_breaking!r}"
            )


class MCTSEdge:
    """Directed edge from parent to child node.

    In mode "state_edge_trajectory": traj and score are populated (the
    trajectory segment and critic score that produced the child).
    In modes "state_only" and "trajectory_node": traj=None, score=None.
    The edge always exists — it is the parent-link for backpropagation.
    """

    __slots__ = ("parent", "child", "traj", "score")

    def __init__(
        self,
        parent: MCTSNode,
        child: MCTSNode,
        traj: Optional[torch.Tensor] = None,
        score: Optional[float] = None,
    ) -> None:
        self.parent = parent
        self.child = child
        self.traj = traj      # (H, obs_dim) CPU tensor or None
        self.score = score    # critic score float or None


class MCTSNode:
    """Single node in the MCTS tree.

    All three storage modes share this class; what is materialised varies:
    - Always stored:        s_norm (obs_dim,) CPU tensor, visit_count, value_sum
    - "trajectory_node":   traj (H, obs_dim) CPU tensor (None for root and other modes)
    - "state_edge_trajectory": trajectory stored on the incoming edge, not the node

    parent_edge is None for the root node.
    """

    __slots__ = (
        "s_norm", "config", "parent_edge",
        "visit_count", "value_sum",
        "traj",
        "_children", "_edges",
    )

    def __init__(
        self,
        s_norm: torch.Tensor,
        config: TreeConfig,
        parent_edge: Optional[MCTSEdge] = None,
        traj: Optional[torch.Tensor] = None,
    ) -> None:
        self.s_norm: torch.Tensor = s_norm.cpu()
        self.config: TreeConfig = config
        self.parent_edge: Optional[MCTSEdge] = parent_edge
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.traj: Optional[torch.Tensor] = (
            traj.cpu() if (config.storage_mode == "trajectory_node" and traj is not None)
            else None
        )
        self._children: List[MCTSNode] = []
        self._edges: List[MCTSEdge] = []   # parallel to _children

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def children(self) -> List[MCTSNode]:
        return self._children

    @property
    def edges(self) -> List[MCTSEdge]:
        return self._edges

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent_edge is None

    # ── Value ──────────────────────────────────────────────────────────────────

    def value(self) -> float:
        """Average critic score seen through this node (0.0 if unvisited)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    # ── UCB1 ───────────────────────────────────────────────────────────────────

    def ucb(self, c: float) -> float:
        """UCB1 score for this node viewed as a child.

        Unvisited nodes return +inf so they are always expanded first.
        Root (no parent) returns its raw value.
        """
        if self.visit_count == 0:
            return float("inf")
        if self.parent_edge is None:
            return self.value()
        parent_n = self.parent_edge.parent.visit_count
        if parent_n == 0:
            return self.value()
        return self.value() + c * math.sqrt(math.log(parent_n) / self.visit_count)

    def best_child(self, c: float) -> MCTSNode:
        """Child with the highest UCB1 score.

        Tie-breaking is controlled by config.ucb_tie_breaking:
        - "random": pick uniformly at random among tied children (default).
          Matters when multiple children are unvisited (UCB=inf); prevents
          greedy depth-first descent before other branches are explored.
        - "greedy": always pick the first tied child (children[0] is the
          highest critic-scored candidate from expand(), so this biases
          toward the best-scored unvisited branch).
        """
        ucb_scores = [ch.ucb(c) for ch in self._children]
        max_ucb = max(ucb_scores)
        tied = [ch for ch, u in zip(self._children, ucb_scores) if u == max_ucb]
        if self.config.ucb_tie_breaking == "greedy":
            return tied[0]
        return tied[torch.randint(len(tied), (1,)).item()]

    # ── Mutation ───────────────────────────────────────────────────────────────

    def add_child(self, child: MCTSNode, edge: MCTSEdge) -> None:
        self._children.append(child)
        self._edges.append(edge)
