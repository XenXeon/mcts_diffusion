"""mcts/mctd_guidance.py

Goal-directed classifier guidance for MCTD — the tree's META-ACTION. Faithful in
spirit to mctd-main/.../df_planning.py::goal_guidance: pull the plan's future
tokens toward the goal position, with a per-node GUIDANCE SCALE selecting how
hard (scale 0 = free/exploratory denoising, larger = goal-committed). In MCTD
the guidance scale is exactly what the tree branches over.

Mechanism. This exposes `.value(x, k) -> (n,)`, the SAME contract the DF sampler's
guidance hook already consumes (mcts/df_model.py / mcts/mctd_denoise.py):

    eps <- eps - w * sqrt(1 - alpha_bar[k]) * grad_x value(x, k)

value(x) = - mean over future tokens of a squashed distance-to-goal, so ascending
it (the sign above does) moves the plan toward the goal. Two differences from the
reference, both because our backbone conditions the START exactly via a clamped
clean history token (mcts/df_model.py), not via guidance:
  * we guide ONLY toward the goal (no start-reconstruction term — the start is
    already pinned by history clamping);
  * we skip the history columns (the caller zeroes their gradient anyway).

The squashing tanh(d / reach_scale) matches the reference's "squashed-gaussian"
trick: far-from-goal tokens get a bounded, well-conditioned gradient rather than
one that blows up with distance.
"""
from __future__ import annotations

from typing import Sequence

import torch


class GoalGuide:
    """Goal-distance guidance over normalized plans.

    goal_norm:  (P,) the goal position in the planner's NORMALIZED coordinates
                (use mcts.specs.normalize_goal_xy to build it) — guidance runs in
                the same normalized space the DF net operates in.
    pos_dims:   which observation channels are the position (e.g. (0, 1)).
    reach_scale: tanh softening length (normalized units); smaller = sharper.
    """

    def __init__(self, goal_norm: Sequence[float], pos_dims=(0, 1),
                 reach_scale: float = 2.0, device: str = "cpu"):
        self.pos_dims = tuple(int(d) for d in pos_dims)
        self.reach_scale = float(reach_scale)
        self.goal = torch.as_tensor(goal_norm, dtype=torch.float32,
                                    device=device).reshape(-1)
        if self.goal.numel() != len(self.pos_dims):
            raise ValueError(
                f"goal_norm has {self.goal.numel()} dims but pos_dims picks "
                f"{len(self.pos_dims)} ({self.pos_dims})")

    def value(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """x: (n, T, D) normalized plan (row 0 = start/history). k: (n, T) int
        noise levels (accepted for signature compatibility; the goal objective
        does not depend on k — the per-token sqrt(1-alpha_bar[k]) annealing is
        applied by the sampler, not here). Returns (n,)."""
        pos = x[..., self.pos_dims]                              # (n, T, P)
        d = torch.linalg.norm(pos - self.goal.view(1, 1, -1), dim=-1)   # (n, T)
        r = torch.tanh(d / self.reach_scale)                    # (n, T) in [0,1)
        # future tokens only: token 0 is the clamped start/history
        return -(r[:, 1:].mean(dim=1))                          # (n,)
