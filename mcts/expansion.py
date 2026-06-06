"""mcts/expansion.py

Single-state expansion primitive for MCTS over the DV planner.

Given a *normalised* observation s_norm (already z-scored via GaussianNormalizer),
generates K candidate trajectories using ContinuousDiffusionSDE, scores each with
DVHorizonCritic, and returns them sorted by critic score (descending).

Normalisation and unnormalisation are the caller's responsibility.
The planner's fix_mask (which clamps trajectory position-0 to the start state) must
be configured at planner construction time — this module does not touch it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExpansionConfig:
    """Immutable configuration for one expansion call.

    Mirrors the maze2d-umaze-v1 production settings by default; override per task.
    """
    K: int            # number of candidate trajectories
    horizon: int      # H — number of jump-step waypoints in each trajectory
    obs_dim: int      # observation dimension
    planner_dim: int  # planner input dim (= obs_dim for 'separate' pipeline)
    solver: str       # diffusion solver, e.g. "ddim"
    sample_steps: int # denoising steps per sample call
    temperature: float
    use_ema: bool
    device: str       # torch device string, e.g. "cpu" or "cuda:0"


@dataclass
class ExpansionResult:
    """Output of one expansion call.

    Both fields are sorted descending by critic score.
    trajs[0] / scores[0] is the highest-valued candidate.
    """
    trajs: torch.Tensor   # (K, H, planner_dim) — normalised
    scores: torch.Tensor  # (K,) — critic values, float

    @property
    def best_traj(self) -> torch.Tensor:
        """Highest-scored trajectory, shape (H, planner_dim)."""
        return self.trajs[0]

    @property
    def best_score(self) -> float:
        return self.scores[0].item()


class PlannerExpansion:
    """Expansion primitive: ranked K-sample rollout from a single start state.

    Wraps the DV planner (ContinuousDiffusionSDE) and critic (DVHorizonCritic)
    to produce a ranked set of candidate trajectories from one normalised start
    state.  The operation is stateless — no tree bookkeeping is done here.

    Both planner and critic must already be in eval mode with weights loaded.

    Args:
        planner: ContinuousDiffusionSDE — fix_mask must clamp position-0 obs dims.
        critic:  DVHorizonCritic — returns (B, 1) scalar per trajectory.
        config:  ExpansionConfig
    """

    def __init__(self, planner, critic, config: ExpansionConfig) -> None:
        self.planner = planner
        self.critic = critic
        self.cfg = config

    def expand(self, s_norm: torch.Tensor) -> ExpansionResult:
        """Generate and rank K candidate trajectories from start state s_norm.

        Args:
            s_norm: (obs_dim,) normalised start observation.

        Returns:
            ExpansionResult — trajs (K, H, planner_dim) and scores (K,) sorted
            descending by critic value.

        Raises:
            ValueError: if s_norm.shape != (obs_dim,).
        """
        cfg = self.cfg
        if s_norm.shape != (cfg.obs_dim,):
            raise ValueError(
                f"s_norm must have shape ({cfg.obs_dim},), got {tuple(s_norm.shape)}"
            )

        # Prior: zeros everywhere; write start state into position-0 obs channels.
        # The planner's fix_mask will re-clamp position-0 after every denoising step.
        prior = torch.zeros(
            (cfg.K, cfg.horizon, cfg.planner_dim), device=cfg.device
        )
        prior[:, 0, : cfg.obs_dim] = (
            s_norm.to(cfg.device).unsqueeze(0).expand(cfg.K, -1)
        )

        with torch.no_grad():
            trajs, _ = self.planner.sample(
                prior,
                solver=cfg.solver,
                n_samples=cfg.K,
                sample_steps=cfg.sample_steps,
                use_ema=cfg.use_ema,
                condition_cfg=None,
                w_cfg=1.0,
                temperature=cfg.temperature,
            )
            # critic returns (K, 1); squeeze to (K,)
            scores = self.critic(trajs).squeeze(-1)

        order = torch.argsort(scores, descending=True)
        return ExpansionResult(trajs=trajs[order], scores=scores[order])

    def expand_batch(self, states: torch.Tensor) -> list:
        """Generate K candidate trajectories for each of N start states in one GPU call.

        Args:
            states: (N, obs_dim) normalised start observations.

        Returns:
            List of N ExpansionResults, each sorted descending by critic score.
        """
        cfg = self.cfg
        N = states.shape[0]
        if states.ndim != 2 or states.shape[1] != cfg.obs_dim:
            raise ValueError(
                f"states must be (N, obs_dim={cfg.obs_dim}), got {tuple(states.shape)}"
            )

        prior = torch.zeros(
            (N * cfg.K, cfg.horizon, cfg.planner_dim), device=cfg.device
        )
        for i in range(N):
            prior[i * cfg.K:(i + 1) * cfg.K, 0, :cfg.obs_dim] = (
                states[i].to(cfg.device)
            )

        with torch.no_grad():
            trajs, _ = self.planner.sample(
                prior,
                solver=cfg.solver,
                n_samples=N * cfg.K,
                sample_steps=cfg.sample_steps,
                use_ema=cfg.use_ema,
                condition_cfg=None,
                w_cfg=1.0,
                temperature=cfg.temperature,
            )
            scores = self.critic(trajs).squeeze(-1)  # (N*K,)

        results = []
        for i in range(N):
            s = scores[i * cfg.K:(i + 1) * cfg.K]
            t = trajs[i * cfg.K:(i + 1) * cfg.K]
            order = torch.argsort(s, descending=True)
            results.append(ExpansionResult(trajs=t[order], scores=s[order]))
        return results
