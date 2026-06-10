"""mcts/rollout.py

Phase 4 — closed-loop episode evaluation for MCTS-guided planning.

Two rollout functions share the same interface and return the same EpisodeResult:

    run_greedy_episode   — one-shot DV-MCSS (mirrors run_one_episode.py exactly)
    run_mcts_episode     — MCTS-guided: builds a fresh tree at every env step,
                           extracts best_path()[1] as the next waypoint, then
                           calls the same inverse-dynamics policy.

Design invariants:
    - mcts/node.py, mcts/tree.py, mcts/expansion.py are NOT modified.
    - cleandiffuser/ and pipelines/ are NOT modified.
    - Policy call is identical to run_one_episode.py (rebase_policy=True,
      position components rebased to current obs, w_cfg=1.0, temperature=0.5).
    - denoising_calls counts DDIM steps (not trajectory-samples), matching
      the Phase 0 accounting: plan_steps per planner call, policy_steps per
      policy call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional

import torch


@dataclass
class RolloutConfig:
    """Configuration for a single episode rollout.

    Args:
        obs_dim:            Observation dimension (4 for maze2d-umaze-v1).
        act_dim:            Action dimension (2 for maze2d-umaze-v1).
        child_state_index:  Trajectory index used as the policy waypoint target.
                            Must match TreeConfig.child_state_index (default 1).
        plan_steps:         DDIM denoising steps for the planner (20).
        policy_steps:       DDIM denoising steps for the policy (10).
        max_t:              Maximum environment steps per episode (300).
        device:             Torch device string.
    """
    obs_dim: int = 4
    act_dim: int = 2
    child_state_index: int = 1
    plan_steps: int = 20
    policy_steps: int = 10
    max_t: int = 300
    device: str = "cpu"


@dataclass
class EpisodeResult:
    """Metrics from one rollout, compatible with the Phase 0 JSON schema.

    MCTS-specific fields (mean_tree_depth, mean_cumulative_best) are None
    for greedy episodes.
    """
    method: str
    seed: int
    raw_return: float
    normalized_score: float
    goal_step: Optional[int]
    episode_length: int
    denoising_calls: int
    wall_seconds: float
    ms_per_step: float
    mcts_budget: Optional[int]
    mean_tree_depth: Optional[float]
    mean_cumulative_best: Optional[float]

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ── Internal helper ────────────────────────────────────────────────────────────

def _policy_action(
    obs_norm: torch.Tensor,
    next_norm: torch.Tensor,
    policy: Any,
    cfg: RolloutConfig,
) -> torch.Tensor:
    """Run one inverse-dynamics policy call and return the action tensor (act_dim,).

    Applies rebase_policy=True: subtract current position from both states so
    the policy sees relative motion. Matches run_one_episode.py lines 117-127.
    """
    obs_r = obs_norm.unsqueeze(0).to(cfg.device).clone()   # (1, obs_dim)
    next_r = next_norm.unsqueeze(0).to(cfg.device).clone() # (1, obs_dim)
    next_r[:, :2] -= obs_r[:, :2]
    obs_r[:, :2] = 0.0
    prior = torch.zeros((1, cfg.act_dim), device=cfg.device)
    with torch.no_grad():
        act, _ = policy.sample(
            prior, solver="ddpm", n_samples=1,
            sample_steps=cfg.policy_steps,
            condition_cfg=torch.cat([obs_r, next_r], dim=-1),
            w_cfg=1.0, use_ema=True, temperature=0.5,
        )
    return act.squeeze(0).cpu()  # (act_dim,)


# ── Greedy baseline ────────────────────────────────────────────────────────────

def run_greedy_episode(
    env: Any,
    expansion: Any,
    policy: Any,
    normalizer: Any,
    cfg: RolloutConfig,
    seed: int = 0,
) -> EpisodeResult:
    """One-shot DV-MCSS rollout — mirrors run_one_episode.py exactly.

    At each step:
        1. Normalise current obs.
        2. Call expansion.expand(s_norm) → K candidates, sorted by critic score.
        3. Take the best trajectory's waypoint at child_state_index.
        4. Run policy to compute action.
        5. Step env.

    denoising_calls counts plan_steps (planner) + policy_steps (policy) per step.
    """
    obs = env.reset()
    ep_reward, finished, t, denoise_calls = 0.0, False, 0, 0
    t0 = time.perf_counter()

    while t < cfg.max_t:
        s_norm = torch.tensor(
            normalizer.normalize(obs[None]), dtype=torch.float32,
        ).squeeze(0)  # (obs_dim,)

        result = expansion.expand(s_norm.to(cfg.device))
        denoise_calls += cfg.plan_steps
        next_s_norm = result.trajs[0, cfg.child_state_index, : cfg.obs_dim].cpu()

        act = _policy_action(s_norm, next_s_norm, policy, cfg)
        denoise_calls += cfg.policy_steps

        obs, rew, done, _ = env.step(act.numpy())
        finished = finished or (rew == 1.0)
        ep_reward += float(finished)
        t += 1
        if done:
            break

    wall = time.perf_counter() - t0
    norm_score = env.get_normalized_score(ep_reward) * 100
    # Formula is correct for fixed-horizon envs where done fires only at max_t
    # (maze2d-umaze-v1 always runs exactly max_t steps).  For early-termination
    # envs use int(t - ep_reward) instead.
    goal_step = int(cfg.max_t - ep_reward) if ep_reward > 0 else None

    return EpisodeResult(
        method="DV-MCSS",
        seed=seed,
        raw_return=ep_reward,
        normalized_score=round(norm_score, 2),
        goal_step=goal_step,
        episode_length=t,
        denoising_calls=denoise_calls,
        wall_seconds=round(wall, 2),
        ms_per_step=round(wall / t * 1000, 1) if t > 0 else 0.0,
        mcts_budget=None,
        mean_tree_depth=None,
        mean_cumulative_best=None,
    )


# ── MCTS-guided rollout ────────────────────────────────────────────────────────

def run_mcts_episode(
    env: Any,
    expansion: Any,
    policy: Any,
    normalizer: Any,
    tree_cfg: Any,
    rollout_cfg: RolloutConfig,
    seed: int = 0,
) -> EpisodeResult:
    """MCTS-guided DV-MCSS rollout.

    At each step:
        1. Normalise current obs.
        2. Build a fresh MCTSTree from s_norm and run tree_cfg.max_expansions steps.
        3. Extract best_path()[1].s_norm as the next waypoint.
           (If tree has only root — budget=0, should not happen — use root itself.)
        4. Run the same policy as greedy to compute action.
        5. Step env.

    denoising_calls = (tree_cfg.max_expansions * rollout_cfg.plan_steps
                       + rollout_cfg.policy_steps) per env step.

    Records mean_tree_depth and mean_cumulative_best across all steps.
    """
    from mcts.tree import MCTSTree

    obs = env.reset()
    ep_reward, finished, t, denoise_calls = 0.0, False, 0, 0
    depths: List[float] = []
    cum_bests: List[float] = []
    t0 = time.perf_counter()

    while t < rollout_cfg.max_t:
        s_norm = torch.tensor(
            normalizer.normalize(obs[None]), dtype=torch.float32,
        ).squeeze(0)

        tree = MCTSTree(s_norm, expansion, tree_cfg)
        records = tree.run()
        # Counts DDIM steps, not trajectory-samples, matching Phase 0 accounting.
        # With leaf_batch_size > 1, multiple leaves share one GPU call but the
        # total DDIM steps is still max_expansions × plan_steps — this is the
        # correct measure of compute budget, not the number of GPU invocations.
        denoise_calls += tree_cfg.max_expansions * rollout_cfg.plan_steps

        path = tree.best_path()
        next_s_norm = path[1].s_norm if len(path) >= 2 else path[0].s_norm

        if records:                       # empty only if max_expansions == 0
            last = records[-1]
            depths.append(float(last.tree_depth))
            cum_bests.append(float(last.cumulative_best))

        act = _policy_action(s_norm, next_s_norm, policy, rollout_cfg)
        denoise_calls += rollout_cfg.policy_steps

        obs, rew, done, _ = env.step(act.numpy())
        finished = finished or (rew == 1.0)
        ep_reward += float(finished)
        t += 1
        if done:
            break

    wall = time.perf_counter() - t0
    norm_score = env.get_normalized_score(ep_reward) * 100
    goal_step = int(rollout_cfg.max_t - ep_reward) if ep_reward > 0 else None  # see greedy

    return EpisodeResult(
        method=f"MCTS-K{tree_cfg.K}-exp{tree_cfg.max_expansions}",
        seed=seed,
        raw_return=ep_reward,
        normalized_score=round(norm_score, 2),
        goal_step=goal_step,
        episode_length=t,
        denoising_calls=denoise_calls,
        wall_seconds=round(wall, 2),
        ms_per_step=round(wall / t * 1000, 1) if t > 0 else 0.0,
        mcts_budget=tree_cfg.max_expansions,
        mean_tree_depth=sum(depths) / len(depths) if depths else 0.0,
        mean_cumulative_best=sum(cum_bests) / len(cum_bests) if cum_bests else 0.0,
    )
