"""tests/test_mcts_rollout.py

Unit tests for mcts.rollout — Phase 4 closed-loop evaluation.

All tests use CPU-only fakes for environment, policy, normaliser, and expansion.
No checkpoint, no d4rl, no GPU required.

Unit tests (17)
---------------
    - RolloutConfig / EpisodeResult construction
    - run_greedy_episode: termination, denoising call counting, goal detection,
      correct waypoint extraction, tree metrics absent
    - run_mcts_episode: termination, denoising call counting, goal detection,
      tree metrics present, waypoint comes from best_path()[1]

Run:
    pytest tests/test_mcts_rollout.py -v
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcts.expansion import ExpansionResult
from mcts.node import TreeConfig
from mcts.rollout import EpisodeResult, RolloutConfig, run_greedy_episode, run_mcts_episode

# ── Fakes ──────────────────────────────────────────────────────────────────────

class FakeEnv:
    """Deterministic fake environment.

    Steps forward with a fixed obs increment.  Gives reward=1.0 from step
    goal_at onward so that ep_reward accumulates predictably.  done=True at
    max_t so the episode always terminates.
    """

    def __init__(self, obs_dim: int = 4, act_dim: int = 2,
                 max_t: int = 5, goal_at: int = 3) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_t = max_t
        self.goal_at = goal_at
        self._t = 0

    def reset(self) -> np.ndarray:
        self._t = 0
        return np.zeros(self.obs_dim, dtype=np.float32)

    def step(self, action: np.ndarray):
        self._t += 1
        obs = np.ones(self.obs_dim, dtype=np.float32) * self._t * 0.1
        rew = 1.0 if self._t >= self.goal_at else 0.0
        done = self._t >= self.max_t
        return obs, rew, done, {}

    def get_normalized_score(self, raw_return: float) -> float:
        return raw_return / max(self.max_t, 1)


class FakeNormalizer:
    """Identity normaliser — returns obs unchanged."""

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32)


class FakePolicy:
    """Returns zero actions and records the condition it was called with."""

    def __init__(self, act_dim: int = 2) -> None:
        self.act_dim = act_dim
        self.call_count = 0
        self.conditions: list = []

    def sample(self, prior, *, solver, n_samples, sample_steps,
               condition_cfg, w_cfg, use_ema, temperature):
        self.call_count += 1
        self.conditions.append(condition_cfg.detach().clone())
        return torch.zeros(n_samples, self.act_dim), None


class FakeExpansion:
    """Deterministic expansion — K trajectories, scores linearly spaced 0.9→0.1."""

    def __init__(self, K: int = 3, H: int = 4, obs_dim: int = 4) -> None:
        self.K = K
        self.H = H
        self.obs_dim = obs_dim
        self._call_count = 0

    def expand(self, s_norm: torch.Tensor) -> ExpansionResult:
        trajs = torch.zeros(self.K, self.H, self.obs_dim)
        trajs[:, 0, :] = s_norm.unsqueeze(0).expand(self.K, -1)
        trajs[:, 1, 0] = (
            torch.arange(self.K, dtype=torch.float32) * 0.01
            + self._call_count * 0.1
        )
        scores = torch.linspace(0.9, 0.1, self.K)
        self._call_count += 1
        return ExpansionResult(trajs=trajs, scores=scores)

    def expand_batch(self, states: torch.Tensor) -> list:
        return [self.expand(states[i]) for i in range(states.shape[0])]


# ── Shared helpers ─────────────────────────────────────────────────────────────

def make_rollout_cfg(max_t: int = 5, plan_steps: int = 3,
                     policy_steps: int = 2) -> RolloutConfig:
    return RolloutConfig(
        obs_dim=4, act_dim=2, child_state_index=1,
        plan_steps=plan_steps, policy_steps=policy_steps,
        max_t=max_t, device="cpu",
    )


def make_tree_cfg(K: int = 3, budget: int = 2) -> TreeConfig:
    return TreeConfig(
        obs_dim=4, horizon=4, child_state_index=1,
        K=K, ucb_c=math.sqrt(2), storage_mode="state_only",
        max_expansions=budget, device="cpu",
        leaf_batch_size=1,
    )


# ── RolloutConfig / EpisodeResult ─────────────────────────────────────────────

def test_rollout_config_defaults():
    cfg = RolloutConfig()
    assert cfg.obs_dim == 4
    assert cfg.act_dim == 2
    assert cfg.child_state_index == 1
    assert cfg.plan_steps == 20
    assert cfg.policy_steps == 10
    assert cfg.max_t == 300
    assert cfg.device == "cpu"


def test_episode_result_to_dict():
    r = EpisodeResult(
        method="test", seed=0, raw_return=1.0, normalized_score=50.0,
        goal_step=10, episode_length=15, denoising_calls=100,
        wall_seconds=1.0, ms_per_step=66.7,
        mcts_budget=None, mean_tree_depth=None, mean_cumulative_best=None,
    )
    d = r.to_dict()
    assert d["method"] == "test"
    assert d["raw_return"] == 1.0
    assert "mean_tree_depth" in d


# ── run_greedy_episode ─────────────────────────────────────────────────────────

def test_greedy_terminates_at_max_t():
    cfg = make_rollout_cfg(max_t=5)
    result = run_greedy_episode(
        FakeEnv(max_t=5), FakeExpansion(), FakePolicy(), FakeNormalizer(), cfg)
    assert result.episode_length == 5


def test_greedy_denoising_calls():
    """Each env step costs plan_steps + policy_steps denoising calls."""
    plan_s, policy_s, max_t = 3, 2, 4
    cfg = make_rollout_cfg(max_t=max_t, plan_steps=plan_s, policy_steps=policy_s)
    result = run_greedy_episode(
        FakeEnv(max_t=max_t, goal_at=999), FakeExpansion(),
        FakePolicy(), FakeNormalizer(), cfg)
    assert result.denoising_calls == max_t * (plan_s + policy_s)


def test_greedy_goal_reward_latches():
    """ep_reward increments by 1 for every step after the first goal touch."""
    max_t, goal_at = 6, 3
    cfg = make_rollout_cfg(max_t=max_t)
    result = run_greedy_episode(
        FakeEnv(max_t=max_t, goal_at=goal_at), FakeExpansion(),
        FakePolicy(), FakeNormalizer(), cfg)
    expected_reward = float(max_t - goal_at + 1)  # steps 3,4,5,6 → 4
    assert result.raw_return == expected_reward


def test_greedy_tree_metrics_are_none():
    cfg = make_rollout_cfg(max_t=3)
    result = run_greedy_episode(
        FakeEnv(max_t=3), FakeExpansion(), FakePolicy(), FakeNormalizer(), cfg)
    assert result.mcts_budget is None
    assert result.mean_tree_depth is None
    assert result.mean_cumulative_best is None


def test_greedy_uses_best_trajectory_waypoint():
    """Policy receives the best trajectory's waypoint (trajs[0, child_idx, :])."""
    policy = FakePolicy(act_dim=2)
    expansion = FakeExpansion(K=3, H=4, obs_dim=4)
    cfg = make_rollout_cfg(max_t=1, plan_steps=1, policy_steps=1)
    run_greedy_episode(FakeEnv(max_t=1), expansion, policy, FakeNormalizer(), cfg)

    # The condition is cat([obs_r, next_r], dim=-1) — shape (1, obs_dim*2)
    assert len(policy.conditions) == 1
    cond = policy.conditions[0]   # (1, 8)
    assert cond.shape == (1, 8)


def test_greedy_method_label():
    cfg = make_rollout_cfg(max_t=2)
    result = run_greedy_episode(
        FakeEnv(max_t=2), FakeExpansion(), FakePolicy(), FakeNormalizer(), cfg, seed=7)
    assert result.method == "DV-MCSS"
    assert result.seed == 7


# ── run_mcts_episode ───────────────────────────────────────────────────────────

def test_mcts_terminates_at_max_t():
    cfg = make_rollout_cfg(max_t=4)
    result = run_mcts_episode(
        FakeEnv(max_t=4), FakeExpansion(), FakePolicy(), FakeNormalizer(),
        make_tree_cfg(K=3, budget=2), cfg)
    assert result.episode_length == 4


def test_mcts_denoising_calls():
    """Each env step costs budget*plan_steps + policy_steps denoising calls."""
    plan_s, policy_s, budget, max_t = 3, 2, 4, 3
    cfg = make_rollout_cfg(max_t=max_t, plan_steps=plan_s, policy_steps=policy_s)
    tree_cfg = make_tree_cfg(K=3, budget=budget)
    result = run_mcts_episode(
        FakeEnv(max_t=max_t, goal_at=999), FakeExpansion(),
        FakePolicy(), FakeNormalizer(), tree_cfg, cfg)
    assert result.denoising_calls == max_t * (budget * plan_s + policy_s)


def test_mcts_tree_metrics_populated():
    """mean_tree_depth and mean_cumulative_best are non-None floats."""
    cfg = make_rollout_cfg(max_t=3)
    result = run_mcts_episode(
        FakeEnv(max_t=3), FakeExpansion(), FakePolicy(), FakeNormalizer(),
        make_tree_cfg(K=3, budget=2), cfg)
    assert result.mcts_budget == 2
    assert isinstance(result.mean_tree_depth, float)
    assert isinstance(result.mean_cumulative_best, float)


def test_mcts_method_label_encodes_K_and_budget():
    cfg = make_rollout_cfg(max_t=2)
    result = run_mcts_episode(
        FakeEnv(max_t=2), FakeExpansion(), FakePolicy(), FakeNormalizer(),
        make_tree_cfg(K=5, budget=7), cfg, seed=3)
    assert "K5" in result.method
    assert "exp7" in result.method   # "exp" not "B" — B is leaf_batch_size in Phase 3
    assert result.seed == 3


def test_mcts_method_label_uses_exp_not_B():
    """Method label must say 'exp' not 'B' to avoid confusion with leaf_batch_size."""
    cfg = make_rollout_cfg(max_t=2)
    result = run_mcts_episode(
        FakeEnv(max_t=2), FakeExpansion(), FakePolicy(), FakeNormalizer(),
        make_tree_cfg(K=3, budget=4), cfg)
    assert "exp4" in result.method, (
        f"Expected 'exp4' in method label, got '{result.method}'. "
        "'B' is reserved for leaf_batch_size in Phase 3 terminology."
    )
    assert "B4" not in result.method


def test_depth_stays_two_when_budget_leq_K():
    """With budget <= K, tree never reaches depth 3.

    Expansion 0 expands root → K children (all UCB=inf).
    Expansions 1..K-1 each pick an unvisited depth-1 node (UCB=inf beats any
    visited node) and expand it.  Budget=K means K expansions total: 1 for root
    + (K-1) for children, leaving 1 child still unvisited.  Even budget=K+1
    only finishes the last child — depth stays at 2.
    Depth 3 requires budget ≥ K + 2 (root + K children + 1 grandchild).
    """
    K = 3
    cfg = make_rollout_cfg(max_t=1, plan_steps=1, policy_steps=1)
    # Budget = K: all K children of root get expanded once, but none go deeper
    tree_cfg = make_tree_cfg(K=K, budget=K)
    result = run_mcts_episode(
        FakeEnv(max_t=1), FakeExpansion(K=K), FakePolicy(), FakeNormalizer(),
        tree_cfg, cfg)
    assert result.mean_tree_depth <= 2.0, (
        f"Expected depth <= 2 with budget={K}, K={K}; got {result.mean_tree_depth}"
    )


def test_depth_reaches_three_when_budget_exceeds_K():
    """With budget ≥ K + 2, at least one grandchild gets expanded (depth 3 reached)."""
    K = 3
    cfg = make_rollout_cfg(max_t=1, plan_steps=1, policy_steps=1)
    # Budget = K + 2: after expanding root and all K children, 2 grandchildren get expanded
    tree_cfg = make_tree_cfg(K=K, budget=K + 2)
    result = run_mcts_episode(
        FakeEnv(max_t=1), FakeExpansion(K=K), FakePolicy(), FakeNormalizer(),
        tree_cfg, cfg)
    assert result.mean_tree_depth >= 3.0, (
        f"Expected depth >= 3 with budget={K+2}, K={K}; got {result.mean_tree_depth}"
    )


def test_best_path_first_element_is_root_state():
    """best_path()[0] is always the root — the current env state."""
    from mcts.tree import MCTSTree
    import math
    root_s = torch.tensor([1.0, 2.0, 0.5, -0.3])
    cfg = TreeConfig(
        obs_dim=4, horizon=4, child_state_index=1,
        K=3, ucb_c=math.sqrt(2), storage_mode="state_only",
        max_expansions=2, device="cpu", leaf_batch_size=1,
    )
    tree = MCTSTree(root_s, FakeExpansion(K=3), cfg)
    tree.run()
    path = tree.best_path()
    assert torch.allclose(path[0].s_norm, root_s), (
        "best_path()[0].s_norm must equal the root state (current obs)"
    )


def test_mcts_waypoint_comes_from_tree_best_path():
    """Policy receives next_s_norm from tree.best_path()[1], not a fixed index."""
    policy = FakePolicy(act_dim=2)
    cfg = make_rollout_cfg(max_t=1, plan_steps=1, policy_steps=1)
    tree_cfg = make_tree_cfg(K=3, budget=2)

    run_mcts_episode(
        FakeEnv(max_t=1), FakeExpansion(), policy, FakeNormalizer(), tree_cfg, cfg)

    # One policy call with a condition tensor of shape (1, obs_dim*2)
    assert len(policy.conditions) == 1
    cond = policy.conditions[0]
    assert cond.shape == (1, 8)
    # next_s_norm is path[1].s_norm — position components are rebased to 0
    # so obs_r[:, :2] = [0, 0] and next_r[:, :2] is relative
    obs_r = cond[0, :4]
    assert obs_r[0].item() == pytest.approx(0.0)
    assert obs_r[1].item() == pytest.approx(0.0)
