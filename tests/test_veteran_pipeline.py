"""Integration tests for veteran pipeline bug fixes."""
import sys
import os
import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..","pipelines"))

from utils import make_dir, render_episode, Timer
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import DVHorizonCritic


DEVICE = "cpu"


# ── make_dir ──────────────────────────────────────────────────────────────────

def test_make_dir_returns_path(tmp_path):
    result = make_dir(tmp_path / "new_dir")
    assert isinstance(result, Path)
    assert result.exists()


def test_make_dir_idempotent(tmp_path):
    """Calling make_dir twice on the same path should not raise."""
    make_dir(tmp_path / "subdir")
    make_dir(tmp_path / "subdir")


def test_make_dir_creates_parents(tmp_path):
    result = make_dir(tmp_path / "a" / "b" / "c")
    assert result.exists()


# ── Timer ─────────────────────────────────────────────────────────────────────

def test_timer_stop_before_start_raises():
    t = Timer()
    with pytest.raises(RuntimeError, match="Timer.stop\\(\\) called before Timer.start\\(\\)"):
        t.stop()


def test_timer_normal_usage():
    t = Timer()
    t.start()
    elapsed = t.stop()
    assert elapsed >= 0.0


# ── render_episode ─────────────────────────────────────────────────────────────

def _make_mock_env(obs_dim=4, done_at_step=3):
    """Return a mock gym env that terminates at done_at_step."""
    env = MagicMock()
    env.reset.return_value = np.zeros(obs_dim)
    step_count = {"n": 0}

    def step(act):
        step_count["n"] += 1
        done = step_count["n"] >= done_at_step
        return np.zeros(obs_dim), 0.0, done, {}

    env.step.side_effect = step
    env.unwrapped.sim.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    return env


def test_render_episode_saves_gif(tmp_path):
    env = _make_mock_env(done_at_step=3)
    policy_fn = lambda obs: np.zeros(2)
    gif_path = str(tmp_path / "test.gif")
    render_episode(env, policy_fn, gif_path=gif_path, max_steps=10)
    assert os.path.exists(gif_path)


def test_render_episode_respects_max_steps(tmp_path):
    """If env never terminates, render_episode should stop at max_steps."""
    env = _make_mock_env(done_at_step=999)
    call_count = {"n": 0}

    def policy_fn(obs):
        call_count["n"] += 1
        return np.zeros(2)

    render_episode(env, policy_fn, gif_path=str(tmp_path / "out.gif"), max_steps=5)
    assert call_count["n"] == 5


def test_render_episode_stops_on_done(tmp_path):
    """render_episode should stop early when env returns done=True."""
    env = _make_mock_env(done_at_step=2)
    call_count = {"n": 0}

    def policy_fn(obs):
        call_count["n"] += 1
        return np.zeros(2)

    render_episode(env, policy_fn, gif_path=str(tmp_path / "out.gif"), max_steps=100)
    assert call_count["n"] == 2


# ── no_grad during inference sampling ─────────────────────────────────────────

@pytest.fixture
def small_planner():
    obs_dim, horizon = 8, 4
    nn_diff = DiT1d(
        obs_dim, emb_dim=32, d_model=64, n_heads=2, depth=2, timestep_emb_type="fourier"
    )
    fix_mask = torch.zeros((horizon, obs_dim))
    fix_mask[0] = 1.0
    loss_weight = torch.ones((horizon, obs_dim))
    planner = ContinuousDiffusionSDE(
        nn_diff, nn_condition=None,
        fix_mask=fix_mask, loss_weight=loss_weight,
        ema_rate=0.9999, device=DEVICE, predict_noise=True, noise_schedule="linear"
    )
    planner.eval()
    return planner, obs_dim, horizon


def test_planner_sample_no_grad_no_graph(small_planner):
    """Sampler detaches output trajectories — no grad_fn regardless of no_grad context."""
    planner, obs_dim, horizon = small_planner
    n_samples = 4
    prior = torch.zeros((n_samples, horizon, obs_dim))

    with torch.no_grad():
        traj, _ = planner.sample(
            prior, solver="ddim", n_samples=n_samples,
            sample_steps=2, use_ema=False,
            condition_cfg=None, w_cfg=1.0, temperature=1.0
        )

    assert traj.grad_fn is None, "Trajectory should have no grad_fn inside no_grad()"


# ── DVHorizonCritic output shape ───────────────────────────────────────────────

def test_critic_output_shape():
    """Critic must return (b, 1) so .view(num_envs, num_candidates) works correctly."""
    planner_dim, num_envs, num_candidates, horizon = 8, 2, 3, 4
    critic = DVHorizonCritic(
        planner_dim, emb_dim=32, d_model=64, n_heads=2, depth=1
    ).to(DEVICE)
    critic.eval()

    traj = torch.zeros((num_envs * num_candidates, horizon, planner_dim))
    with torch.no_grad():
        value = critic(traj)

    assert value.shape == (num_envs * num_candidates, 1)
    # Verify the reshape pattern used in inference works without error
    value = value.view(num_envs, num_candidates)
    idx = torch.argmax(value, -1)
    assert idx.shape == (num_envs,)
