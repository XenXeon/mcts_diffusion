"""Smoke tests to verify the Docker image installation is correct."""
import pytest


def test_torch_cuda():
    import torch
    assert torch.cuda.is_available(), "CUDA is not available"
    print(f"torch version: {torch.__version__}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


def test_torch_blackwell_support():
    import torch
    # RTX 50-series (Blackwell) requires sm_120 support
    assert "sm_120" in torch.cuda.get_arch_list(), (
        f"sm_120 (Blackwell) not supported in this PyTorch build. Arch list: {torch.cuda.get_arch_list()}"
    )


def test_cleandiffuser_import():
    import cleandiffuser
    print(f"cleandiffuser location: {cleandiffuser.__file__}")
    # Should be the live-mounted workspace, not a site-packages copy
    assert "/workspace" in cleandiffuser.__file__, (
        f"cleandiffuser is not loading from /workspace: {cleandiffuser.__file__}"
    )


def test_d4rl_import():
    import gym
    import d4rl  # noqa: F401 — import registers envs as a side effect
    env = gym.make("hopper-medium-v2")
    env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())
    assert obs is not None


def test_mujoco_py_import():
    import mujoco_py  # noqa: F401


def test_mujoco_import():
    import mujoco  # noqa: F401
