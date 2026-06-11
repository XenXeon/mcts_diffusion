"""mcts/specs.py

Single source of truth for the env-family constants shared by the MCTS-sampler
tooling (mcts/mcts_loop.py, scripts/train_state_value.py, scripts/eval_state_value.py).

These dicts were previously copy-pasted into each consumer; any drift between the
copies would silently change which checkpoint / target config an experiment uses.
Import from here instead.

This module is importable WITHOUT torch/numpy/gym (constants only at import time);
the helpers that need heavyweight deps import them lazily inside the function, so
torch-free environments (e.g. the local Windows box running unit tests) can still
import `mcts.specs`.

Checkpoint-step conventions (verified against configs + saved checkpoints):
- planner / policy: 1000000 (configs/veteran/*/: planner_ckpt, policy_ckpt).
- MCSS trajectory critic: the official DV config default is critic_ckpt=200000
  (configs/veteran/antmaze/antmaze.yaml:62, maze2d.yaml:62), but checkpoints exist
  every 100k up to 1M and this harness has always loaded critic_ckpt_1000000.pt.
  Harness validation showed the two are empirically equivalent (MCSS k=50 reach
  76.0% vs the pipeline's 76.9% baseline) — state this in any write-up that
  compares against the official DV numbers.
- state value V(s): `state_value_ckpt_latest.pt`, co-located with the planner.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def env_family(env_name: str) -> str:
    """'maze2d' or 'antmaze' — the two trained DV checkpoint families."""
    return "maze2d" if env_name.startswith("maze2d") else "antmaze"


# Per-family geometry + checkpoint roots (matches the training pipelines).
# `max_path_length` is a FALLBACK only — prefer max_episode_steps(env), because the
# family-level value is correct only for the largest maze (maze2d: umaze=300,
# medium=600, large=800; antmaze: all 1000).
SPECS: Dict[str, Dict[str, Any]] = {
    "maze2d": dict(
        H=32, stride=15, planner_depth=2, max_path_length=800,
        ckpt=("results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer"
              "_d2_width256_separate_dpTrue")),
    "antmaze": dict(
        H=40, stride=25, planner_depth=8, max_path_length=1000,
        ckpt=("results/veteran_d4rl_antmaze_H40_Jump25_next1_MCSS_transformer"
              "_d8_width256_separate_dp1")),
}

# Dataset value-target config — MUST match the training pipeline
# (configs/veteran/*/reward_mode/linear.yaml; center_mapping=True because
# guidance_type=MCSS != "cfg").  With these settings seq_val is the normalised
# NEGATIVE time-to-terminus in [-1, 1] (1 == at a terminus).
TARGET_CFG: Dict[str, Any] = dict(discount=1.0, continous_reward_at_done=True,
                                  reward_tune="iql", center_mapping=True)


def spec_for(env_name: str) -> Dict[str, Any]:
    return SPECS[env_family(env_name)]


def ckpt_dir(env_name: str, override: Optional[str] = None) -> str:
    """Per-env checkpoint directory (planner/critic/policy/state-value live here)."""
    return (override or spec_for(env_name)["ckpt"]) + f"/{env_name}"


def max_episode_steps(env: Any, env_name: str) -> int:
    """Episode length from the env's own TimeLimit, falling back to the family spec.

    Reading the TimeLimit is what keeps maze2d-umaze (300) / medium (600) from being
    silently run at the family-level 800 — the same bug class the Phase-0 baseline
    script had to fix (see notes/writeup_phases_0_to_4.md §2).
    """
    v = getattr(env, "_max_episode_steps", None)
    if v:
        return int(v)
    return int(spec_for(env_name)["max_path_length"])


def get_goal(e: Any):
    """Eval goal (x, y) of a single (non-vector) env, robust across d4rl versions.

    antmaze exposes `target_goal`; maze2d exposes `_target` (or `get_target()`).
    Returns a float32 numpy array of shape (2,).
    """
    import numpy as np
    u = e.unwrapped
    for attr in ("target_goal", "_target", "target"):
        if hasattr(u, attr):
            g = np.asarray(getattr(u, attr), dtype=np.float32).reshape(-1)
            if g.size >= 2:
                return g[:2]
    if hasattr(u, "get_target"):
        return np.asarray(u.get_target(), dtype=np.float32).reshape(-1)[:2]
    raise RuntimeError("could not locate the goal for this env")


def make_dataset(env_name: str, H: Optional[int] = None,
                 stride: Optional[int] = None) -> Tuple[Any, Any]:
    """Build (env, DV dataset) with the family geometry and the pipeline TARGET_CFG.

    Heavyweight imports (gym/d4rl/cleandiffuser) happen here, not at module import.
    """
    import d4rl  # noqa: F401  (registers envs)
    import gym

    fam = env_family(env_name)
    spec = SPECS[fam]
    H = H or spec["H"]
    stride = stride or spec["stride"]
    env = gym.make(env_name)
    raw = env.get_dataset()
    if fam == "maze2d":
        from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset as DS
    else:
        from cleandiffuser.dataset.d4rl_antmaze_dataset import DV_D4RLAntmazeSeqDataset as DS
    ds = DS(raw, horizon=H, stride=stride, learn_policy=False, **TARGET_CFG)
    return env, ds
