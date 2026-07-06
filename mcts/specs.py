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
    """'maze2d', 'antmaze', or 'kitchen' — the trained DV checkpoint families."""
    if env_name.startswith("maze2d"):
        return "maze2d"
    if env_name.startswith("kitchen"):
        return "kitchen"
    return "antmaze"


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
    # kitchen: long-horizon SEQUENTIAL manipulation (complete 4 subtasks), NO locomotion.
    # Value target = normalised DISCOUNTED return of subtask completions (a CLEAN signal,
    # unlike nav's noisy behaviour-time) -> V(s) should correlate well here. discount=0.997
    # matches configs/veteran/kitchen/kitchen.yaml (nav uses 1.0).
    "kitchen": dict(
        H=32, stride=4, planner_depth=2, max_path_length=280, discount=0.997,
        ckpt=("results/veteran_d4rl_kitchen_H32_Jump4_next1_MCSS_transformer"
              "_d2_width256_separate_dp1")),
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


def normalize_goal_xy(normalizer: Any, xy: Any):
    """Normalise a raw goal (x, y) to the critic's goal-input scale.

    THE single source for goal normalisation (plan v5.1 C1): training uses
    goal = seq_obs[..., :2], i.e. the GaussianNormalizer-normalised xy dims, so
    inference must normalise the raw goal with the SAME [0:2] statistics. D1, D4,
    and the Stage-2 sampler all call this — a second, subtly-different inlined
    version (wrong dims / forgotten normalisation) is the most likely train/deploy
    skew, so there must be exactly one.

    Accepts (2,) or (N, 2); returns the same shape, float32.
    """
    import numpy as np
    arr = np.asarray(xy, dtype=np.float32)
    single = arr.ndim == 1
    pts = arr.reshape(-1, 2)
    obs_dim = np.asarray(normalizer.mean).reshape(-1).shape[0]
    padded = np.zeros((pts.shape[0], obs_dim), dtype=np.float32)
    padded[:, :2] = pts
    g = normalizer.normalize(padded)[:, :2].astype(np.float32)
    return g[0] if single else g


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
                 stride: Optional[int] = None,
                 learn_policy: bool = False) -> Tuple[Any, Any]:
    """Build (env, DV dataset) with the family geometry and the pipeline TARGET_CFG.

    `learn_policy=False` (default) keeps only terminus-reaching trajectories — the
    DV critic / V(s) regime. `learn_policy=True` ALSO keeps timeout trajectories
    (~89% more data on antmaze-large-diverse), which carry valid relabeling triples
    even though they never reach a goal — the full-data V(s,g) regime (plan v5.1).
    Either way seq_obs uses the same GaussianNormalizer, so the goal scale is
    identical across modes.

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
    if fam == "kitchen":
        # Different dataset signature: discounted MC return of subtask completions, NO
        # learn_policy / reward_tune / continous_reward_at_done (those are the nav target).
        from cleandiffuser.dataset.d4rl_kitchen_dataset import DV_D4RLKitchenSeqDataset as DS
        ds = DS(raw, horizon=H, stride=stride, discount=spec.get("discount", 0.997),
                center_mapping=True)
        return env, ds
    if fam == "maze2d":
        from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset as DS
    else:
        from cleandiffuser.dataset.d4rl_antmaze_dataset import DV_D4RLAntmazeSeqDataset as DS
    ds = DS(raw, horizon=H, stride=stride, learn_policy=learn_policy, **TARGET_CFG)
    return env, ds
