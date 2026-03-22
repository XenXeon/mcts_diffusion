"""Quick benchmark for the veteran pipeline with MCSS guidance.

Runs a small number of episodes with reduced compute to get a fast,
reasonably meaningful performance signal. Use this to check whether
a code change broke things or shifted performance before committing
to a full evaluation run.

Usage:
    # Defaults: 5 envs, 3 episodes, antmaze-medium-play-v2
    python pipelines/quick_benchmark.py

    # Specific task
    python pipelines/quick_benchmark.py --task antmaze-large-diverse-v2

    # Compare two configs (A/B)
    python pipelines/quick_benchmark.py --compare \\
        --sampling-steps-a 20 --candidates-a 50 \\
        --sampling-steps-b 5  --candidates-b 10

    # Custom checkpoint paths
    python pipelines/quick_benchmark.py \\
        --planner-ckpt /path/to/planner.pt \\
        --critic-ckpt /path/to/critic.pt \\
        --policy-ckpt /path/to/policy.pt
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.dataset.d4rl_antmaze_dataset import DV_D4RLAntmazeSeqDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DiT1d, DVInvMlp
from cleandiffuser.utils import DVHorizonCritic

# ──────────────────────────────────────────────────────────────────
# Task registry — mirrors the hydra task configs
# ──────────────────────────────────────────────────────────────────
TASKS = {
    "antmaze-medium-play-v2":    {"max_path_length": 1000, "planner_horizon": 40, "stride": 25, "planner_temperature": 1.0},
    "antmaze-medium-diverse-v2": {"max_path_length": 1000, "planner_horizon": 40, "stride": 25, "planner_temperature": 1.0},
    "antmaze-large-play-v2":     {"max_path_length": 1000, "planner_horizon": 40, "stride": 25, "planner_temperature": 1.0},
    "antmaze-large-diverse-v2":  {"max_path_length": 1000, "planner_horizon": 40, "stride": 25, "planner_temperature": 1.0},
}

# Default checkpoint path pattern (relative to repo root)
DEFAULT_CKPT_BASE = "results/veteran_d4rl_antmaze_H{horizon}_Jump{stride}_next1_MCSS_transformer_d8_width256_separate_dpTrue/{task}/"


@dataclass
class BenchConfig:
    """All knobs for a single benchmark run."""
    task: str = "antmaze-medium-play-v2"
    seed: int = 42
    device: str = "cuda:0"
    # Eval scale — these are the "quick" defaults
    num_envs: int = 5
    num_episodes: int = 3
    max_path_length: int = 0  # 0 = use task default
    # Planner
    planner_sampling_steps: int = 20
    planner_num_candidates: int = 50
    planner_solver: str = "ddim"
    planner_use_ema: bool = True
    planner_temperature: float = 0.0  # 0 = use task default
    # Policy
    policy_sampling_steps: int = 10
    policy_solver: str = "ddpm"
    policy_use_ema: bool = True
    policy_temperature: float = 0.5
    rebase_policy: bool = True
    # Network dims (must match training)
    planner_emb_dim: int = 128
    planner_d_model: int = 256
    planner_depth: int = 8
    policy_hidden_dim: int = 256
    policy_diffusion_steps: int = 10
    policy_ema_rate: float = 0.995
    planner_ema_rate: float = 0.9999
    # Checkpoint paths (auto-resolved if None)
    planner_ckpt: Optional[str] = None
    critic_ckpt: Optional[str] = None
    policy_ckpt: Optional[str] = None
    # Label for display
    label: str = ""


def resolve_ckpt_paths(cfg: BenchConfig):
    """Fill in default checkpoint paths if not explicitly provided."""
    task_info = TASKS[cfg.task]
    base = DEFAULT_CKPT_BASE.format(
        horizon=task_info["planner_horizon"],
        stride=task_info["stride"],
        task=cfg.task,
    )
    if cfg.planner_ckpt is None:
        cfg.planner_ckpt = base + "planner_ckpt_latest.pt"
    if cfg.critic_ckpt is None:
        cfg.critic_ckpt = base + "critic_ckpt_latest.pt"
    if cfg.policy_ckpt is None:
        cfg.policy_ckpt = base + "policy_ckpt_latest.pt"


# ──────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────

def load_models(cfg: BenchConfig, obs_dim: int, act_dim: int):
    """Instantiate and load planner, critic, and policy from checkpoints."""
    task_info = TASKS[cfg.task]
    planner_horizon = task_info["planner_horizon"]
    planner_dim = obs_dim  # separate pipeline = obs only

    # Planner
    nn_diffusion_planner = DiT1d(
        planner_dim, emb_dim=cfg.planner_emb_dim,
        d_model=cfg.planner_d_model, n_heads=cfg.planner_d_model // 64,
        depth=cfg.planner_depth, timestep_emb_type="fourier")

    fix_mask = torch.zeros((planner_horizon, planner_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((planner_horizon, planner_dim))
    loss_weight[1] = 1  # planner_next_obs_loss_weight=1

    planner = ContinuousDiffusionSDE(
        nn_diffusion_planner, nn_condition=None,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=None,
        ema_rate=cfg.planner_ema_rate,
        device=cfg.device, predict_noise=True, noise_schedule="linear")
    planner.load(cfg.planner_ckpt)
    planner.eval()

    # Critic
    critic = DVHorizonCritic(
        planner_dim, emb_dim=cfg.planner_emb_dim,
        d_model=cfg.planner_d_model, n_heads=cfg.planner_d_model // 64,
        depth=2, norm_type="pre").to(cfg.device)
    critic_ckpt = torch.load(cfg.critic_ckpt, map_location=cfg.device)
    critic.load_state_dict(critic_ckpt["critic"])
    critic.eval()

    # Policy (diffusion inverse dynamics)
    nn_diffusion_invdyn = DVInvMlp(
        obs_dim, act_dim, emb_dim=64,
        hidden_dim=cfg.policy_hidden_dim, timestep_emb_type="positional").to(cfg.device)
    nn_condition_invdyn = IdentityCondition(dropout=0.0).to(cfg.device)
    policy = DiscreteDiffusionSDE(
        nn_diffusion_invdyn, nn_condition_invdyn, predict_noise=True,
        optim_params={"lr": 3e-4},
        x_max=+1. * torch.ones((1, act_dim), device=cfg.device),
        x_min=-1. * torch.ones((1, act_dim), device=cfg.device),
        diffusion_steps=cfg.policy_diffusion_steps,
        ema_rate=cfg.policy_ema_rate, device=cfg.device)
    policy.load(cfg.policy_ckpt)
    policy.eval()

    return planner, critic, policy, planner_dim


# ──────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────

def run_eval(cfg: BenchConfig):
    """Run evaluation and return per-environment normalized scores + timing."""
    resolve_ckpt_paths(cfg)
    task_info = TASKS[cfg.task]
    max_path_length = cfg.max_path_length or task_info["max_path_length"]
    planner_horizon = task_info["planner_horizon"]
    planner_temperature = cfg.planner_temperature or task_info["planner_temperature"]

    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Dataset (needed for normalizer + dims)
    env_tmp = gym.make(cfg.task)
    dataset = DV_D4RLAntmazeSeqDataset(
        env_tmp.get_dataset(), horizon=planner_horizon,
        discount=0.997, continous_reward_at_done=True,
        reward_tune="iql", stride=task_info["stride"],
        learn_policy=False, center_mapping=False)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    normalizer = dataset.get_normalizer()
    planner_dim = obs_dim

    # Models
    planner, critic, policy, _ = load_models(cfg, obs_dim, act_dim)

    # Vectorized env
    env_eval = gym.vector.make(cfg.task, cfg.num_envs)

    all_scores = []
    step_times = []

    for ep in range(cfg.num_episodes):
        obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
        ep_start = time.time()

        while not np.all(cum_done) and t < max_path_length + 1:
            step_t0 = time.time()

            # --- Planner: sample + rerank with critic ---
            planner_prior = torch.zeros(
                (cfg.num_envs * cfg.planner_num_candidates, planner_horizon, planner_dim),
                device=cfg.device)

            obs_t = torch.tensor(normalizer.normalize(obs), device=cfg.device, dtype=torch.float32)
            obs_repeat = obs_t.unsqueeze(1).repeat(1, cfg.planner_num_candidates, 1).view(-1, obs_dim)
            planner_prior[:, 0, :obs_dim] = obs_repeat

            with torch.no_grad():
                traj, _ = planner.sample(
                    planner_prior, solver=cfg.planner_solver,
                    n_samples=cfg.num_envs * cfg.planner_num_candidates,
                    sample_steps=cfg.planner_sampling_steps,
                    use_ema=cfg.planner_use_ema,
                    condition_cfg=None, w_cfg=1.0,
                    temperature=planner_temperature)

            with torch.no_grad():
                value = critic(traj).view(cfg.num_envs, cfg.planner_num_candidates)
                idx = torch.argmax(value, -1)
                traj = traj.reshape(cfg.num_envs, cfg.planner_num_candidates, planner_horizon, planner_dim)
                traj = traj[torch.arange(cfg.num_envs), idx]

            # --- Policy: inverse dynamics ---
            policy_prior = torch.zeros((cfg.num_envs, act_dim), device=cfg.device)
            with torch.no_grad():
                next_obs_plan = traj[:, 1, :]
                obs_policy = obs_t.clone()
                next_obs_policy = next_obs_plan.clone()
                if cfg.rebase_policy:
                    next_obs_policy[:, :2] -= obs_policy[:, :2]
                    obs_policy[:, :2] = 0
                act, _ = policy.sample(
                    policy_prior, solver=cfg.policy_solver,
                    n_samples=cfg.num_envs,
                    sample_steps=cfg.policy_sampling_steps,
                    condition_cfg=torch.cat([obs_policy, next_obs_policy], dim=-1),
                    w_cfg=1.0, use_ema=cfg.policy_use_ema,
                    temperature=cfg.policy_temperature)
                act = act.cpu().numpy()

            obs, rew, done, info = env_eval.step(act)
            t += 1
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)
            ep_reward += rew
            step_times.append(time.time() - step_t0)

        ep_elapsed = time.time() - ep_start
        scores = np.clip(ep_reward, 0., 1.)
        scores = np.array([env_tmp.get_normalized_score(s) for s in scores]) * 100
        all_scores.append(scores)
        print(f"  Episode {ep+1}/{cfg.num_episodes}: "
              f"mean={scores.mean():.1f}  std={scores.std():.1f}  "
              f"time={ep_elapsed:.1f}s")

    env_eval.close()
    env_tmp.close()

    all_scores = np.concatenate(all_scores)
    timing = {
        "mean_step_ms": np.mean(step_times) * 1000,
        "median_step_ms": np.median(step_times) * 1000,
        "p95_step_ms": np.percentile(step_times, 95) * 1000,
    }
    return all_scores, timing


# ──────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────

def report(label: str, scores: np.ndarray, timing: dict):
    n = len(scores)
    mean = scores.mean()
    std = scores.std()
    sem = std / np.sqrt(n)
    ci95 = 1.96 * sem
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Episodes evaluated : {n}")
    print(f"  Normalized score   : {mean:.2f} +/- {ci95:.2f}  (95% CI)")
    print(f"  Std                : {std:.2f}")
    print(f"  Min / Max          : {scores.min():.2f} / {scores.max():.2f}")
    print(f"  Step latency (ms)  : mean={timing['mean_step_ms']:.1f}  "
          f"median={timing['median_step_ms']:.1f}  p95={timing['p95_step_ms']:.1f}")
    print(f"{'=' * 60}\n")
    return {"label": label, "n": n, "mean": mean, "std": std, "ci95": ci95,
            "min": float(scores.min()), "max": float(scores.max()), **timing}


def compare_report(result_a: dict, result_b: dict):
    """Print a comparison and a simple two-sample t-test."""
    delta = result_b["mean"] - result_a["mean"]
    pooled_se = np.sqrt(result_a["std"]**2 / result_a["n"] + result_b["std"]**2 / result_b["n"])
    if pooled_se > 0:
        t_stat = delta / pooled_se
    else:
        t_stat = 0.0

    print(f"{'=' * 60}")
    print(f"  A/B Comparison")
    print(f"{'=' * 60}")
    print(f"  A ({result_a['label']}): {result_a['mean']:.2f} +/- {result_a['ci95']:.2f}")
    print(f"  B ({result_b['label']}): {result_b['mean']:.2f} +/- {result_b['ci95']:.2f}")
    print(f"  Delta (B - A)     : {delta:+.2f}")
    print(f"  t-statistic       : {t_stat:.3f}")
    if abs(t_stat) > 1.96:
        print(f"  ** Significant at p < 0.05 **")
    elif abs(t_stat) > 1.645:
        print(f"  * Marginally significant (p < 0.10) *")
    else:
        print(f"  Not significant (|t| = {abs(t_stat):.2f} < 1.96)")
    speed_ratio = result_b["mean_step_ms"] / max(result_a["mean_step_ms"], 1e-6)
    print(f"  Step latency ratio: {speed_ratio:.2f}x  "
          f"(A={result_a['mean_step_ms']:.1f}ms, B={result_b['mean_step_ms']:.1f}ms)")
    print(f"{'=' * 60}\n")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Quick benchmark for veteran pipeline (MCSS)")
    p.add_argument("--task", default="antmaze-medium-play-v2", choices=list(TASKS.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=5)
    p.add_argument("--num-episodes", type=int, default=3)
    p.add_argument("--max-path-length", type=int, default=0,
                    help="0 = use task default")

    # Planner
    p.add_argument("--sampling-steps", type=int, default=20)
    p.add_argument("--candidates", type=int, default=50)
    p.add_argument("--planner-solver", default="ddim")

    # Policy
    p.add_argument("--policy-sampling-steps", type=int, default=10)
    p.add_argument("--policy-solver", default="ddpm")
    p.add_argument("--policy-temperature", type=float, default=0.5)

    # Checkpoints
    p.add_argument("--planner-ckpt", default=None)
    p.add_argument("--critic-ckpt", default=None)
    p.add_argument("--policy-ckpt", default=None)

    # A/B comparison mode
    p.add_argument("--compare", action="store_true",
                    help="Run an A/B comparison with different inference params")
    p.add_argument("--sampling-steps-a", type=int, default=None)
    p.add_argument("--candidates-a", type=int, default=None)
    p.add_argument("--policy-steps-a", type=int, default=None)
    p.add_argument("--sampling-steps-b", type=int, default=None)
    p.add_argument("--candidates-b", type=int, default=None)
    p.add_argument("--policy-steps-b", type=int, default=None)

    # Output
    p.add_argument("--save-json", default=None,
                    help="Save results to a JSON file")
    return p


def cfg_from_args(args) -> BenchConfig:
    return BenchConfig(
        task=args.task, seed=args.seed, device=args.device,
        num_envs=args.num_envs, num_episodes=args.num_episodes,
        max_path_length=args.max_path_length,
        planner_sampling_steps=args.sampling_steps,
        planner_num_candidates=args.candidates,
        planner_solver=args.planner_solver,
        policy_sampling_steps=args.policy_sampling_steps,
        policy_solver=args.policy_solver,
        policy_temperature=args.policy_temperature,
        planner_ckpt=args.planner_ckpt,
        critic_ckpt=args.critic_ckpt,
        policy_ckpt=args.policy_ckpt,
    )


def main():
    args = build_parser().parse_args()

    if args.compare:
        # ── A/B mode ──
        cfg_a = cfg_from_args(args)
        cfg_a.label = "A (baseline)"
        if args.sampling_steps_a is not None:
            cfg_a.planner_sampling_steps = args.sampling_steps_a
        if args.candidates_a is not None:
            cfg_a.planner_num_candidates = args.candidates_a
        if args.policy_steps_a is not None:
            cfg_a.policy_sampling_steps = args.policy_steps_a

        cfg_b = cfg_from_args(args)
        cfg_b.label = "B (variant)"
        if args.sampling_steps_b is not None:
            cfg_b.planner_sampling_steps = args.sampling_steps_b
        if args.candidates_b is not None:
            cfg_b.planner_num_candidates = args.candidates_b
        if args.policy_steps_b is not None:
            cfg_b.policy_sampling_steps = args.policy_steps_b

        print(f"\n>>> Running config A: steps={cfg_a.planner_sampling_steps}, "
              f"candidates={cfg_a.planner_num_candidates}, "
              f"policy_steps={cfg_a.policy_sampling_steps}")
        scores_a, timing_a = run_eval(cfg_a)
        res_a = report(cfg_a.label, scores_a, timing_a)

        print(f"\n>>> Running config B: steps={cfg_b.planner_sampling_steps}, "
              f"candidates={cfg_b.planner_num_candidates}, "
              f"policy_steps={cfg_b.policy_sampling_steps}")
        scores_b, timing_b = run_eval(cfg_b)
        res_b = report(cfg_b.label, scores_b, timing_b)

        compare_report(res_a, res_b)

        if args.save_json:
            with open(args.save_json, "w") as f:
                json.dump({"config_a": res_a, "config_b": res_b}, f, indent=2)
            print(f"Results saved to {args.save_json}")

    else:
        # ── Single run mode ──
        cfg = cfg_from_args(args)
        cfg.label = f"{cfg.task} (steps={cfg.planner_sampling_steps}, cand={cfg.planner_num_candidates})"

        print(f"\n>>> Quick benchmark: {cfg.task}")
        print(f"    envs={cfg.num_envs}, episodes={cfg.num_episodes}, "
              f"planner_steps={cfg.planner_sampling_steps}, "
              f"candidates={cfg.planner_num_candidates}, "
              f"policy_steps={cfg.policy_sampling_steps}")
        print()

        scores, timing = run_eval(cfg)
        result = report(cfg.label, scores, timing)

        if args.save_json:
            with open(args.save_json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.save_json}")


if __name__ == "__main__":
    main()
