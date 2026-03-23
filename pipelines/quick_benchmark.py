"""Quick benchmark for the veteran MCSS pipeline.

Runs a small number of episodes to get a fast performance signal.
Use this to verify a code change didn't break things or noticeably
shift performance before doing a full evaluation.

The script auto-detects the environment domain (maze2d / antmaze)
from the task name and uses the correct dataset, reward logic, and
network config for each.

Usage:
    # Default: maze2d-umaze-v1, 5 envs, 3 episodes
    python pipelines/quick_benchmark.py

    # Different task
    python pipelines/quick_benchmark.py --task antmaze-large-diverse-v2

    # Custom checkpoints
    python pipelines/quick_benchmark.py \\
        --planner-ckpt results/.../planner_ckpt_latest.pt \\
        --critic-ckpt results/.../critic_ckpt_latest.pt \\
        --policy-ckpt results/.../policy_ckpt_latest.pt

    # Save results to JSON for later comparison
    python pipelines/quick_benchmark.py --save-json baseline.json

    # Compare against a saved baseline
    python pipelines/quick_benchmark.py --baseline baseline.json
"""

import argparse
import json
import os
import time
from typing import Optional

import d4rl
import gym
import numpy as np
import torch

from cleandiffuser.diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DiT1d, DVInvMlp
from cleandiffuser.utils import DVHorizonCritic


# ──────────────────────────────────────────────────────────────────
# Task registry — mirrors configs/veteran/{domain}/task/*.yaml
# ──────────────────────────────────────────────────────────────────

TASKS = {
    # maze2d
    "maze2d-umaze-v1":           {"domain": "maze2d", "max_path_length": 300,  "planner_horizon": 32, "stride": 15, "planner_temperature": 1.0, "discount": 1.0,   "planner_depth": 2},
    "maze2d-medium-v1":          {"domain": "maze2d", "max_path_length": 600,  "planner_horizon": 32, "stride": 15, "planner_temperature": 1.0, "discount": 1.0,   "planner_depth": 2},
    "maze2d-large-v1":           {"domain": "maze2d", "max_path_length": 800,  "planner_horizon": 32, "stride": 15, "planner_temperature": 1.0, "discount": 1.0,   "planner_depth": 2},
    # antmaze
    "antmaze-medium-play-v2":    {"domain": "antmaze", "max_path_length": 1000, "planner_horizon": 40, "stride": 25, "planner_temperature": 1.0, "discount": 0.997, "planner_depth": 8},
    "antmaze-medium-diverse-v2": {"domain": "antmaze", "max_path_length": 1000, "planner_horizon": 40, "stride": 25, "planner_temperature": 1.0, "discount": 0.997, "planner_depth": 8},
    "antmaze-large-play-v2":     {"domain": "antmaze", "max_path_length": 1000, "planner_horizon": 40, "stride": 25, "planner_temperature": 1.0, "discount": 0.997, "planner_depth": 8},
    "antmaze-large-diverse-v2":  {"domain": "antmaze", "max_path_length": 1000, "planner_horizon": 40, "stride": 25, "planner_temperature": 1.0, "discount": 0.997, "planner_depth": 8},
}

def resolve_ckpt(provided, base, name):
    # If the user passed a specific path via args, use it
    if provided:
        return provided
    
    latest = os.path.join(base, f"{name}_ckpt_latest.pt")
    milestone = os.path.join(base, f"{name}_ckpt_1000000.pt")
    
    # Check if 'latest' exists, otherwise fall back to '1000000'
    return latest if os.path.exists(latest) else milestone

def get_ckpt_base(task: str) -> str:
    """Build the default checkpoint directory for a task."""
    info = TASKS[task]
    domain = info["domain"]
    depth = info["planner_depth"]
    return (
        f"results/veteran_d4rl_{domain}"
        f"_H{info['planner_horizon']}_Jump{info['stride']}"
        f"_next1_MCSS_transformer_d{depth}_width256_separate_dpTrue"
        f"/{task}/"
    )


def make_dataset(task: str, env):
    """Create the correct dataset class for the domain."""
    info = TASKS[task]
    kwargs = dict(
        horizon=info["planner_horizon"],
        discount=info["discount"],
        continous_reward_at_done=True,
        reward_tune="iql",
        stride=info["stride"],
        learn_policy=False,
        center_mapping=False,
    )
    if info["domain"] == "maze2d":
        from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset
        return DV_D4RLMaze2DSeqDataset(env.get_dataset(), **kwargs)
    else:
        from cleandiffuser.dataset.d4rl_antmaze_dataset import DV_D4RLAntmazeSeqDataset
        return DV_D4RLAntmazeSeqDataset(env.get_dataset(), **kwargs)


# ──────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────

def load_models(task, device, obs_dim, act_dim,
                planner_ckpt, critic_ckpt, policy_ckpt):
    """Instantiate and load planner, critic, and policy."""
    info = TASKS[task]
    planner_horizon = info["planner_horizon"]
    planner_dim = obs_dim  # separate pipeline

    emb_dim = 128
    d_model = 256

    # Planner
    nn_planner = DiT1d(
        planner_dim, emb_dim=emb_dim,
        d_model=d_model, n_heads=d_model // 64,
        depth=info["planner_depth"], timestep_emb_type="fourier")

    fix_mask = torch.zeros((planner_horizon, planner_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((planner_horizon, planner_dim))
    loss_weight[1] = 1

    planner = ContinuousDiffusionSDE(
        nn_planner, nn_condition=None,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=None,
        ema_rate=0.9999, device=device, predict_noise=True,
        noise_schedule="linear")
    planner.load(planner_ckpt)
    planner.eval()

    # Critic
    critic = DVHorizonCritic(
        planner_dim, emb_dim=emb_dim,
        d_model=d_model, n_heads=d_model // 64,
        depth=2, norm_type="pre").to(device)
    ckpt = torch.load(critic_ckpt, map_location=device)
    critic.load_state_dict(ckpt["critic"])
    critic.eval()

    # Policy (diffusion inverse dynamics)
    nn_invdyn = DVInvMlp(
        obs_dim, act_dim, emb_dim=64,
        hidden_dim=256, timestep_emb_type="positional").to(device)
    nn_cond = IdentityCondition(dropout=0.0).to(device)
    policy = DiscreteDiffusionSDE(
        nn_invdyn, nn_cond, predict_noise=True,
        optim_params={"lr": 3e-4},
        x_max=+1. * torch.ones((1, act_dim), device=device),
        x_min=-1. * torch.ones((1, act_dim), device=device),
        diffusion_steps=10, ema_rate=0.995, device=device)
    policy.load(policy_ckpt)
    policy.eval()

    return planner, critic, policy


# ──────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────

def run_eval(task, device, num_envs, num_episodes, seed,
             planner_ckpt, critic_ckpt, policy_ckpt):
    """Run evaluation episodes and return (scores, timing_dict)."""
    info = TASKS[task]
    max_path_length = info["max_path_length"]
    planner_horizon = info["planner_horizon"]
    planner_temperature = info["planner_temperature"]
    domain = info["domain"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Dataset for normalizer + dims
    env_tmp = gym.make(task)
    dataset = make_dataset(task, env_tmp)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    normalizer = dataset.get_normalizer()
    planner_dim = obs_dim

    # Models
    planner, critic, policy = load_models(
        task, device, obs_dim, act_dim,
        planner_ckpt, critic_ckpt, policy_ckpt)

    # Vectorized env
    env_eval = gym.vector.make(task, num_envs)

    all_scores = []
    step_times = []

    for ep in range(num_episodes):
        obs = env_eval.reset()
        ep_reward = np.zeros(num_envs, dtype=np.float64)
        finished = np.zeros(num_envs, dtype=bool)
        cum_done = np.zeros(num_envs, dtype=bool)
        t = 0
        ep_start = time.time()

        while not np.all(cum_done) and t < max_path_length + 1:
            step_t0 = time.time()

            # Planner: sample + rerank
            planner_prior = torch.zeros(
                (num_envs * 50, planner_horizon, planner_dim),
                device=device)
            obs_t = torch.tensor(
                normalizer.normalize(obs), device=device, dtype=torch.float32)
            obs_repeat = obs_t.unsqueeze(1).repeat(1, 50, 1).view(-1, obs_dim)
            planner_prior[:, 0, :obs_dim] = obs_repeat

            with torch.no_grad():
                traj, _ = planner.sample(
                    planner_prior, solver="ddim",
                    n_samples=num_envs * 50,
                    sample_steps=20, use_ema=True,
                    condition_cfg=None, w_cfg=1.0,
                    temperature=planner_temperature)

                value = critic(traj).view(num_envs, 50)
                idx = torch.argmax(value, -1)
                traj = traj.reshape(num_envs, 50, planner_horizon, planner_dim)
                traj = traj[torch.arange(num_envs), idx]

            # Policy: inverse dynamics
            policy_prior = torch.zeros((num_envs, act_dim), device=device)
            with torch.no_grad():
                next_obs_plan = traj[:, 1, :]
                obs_policy = obs_t.clone()
                next_obs_policy = next_obs_plan.clone()
                next_obs_policy[:, :2] -= obs_policy[:, :2]
                obs_policy[:, :2] = 0
                act, _ = policy.sample(
                    policy_prior, solver="ddpm",
                    n_samples=num_envs, sample_steps=10,
                    condition_cfg=torch.cat([obs_policy, next_obs_policy], dim=-1),
                    w_cfg=1.0, use_ema=True, temperature=0.5)
                act = act.cpu().numpy()

            obs, rew, done, info_env = env_eval.step(act)
            t += 1
            cum_done = np.logical_or(cum_done, done)
            step_times.append(time.time() - step_t0)

            # Reward accumulation differs by domain
            if domain == "maze2d":
                finished |= (rew == 1.0)
                ep_reward += finished
            else:
                ep_reward += rew

        ep_elapsed = time.time() - ep_start

        # Normalize scores
        if domain == "maze2d":
            scores = np.array([env_tmp.get_normalized_score(r) for r in ep_reward]) * 100
        else:
            scores = np.array([env_tmp.get_normalized_score(r) for r in np.clip(ep_reward, 0., 1.)]) * 100

        all_scores.append(scores)
        print(f"  Episode {ep+1}/{num_episodes}: "
              f"mean={scores.mean():.1f}  min={scores.min():.1f}  "
              f"max={scores.max():.1f}  time={ep_elapsed:.1f}s")

    env_eval.close()
    env_tmp.close()

    all_scores = np.concatenate(all_scores)
    timing = {
        "mean_step_ms": float(np.mean(step_times) * 1000),
        "median_step_ms": float(np.median(step_times) * 1000),
        "p95_step_ms": float(np.percentile(step_times, 95) * 1000),
    }
    return all_scores, timing


# ──────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────

def report(task, scores, timing):
    n = len(scores)
    mean = float(scores.mean())
    std = float(scores.std())
    sem = std / np.sqrt(n)
    ci95 = 1.96 * sem

    print(f"\n{'=' * 60}")
    print(f"  {task}  ({n} envs total)")
    print(f"{'=' * 60}")
    print(f"  Normalized score : {mean:.2f} +/- {ci95:.2f}  (95% CI)")
    print(f"  Std              : {std:.2f}")
    print(f"  Min / Max        : {scores.min():.2f} / {scores.max():.2f}")
    print(f"  Step latency     : mean={timing['mean_step_ms']:.1f}ms  "
          f"p50={timing['median_step_ms']:.1f}ms  "
          f"p95={timing['p95_step_ms']:.1f}ms")
    print(f"{'=' * 60}\n")

    return {
        "task": task, "n": n, "mean": mean, "std": std, "ci95": ci95,
        "min": float(scores.min()), "max": float(scores.max()),
        **timing,
    }


def compare_with_baseline(current, baseline):
    """Compare current results against a saved baseline JSON."""
    delta = current["mean"] - baseline["mean"]
    pooled_se = np.sqrt(
        baseline["std"]**2 / baseline["n"] +
        current["std"]**2 / current["n"])
    t_stat = delta / pooled_se if pooled_se > 0 else 0.0

    print(f"{'=' * 60}")
    print(f"  Comparison vs baseline")
    print(f"{'=' * 60}")
    print(f"  Baseline : {baseline['mean']:.2f} +/- {baseline['ci95']:.2f}  (n={baseline['n']})")
    print(f"  Current  : {current['mean']:.2f} +/- {current['ci95']:.2f}  (n={current['n']})")
    print(f"  Delta    : {delta:+.2f}")
    print(f"  t-stat   : {t_stat:.3f}")
    if abs(t_stat) > 1.96:
        print(f"  ** Significant change (p < 0.05) **")
    elif abs(t_stat) > 1.645:
        print(f"  * Marginal change (p < 0.10) *")
    else:
        print(f"  No significant change")
    if baseline.get("mean_step_ms"):
        ratio = current["mean_step_ms"] / max(baseline["mean_step_ms"], 1e-6)
        print(f"  Latency  : {ratio:.2f}x  "
              f"(baseline={baseline['mean_step_ms']:.1f}ms, "
              f"current={current['mean_step_ms']:.1f}ms)")
    print(f"{'=' * 60}\n")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Quick benchmark for veteran MCSS pipeline")
    p.add_argument("--task", default="maze2d-umaze-v1",
                   help="Any d4rl task. Defaults to maze2d-umaze-v1. "
                        f"Known tasks: {', '.join(TASKS.keys())}")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=5,
                   help="Parallel environments (default: 5)")
    p.add_argument("--num-episodes", type=int, default=3,
                   help="Number of eval episodes (default: 3)")

    # Checkpoints
    p.add_argument("--planner-ckpt", default=None,
                   help="Path to planner checkpoint (auto-resolved if omitted)")
    p.add_argument("--critic-ckpt", default=None,
                   help="Path to critic checkpoint (auto-resolved if omitted)")
    p.add_argument("--policy-ckpt", default=None,
                   help="Path to policy checkpoint (auto-resolved if omitted)")

    # Output / comparison
    p.add_argument("--save-json", default=None,
                   help="Save results to JSON for later comparison")
    p.add_argument("--baseline", default=None,
                   help="Path to a baseline JSON to compare against")

    args = p.parse_args()

    task = args.task
    if task not in TASKS:
        print(f"Warning: '{task}' not in known tasks. "
              f"Will try anyway but checkpoint auto-resolution won't work.")

    # Resolve checkpoints
    if task in TASKS:
        base = get_ckpt_base(task)
    else:
        base = ""

    planner_ckpt = resolve_ckpt(args.planner_ckpt, base, "planner")
    critic_ckpt  = resolve_ckpt(args.critic_ckpt,  base, "critic")
    policy_ckpt  = resolve_ckpt(args.policy_ckpt,  base, "policy")

    # Verify checkpoints exist
    for name, path in [("planner", planner_ckpt), ("critic", critic_ckpt), ("policy", policy_ckpt)]:
        if not os.path.isfile(path):
            print(f"ERROR: {name} checkpoint not found: {path}")
            print(f"  Pass --{name}-ckpt explicitly or train first.")
            return

    print(f"\n>>> Quick benchmark: {task}")
    print(f"    envs={args.num_envs}, episodes={args.num_episodes}, "
          f"seed={args.seed}, device={args.device}")
    print(f"    planner: {planner_ckpt}")
    print(f"    critic:  {critic_ckpt}")
    print(f"    policy:  {policy_ckpt}")
    print()

    scores, timing = run_eval(
        task=task, device=args.device,
        num_envs=args.num_envs, num_episodes=args.num_episodes,
        seed=args.seed,
        planner_ckpt=planner_ckpt,
        critic_ckpt=critic_ckpt,
        policy_ckpt=policy_ckpt)

    result = report(task, scores, timing)

    # Save
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.save_json}")

    # Compare
    if args.baseline:
        with open(args.baseline) as f:
            baseline = json.load(f)
        compare_with_baseline(result, baseline)


if __name__ == "__main__":
    main()
