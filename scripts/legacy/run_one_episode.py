"""Run one maze2d episode with the DV/MCSS pipeline and print stats.

Works for umaze / medium / large — episode length is read from the env's
TimeLimit, so it is never silently truncated.

Prints: denoising-call count, raw return, normalised score, episode length, wall time.
Optionally writes a structured JSON artefact via --save-json <path>.

Usage:
    python scripts/run_one_episode.py                       # umaze, seed 0
    python scripts/run_one_episode.py --env maze2d-large-v1 --seed 3
    python scripts/run_one_episode.py --save-json results/phase0_baseline.json
"""
import argparse, json, os, subprocess, sys, time
sys.path.insert(0, ".")

import d4rl  # noqa: F401 — registers D4RL envs
import gym
import numpy as np
import torch

from cleandiffuser.diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE
from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DiT1d, DVInvMlp
from cleandiffuser.utils import DVHorizonCritic
from pipelines.utils import set_seed

# ── CLI ──
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--env", type=str, default="maze2d-umaze-v1",
                    choices=["maze2d-umaze-v1", "maze2d-medium-v1", "maze2d-large-v1"],
                    help="D4RL maze2d env; selects checkpoint and episode length")
parser.add_argument("--save-json", type=str, default=None,
                    help="Append one result row to this JSON file (list of dicts)")
args = parser.parse_args()
SEED = args.seed

# ── Config (mirrors configs/veteran/maze2d/task/<env>.yaml + maze2d.yaml) ──
DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"
ENV          = args.env
H            = 32    # planner_horizon
M            = 15    # stride (jump step)
K            = 50    # planner_num_candidates
D_MODEL      = 256
DEPTH        = 2
EMB_DIM      = 128
PLAN_STEPS   = 20    # planner_sampling_steps
POLICY_STEPS = 10    # policy_sampling_steps
# MAX_T is read from the env's own TimeLimit after creation (300 umaze /
# 600 medium / 800 large) so episodes are never silently truncated.
CKPT         = (
    "results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer"
    f"_d2_width256_separate_dpTrue/{ENV}"
)

set_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Dataset (needed for GaussianNormalizer) ──
env = gym.make(ENV)
env.seed(SEED)
env.action_space.seed(SEED)
# Episode length straight from the env's TimeLimit — guarantees the rollout
# runs the full task (umaze 300 / medium 600 / large 800), never truncated.
MAX_T = env._max_episode_steps
dataset = DV_D4RLMaze2DSeqDataset(
    env.get_dataset(), horizon=H, stride=M,
    learn_policy=False, center_mapping=False,
    discount=1.0, continous_reward_at_done=True, reward_tune="iql")
normalizer = dataset.get_normalizer()
obs_dim, act_dim = dataset.o_dim, dataset.a_dim  # 4, 2 for maze2d-umaze-v1

# ── Planner ──
fix_mask = torch.zeros((H, obs_dim))
fix_mask[0, :obs_dim] = 1.0
planner = ContinuousDiffusionSDE(
    DiT1d(obs_dim, emb_dim=EMB_DIM, d_model=D_MODEL,
          n_heads=D_MODEL // 64, depth=DEPTH, timestep_emb_type="fourier"),
    fix_mask=fix_mask, noise_schedule="linear", device=DEVICE)
planner.load(f"{CKPT}/planner_ckpt_1000000.pt")
planner.eval()

# ── Critic ──
critic = DVHorizonCritic(
    obs_dim, emb_dim=EMB_DIM, d_model=D_MODEL,
    n_heads=D_MODEL // 64, depth=2, norm_type="pre").to(DEVICE)
critic.load_state_dict(
    torch.load(f"{CKPT}/critic_ckpt_1000000.pt", map_location=DEVICE)["critic"])
critic.eval()

# ── Policy (diffusion inverse dynamics) ──
policy = DiscreteDiffusionSDE(
    DVInvMlp(obs_dim, act_dim, emb_dim=64, hidden_dim=256,
             timestep_emb_type="positional").to(DEVICE),
    IdentityCondition(dropout=0.0).to(DEVICE),
    x_max=+torch.ones((1, act_dim), device=DEVICE),
    x_min=-torch.ones((1, act_dim), device=DEVICE),
    diffusion_steps=POLICY_STEPS, device=DEVICE)
policy.load(f"{CKPT}/policy_ckpt_1000000.pt")
policy.eval()

# ── Episode rollout ──
obs = env.reset()
ep_reward, finished, t, denoise_calls = 0.0, False, 0, 0
t0 = time.perf_counter()

while t < MAX_T:
    obs_t = torch.tensor(
        normalizer.normalize(obs[None]), device=DEVICE, dtype=torch.float32)  # (1, 4)

    # 1) Plan: sample K trajectories, critic-rerank, keep best
    prior = torch.zeros((K, H, obs_dim), device=DEVICE)
    prior[:, 0, :] = obs_t.expand(K, -1)
    with torch.no_grad():
        traj, _ = planner.sample(
            prior, solver="ddim", n_samples=K,
            sample_steps=PLAN_STEPS, use_ema=True, temperature=1.0)
        denoise_calls += PLAN_STEPS
        value = critic(traj).squeeze(-1)              # (K,)
        best = traj[torch.argmax(value)]              # (H, obs_dim)

    # 2) Action: inverse dynamics from (obs, next_obs_plan)
    next_obs = best[1:2, :].clone()                   # (1, obs_dim)
    obs_r, next_r = obs_t.clone(), next_obs.clone()
    next_r[:, :2] -= obs_r[:, :2]                    # rebase_policy=True
    obs_r[:, :2] = 0.0
    policy_prior = torch.zeros((1, act_dim), device=DEVICE)
    with torch.no_grad():
        act, _ = policy.sample(
            policy_prior, solver="ddpm", n_samples=1,
            sample_steps=POLICY_STEPS,
            condition_cfg=torch.cat([obs_r, next_r], dim=-1),
            w_cfg=1.0, use_ema=True, temperature=0.5)
        denoise_calls += POLICY_STEPS

    obs, rew, done, _ = env.step(act.squeeze(0).cpu().numpy())
    finished = finished or (rew == 1.0)   # latch: mirrors reference pipeline line 445
    ep_reward += float(finished)           # +1 every step after first goal-touch
    t += 1
    if done:
        break

wall = time.perf_counter() - t0
score = env.get_normalized_score(ep_reward) * 100
goal_step = int(MAX_T - ep_reward) if ep_reward > 0 else None

print(
    f"seed={SEED}  "
    f"denoising_calls={denoise_calls}  "
    f"return={ep_reward:.1f}  "
    f"normalized_score={score:.1f}  "
    f"length={t}  "
    f"wall={wall:.2f}s"
)

# ── Optional JSON output ──
if args.save_json:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"

    row = {
        "method": "DV-MCSS",
        "env": ENV,
        "seed": SEED,
        "normalized_score": round(score, 2),
        "raw_return": ep_reward,
        "goal_step": goal_step,
        "episode_length": t,
        "denoising_calls": denoise_calls,
        "wall_seconds": round(wall, 2),
        "ms_per_step": round(wall / t * 1000, 1),
        "git_commit": commit,
    }

    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    existing = []
    if os.path.exists(args.save_json):
        with open(args.save_json) as f:
            existing = json.load(f)
    existing.append(row)
    with open(args.save_json, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"→ wrote {args.save_json}")
