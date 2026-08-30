"""scripts/phase5_critic_depth_calibration.py

Depth-stratified critic calibration test.

Tests whether the critic remains calibrated when scoring plans generated from
depth-d imagined states — i.e., states reached by chaining d planner jumps
from a real start state (exactly what MCTS does when it descends the tree).

For each depth d in {0, 1, 2, 3}:
  1. Build the depth-d imagined state by chaining d planner jumps, each taking
     waypoint 1 of the generated trajectory as the next state (the current MCTS
     child_state_index=1 design).
  2. Generate K_EVAL candidate plans from the imagined state; record critic scores.
  3. set_state the env to the imagined state; execute each candidate plan via the
     inverse-dynamics policy for N_TRANSITIONS=31 jump-steps; record true return.
  4. Compute Pearson and Spearman correlation(critic_score, true_return).
  5. Compute OOD proxy: nearest-neighbour L2 distance to the training dataset.

Prediction under the critic-exploitation hypothesis:
  - As d increases: nn_dist ↑ (states leave the training manifold)
  - Critic score may stay high or inflate (searching finds critic-maximising inputs)
  - True return decreases (imagined states are harder to navigate from)
  - Calibration degrades: corr(critic, true_return) → 0 or negative

Outputs:
  results/phase5/critic_depth_calibration.csv   (one row per plan)
  results/phase5/critic_depth_summary.csv        (one row per depth)
  results/phase5/plots/critic_depth_scatter.png  (4-panel scatter)
  results/phase5/plots/critic_depth_summary.png  (metric vs depth line plot)

Usage:
    python scripts/phase5_critic_depth_calibration.py
    python scripts/phase5_critic_depth_calibration.py --n-starts 10 --k-eval 5
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, ".")

import d4rl  # noqa: F401
import gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DiT1d, DVInvMlp
from cleandiffuser.utils import DVHorizonCritic
from pipelines.utils import set_seed

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n-starts", type=int, default=5,
                    help="Number of real start states (default 5; increase for more data)")
parser.add_argument("--k-eval", type=int, default=5,
                    help="Plans evaluated per imagined state (default 5)")
parser.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 3])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--env", default="maze2d-umaze-v1",
                    choices=["maze2d-umaze-v1", "maze2d-medium-v1", "maze2d-large-v1"])
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
ENV_NAME      = args.env
ENV_TAG       = ENV_NAME.replace("maze2d-", "").replace("-v1", "")
H             = 32
M             = 15
OBS_DIM       = 4
ACT_DIM       = 2
PLAN_STEPS    = 20
POLICY_STEPS  = 10
N_TRANSITIONS = H - 1   # 31 jump-steps per plan rollout
N_OOD_REF     = 5000    # dataset states used for nearest-neighbour proxy
CKPT = (
    "results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer"
    f"_d2_width256_separate_dpTrue/{ENV_NAME}"
)
OUT_DIR = "results/phase5"
os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)

set_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Dataset + normalizer ──────────────────────────────────────────────────────
print("Loading dataset …")
env_data   = gym.make(ENV_NAME)
raw_ds     = env_data.get_dataset()
dataset    = DV_D4RLMaze2DSeqDataset(
    raw_ds, horizon=H, stride=M,
    learn_policy=False, center_mapping=False,
    discount=1.0, continous_reward_at_done=True, reward_tune="iql")
normalizer = dataset.get_normalizer()

def unnormalise(s_norm_np: np.ndarray) -> np.ndarray:
    """Inverse of normalizer.normalize — returns raw (x,y,vx,vy)."""
    return normalizer.unnormalize(s_norm_np)

# Verify round-trip
_s0    = raw_ds["observations"][0]
_s0_rt = unnormalise(normalizer.normalize(_s0[None]))[0]
assert np.allclose(_s0, _s0_rt, atol=1e-4), \
    f"Normalizer round-trip failed: {_s0} vs {_s0_rt}"

# OOD reference: uniform random sample so all maze regions are represented
# (first-N_OOD_REF observations are contiguous trajectories that may miss corners)
ood_idx  = np.random.choice(len(raw_ds["observations"]), N_OOD_REF, replace=False)
ref_norm = normalizer.normalize(raw_ds["observations"][ood_idx])   # (N_OOD_REF, obs_dim)

# Goal position: inferred from dataset states where D4RL sparse reward fires
_rew1 = raw_ds["rewards"] == 1.0
goal_pos_raw = (raw_ds["observations"][_rew1][:, :2].mean(0)
                if _rew1.sum() > 0 else np.array([3.5, 3.5]))
print(f"Goal position (raw x,y): {goal_pos_raw}")

# ── Models ────────────────────────────────────────────────────────────────────
print("Loading models …")
fix_mask = torch.zeros((H, OBS_DIM))
fix_mask[0, :OBS_DIM] = 1.0
planner = ContinuousDiffusionSDE(
    DiT1d(OBS_DIM, emb_dim=128, d_model=256, n_heads=4, depth=2,
          timestep_emb_type="fourier"),
    fix_mask=fix_mask, noise_schedule="linear", device=DEVICE,
    predict_noise=True, ema_rate=0.9999,
    loss_weight=torch.ones((H, OBS_DIM)),
)
planner.load(f"{CKPT}/planner_ckpt_1000000.pt")
planner.eval()

critic_ckpt = torch.load(f"{CKPT}/critic_ckpt_1000000.pt", map_location=DEVICE)
critic = DVHorizonCritic(
    OBS_DIM, emb_dim=128, d_model=256, n_heads=4, depth=2, norm_type="pre",
).to(DEVICE)
critic.load_state_dict(critic_ckpt["critic"])
critic.eval()

policy = DiscreteDiffusionSDE(
    DVInvMlp(OBS_DIM, ACT_DIM, emb_dim=64, hidden_dim=256,
              timestep_emb_type="positional").to(DEVICE),
    IdentityCondition(dropout=0.0).to(DEVICE),
    x_max=+torch.ones((1, ACT_DIM), device=DEVICE),
    x_min=-torch.ones((1, ACT_DIM), device=DEVICE),
    diffusion_steps=POLICY_STEPS, device=DEVICE,
)
policy.load(f"{CKPT}/policy_ckpt_1000000.pt")
policy.eval()
print(f"Models loaded on {DEVICE}.")

# ── Rollout environment ───────────────────────────────────────────────────────
env_rollout = gym.make(ENV_NAME)
env_rollout._max_episode_steps = N_TRANSITIONS * M + 50   # prevent TimeLimit

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_imagined_state(s_norm_np, depth):
    """Chain 'depth' planner jumps from s_norm_np (waypoint 1 each step)."""
    current = torch.tensor(s_norm_np, dtype=torch.float32).to(DEVICE)
    for _ in range(depth):
        prior = torch.zeros((1, H, OBS_DIM), device=DEVICE)
        prior[0, 0, :OBS_DIM] = current
        with torch.no_grad():
            traj, _ = planner.sample(
                prior, solver="ddim", n_samples=1,
                sample_steps=PLAN_STEPS, use_ema=True, temperature=1.0)
        current = traj[0, 1, :OBS_DIM]   # waypoint 1 — mirrors MCTS child_state_index=1
    return current.cpu().numpy()   # (obs_dim,) normalised


def nn_dist(s_norm_np):
    """Nearest-neighbour L2 distance from s_norm_np to the OOD reference set."""
    diff = ref_norm - s_norm_np[None, :]   # (N_OOD_REF, obs_dim)
    return float(np.min(np.sqrt((diff ** 2).sum(axis=1))))


def rollout_plan(plan_norm, start_raw):
    """Execute a plan from start_raw via the inverse-dynamics policy.

    Args:
        plan_norm: (H, obs_dim) normalised plan (from planner.sample)
        start_raw: (obs_dim,) unnormalised start state

    Returns:
        true_return: sum of D4RL rewards over N_TRANSITIONS jump-steps
    """
    env_rollout.reset()
    env_rollout.unwrapped.set_state(
        start_raw[:2].copy(),   # qpos [x, y]
        start_raw[2:].copy())   # qvel [vx, vy]

    current_raw = start_raw.copy()
    total_rew   = 0.0

    for t_jump in range(N_TRANSITIONS):
        obs_norm  = torch.tensor(
            normalizer.normalize(current_raw[None]),
            device=DEVICE, dtype=torch.float32)   # (1, obs_dim)
        next_norm = torch.tensor(
            plan_norm[t_jump + 1][None],
            device=DEVICE, dtype=torch.float32)   # (1, obs_dim)

        obs_r  = obs_norm.clone()
        next_r = next_norm.clone()
        next_r[:, :2] -= obs_r[:, :2]   # rebase_policy=True
        obs_r[:, :2]   = 0.0

        prior = torch.zeros((1, ACT_DIM), device=DEVICE)
        with torch.no_grad():
            act, _ = policy.sample(
                prior, solver="ddpm", n_samples=1,
                sample_steps=POLICY_STEPS,
                condition_cfg=torch.cat([obs_r, next_r], dim=-1),
                w_cfg=1.0, use_ema=True, temperature=0.5)

        action = act.squeeze(0).cpu().numpy()
        for _ in range(M):
            current_raw, rew, _, _ = env_rollout.step(action)
            total_rew += rew

    return float(total_rew)


# ── Sample real start states ──────────────────────────────────────────────────
raw_obs = raw_ds["observations"]
step    = max(1, len(raw_obs) // args.n_starts)
indices = np.arange(0, min(len(raw_obs), step * args.n_starts), step)[: args.n_starts]
starts_raw  = raw_obs[indices]                      # (N, obs_dim) unnormalised
starts_norm = normalizer.normalize(starts_raw)      # (N, obs_dim) normalised

print(f"\nDepth-stratified critic calibration:")
print(f"  N starts={args.n_starts}, K_eval={args.k_eval}, "
      f"depths={args.depths}, N_transitions={N_TRANSITIONS}\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
all_rows = []   # per-plan records
t_total  = time.perf_counter()

for depth in args.depths:
    print(f"{'='*60}")
    print(f"Depth {depth}:")
    depth_critics, depth_returns = [], []

    for si in range(args.n_starts):
        s0_norm = starts_norm[si]

        # Build depth-d imagined state
        sd_norm   = build_imagined_state(s0_norm, depth)   # (obs_dim,) normalised
        sd_raw    = unnormalise(sd_norm[None])[0]          # (obs_dim,) raw
        nn_d      = nn_dist(sd_norm)
        # Distance to goal in raw maze units — needed to control for the
        # goal-proximity confound: chained planner jumps are goal-directed, so
        # depth-d states are systematically closer to the goal than depth-0 states.
        dist_goal = float(np.linalg.norm(sd_raw[:2] - goal_pos_raw))

        # Generate K_EVAL candidate plans from sd
        prior = torch.zeros((args.k_eval, H, OBS_DIM), device=DEVICE)
        prior[:, 0, :OBS_DIM] = torch.tensor(
            sd_norm, dtype=torch.float32).to(DEVICE).unsqueeze(0).expand(args.k_eval, -1)
        with torch.no_grad():
            trajs, _ = planner.sample(
                prior, solver="ddim", n_samples=args.k_eval,
                sample_steps=PLAN_STEPS, use_ema=True, temperature=1.0)
            scores = critic(trajs).squeeze(-1).cpu().numpy()   # (K_eval,)
        trajs_np = trajs.cpu().numpy()   # (K_eval, H, obs_dim)

        # Rollout each plan from sd_raw
        for k in range(args.k_eval):
            t0 = time.perf_counter()
            true_ret = rollout_plan(trajs_np[k], sd_raw)
            elapsed  = time.perf_counter() - t0

            depth_critics.append(float(scores[k]))
            depth_returns.append(true_ret)
            all_rows.append({
                "depth":        depth,
                "start_idx":    int(indices[si]),
                "plan_k":       k,
                "critic_score": round(float(scores[k]), 6),
                "true_return":  round(true_ret, 2),
                "nn_dist":      round(nn_d, 5),
                "dist_to_goal": round(dist_goal, 5),
                "wall_s":       round(elapsed, 2),
            })

        print(f"  start {si+1}/{args.n_starts}  "
              f"nn={nn_d:.4f}  d_goal={dist_goal:.4f}  "
              f"scores={[round(float(s),3) for s in scores]}  "
              f"returns={[round(r,1) for r in depth_returns[-args.k_eval:]]}")

    # Per-depth summary
    dc = np.array(depth_critics)
    dr = np.array(depth_returns)
    if dc.std() > 1e-8 and dr.std() > 1e-8:
        pr, _ = pearsonr(dc, dr)
        sr, _ = spearmanr(dc, dr)
    else:
        pr = sr = float("nan")

    _depth_goals = [r["dist_to_goal"] for r in all_rows if r["depth"] == depth]
    print(f"\n  depth={depth}  n={len(dc)}  "
          f"mean_critic={dc.mean():.4f}  mean_return={dr.mean():.2f}±{dr.std():.2f}  "
          f"pearson_r={pr:.3f}  mean_dist_goal={np.mean(_depth_goals):.4f}\n")

elapsed_total = time.perf_counter() - t_total
print(f"Total wall time: {elapsed_total:.1f}s")

# ── Save per-plan CSV ─────────────────────────────────────────────────────────
csv_path = f"{OUT_DIR}/critic_depth_calibration_{ENV_TAG}.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    writer.writeheader()
    writer.writerows(all_rows)
print(f"\n→ {csv_path}")

# ── Compute summary per depth ─────────────────────────────────────────────────
summary_rows = []
for depth in args.depths:
    subset  = [r for r in all_rows if r["depth"] == depth]
    dc      = np.array([r["critic_score"] for r in subset])
    dr      = np.array([r["true_return"]  for r in subset])
    nn_mean = float(np.mean([r["nn_dist"] for r in subset]))
    if dc.std() > 1e-8 and dr.std() > 1e-8:
        pr, _ = pearsonr(dc, dr)
        sr, _ = spearmanr(dc, dr)
    else:
        pr = sr = float("nan")
    summary_rows.append({
        "depth":             depth,
        "n_plans":           len(dc),
        "mean_critic":       round(float(dc.mean()), 5),
        "std_critic":        round(float(dc.std()),  5),
        "mean_true_return":  round(float(dr.mean()), 2),
        "std_true_return":   round(float(dr.std()),  2),
        "pearson_r":         round(pr, 4) if not np.isnan(pr) else "nan",
        "spearman_r":        round(sr, 4) if not np.isnan(sr) else "nan",
        "mean_nn_dist":      round(nn_mean, 5),
        "mean_dist_to_goal": round(float(np.mean(
            [r["dist_to_goal"] for r in subset])), 5),
    })

summary_csv = f"{OUT_DIR}/critic_depth_summary_{ENV_TAG}.csv"
with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"→ {summary_csv}")

# Print summary table
print(f"\n{'─'*100}")
print(f"{'Depth':>6}  {'n':>4}  {'mean_critic':>11}  {'mean_return':>11}  "
      f"{'std_return':>10}  {'pearson_r':>10}  {'spearman_r':>11}  "
      f"{'nn_dist':>8}  {'dist_goal':>10}")
print(f"{'─'*100}")
for r in summary_rows:
    print(f"{r['depth']:>6}  {r['n_plans']:>4}  {float(r['mean_critic']):>11.4f}  "
          f"{float(r['mean_true_return']):>11.2f}  "
          f"{float(r['std_true_return']):>10.2f}  "
          f"{str(r['pearson_r']):>10}  {str(r['spearman_r']):>11}  "
          f"{float(r['mean_nn_dist']):>8.5f}  "
          f"{float(r['mean_dist_to_goal']):>10.5f}")
print(f"{'─'*100}")
print()
print("INTERPRETATION NOTE: mean_true_return may rise with depth because the planner")
print("is goal-directed — imagined states at depth d are systematically closer to the")
print("goal (~M*d dense steps closer).  Check dist_to_goal trend and std_true_return")
print("(range restriction → near-zero std at depth 3 flags proximity, not calibration).")
print("A valid OOD-miscalibration finding requires: nn_dist↑, critic_score stays high,")
print("true_return↓ *after controlling for proximity*, and pearson_r → 0 or negative.")

# ── Plot 1: 4-panel scatter (critic score vs true return per depth) ───────────
fig, axes = plt.subplots(1, len(args.depths), figsize=(4 * len(args.depths), 4),
                          sharey=False)
if len(args.depths) == 1:
    axes = [axes]

colours = ["#2196F3", "#FF9800", "#F44336", "#4CAF50"]
for ax, depth, sr_row in zip(axes, args.depths, summary_rows):
    subset = [r for r in all_rows if r["depth"] == depth]
    xs = [r["critic_score"] for r in subset]
    ys = [r["true_return"]  for r in subset]
    c  = colours[depth % len(colours)]
    ax.scatter(xs, ys, color=c, alpha=0.7, s=55, edgecolors="white",
               linewidths=0.4, zorder=3)
    # Regression line
    if len(xs) >= 2 and np.std(xs) > 1e-8:
        m, b = np.polyfit(xs, ys, 1)
        xr = np.linspace(min(xs), max(xs), 50)
        ax.plot(xr, m * xr + b, "--", color=c, linewidth=1.2, alpha=0.8)
    ax.set_title(
        f"Depth {depth}  (d={depth} jumps)\n"
        f"r={sr_row['pearson_r']}  ρ={sr_row['spearman_r']}\n"
        f"nn={sr_row['mean_nn_dist']:.4f}  d_goal={sr_row['mean_dist_to_goal']:.3f}",
        fontsize=8)
    ax.set_xlabel("Critic score", fontsize=8)
    ax.set_ylabel("True return (dense reward)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)

fig.suptitle(
    f"Critic calibration at each tree depth\n"
    f"(N_starts={args.n_starts}, K_eval={args.k_eval}, {ENV_NAME})",
    fontsize=9)
fig.tight_layout()
p1 = f"{OUT_DIR}/plots/critic_depth_scatter_{ENV_TAG}.png"
fig.savefig(p1, dpi=150)
plt.close(fig)
print(f"→ {p1}")

# ── Plot 2: Summary metrics vs depth (2×3 grid) ──────────────────────────────
depths_vals     = [r["depth"]               for r in summary_rows]
mean_critics    = [float(r["mean_critic"])   for r in summary_rows]
mean_returns    = [float(r["mean_true_return"]) for r in summary_rows]
std_returns     = [float(r["std_true_return"])  for r in summary_rows]
pearson_vals    = [float(r["pearson_r"]) if r["pearson_r"] != "nan" else float("nan")
                   for r in summary_rows]
nn_dists        = [float(r["mean_nn_dist"])      for r in summary_rows]
dist_goal_vals  = [float(r["mean_dist_to_goal"]) for r in summary_rows]

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes = axes.ravel()

axes[0].plot(depths_vals, mean_critics, "o-", color="#1976D2")
axes[0].set_title("Mean critic score vs depth")
axes[0].set_xlabel("Depth"); axes[0].set_ylabel("Mean critic score")
axes[0].grid(True, alpha=0.3)

axes[1].plot(depths_vals, mean_returns, "o-", color="#388E3C")
axes[1].set_title("Mean true return vs depth\n(⚠ rises if imagined state nears goal)")
axes[1].set_xlabel("Depth"); axes[1].set_ylabel("True return (sum rewards)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(depths_vals, pearson_vals, "o-", color="#F44336")
axes[2].axhline(0, linestyle="--", color="gray", linewidth=0.8)
axes[2].set_title("Pearson r(critic, true_return) vs depth")
axes[2].set_xlabel("Depth"); axes[2].set_ylabel("Pearson r")
axes[2].set_ylim(-1.1, 1.1)
axes[2].grid(True, alpha=0.3)

axes[3].plot(depths_vals, nn_dists, "o-", color="#FF9800")
axes[3].set_title("OOD proxy (mean nn dist) vs depth")
axes[3].set_xlabel("Depth"); axes[3].set_ylabel("Nearest-neighbour L2")
axes[3].grid(True, alpha=0.3)

axes[4].plot(depths_vals, dist_goal_vals, "o-", color="#7B1FA2")
axes[4].set_title("Distance to goal vs depth\n(confound control: chained jumps → goal)")
axes[4].set_xlabel("Depth"); axes[4].set_ylabel("L2 dist to goal (raw maze units)")
axes[4].grid(True, alpha=0.3)

axes[5].plot(depths_vals, std_returns, "o-", color="#5D4037")
axes[5].set_title("Std(true return) vs depth\n(→ 0 = range restriction from proximity)")
axes[5].set_xlabel("Depth"); axes[5].set_ylabel("Std of true return")
axes[5].grid(True, alpha=0.3)

fig.suptitle(
    f"Depth-stratified critic calibration summary\n({ENV_NAME}, "
    f"N_starts={args.n_starts}, K_eval={args.k_eval})",
    fontsize=10)
fig.tight_layout()
p2 = f"{OUT_DIR}/plots/critic_depth_summary_{ENV_TAG}.png"
fig.savefig(p2, dpi=150)
plt.close(fig)
print(f"→ {p2}")
print("\nDone.")
