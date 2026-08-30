"""scripts/phase5_plan_diversity.py

Diagnostic: measures K-plan spatial diversity at each waypoint index.

For N real start states, generates K candidate plans and computes the mean
pairwise L2 distance among the K plan positions (x,y) at waypoints
{1, 2, 4, 8, 16, 31}.

This directly quantifies the core MCTS branching problem: plans are nearly
identical at waypoint 1 (~one-step jitter) and diverge substantially only at
later waypoints where genuine route alternatives appear.  If the tree branches
on waypoint 1, all K children represent the same next-step with minor noise.

Outputs:
  results/phase5/plan_diversity.csv
  results/phase5/plots/plan_diversity.png

Usage:
    python scripts/phase5_plan_diversity.py
    python scripts/phase5_plan_diversity.py --n-starts 50 --K 50
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, ".")

import d4rl  # noqa: F401
import gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import DiT1d
from pipelines.utils import set_seed

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n-starts", type=int, default=30,
                    help="Number of real start states to sample")
parser.add_argument("--K", type=int, default=50,
                    help="Candidate plans per start state")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--env", default="maze2d-umaze-v1",
                    choices=["maze2d-umaze-v1", "maze2d-medium-v1", "maze2d-large-v1"])
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
ENV_NAME   = args.env
ENV_TAG    = ENV_NAME.replace("maze2d-", "").replace("-v1", "")
H          = 32
M          = 15
OBS_DIM    = 4
PLAN_STEPS = 20
CKPT = (
    "results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer"
    f"_d2_width256_separate_dpTrue/{ENV_NAME}"
)
WAYPOINTS = [1, 2, 4, 8, 16, 31]
OUT_DIR   = "results/phase5"
os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)

set_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Dataset + normalizer ──────────────────────────────────────────────────────
print("Loading dataset …")
env = gym.make(ENV_NAME)
raw_dataset = env.get_dataset()
dataset = DV_D4RLMaze2DSeqDataset(
    raw_dataset, horizon=H, stride=M,
    learn_policy=False, center_mapping=False,
    discount=1.0, continous_reward_at_done=True, reward_tune="iql")
normalizer = dataset.get_normalizer()

# ── Planner ───────────────────────────────────────────────────────────────────
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
print(f"Planner loaded on {DEVICE}.")

# ── Sample N real start states ────────────────────────────────────────────────
raw_obs = raw_dataset["observations"]   # (N_total, obs_dim) unnormalised
step = max(1, len(raw_obs) // args.n_starts)
indices = np.arange(0, min(len(raw_obs), step * args.n_starts), step)[: args.n_starts]
starts_raw  = raw_obs[indices]                             # (N, obs_dim)
starts_norm = normalizer.normalize(starts_raw)             # (N, obs_dim)

print(f"\nMeasuring plan diversity:")
print(f"  N starts = {len(starts_norm)}, K plans each, "
      f"waypoints = {WAYPOINTS}\n")

# ── Measure mean pairwise L2 spread at each waypoint ─────────────────────────
spread = {wp: [] for wp in WAYPOINTS}    # wp → list of per-start mean-pairwise-L2

for i, s0_norm in enumerate(starts_norm):
    s0_t = torch.tensor(s0_norm, dtype=torch.float32)
    prior = torch.zeros((args.K, H, OBS_DIM), device=DEVICE)
    prior[:, 0, :OBS_DIM] = s0_t.to(DEVICE).unsqueeze(0).expand(args.K, -1)

    with torch.no_grad():
        trajs, _ = planner.sample(
            prior, solver="ddim", n_samples=args.K,
            sample_steps=PLAN_STEPS, use_ema=True, temperature=1.0)

    trajs_np = trajs.cpu().numpy()   # (K, H, obs_dim)

    for wp in WAYPOINTS:
        pts = trajs_np[:, wp, :2]   # (K, 2) — x,y normalised
        # Mean over all K*(K-1)/2 unique pairs
        diffs = pts[:, None, :] - pts[None, :, :]   # (K, K, 2)
        dists = np.sqrt((diffs ** 2).sum(-1))        # (K, K)
        upper = dists[np.triu_indices(args.K, k=1)]  # upper triangle, no diagonal
        spread[wp].append(float(upper.mean()) if len(upper) > 0 else 0.0)

    if i % 5 == 0 or i == len(starts_norm) - 1:
        row = "  ".join(f"wp{wp}={np.mean(spread[wp]):.3f}" for wp in WAYPOINTS)
        print(f"  [{i+1:2d}/{len(starts_norm)}]  {row}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"{'Waypoint':>10}  {'Dense steps':>12}  {'Mean pairwise L2':>18}  "
      f"{'Std':>8}  {'vs wp1':>8}")
print(f"{'─'*70}")

rows_csv = []
wp1_mean = float(np.mean(spread[1]))
for wp in WAYPOINTS:
    vals     = np.array(spread[wp])
    dense    = wp * M
    ratio    = vals.mean() / (wp1_mean + 1e-12)
    print(f"{wp:>10}  {dense:>12}  {vals.mean():>18.4f}  "
          f"{vals.std():>8.4f}  {ratio:>7.1f}×")
    rows_csv.append({
        "waypoint":         wp,
        "dense_steps":      dense,
        "mean_pairwise_l2": round(float(vals.mean()), 6),
        "std":              round(float(vals.std()),  6),
        "ratio_vs_wp1":     round(ratio, 3),
    })
print(f"{'─'*70}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = f"{OUT_DIR}/plan_diversity_{ENV_TAG}.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows_csv[0].keys())
    writer.writeheader()
    writer.writerows(rows_csv)
print(f"\n→ {csv_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────
means        = [float(np.mean(spread[wp])) for wp in WAYPOINTS]
stds         = [float(np.std(spread[wp]))  for wp in WAYPOINTS]
dense_steps  = [wp * M for wp in WAYPOINTS]

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(dense_steps, means, yerr=stds, marker="o", capsize=4,
            color="#1976D2", linewidth=1.8, markersize=7, label="mean ± 1 std")

# Annotate current MCTS branching point (waypoint 1)
idx1 = WAYPOINTS.index(1)
ax.annotate(
    f"Current MCTS\nbranch (wp=1)\n{means[idx1]:.4f}",
    xy=(dense_steps[idx1], means[idx1]),
    xytext=(dense_steps[idx1] + 25, means[idx1] + 0.12),
    fontsize=8, color="#D32F2F",
    arrowprops=dict(arrowstyle="->", color="#D32F2F"),
)
# Annotate last waypoint
ax.annotate(
    f"wp={WAYPOINTS[-1]}\n{means[-1]:.4f}",
    xy=(dense_steps[-1], means[-1]),
    xytext=(dense_steps[-1] - 55, means[-1] + 0.12),
    fontsize=8, color="#388E3C",
    arrowprops=dict(arrowstyle="->", color="#388E3C"),
)

ax.set_xlabel(f"Waypoint index × M={M}  (dense env steps per waypoint)")
ax.set_ylabel("Mean pairwise L2 (normalised obs space)")
ax.set_title(
    f"K-plan spatial diversity vs waypoint index\n"
    f"(K={args.K}, N={args.n_starts} real starts, {ENV_NAME})")
ax.set_xticks(dense_steps)
ax.set_xticklabels(
    [f"wp{wp}\n({wp*M})" for wp in WAYPOINTS], fontsize=8)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()

plot_path = f"{OUT_DIR}/plots/plan_diversity_{ENV_TAG}.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"→ {plot_path}")
print("\nDone.")
