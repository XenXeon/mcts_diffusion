"""
test_start_state_generalisation.py

Tests whether the DV planner generalises to held-out DATASET start states.
This is NOT a test of generalisation to imagined intermediate states from MCTS
expansions — that is a Phase 5 question. Here we verify that conditioning the
planner on an arbitrary normalised observation from a held-out trajectory
produces a coherent, critic-valued plan that is structurally similar to the real
dataset continuation.

Protocol:
  - Reproduces the same 10% held-out split as the Phase 1 preflight (rng seed 0).
  - Samples 100 (traj_idx, offset) pairs from the usable held-out trajectories
    (rng seed 42 for offsets).
  - For each pair, conditions the DV planner on seq_obs[traj_idx, offset]
    (already z-score normalised — no double-normalisation).
  - Runs MCSS: K=50 candidates, DDIM 20 steps, critic argmax.
  - Asserts that fix_mask clamped position-0 of the plan to s_0 (tolerance 1e-4).
  - Generates a second independent draw from the same prior to obtain a
    planner self-consistency baseline.
  - Subsamples the real dataset continuation at stride=15 to obtain a (32, 4)
    ground-truth jump-step trajectory for direct comparison.
  - Scores all plans with DVHorizonCritic and computes L2 distances.

Artefacts written:
  results/phase1/per_state_results.csv  — one row per sample
  results/phase1/plans.npz             — raw plan arrays (100, 32, 4) × 3
  results/phase1/heldout_ids.json      — held-out trajectory indices

Usage:
    python scripts/test_start_state_generalisation.py
"""
import csv
import json
import math
import os
import sys

sys.path.insert(0, ".")

import d4rl  # noqa: F401 — registers D4RL envs
import gym
import numpy as np
import torch

from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import DVHorizonCritic

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

# ── Config (mirrors maze2d-umaze-v1 production config) ───────────────────────
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
ENV_NAME   = "maze2d-umaze-v1"
H          = 32     # planner_horizon (jump-steps)
M          = 15     # stride (dense env steps per jump-step)
K          = 50     # planner_num_candidates
D_MODEL    = 256
DEPTH      = 2
EMB_DIM    = 128
PLAN_STEPS = 20
OBS_DIM    = 4
CKPT = (
    "results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer"
    "_d2_width256_separate_dpTrue/maze2d-umaze-v1"
)
MAZE_XY_BOUNDS = (0.0, 5.0)   # generous bounding box for maze2d-umaze physical xy
REQUIRED_DENSE = H * M         # 480 dense steps needed to extract one full real plan

# ── Dataset and normalizer ────────────────────────────────────────────────────
print("Loading dataset …")
env = gym.make(ENV_NAME)
dataset = DV_D4RLMaze2DSeqDataset(
    env.get_dataset(), horizon=H, stride=M,
    learn_policy=False, center_mapping=False,
    discount=1.0, continous_reward_at_done=True, reward_tune="iql")

normalizer = dataset.get_normalizer()
# seq_obs: (n_traj, max_padded_len, 4) — already z-score normalised at construction
obs = dataset.seq_obs
true_lengths = np.load("results/phase1/path_lengths.npy")

# ── Held-out split ────────────────────────────────────────────────────────────
# rng(0) reproduces the same split as the Phase 1 preflight sizing script.
split_rng  = np.random.default_rng(0)
offset_rng = np.random.default_rng(42)

n_traj = len(true_lengths)
ids    = split_rng.permutation(n_traj)

heldout_ids        = ids[-int(n_traj * 0.10):]
usable_mask        = true_lengths[heldout_ids] >= REQUIRED_DENSE
usable_heldout_ids = heldout_ids[usable_mask]

n_usable         = len(usable_heldout_ids)
samples_per_traj = max(2, math.ceil(100 / n_usable))

print(f"Total trajectories : {n_traj}")
print(f"Held-out pool      : {len(heldout_ids)} ({len(heldout_ids)/n_traj:.1%})")
print(f"Usable (≥{REQUIRED_DENSE} steps): {n_usable}")
print(f"Samples per traj   : {samples_per_traj}  (guarantees ≥100 total)")

# ── Sample 100 (traj_idx, offset) pairs ──────────────────────────────────────
samples: list[tuple[int, int]] = []
for traj_idx in usable_heldout_ids:
    max_offset = int(true_lengths[traj_idx]) - REQUIRED_DENSE
    offsets = offset_rng.integers(0, max_offset + 1, size=samples_per_traj)
    for offset in offsets:
        samples.append((int(traj_idx), int(offset)))
        if len(samples) == 100:
            break
    if len(samples) == 100:
        break

print(f"Successfully sampled {len(samples)} valid start states.")

# ── Planner initialisation ────────────────────────────────────────────────────
# fix_mask: clamp position-0 of the trajectory to the start state (obs_dim=4 dims).
# Mirrors pipelines/veteran_d4rl_maze2d.py lines 145–146.
fix_mask = torch.zeros((H, OBS_DIM))
fix_mask[0, :OBS_DIM] = 1.0

planner = ContinuousDiffusionSDE(
    DiT1d(OBS_DIM, emb_dim=EMB_DIM, d_model=D_MODEL,
          n_heads=D_MODEL // 64, depth=DEPTH, timestep_emb_type="fourier"),
    fix_mask=fix_mask, noise_schedule="linear", device=DEVICE)
planner.load(f"{CKPT}/planner_ckpt_1000000.pt")
planner.eval()

# ── Critic initialisation ─────────────────────────────────────────────────────
critic = DVHorizonCritic(
    OBS_DIM, emb_dim=EMB_DIM, d_model=D_MODEL,
    n_heads=D_MODEL // 64, depth=DEPTH, norm_type="pre").to(DEVICE)
critic.load_state_dict(
    torch.load(f"{CKPT}/critic_ckpt_1000000.pt", map_location=DEVICE)["critic"])
critic.eval()

print(f"Models loaded on {DEVICE}.\n")

# ── Verification loop ─────────────────────────────────────────────────────────
rows: list[dict] = []
imagined_plans_arr = np.zeros((100, H, OBS_DIM), dtype=np.float32)
real_plans_arr     = np.zeros((100, H, OBS_DIM), dtype=np.float32)
imagined_plans_2nd = np.zeros((100, H, OBS_DIM), dtype=np.float32)

for idx, (traj_idx, offset) in enumerate(samples):

    # A. Start state — already normalised; do NOT call normalizer.normalize again.
    s_0_np = obs[traj_idx, offset]                                     # (4,) numpy
    s_0    = torch.tensor(s_0_np, device=DEVICE, dtype=torch.float32)  # (4,) tensor

    # B. Real dataset continuation subsampled at stride M → exactly (H, 4).
    #    REQUIRED_DENSE = H * M = 480, so [::M] yields 480//15 = 32 rows.
    real_dense = obs[traj_idx, offset : offset + REQUIRED_DENSE]       # (480, 4)
    real_jump  = real_dense[::M]                                        # (32, 4)

    # C. Build the MCSS prior: zeros everywhere, s_0 clamped at position 0.
    prior = torch.zeros((K, H, OBS_DIM), device=DEVICE)
    prior[:, 0, :] = s_0                                               # broadcast → (K, 4)

    # First imagined plan: K=50 candidates, DDIM 20 steps, critic argmax.
    with torch.no_grad():
        traj1, _ = planner.sample(
            prior, solver="ddim", n_samples=K,
            sample_steps=PLAN_STEPS, use_ema=True, temperature=1.0)
        val1  = critic(traj1).squeeze(-1)          # (K,)
        best1 = traj1[torch.argmax(val1)]           # (H, 4)

    # D. Verify fix_mask clamped position-0 of the best plan to s_0.
    clamp_err = (best1[0] - s_0).abs().max().item()
    assert clamp_err < 1e-4, (
        f"fix_mask assertion failed at sample {idx} "
        f"(traj={traj_idx}, off={offset}): err={clamp_err:.2e}. "
        "Planner did not clamp position-0 to s_0.")

    # E. Second independent draw — same prior, different internal torch RNG state.
    with torch.no_grad():
        traj2, _ = planner.sample(
            prior, solver="ddim", n_samples=K,
            sample_steps=PLAN_STEPS, use_ema=True, temperature=1.0)
        val2  = critic(traj2).squeeze(-1)
        best2 = traj2[torch.argmax(val2)]           # (H, 4)

    # F. Critic scores (normalised space, range ≈ [-1, 1]).
    real_t = torch.tensor(real_jump, device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        score_gen  = critic(best1.unsqueeze(0)).item()
        score_real = critic(real_t.unsqueeze(0)).item()

    # G. L2 metrics (in normalised observation space).
    best1_np = best1.cpu().numpy()   # (32, 4)
    best2_np = best2.cpu().numpy()   # (32, 4)
    mean_l2  = float(np.linalg.norm(best1_np - real_jump,  axis=-1).mean())
    self_l2  = float(np.linalg.norm(best1_np - best2_np,   axis=-1).mean())

    # H. in_bounds: unnormalise position dims and check against maze bounding box.
    imagined_unnorm = normalizer.unnormalize(best1_np)   # (32, 4)
    lo, hi = MAZE_XY_BOUNDS
    in_bounds = bool(np.all(
        (imagined_unnorm[:, :2] >= lo) & (imagined_unnorm[:, :2] <= hi)))

    # I. Accumulate.
    rows.append(dict(
        traj_idx=traj_idx,
        offset=offset,
        score_gen=round(score_gen, 6),
        score_real=round(score_real, 6),
        mean_l2=round(mean_l2, 6),
        planner_self_l2=round(self_l2, 6),
        in_bounds=int(in_bounds),
    ))
    imagined_plans_arr[idx] = best1_np
    real_plans_arr[idx]     = real_jump
    imagined_plans_2nd[idx] = best2_np

    if idx % 10 == 0:
        print(f"  [{idx:3d}/100] traj={traj_idx:4d} off={offset:4d}  "
              f"s_gen={score_gen:+.4f}  s_real={score_real:+.4f}  "
              f"l2={mean_l2:.4f}  self_l2={self_l2:.4f}  "
              f"in_bounds={bool(in_bounds)}")

# ── Final aggregate stats ─────────────────────────────────────────────────────
scores_gen   = [r["score_gen"]        for r in rows]
scores_real  = [r["score_real"]       for r in rows]
l2s          = [r["mean_l2"]          for r in rows]
self_l2s     = [r["planner_self_l2"]  for r in rows]
in_b         = [r["in_bounds"]        for r in rows]

print(f"\n{'─'*60}")
print(f"Mean Generated Score   : {np.mean(scores_gen):.4f}  ± {np.std(scores_gen):.4f}")
print(f"Mean Real Score        : {np.mean(scores_real):.4f}  ± {np.std(scores_real):.4f}")
print(f"Mean L2 (gen vs real)  : {np.mean(l2s):.4f}  ± {np.std(l2s):.4f}")
print(f"Mean L2 (self-consist) : {np.mean(self_l2s):.4f}  ± {np.std(self_l2s):.4f}")
print(f"In-bounds fraction     : {np.mean(in_b):.1%}")
print(f"{'─'*60}\n")

# ── Save artefacts ────────────────────────────────────────────────────────────
os.makedirs("results/phase1", exist_ok=True)

with open("results/phase1/per_state_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print("→ results/phase1/per_state_results.csv")

# plans.npz: required arrays (100, 32, 4) × 3, plus normalizer params for plots.
np.savez(
    "results/phase1/plans.npz",
    imagined_plans=imagined_plans_arr,
    real_plans=real_plans_arr,
    imagined_plans_2nd_draw=imagined_plans_2nd,
    norm_mean=normalizer.mean,    # (4,) — stored for unnormalisation in plots script
    norm_std=normalizer.std,      # (4,)
)
print("→ results/phase1/plans.npz")

with open("results/phase1/heldout_ids.json", "w") as f:
    json.dump(heldout_ids.tolist(), f)
print("→ results/phase1/heldout_ids.json")

print("\nDone.")
