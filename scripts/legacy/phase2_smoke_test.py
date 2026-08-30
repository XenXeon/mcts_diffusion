"""scripts/phase2_smoke_test.py

End-to-end smoke test for the Phase 2 expansion primitive.

Loads the real maze2d-umaze-v1 checkpoint, picks the first held-out start state
from Phase 1 (results/phase1/per_state_results.csv), and runs one expand() call.
Asserts the fix_mask invariant, prints a result summary, and exits non-zero on
any failure.

Run inside Docker (or any env with torch + d4rl + cleandiffuser):
    python scripts/phase2_smoke_test.py

Dependencies: torch, d4rl, gym, cleandiffuser, mcts
"""
import csv
import os
import sys
import time

sys.path.insert(0, ".")

import d4rl  # noqa: F401 — registers D4RL envs
import gym
import torch

from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import DVHorizonCritic
from mcts.expansion import ExpansionConfig, PlannerExpansion

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
ENV_NAME = "maze2d-umaze-v1"
H = 32
M = 15
OBS_DIM = 4
CKPT = (
    "results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer"
    "_d2_width256_separate_dpTrue/maze2d-umaze-v1"
)

REQUIRED = [
    f"{CKPT}/planner_ckpt_1000000.pt",
    f"{CKPT}/critic_ckpt_1000000.pt",
    "results/phase1/per_state_results.csv",
]
for p in REQUIRED:
    if not os.path.exists(p):
        print(f"[FAIL] Required file not found: {p}")
        sys.exit(1)

# ── Load dataset (for normalizer and held-out start state) ────────────────────
print("Loading dataset …")
env_data = gym.make(ENV_NAME)
dataset = DV_D4RLMaze2DSeqDataset(
    env_data.get_dataset(), horizon=H, stride=M,
    learn_policy=False, center_mapping=False,
    discount=1.0, continous_reward_at_done=True, reward_tune="iql",
)
normalizer = dataset.get_normalizer()

# Pick first row from Phase 1 CSV → (traj_idx, offset) → s_0 already normalised
rows = list(csv.DictReader(open("results/phase1/per_state_results.csv")))
row = rows[0]
traj_idx, offset = int(row["traj_idx"]), int(row["offset"])
s_norm = torch.tensor(
    dataset.seq_obs[traj_idx, offset], dtype=torch.float32, device=DEVICE
)
print(f"Start state from Phase 1 row 0: traj={traj_idx} offset={offset}")
print(f"  s_norm (normalised) : {s_norm.cpu().numpy()}")
print(f"  s_raw  (unnormed)   : {normalizer.unnormalize(s_norm.unsqueeze(0).cpu().numpy())[0]}")

# ── Build expansion primitive ─────────────────────────────────────────────────
print("\nBuilding planner and critic …")
nn_diff = DiT1d(
    OBS_DIM, emb_dim=128, d_model=256, n_heads=4, depth=2,
    timestep_emb_type="fourier",
)
fix_mask = torch.zeros((H, OBS_DIM))
fix_mask[0, :OBS_DIM] = 1.0

planner = ContinuousDiffusionSDE(
    nn_diff, nn_condition=None,
    fix_mask=fix_mask,
    loss_weight=torch.ones((H, OBS_DIM)),
    ema_rate=0.9999, device=DEVICE,
    predict_noise=True, noise_schedule="linear",
)
planner.load(f"{CKPT}/planner_ckpt_1000000.pt")
planner.eval()

critic = DVHorizonCritic(
    OBS_DIM, emb_dim=128, d_model=256, n_heads=4, depth=2, norm_type="pre",
).to(DEVICE)
critic_ckpt = torch.load(f"{CKPT}/critic_ckpt_1000000.pt", map_location=DEVICE)
critic.load_state_dict(critic_ckpt["critic"])
critic.eval()

cfg = ExpansionConfig(
    K=50,
    horizon=H,
    obs_dim=OBS_DIM,
    planner_dim=OBS_DIM,
    solver="ddim",
    sample_steps=20,
    temperature=1.0,
    use_ema=True,
    device=DEVICE,
)
expansion = PlannerExpansion(planner, critic, cfg)
print(f"  Planner + critic loaded on {DEVICE}.")

# ── Run one expansion ─────────────────────────────────────────────────────────
print("\nRunning expand() …")
t0 = time.time()
result = expansion.expand(s_norm)
elapsed = time.time() - t0

print(f"  Wall time : {elapsed:.2f}s")
print(f"  trajs shape : {tuple(result.trajs.shape)}")
print(f"  scores shape: {tuple(result.scores.shape)}")
print(f"  best_score  : {result.best_score:.4f}")
print(f"  scores (top-5): {result.scores[:5].cpu().numpy()}")

# ── Mandatory assertions ──────────────────────────────────────────────────────
print("\nRunning assertions …")
errors = []

# 1. fix_mask: position-0 must equal s_norm within 1e-4
diff = (result.trajs[:, 0, :OBS_DIM] - s_norm).abs().max().item()
if diff >= 1e-4:
    errors.append(f"fix_mask VIOLATED: max-abs = {diff:.2e} >= 1e-4")
else:
    print(f"  [PASS] fix_mask: max-abs = {diff:.2e} < 1e-4")

# 2. scores descending
scores_cpu = result.scores.cpu()
desc_ok = all(scores_cpu[i] >= scores_cpu[i + 1] for i in range(len(scores_cpu) - 1))
if not desc_ok:
    errors.append("scores NOT descending")
else:
    print(f"  [PASS] scores descending")

# 3. shapes
if result.trajs.shape != (cfg.K, H, OBS_DIM):
    errors.append(f"trajs shape wrong: {tuple(result.trajs.shape)}")
else:
    print(f"  [PASS] trajs shape: {tuple(result.trajs.shape)}")
if result.scores.shape != (cfg.K,):
    errors.append(f"scores shape wrong: {tuple(result.scores.shape)}")
else:
    print(f"  [PASS] scores shape: {tuple(result.scores.shape)}")

# 4. second call produces different trajectories (stochastic)
result2 = expansion.expand(s_norm)
l2 = (result.trajs[:, 1:, :] - result2.trajs[:, 1:, :]).norm(dim=-1).mean().item()
if l2 < 0.01:
    errors.append(f"planner near-deterministic: planner_self_l2 = {l2:.4f}")
else:
    print(f"  [PASS] planner stochastic: planner_self_l2 = {l2:.4f}")

# 5. Compare scores[0] against Phase 1 score_gen for this row
phase1_score_gen = float(row["score_gen"])
print(f"\n  Phase 1 recorded score_gen for this state: {phase1_score_gen:.6f}")
print(f"  expand() best_score (may differ — different RNG state): {result.best_score:.6f}")

# ── Result ────────────────────────────────────────────────────────────────────
print()
if errors:
    for e in errors:
        print(f"[FAIL] {e}")
    sys.exit(1)
else:
    print("[PASS] All smoke test assertions passed.")
