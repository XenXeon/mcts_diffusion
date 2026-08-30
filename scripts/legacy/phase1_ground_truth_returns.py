"""
phase1_ground_truth_returns.py

⚠ INTERPRETATION WARNING (recorded so the original mistake is not repeated):
this script's headline number, Pearson r(score_gen, true_return_gen) = −0.116,
was initially read as "the critic cannot rank plans". That reading was WRONG.
Re-analysis (notes/writeup_phases_0_to_4.md §3d, §9.1) showed: n=30 with a 95% CI
of [−0.457, +0.255] (spans zero), range-restricted (all 30 plans are planner
outputs), and OPEN-LOOP (no re-planning — not the deployment setting). The critic
actually reads plans near-perfectly: r(critic_score, plan reach-time) = −0.995.
Never treat a single small-n open-loop correlation from this script as a verdict.

Validates DV critic calibration by comparing critic predictions against true
environment returns obtained without re-running the planner.

For 30 samples (every 3rd of the 100 from per_state_results.csv):

  Real continuation true return
    Read directly from dataset.seq_rew over the 31-transition window
    (31 × M=15 = 465 dense steps). seq_rew stores raw_D4RL_reward − 1;
    adding 1 recovers the sparse +1 goal signal.

  Imagined plan true return
    Reset maze2d env to s_0 via set_state(qpos, qvel). For each of the 31
    jump-step transitions, run the inverse-dynamics policy on
    (current_obs_norm, imagined_plan[t+1]) with the same rebase used at
    inference, hold the resulting action for M=15 dense env steps, accumulate
    raw D4RL rewards. The imagined plan is loaded from plans.npz — the planner
    is NOT re-run.

Saves:
  results/phase1/ground_truth_returns.csv
  results/phase1/plots/plot5_critic_vs_true_return.png
  results/phase1/plots/plot6_four_quadrant_breakdown.png

Usage:
    python scripts/phase1_ground_truth_returns.py
"""
import csv
import os
import sys

sys.path.insert(0, ".")

import d4rl  # noqa: F401 — registers D4RL envs
import gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DVInvMlp

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"
ENV_NAME     = "maze2d-umaze-v1"
H            = 32
M            = 15
N_TRANSITIONS = H - 1         # 31 jump-step transitions per rollout
N_DENSE       = N_TRANSITIONS * M  # 465 dense env steps
OBS_DIM      = 4
ACT_DIM      = 2
POLICY_STEPS = 10
CKPT = (
    "results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer"
    "_d2_width256_separate_dpTrue/maze2d-umaze-v1"
)
PLOTS_DIR = "results/phase1/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load Phase 1 artefacts ────────────────────────────────────────────────────
npz_path = "results/phase1/plans.npz"
csv_path = "results/phase1/per_state_results.csv"

for p in (npz_path, csv_path):
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found. Run scripts/test_start_state_generalisation.py first.")

npz = np.load(npz_path)
imagined_plans = npz["imagined_plans"]   # (100, 32, 4) normalised
norm_mean      = npz["norm_mean"]        # (4,)
norm_std       = npz["norm_std"]         # (4,)

with open(csv_path, newline="") as f:
    all_rows = list(csv.DictReader(f))

# ── Dataset (for seq_rew and normalizer) ──────────────────────────────────────
print("Loading dataset …")
env_data = gym.make(ENV_NAME)
dataset  = DV_D4RLMaze2DSeqDataset(
    env_data.get_dataset(), horizon=H, stride=M,
    learn_policy=False, center_mapping=False,
    discount=1.0, continous_reward_at_done=True, reward_tune="iql")

normalizer = dataset.get_normalizer()
seq_rew    = dataset.seq_rew   # (n_traj, 1265, 1) — stores raw_D4RL_reward − 1

# ── Inverse-dynamics policy ───────────────────────────────────────────────────
policy = DiscreteDiffusionSDE(
    DVInvMlp(OBS_DIM, ACT_DIM, emb_dim=64, hidden_dim=256,
             timestep_emb_type="positional").to(DEVICE),
    IdentityCondition(dropout=0.0).to(DEVICE),
    x_max=+torch.ones((1, ACT_DIM), device=DEVICE),
    x_min=-torch.ones((1, ACT_DIM), device=DEVICE),
    diffusion_steps=POLICY_STEPS, device=DEVICE)
policy.load(f"{CKPT}/policy_ckpt_1000000.pt")
policy.eval()
print(f"Policy loaded on {DEVICE}.")

# ── Rollout environment ───────────────────────────────────────────────────────
# Extend TimeLimit beyond N_DENSE so it never fires during our 465-step rollout.
env_rollout = gym.make(ENV_NAME)
if hasattr(env_rollout, "_max_episode_steps"):
    env_rollout._max_episode_steps = N_DENSE + 50   # 515

# ── Pick 30 samples: every 3rd (indices 0, 3, 6, …, 87) ─────────────────────
sample_indices = list(range(0, 90, 3))
assert len(sample_indices) == 30
print(f"\nEvaluating ground-truth returns for {len(sample_indices)} samples …\n")

rows_out = []

for si, sample_i in enumerate(sample_indices):

    row       = all_rows[sample_i]
    traj_idx  = int(row["traj_idx"])
    offset    = int(row["offset"])
    score_gen  = float(row["score_gen"])
    score_real = float(row["score_real"])

    # ── A. Real continuation true return ─────────────────────────────────────
    # seq_rew stores raw_D4RL_reward − 1 (IQL shift applied at construction).
    # +1 recovers the sparse 0/1 goal reward. Usable trajectories guarantee
    # offset + N_DENSE ≤ path_length, so these are genuine steps, not padding.
    raw_rews_real    = seq_rew[traj_idx, offset : offset + N_DENSE, 0] + 1.0
    true_return_real = float(raw_rews_real.sum())

    # ── B. Imagined plan true return via inverse-dynamics rollout ─────────────
    imagined_plan_np = imagined_plans[sample_i]   # (32, 4) normalised

    # s_0 is stored as imagined_plan[0] — fix_mask guarantees it equals the
    # conditioned start state. Unnormalise to get raw qpos/qvel.
    s_0_raw = imagined_plan_np[0] * norm_std + norm_mean   # (4,) unnormalised

    env_rollout.reset()
    env_rollout.unwrapped.set_state(
        s_0_raw[:2].copy(),   # qpos: [x, y]
        s_0_raw[2:].copy())   # qvel: [vx, vy]

    current_obs_raw  = s_0_raw.copy()
    true_return_gen  = 0.0

    for t_jump in range(N_TRANSITIONS):
        # Normalise current raw observation.
        current_obs_norm = torch.tensor(
            normalizer.normalize(current_obs_raw[None]),
            device=DEVICE, dtype=torch.float32)              # (1, 4)

        # Next planned waypoint (already normalised — loaded from plans.npz).
        next_obs_norm = torch.tensor(
            imagined_plan_np[t_jump + 1][None],
            device=DEVICE, dtype=torch.float32)              # (1, 4)

        # Rebase: mirrors the training pre-processing and production inference
        # (pipelines/veteran_d4rl_maze2d.py lines 418–419).
        obs_r  = current_obs_norm.clone()
        next_r = next_obs_norm.clone()
        next_r[:, :2] -= obs_r[:, :2]
        obs_r[:, :2]   = 0.0

        policy_prior = torch.zeros((1, ACT_DIM), device=DEVICE)
        with torch.no_grad():
            act, _ = policy.sample(
                policy_prior, solver="ddpm", n_samples=1,
                sample_steps=POLICY_STEPS,
                condition_cfg=torch.cat([obs_r, next_r], dim=-1),
                w_cfg=1.0, use_ema=True, temperature=0.5)

        action = act.squeeze(0).cpu().numpy()   # (2,)

        # Hold action for M=15 dense steps. maze2d done fires only from
        # TimeLimit (which we extended above) — no break needed.
        for _ in range(M):
            current_obs_raw, rew, _, _ = env_rollout.step(action)
            true_return_gen += rew

    rows_out.append(dict(
        traj_idx=traj_idx,
        offset=offset,
        score_gen=round(score_gen, 6),
        score_real=round(score_real, 6),
        true_return_gen=round(true_return_gen, 2),
        true_return_real=round(true_return_real, 2),
    ))

    if si % 5 == 0:
        print(f"  [{si+1:2d}/30] traj={traj_idx:4d} off={offset:4d}  "
              f"s_gen={score_gen:+.4f} → true_gen={true_return_gen:6.1f}  |  "
              f"s_real={score_real:+.4f} → true_real={true_return_real:6.1f}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
out_csv = "results/phase1/ground_truth_returns.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
    writer.writeheader()
    writer.writerows(rows_out)
print(f"\n→ {out_csv}")

# ── Extract arrays ────────────────────────────────────────────────────────────
score_gen_arr  = np.array([r["score_gen"]         for r in rows_out])
score_real_arr = np.array([r["score_real"]        for r in rows_out])
tr_gen_arr     = np.array([r["true_return_gen"]   for r in rows_out])
tr_real_arr    = np.array([r["true_return_real"]  for r in rows_out])

def _safe_corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

corr_gen  = _safe_corr(score_gen_arr,  tr_gen_arr)
corr_real = _safe_corr(score_real_arr, tr_real_arr)

print(f"\n{'─'*60}")
print(f"Mean true return — imagined : {tr_gen_arr.mean():.2f}  ± {tr_gen_arr.std():.2f}")
print(f"Mean true return — real     : {tr_real_arr.mean():.2f}  ± {tr_real_arr.std():.2f}")
print(f"Corr(score_gen,  true_gen)  : {corr_gen:.4f}")
print(f"Corr(score_real, true_real) : {corr_real:.4f}")
print(f"{'─'*60}\n")

# ── Plot 5: Critic score vs true return scatter ───────────────────────────────
# Normalise both axes to [0,1] so the y=x perfect-calibration line is meaningful.
all_scores = np.concatenate([score_gen_arr, score_real_arr])
all_tr     = np.concatenate([tr_gen_arr, tr_real_arr])
s_lo, s_hi = all_scores.min(), all_scores.max()
t_lo, t_hi = all_tr.min(),     all_tr.max()

def _n01(x, lo, hi):
    return (x - lo) / max(hi - lo, 1e-8)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(_n01(score_gen_arr,  s_lo, s_hi),
           _n01(tr_gen_arr,     t_lo, t_hi),
           color="#2196F3", alpha=0.75, s=55, zorder=3,
           label=f"Generated   (r = {corr_gen:.3f})")
ax.scatter(_n01(score_real_arr, s_lo, s_hi),
           _n01(tr_real_arr,    t_lo, t_hi),
           color="#FF9800", alpha=0.75, s=55, zorder=3,
           label=f"Real cont.  (r = {'n/a — zero var' if np.isnan(corr_real) else f'{corr_real:.3f}'})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1.0,
        label="y = x  (perfect calibration)", zorder=2)

ax.set_xlabel(f"Critic score (norm. to [0,1];  raw range [{s_lo:.3f}, {s_hi:.3f}])")
ax.set_ylabel(f"True return (norm. to [0,1];  raw range [{t_lo:.0f}, {t_hi:.0f}])")
ax.set_title("Critic prediction vs true env return  (n=30)")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_aspect("equal")
ax.legend(framealpha=0.9, fontsize=9)
fig.tight_layout()
path = f"{PLOTS_DIR}/plot5_critic_vs_true_return.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"→ {path}")

# ── Plot 6: Four-quadrant breakdown ──────────────────────────────────────────
# x = true_return_gen − true_return_real  (positive → imagined outperforms real)
# y = score_gen − score_real              (positive → critic prefers generated)
delta_tr = tr_gen_arr    - tr_real_arr
delta_sc = score_gen_arr - score_real_arr

colors = []
for dt, ds in zip(delta_tr, delta_sc):
    if   dt >= 0 and ds >= 0: colors.append("#4CAF50")   # Q1: gen better, critic correct
    elif dt <  0 and ds >= 0: colors.append("#F44336")   # Q2: real better, critic wrong
    elif dt <  0 and ds <  0: colors.append("#4CAF50")   # Q3: real better, critic correct
    else:                     colors.append("#F44336")   # Q4: gen better, critic wrong

q1 = sum(1 for dt, ds in zip(delta_tr, delta_sc) if dt >= 0 and ds >= 0)
q2 = sum(1 for dt, ds in zip(delta_tr, delta_sc) if dt <  0 and ds >= 0)
q3 = sum(1 for dt, ds in zip(delta_tr, delta_sc) if dt <  0 and ds <  0)
q4 = sum(1 for dt, ds in zip(delta_tr, delta_sc) if dt >= 0 and ds <  0)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(delta_tr, delta_sc, c=colors, alpha=0.75, s=60, zorder=3,
           edgecolors="white", linewidths=0.5)
ax.axhline(0, color="black", linewidth=0.9, zorder=2)
ax.axvline(0, color="black", linewidth=0.9, zorder=2)

ax.set_xlabel("True return gap  (imagined − real, dense steps at goal)")
ax.set_ylabel("Critic score gap  (imagined − real)")
ax.set_title(
    f"Four-quadrant critic calibration  (n=30)\n"
    f"Q1={q1}  Q2={q2}  Q3={q3}  Q4={q4}  "
    f"— correct: {q1+q3}/30  wrong: {q2+q4}/30")

kw = dict(transform=ax.transAxes, fontsize=8)
ax.text(0.97, 0.97, "Q1\ngen better\ncritic agrees",
        ha="right", va="top",    color="#2E7D32", **kw)
ax.text(0.03, 0.97, "Q2\nreal better\ncritic wrong",
        ha="left",  va="top",    color="#B71C1C", **kw)
ax.text(0.03, 0.03, "Q3\nreal better\ncritic agrees",
        ha="left",  va="bottom", color="#2E7D32", **kw)
ax.text(0.97, 0.03, "Q4\ngen better\ncritic wrong",
        ha="right", va="bottom", color="#B71C1C", **kw)

fig.tight_layout()
path = f"{PLOTS_DIR}/plot6_four_quadrant_breakdown.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"→ {path}")

print("\nDone.")
