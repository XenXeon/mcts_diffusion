"""
phase1_make_plots.py

Loads results/phase1/per_state_results.csv and results/phase1/plans.npz
and produces four diagnostic plots WITHOUT re-running the planner.

Plot 1 — Critic score histograms
    Overlapping distributions of DVHorizonCritic scores for generated plans
    and real dataset continuations. Tests whether the planner produces plans
    that are scored comparably to ground-truth data.

Plot 2 — Paired scatter
    Each of the 100 samples as a point (score_real, score_gen) with the
    y=x diagonal. Positive correlation indicates the planner correctly
    assigns higher value to start states near the goal.

Plot 3 — L2 divergence vs horizon step
    Per-jump-step mean L2 distance between imagined and real plans (red),
    with the planner self-consistency baseline (two independent draws, blue).
    The self-consistency curve is the lower bound: if imagined-vs-real L2
    greatly exceeds it, the plans are genuinely diverging from ground-truth,
    not just showing inherent diffusion stochasticity.

Plot 4 — 8-panel trajectory overlays on the maze
    4 samples with lowest imagined-vs-real L2 (best generalisation) and 4
    with highest L2 (worst generalisation). Each panel overlays the imagined
    plan (blue) and real continuation (orange) on a grey point-cloud of all
    maze2d-umaze-v1 observations, with the conditioned start state marked.

Output:
    results/phase1/plots/plot1_critic_histograms.png
    results/phase1/plots/plot2_paired_scatter.png
    results/phase1/plots/plot3_l2_vs_horizon.png
    results/phase1/plots/plot4_trajectory_overlays.png

Usage:
    python scripts/phase1_make_plots.py
"""
import csv
import os
import sys

sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HORIZON  = 32
OUT_DIR  = "results/phase1/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load artefacts ────────────────────────────────────────────────────────────
csv_path = "results/phase1/per_state_results.csv"
npz_path = "results/phase1/plans.npz"

if not os.path.exists(csv_path) or not os.path.exists(npz_path):
    raise FileNotFoundError(
        "Artefacts not found. Run scripts/test_start_state_generalisation.py first.")

with open(csv_path, newline="") as f:
    _rows = list(csv.DictReader(f))
_float = lambda col: np.array([float(r[col]) for r in _rows])
score_gen   = _float("score_gen")
score_real  = _float("score_real")
mean_l2_arr = _float("mean_l2")
self_l2_arr = _float("planner_self_l2")

npz = np.load(npz_path)

imagined_plans     = npz["imagined_plans"]           # (100, 32, 4) normalised
real_plans         = npz["real_plans"]               # (100, 32, 4) normalised
imagined_plans_2nd = npz["imagined_plans_2nd_draw"]  # (100, 32, 4) normalised
norm_mean          = npz["norm_mean"]                # (4,)
norm_std           = npz["norm_std"]                 # (4,)

n_samples = len(_rows)

def unnormalise(x: np.ndarray) -> np.ndarray:
    """Invert GaussianNormalizer: x_raw = x_norm * std + mean."""
    return x * norm_std + norm_mean


# ── Plot 1: Critic score histograms ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

all_scores = np.concatenate([score_gen, score_real])
bins = np.linspace(all_scores.min() - 0.02, all_scores.max() + 0.02, 30)

ax.hist(score_gen,  bins=bins, alpha=0.65,
        label=f"Generated  (μ={score_gen.mean():.3f})",
        color="#2196F3", edgecolor="white", linewidth=0.4)
ax.hist(score_real, bins=bins, alpha=0.65,
        label=f"Real cont. (μ={score_real.mean():.3f})",
        color="#FF9800", edgecolor="white", linewidth=0.4)

ax.axvline(score_gen.mean(),  color="#1565C0", linestyle="--", linewidth=1.5)
ax.axvline(score_real.mean(), color="#E65100", linestyle="--", linewidth=1.5)

ax.set_xlabel("DVHorizonCritic score  (MC return in [−1, 1])")
ax.set_ylabel("Count")
ax.set_title("Critic score distribution: generated vs real plans  (n=100)")
ax.legend(framealpha=0.9)
fig.tight_layout()
path = f"{OUT_DIR}/plot1_critic_histograms.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"→ {path}")


# ── Plot 2: Paired scatter ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5.5))

ax.scatter(score_real, score_gen,
           alpha=0.55, s=30, color="#6200EA", zorder=3)

lim_lo = min(score_real.min(), score_gen.min()) - 0.05
lim_hi = max(score_real.max(), score_gen.max()) + 0.05
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
        "k--", linewidth=1.0, label="y = x  (perfect agreement)", zorder=2)

corr = np.corrcoef(score_real, score_gen)[0, 1]
ax.set_xlabel("Real-continuation critic score")
ax.set_ylabel("Generated-plan critic score")
ax.set_title(f"Per-sample paired critic scores  (r = {corr:.3f})")
ax.set_xlim(lim_lo, lim_hi)
ax.set_ylim(lim_lo, lim_hi)
ax.set_aspect("equal")
ax.legend(framealpha=0.9)
fig.tight_layout()
path = f"{OUT_DIR}/plot2_paired_scatter.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"→ {path}")


# ── Plot 3: L2 divergence vs horizon step ────────────────────────────────────
# Per-step L2 across all 100 samples.
l2_vs_real = np.linalg.norm(imagined_plans - real_plans,         axis=-1)  # (100, 32)
l2_self    = np.linalg.norm(imagined_plans - imagined_plans_2nd, axis=-1)  # (100, 32)

steps    = np.arange(HORIZON)
mean_r   = l2_vs_real.mean(axis=0)
std_r    = l2_vs_real.std(axis=0)
mean_s   = l2_self.mean(axis=0)
std_s    = l2_self.std(axis=0)

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(steps, mean_r, color="#F44336", linewidth=2.0, label="Imagined vs real")
ax.fill_between(steps, mean_r - std_r, mean_r + std_r,
                alpha=0.18, color="#F44336")

ax.plot(steps, mean_s, color="#2196F3", linewidth=2.0, linestyle="--",
        label="Planner self-consistency (2nd draw)")
ax.fill_between(steps, mean_s - std_s, mean_s + std_s,
                alpha=0.18, color="#2196F3")

ax.set_xlabel("Jump-step index  (1 step = 15 dense env steps)")
ax.set_ylabel("Mean L2 distance  (normalised obs space)")
ax.set_title("L2 divergence vs planning horizon  (mean ± 1 std, n=100)")
ax.set_xlim(0, HORIZON - 1)
ax.legend(framealpha=0.9)
fig.tight_layout()
path = f"{OUT_DIR}/plot3_l2_vs_horizon.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"→ {path}")


# ── Plot 4: 8-panel trajectory overlays ──────────────────────────────────────
# Select 4 samples with lowest mean_l2 (best generalisation) and 4 with
# highest mean_l2 (worst generalisation).
sorted_idx  = np.argsort(mean_l2_arr)
panel_indices = np.concatenate([sorted_idx[:4], sorted_idx[-4:]])
panel_labels  = (["Best L2"] * 4) + (["Worst L2"] * 4)

# Build a maze point-cloud from unnormalised plan positions.
# Use both imagined and real plans (already loaded) to create background coverage.
all_norm_xy = np.concatenate([
    imagined_plans[:, :, :2].reshape(-1, 2),
    real_plans[:, :, :2].reshape(-1, 2),
    imagined_plans_2nd[:, :, :2].reshape(-1, 2),
], axis=0)
all_unnorm_xy = all_norm_xy * norm_std[:2] + norm_mean[:2]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for panel, (sample_i, label) in enumerate(zip(panel_indices, panel_labels)):
    ax = axes[panel]

    # Gray background: all trajectory positions in this dataset sample.
    mask = (
        (all_unnorm_xy[:, 0] > 0.2) & (all_unnorm_xy[:, 0] < 4.8) &
        (all_unnorm_xy[:, 1] > 0.2) & (all_unnorm_xy[:, 1] < 4.8)
    )
    ax.scatter(all_unnorm_xy[mask, 0], all_unnorm_xy[mask, 1],
               c="lightgray", s=1, zorder=1, rasterized=True)

    # Unnormalise the two plans for this sample.
    img_plan  = unnormalise(imagined_plans[sample_i])   # (32, 4)
    real_plan = unnormalise(real_plans[sample_i])        # (32, 4)

    ax.plot(real_plan[:, 0], real_plan[:, 1],
            color="#FF9800", linewidth=1.8, label="Real", zorder=2, alpha=0.85)
    ax.plot(img_plan[:, 0],  img_plan[:, 1],
            color="#1565C0", linewidth=1.8, label="Imagined", zorder=3, alpha=0.85)

    # Mark the conditioned start state.
    ax.scatter(img_plan[0, 0], img_plan[0, 1],
               color="#00C853", s=80, zorder=5, marker="*", label="s₀")

    ax.set_title(
        f"{label} #{sample_i}\n"
        f"L2={mean_l2_arr[sample_i]:.3f}  "
        f"s_gen={score_gen[sample_i]:.3f}  "
        f"s_real={score_real[sample_i]:.3f}",
        fontsize=8)
    ax.set_xlim(0.2, 4.8)
    ax.set_ylim(0.2, 4.8)
    ax.set_aspect("equal")
    ax.axis("off")

    if panel == 0:
        ax.legend(fontsize=7, loc="lower right", framealpha=0.85)

fig.suptitle(
    "Trajectory overlays on maze2d-umaze-v1  "
    "— 4 best (left) + 4 worst (right) by imagined-vs-real L2\n"
    "Blue = imagined plan, Orange = real continuation, ★ = conditioned start state",
    fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
path = f"{OUT_DIR}/plot4_trajectory_overlays.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"→ {path}")

print(f"\nAll four plots saved to {OUT_DIR}/")
