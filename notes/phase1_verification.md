# Phase 1 Verification Report

## 1. Setup

| Field | Value |
|---|---|
| Environment | `maze2d-umaze-v1` |
| n_states | 100 held-out (traj_idx, offset) pairs |
| Planner checkpoint | 1 000 000 gradient steps |
| Policy checkpoint | 1 000 000 gradient steps |
| Critic checkpoint | 1 000 000 gradient steps |
| Checkpoint path | `results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer_d2_width256_separate_dpTrue/maze2d-umaze-v1/` |
| Seeds | trajectory permutation rng=0; offset sampling rng=42; torch seed=0 |
| Git commit | `5ae067313c2ffe1b7211208edb7bdc2a3db0e529` |
| Reproducibility | `cudnn.deterministic=True`, `cudnn.benchmark=False` |
| Scripts | `test_start_state_generalisation.py`, `phase1_make_plots.py`, `phase1_ground_truth_returns.py` |
| Wall time | not recorded (scripts run independently) |

## 2. Pre-flight Summary

See [`notes/phase1_preflight.md`](phase1_preflight.md) for full findings. Nothing changed
relative to that document:

- `dataset.seq_obs` is pre-normalised (z-scored). Env observations are raw and normalised
  at the rollout boundary only. Double-normalisation was confirmed to blow values to abs-max ≈ 4.
- `seq_obs` shape `(1566, 1265, 4)` with heavy terminal-state padding. True path lengths
  recovered by consecutive-state change detection (threshold 1e-5) and saved to
  `results/phase1/path_lengths.npy`: mean 482.8 dense steps, range [47, 800].
- 10% held-out split (rng=0): 156 trajectories, 74 usable (length ≥ H×M = 480), 17,676
  valid offset slots. 100-sample draw is well-distributed across trajectories.
- `cudnn.deterministic = True` confirmed set in all Phase 1 runner scripts.

## 3. Quantitative Results

### 3a. Critic score distributions (n=100)

| Statistic | score_gen | score_real |
|---|---|---|
| mean | +0.6930 | −0.5131 |
| stdev | 0.1146 | 0.2236 |
| min | +0.5029 | −0.9651 |
| max | +0.9674 | −0.1975 |

**Mean score gap** (gen − real): **+1.2062**
**K–S statistic**: **1.0000** (p ≈ 0; the two distributions have zero overlap)

The distributions are completely separated: every generated plan score is positive, every
real-continuation score is negative.

### 3b. Trajectory geometry

| Metric | Mean | Max |
|---|---|---|
| mean_l2 (imagined vs real) | 2.7919 | 3.3089 |
| planner_self_l2 (imagined vs 2nd draw) | 0.3054 | 1.0434 |

Plans diverge from real continuations by ~9× the planner's own intrinsic stochasticity,
confirming systematic plan/data-manifold separation. The planner is stochastic
(`planner_self_l2` mean 0.305, non-zero across all 100 samples), so stochastic
branching for MCTS expansion is viable.

### 3c. In-bounds rate

**100/100** — every position in every generated plan (after unnormalising) lies within
the maze bounds.

### 3d. Ground-truth return comparison (n=30, every 3rd sample)

| Metric | Generated plans | Real continuations |
|---|---|---|
| true_return mean | 58.50 | 0.00 |
| true_return stdev | 54.03 | 0.00 |
| true_return range | [17, 222] | [0, 0] |

All 30 generated plans achieved nonzero true return when rolled out via the
inverse-dynamics policy. All 30 real continuations had true return = 0, meaning none
of the sampled held-out windows contained a goal visit in the next 465 dense steps
(31 jump-step transitions × M=15). See caveat in §6.

**Critic calibration within generated plans:**
Pearson r(score_gen, true_return_gen) = **−0.116**;
Spearman ρ = **−0.081** (n=30).
The correlation is weakly negative: the highest-scoring generated plans are not the
ones that achieve the most goal-time in the actual rollout (e.g. score=0.967 →
true_return=22; score=0.748 → true_return=193).

## 4. Plots

All plots saved to `results/phase1/plots/`.

| File | Description |
|---|---|
| `plot1_critic_histograms.png` | Overlaid histograms of score_gen and score_real with means and K–S annotation |
| `plot2_paired_scatter.png` | Per-state scatter: x=score_real, y=score_gen, y=x diagonal |
| `plot3_l2_vs_horizon.png` | Mean L2 ± 1 std vs jump-step index (imagined-vs-real and self-consistency) |
| `plot4_trajectory_overlays.png` | 2×4 grid: 4 best + 4 worst by imagined-vs-real L2, real (blue) vs generated (red) |
| `plot5_critic_vs_true_return.png` | Critic score vs true env return (normalised axes), n=30 |
| `plot6_four_quadrant_breakdown.png` | (true_return_gen − true_return_real) vs (score_gen − score_real) quadrant chart |

## 5. Verdict: PASS-WITH-LIMITATION

**This is an explicit override of the metric-table decision rule, justified by the
pre-planned anomaly interpretation in the verification plan.**

The pass-criteria table marks K–S = 1.0 and mean gap = +1.206 as FAIL. However,
[`notes/phase1_verification_plan.md`](phase1_verification_plan.md) §"What to Do on Each
Failure Mode" explicitly anticipates this pattern:

> *K–S > 0.40, score_gen >> score_real → "Not a Phase 1 fail. Proceed to Phase 2 but
> flag prominently for Phase 5."*

The structural checks that would constitute a true Phase 1 blocker all passed:

- `fix_mask` assertion: imagined_plan[0] − s_0 max-abs < 1e-4 for all 100 samples
- In-bounds rate: 100%
- Planner stochasticity: planner_self_l2 > 0 for all 100 samples

The K–S = 1.0 / gap = +1.206 pattern reflects model exploitation at depth 0: the
planner generates goal-directed plans from any conditioned start state, while the sampled
real continuations happen not to visit the goal in the 465-step window. The ground-truth
rollouts confirm the score separation has a real behavioural basis — generated plans do
reach the goal, real continuations did not — but the critic's score does not provide
fine-grained ranking within the generated set (r = −0.116 on n=30).

**Justification for proceeding:** the foundation is structurally sound. The exploitation
signature is a known property of trajectory-value critics trained on sparse rewards; it
motivates, but does not invalidate, the Phase 5 uncertainty-penalty work.

## 6. Anomalies for Phase 5

### A. Model exploitation at depth 0 (primary finding)

K–S = 1.0 and gap = +1.206 are the clearest possible exploitation signal. Even at
depth-0 of any future MCTS tree, the critic will systematically prefer generated plans
over dataset plans. Within generated plans, score rank does not correlate with true
return (r = −0.116). This means the critic cannot reliably rank candidate trajectories
by quality — on this 30-sample subset, the critic does not show reliable fine-grained ranking of generated trajectories by true return, even though it strongly separates generated from real continuations.

**Implication for MCTS:** without an uncertainty penalty or calibration correction,
tree search will exploit the critic score, not optimise actual return. Phase 5 should
treat this as the primary risk and design accordingly.

### B. true_return_real = 0 for all 30 ground-truth samples

Every sampled real continuation had zero reward in the 465-step window. Two hypotheses:

1. **Expected:** the held-out start states are from trajectories where the next 465
   dense steps do not contain a goal visit. Given maze2d-umaze is a sparse-reward random
   walk dataset (mean trajectory length 482.8, many trajectories reaching the goal only
   near the end), this is plausible. Offsets are sampled from `[0, path_length − 480]`,
   which can place the window entirely before the terminal goal visit.
2. **Bug:** the `seq_rew` extraction in `phase1_ground_truth_returns.py` (line 136,
   `seq_rew[traj_idx, offset:offset+N_DENSE, 0] + 1.0`) might silently read padding
   entries. The IQL reward shift (raw_D4RL_reward − 1 stored, +1 added back) should
   recover the sparse 0/1 signal, but this has not been spot-checked against raw D4RL
   reward values for the same (traj_idx, offset) pairs.

**Verified (hypothesis 1 confirmed):** for all 30 rows, the goal step (index
`path_lengths[traj_idx] − 1`) lies 23–305 dense steps past the window end
(`offset + 464`), so no sampled window can contain a goal reward; `true_return_real = 0`
is mathematically guaranteed by the offset sampling constraint and is not a data bug.

### C. Planner-self L2 outlier

`planner_self_l2` max = 1.043 (one sample) against a mean of 0.305. This single
high-variance draw is worth noting but is not unexpected given diffusion stochasticity
and does not indicate a systematic problem.
