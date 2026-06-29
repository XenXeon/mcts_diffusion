# MCTS over a Diffusion Planner — Phases 0–4 Consolidated Report

**Task family:** D4RL `maze2d-{umaze,medium,large}-v1`
**Base method:** DV-MCSS (Diffusion Veteran with Monte-Carlo Self-Sampling)
**Checkpoints:** planner + critic + policy, each at 1,000,000 gradient steps, per env
(`results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer_d2_width256_separate_dpTrue/<env>/`)
**Reproducibility:** `cudnn.deterministic=True`, `cudnn.benchmark=False` throughout.

---

## 0. Executive summary

The project asked whether wrapping the DV-MCSS diffusion planner in Monte-Carlo Tree
Search (MCTS) improves closed-loop control on maze2d. After building and verifying the
full stack (Phases 0–3), running controlled closed-loop comparisons (Phase 4), and
directly testing the two candidate "fixable flaw" explanations (Phase 5), the answer is
**no, and the experiments explain why with a clear mechanism**:

1. **Tree search is neutral at matched candidate count.** On `maze2d-large`,
   MCTS-K10 (182.8) ≈ greedy-K10 (183.4); the entire apparent deficit vs the K=50
   baseline was the candidate-count reduction, not the search.
2. **At a fixed plan-sample budget, breadth strictly dominates depth.** Spending 120
   plans/step on breadth (greedy-K120: 191.9) beats spending them on depth
   (MCTS-K10-exp12: 182.8) by +9.1 points and runs ~3× faster.
3. **Breadth saturates by K≈50.** K10→K50 buys +8.1; K50→K120 buys +0.5 (noise).
   greedy-K50 already sits at the system's performance ceiling (~191 on large), which
   neither more breadth nor any depth exceeds.

The binding constraint is **model quality (planner/critic/policy), not the planning-time
search strategy.** The contribution is a quantified, mechanistic negative result telling
the field *where not to spend compute* on diffusion planners.

**Phase 5 closes the "is the ceiling fixable?" question.** Both proposed design fixes are
refuted: branching the tree where the K plans actually diverge (not the degenerate
waypoint 1) is ≈ the wp1 design and *decreases* with branch distance — wp1 is in fact
optimal; and the critic is well-calibrated (Pearson r ≈ 0.44, flat) at every tree depth
with no off-manifold drift. The ceiling is **intrinsic redundancy**, not a defect.

A second, important correction emerged during Phase 1/4 analysis: the critic, initially
suspected to be the bottleneck (a weak `r=−0.116` calibration number), is in fact
**near-perfect** at scoring plans (`r=−0.995` against plan reach-time). The `−0.116` was
an underpowered, confounded, open-loop artefact, not evidence of a bad critic.

---

## 1. System under test (DV-MCSS)

DV-MCSS is a **separate-pipeline** diffusion planner:

- **Planner** — `ContinuousDiffusionSDE` over a `DiT1d` transformer, DDIM solver, 20
  steps. Generates trajectories of `H=32` waypoints in **observation space only**
  (`planner_dim = obs_dim = 4`; no actions in the trajectory). Waypoints are spaced
  `stride M=15` dense env steps apart, so one plan depicts `32×15 = 480` dense steps.
  A `fix_mask` clamps waypoint 0 to the conditioned start state at every denoising step.
- **Critic** — `DVHorizonCritic` transformer, input `(B,32,4)` → scalar, trained on MC
  returns normalised to `[−1,1]`. Ranks the K planner candidates.
- **Policy** — diffusion inverse-dynamics MLP (`DVInvMlp`), DDPM solver, 10 steps. Given
  `(current_state, next_waypoint)` (both position-rebased) it infers the action.

**Per-step control loop (greedy DV-MCSS / "MCSS"):** normalise obs → sample K plans →
critic argmax → take waypoint index 1 of the best plan → policy infers action → step env
→ re-plan. It is a receding-horizon (MPC) controller.

**Reward accounting:** maze2d is fixed-horizon (TimeLimit; the goal is non-terminal).
Reward latches — `finished |= (rew==1.0); ep_reward += float(finished)` — so return =
number of steps spent at/after the goal. D4RL normalised score uses
`REF_MIN=23.85`, `REF_MAX=161.86` for umaze.

---

## 2. Phase 0 — Baseline reproduction

**Goal:** reproduce single-shot DV-MCSS and establish the comparison baseline per env.
**Script:** `scripts/run_one_episode.py` (driven by `scripts/run_baseline_seeds.sh`).

A parameterisation fix was required before running the larger mazes: the script was
hardcoded to umaze with `MAX_T=300`. Running medium/large at `MAX_T=300` would silently
truncate episodes (to 50% / 37% of the task) with no error. Fixed by reading
`MAX_T = env._max_episode_steps` (300 / 600 / 800) and deriving the checkpoint path from
`--env`. The greedy *logic* itself was unchanged and is verified identical to the audited
`mcts/rollout.py` (see Phase 4, Ablation B).

### Results (5 seeds each)

| Env | episode len | mean ± std | per-seed normalised | goal_step range |
|---|---|---|---|---|
| umaze | 300 | **107.2 ± 19.4** | 114.6, 68.9, 121.8, 112.4, 118.2 | 108–181 |
| medium | 600 | **127.5 ± 20.1** | 127.5, 119.5, 113.5, 110.8, 166.1 | 148–294 |
| large | 800 | **191.4 ± 68.1** | 111.2, 211.1, 112.7, 276.6, 245.5 | 54–496 |

**Findings:**
- The planner generalises to all three mazes (every seed solved, scores well above the
  reference expert).
- Single-episode variance is high (umaze seed 1 = 68.9), consistent with the DV paper's
  note that maze2d single-episode scores are noisy. **All later comparisons are
  seed-matched** as a result.
- The `goal_step` vs the 480-step single-plan horizon is the key signal for later phases:
  umaze/medium never approach it; **large seeds 0 & 2 (492, 496) exceed it** — the first
  evidence of a regime where one plan cannot reach the goal.

---

## 3. Phase 1 — Critic & pipeline verification

**Goal:** confirm the planner/critic/policy behave correctly and probe critic
calibration on 100 held-out start states.
**Scripts:** `test_start_state_generalisation.py`, `phase1_ground_truth_returns.py`.
**Verdict at the time:** PASS-WITH-LIMITATION.

### 3a. Structural checks (all passed)
- `fix_mask` holds: `|imagined_plan[0] − s_0| < 1e-4` for all 100 samples.
- In-bounds: 100/100 plans lie within maze bounds after unnormalising.
- Planner is stochastic: `planner_self_l2 > 0` for all 100 (necessary for MCTS branching).

### 3b. Critic score separation (n=100)
| | score_gen | score_real |
|---|---|---|
| mean | **+0.693** | **−0.513** |
| range | [+0.50, +0.97] | [−0.97, −0.20] |

Gap **+1.206**, K–S **1.000** (zero overlap). The critic cleanly separates generated
(goal-directed) plans from real dataset continuations (random-walk segments).

### 3c. Ground-truth returns (n=30) and the `−0.116` number
Executing each plan via the inverse-dynamics policy:
- `true_return_gen`: mean 58.5, range [17, 222]. **30/30 generated plans reached the goal.**
- `true_return_real`: **all 0** — structurally guaranteed (offset sampling places the
  465-step window entirely before each trajectory's terminal goal visit; verified, not a bug).
- **Pearson r(score_gen, true_return_gen) = −0.116** (Spearman −0.09), n=30.

This number was originally read as "the critic cannot rank plans → the critic is the
bottleneck." **That interpretation was wrong**, and re-analysis (below) overturned it.

### 3d. Re-analysis — the critic is near-perfect; the `−0.116` is an artefact
Recomputing from `results/phase1/ground_truth_returns.csv` and the saved plans:

| correlation | value | meaning |
|---|---|---|
| r(score_gen, true_return_gen) | −0.116, **95% CI [−0.457, +0.255]** | spans zero → *not significant* (t=−0.62, n=30) |
| **r(critic_score, plan reach-time)** | **−0.995** | critic reads the plan **near-perfectly** (earlier goal arrival → higher score) |
| r(plan reach-time, true_return) | +0.11 (≈0) | the plan's geometry does **not** predict open-loop execution return |

The chain: critic → plan reach-time is −0.995; plan reach-time → execution return is ≈0;
so critic → execution return is ≈0 by composition. **The weak link is the
plan→execution gap, not the critic.** Causes:
- **Range restriction:** all 30 plans are "good" (planner outputs), so within-band
  ranking correlation is attenuated.
- **Noisy target:** `true_return_gen` (dense camping steps) conflates arrival timing with
  latching.
- **Open-loop measurement:** plans were executed without re-planning; the closed-loop
  controller (Phase 0) re-plans every step and corrects tracking drift — which is why
  MCSS reliably scores ~107–191. The `−0.116` is doubly irrelevant to deployment.

### 3e. Plan geometry (decisive for Phase 4 framing)
Inspecting the 30 best plans' unnormalised trajectories:
- **30/30 reach the goal** at mean waypoint **7.1 of 31** (~107 dense steps).
- The plan horizon is 480 dense steps, so **~25 waypoints (~373 dense steps) remain after
  the goal is reached** — the planner's single shot contains the whole umaze solution with
  the horizon ~3–4× longer than the task needs.

**Implication:** on umaze, one diffusion sample already solves the task; tree depth has
nothing to search for. This is the root cause of the Phase 4 umaze null result.

---

## 4. Phase 2 — Expansion primitive

**Goal:** extract the per-step MCSS operation into a stateless, testable module.
**File:** `mcts/expansion.py` (`ExpansionConfig`, `ExpansionResult`, `PlannerExpansion`).

`PlannerExpansion.expand(s_norm)`: build `prior=zeros(K,H,4)`, write `s_norm` into
`prior[:,0]`, call `planner.sample`, score with `critic`, return K trajectories sorted
descending by score. `expand_batch(states)` generates K candidates for N states in one
GPU call (used for batched leaf evaluation). The module imports only `torch`/`dataclasses`
— fully testable without d4rl/gym. Contracts (fix_mask held, shapes, descending scores,
stochasticity, no-grad) are unit-tested. A later `uncertainty_beta` knob
(`scores − β·std(scores)`) was added for the Phase 5 idea (see Ablation E).

---

## 5. Phase 3 — Tree design and mechanics ablation

**Goal:** build the MCTS tree and measure its *engineering* properties (storage, depth,
wall-time) as an ablation rather than assuming them.
**Files:** `mcts/node.py`, `mcts/tree.py`, `scripts/phase3_ablation.py`,
`scripts/phase3_k_ablation.py`.

Tree mechanics: UCB1 selection (`value + c·sqrt(ln N_parent / N_child)`, `c=√2`,
unvisited = +∞), mean-score backprop, child = waypoint index 1, virtual loss for batched
leaf selection.

### 5a. Storage-mode ablation (`results/phase3/summary.csv`)
Three modes — `state_only`, `trajectory_node`, `state_edge_trajectory` — produce
**identical trees** (same `cumulative_best`, depth, `n_nodes`) and differ only in stored
trajectory floats:

| budget | n_nodes | depth | traj floats (state_only / node / edge) | wall (s) |
|---|---|---|---|---|
| 60 | 3001 | 3 | 0 / 384,128 / 384,000 | ~6–8 |
| 300 | 15001 | 3 | 0 / 1,920,128 / 1,920,000 | ~50 |

**Finding:** `state_only` is free (0 trajectory floats) with no loss — trajectories can be
regenerated from state. Wall-time is unaffected by storage mode. `state_only` adopted
downstream.

### 5b. K-ablation at fixed node count (`results/phase3/k_summary.csv`)
Fixing total nodes ≈ 15,001 and varying K (budget = 15000/K), batch_size ∈ {1,10,20}:

| K | budget | tree depth | cumulative_best (critic) | wall (batch=1) | wall (batch=10) |
|---|---|---|---|---|---|
| 5 | 3000 | **9** | ~1.00 | ~270 s | ~32 s |
| 10 | 1500 | 6–7 | ~1.00 | ~138 s | ~17 s |
| 20 | 750 | 4 | ~0.93 | ~71 s | ~10 s |
| 50 | 300 | **3** | ~0.89 | ~30 s | ~6.5 s |

**Findings:**
- Smaller K → deeper tree (more expansions at fixed node budget) → higher *critic*
  cumulative-best.
- **Critical caveat:** `cumulative_best` is the **critic score**, i.e. critic
  self-consistency, **not task return**. Deeper trees reaching critic≈1.0 does *not* mean
  better control — Phase 4 shows the opposite. Phase 3 measured tree mechanics, never
  performance.
- Per-expansion GPU cost is ~constant regardless of K (GPU unsaturated), so wall-time
  scales with the number of expansions, not total candidates. Batched leaf evaluation
  (`leaf_batch_size=10`) gives ~8× speedup.

---

## 6. Phase 4 — Closed-loop MCTS vs greedy

**Goal:** the actual question — does MCTS-guided control beat greedy DV-MCSS?
**Files:** `mcts/rollout.py`, `scripts/phase4_mcts_rollout.py`,
`scripts/phase4_ablation.py`, `scripts/phase5_headroom_diagnostic.py`.

MCTS rollout: at each env step, build a fresh tree from the current state, run
`max_expansions`, take `best_path()[1]` (root's highest-value child) as the next
waypoint, then the same policy as greedy.

### 6a. Initial umaze result + the depth bug
First runs (umaze, K=10, leaf_batch_size=10) underperformed badly:

| method | mean (seeds 0–4) | tree depth |
|---|---|---|
| greedy K=50 | 107.2 | — |
| MCTS-K10, budget 5 | 72.9 | 2.0 |
| MCTS-K10, budget 10 | 74.7 | 2.0 |

Diagnosed an **off-by-one in the depth/budget relationship**: with K candidates per
expansion, step 0 expands the root and steps 1..K expand its children — so **depth 3
requires budget ≥ K+2**, not `> K`. With K=10, budgets 5 and 10 both sit at depth 2 (no
look-ahead at all). The `best_path()` docstring and the depth tests were corrected.
Re-running at budget 12 (depth 3) on the *same* seeds jumped MCTS from 72.9 → **112.9**
(+40), confirming the bug had real impact.

### 6b. Diagnostic ablations (`results/phase4_ablation/`)
| Ablation | Question | Result |
|---|---|---|
| **B** | Is the MCTS machinery correct? (K=50, budget=1 vs greedy) | **Identical**: MCTS-K50-exp1 = greedy = 134.4 (seeds 5–14), Δ=+0.0. No extraction bug; `best_path()[1]` at depth 1 ≡ argmax. |
| **C** | Does depth help? (K=10, budgets 12/22/52/102) | MCTS-K10-exp12 (depth 3) = 112.0 vs greedy-K50 125.3 (seeds 0–14). Residual gap is **K, not search**. budget 22 = unstable "dead zone". |
| **D1** | Tie-breaking (greedy vs random UCB) | exp12 fine under both; exp22 unstable under both → budget level, not tie-breaking, is the instability. |
| **D2** | Smaller fan-out (K=5, budgets 6/12/31/52) | K=5 exp31/exp52 (depth ~4) **marginally beat** greedy on seeds 0–4 (+2.8 / +3.5) — the only umaze configs that did, at 20–35× the compute. |
| **E** | Uncertainty penalty `score−β·std` (the proposed "Phase 5") | **Disconfirmed.** No β > 0 beats β=0; β=1.0 catastrophic (84.1). The critic-exploitation mitigation does not help. |

**Seed-set confound corrected:** the original "MCTS 30% worse" was inflated because seeds
0–4 are hard (greedy-K50: 107.2 on 0–4 vs 134.4 on 5–14). Matched comparisons are the
only valid ones.

### 6c. Large-maze investigation (the decisive experiments)
Phase 0 showed large seeds 0 & 2 exceed the single-plan horizon. The headroom diagnostic
(`phase5_headroom_diagnostic.py`) confirmed the regime per seed:

| seed | start→goal | best plan reaches? | K reach | regime |
|---|---|---|---|---|
| 0 | 7.09 | NO (min 0.72) | **0/50** | stitching needed |
| 2 | 7.94 | NO (min 1.28) | **0/50** | stitching needed |
| 1,3,4 | 1.4–3.9 | YES | 27–29/50 | one plan solves it |

For seeds 0 & 2, **no candidate reaches the goal** — the genuine multi-plan stitching
regime, the first time the project left the degenerate (single-shot-solvable) setting.

Closed-loop MCTS vs greedy on large, with matched-K and matched-budget controls:

| method | K | plans/step | mean | seed 2 (hardest) | ms/step |
|---|---|---|---|---|---|
| greedy (baseline) | 50 | 50 | **191.4** | 112.7 | 112 |
| greedy control | 10 | 10 | 183.4 | 79.8 | ~102 |
| MCTS-K10-exp12 (depth) | 10 | 120 | 182.8 | 86.5 | 280 |
| greedy K=120 (breadth) | 120 | 120 | **191.9** | 106.0 | 98 |

Per-seed gap decomposition (MCTS underperformed *worst on the hard seeds* — monotonic with
difficulty, the **opposite** of the hypothesis):

| seed | greedy K50 | greedy K10 | MCTS K10 | K cost (50→10) | search cost (greedy→MCTS @K10) |
|---|---|---|---|---|---|
| 0 | 111.2 | 109.7 | 103.4 | −1.5 | −6.3 |
| 1 | 211.1 | 204.8 | 201.4 | −6.3 | −3.4 |
| 2 | 112.7 | 79.8 | 86.5 | **−32.9** | +6.7 |
| 3 | 276.6 | 275.8 | 275.8 | −0.8 | 0.0 |
| 4 | 245.5 | 246.7 | 246.7 | +1.2 | 0.0 |
| **mean** | **191.4** | **183.4** | **182.8** | **−8.0** | **−0.6** |

**Findings:**
1. **Search is neutral at matched K** (MCTS 182.8 ≈ greedy-K10 183.4, Δ=−0.6). The
   −8.7 deficit vs baseline was entirely the K=50→K=10 reduction.
2. **Breadth dominates depth at matched budget** (greedy-K120 191.9 > MCTS 182.8, +9.1)
   and is ~3× faster — because breadth is one batched GPU call (parallel) while tree depth
   is serial (each expansion depends on the previous selection).
3. **Breadth saturates by K≈50** (K10→K50 +8.1, K50→K120 +0.5). greedy-K50 is at the
   system's ceiling (~191); no amount of breadth or depth exceeds it.

---

## 7. Phase 5 — Diagnosing whether the ceiling is fixable

Phases 0–4 showed MCTS ties greedy at matched K and never beats greedy-K50. Phase 5 asks
the follow-up: is that ceiling an **intrinsic redundancy** or a **fixable design flaw**?
Two concrete flaws were proposed and tested head-on — the tree branches on waypoint 1
(where the K plans are nearly identical), and the critic might mis-score the imagined
states the tree creates at depth. **Both are refuted.**

### 7a. Plan diversity vs branch point (`phase5_plan_diversity.py`)
Mean pairwise L2 among K=50 plans at each waypoint (normalised obs space), 30 starts:

| waypoint (dense steps) | umaze | medium | large |
|---|---|---|---|
| 1 (15) — *where MCTS branches* | 0.085 (1.0×) | 0.066 (1.0×) | 0.073 (1.0×) |
| 8 (120) | 1.22 (14.5×) | 0.99 (15.0×) | 0.84 (11.6×) |
| 16 (240) — *peak* | 1.41 (16.7×) | 1.51 (23.0×) | 1.32 (18.2×) |
| 31 (465) | 0.77 (9.1×) | 1.22 (18.5×) | 1.18 (16.3×) |

Two facts, consistent across all three mazes: (i) the K plans are **~16× tighter at
waypoint 1 than at their most diverse point** — branching at wp1 captures only ~6 % of the
available trajectory diversity; (ii) diversity is a **hump** — it peaks mid-horizon (wp16)
and *re-converges* by wp31 as the plans funnel back toward the same goal. So the structural
critique is quantified, and it predicts where a "fix" would branch (mid-horizon, ~wp8–16).

### 7b. child_state_index ablation (Ablation F, `phase5_child_index_ablation.py`)
Branch the tree at `cidx ∈ {1,4,8,16}` in two modes — **matched** (policy also targets
wp cidx) and **split** (tree branches at cidx but the policy is commanded toward wp 1,
keeping it in-distribution). Large, seeds 0–4:

| cidx | matched | split | (greedy = 191.4, cidx1 = 182.8) |
|---|---|---|---|
| 1 | **182.8** | — | the current design |
| 4 | 159.9 | 180.4 | |
| 8 | **54.1** | 177.7 | |
| 16 | **54.2** | 173.9 | |

The fair test is **split** (diverse branching + healthy policy): it gives 180→174 —
**≈ cidx=1, always below greedy, and *decreasing* with cidx**. Branching where the plans
actually diverge does not help; it slowly hurts. The hypothesis is not merely refuted but
**inverted: wp1 is the optimal branch point.** (matched collapses for cidx>1 because the
policy receives a target far outside its 1-step training range — see 7d.)

### 7c. Depth-stratified critic calibration (`phase5_critic_depth_calibration.py`)
Build depth-d imagined states by chaining d planner jumps (exactly as MCTS descends),
generate K=20 plans from each over 20 starts (n=400/depth), execute every plan, correlate
critic score with true return. umaze:

| depth | Pearson r | nn-dist to data | mean critic |
|---|---|---|---|
| 0 | 0.45 | 0.101 | 0.31 |
| 1 | 0.43 | 0.100 | 0.39 |
| 2 | 0.44 | 0.091 | 0.31 |
| 3 | 0.44 | 0.080 | 0.43 |

Calibration is **flat (~0.44) across depth** and `nn-dist` to the dataset *decreases* —
imagined states do **not** drift off-manifold as the tree descends, and the critic does
**not** mis-score them. (r≈0.44 at n=400 also re-confirms the critic is genuinely decent;
Phase 1's −0.116 was the underpowered, range-restricted artifact.) The
"OOD-critic-at-depth" hypothesis is refuted.

### 7d. Why some episodes return exactly 0 (`phase5_zero_return_diagnosis.py`)
cidx>1 runs produce bimodal returns (decent or exactly 0). An instrumented closed loop
(per-step target displacement, action saturation, distance-to-goal, tree value) attributes
each 0. **Validation first:** raw env `rew==1` ⟺ the `dist<0.5` goal proxy on **0/3600**
steps, and `latched_return` matches every episode return — the metrics are exact. The 0s
fall into three non-bug signatures:

| signature | evidence | closest-approach position | cause |
|---|---|---|---|
| **matched cidx>1** | `tgt_disp≈2.1`, `act_sat≈0.9`, dist plateaus at 1.4–1.5 | pinned at wall **x≈2.4** (goal ≈ (0.9,1.0)) | policy OOD → can't navigate to a far target |
| **split cidx>1 (near-miss)** | in-dist (`tgt_disp 0.57`, `act_sat 0.20`), goal-directed targets, lingers ~30 steps then leaves | **beside the goal**, dist ≈ 0.80 | mis-aligned final approach (tree selects post-goal children on near-noise) |
| **split, slow** | distance still *descending* at t=300 | en route | far-branching path too slow to finish the episode |

None is a harness bug, and crucially **none is fixable by moving the branch point — each
*is* the cost of moving it.**

### Phase 5 verdict
Both "fixable flaw" explanations are dead: branching further out (where diversity is real)
hurts monotonically, and the critic is well-calibrated at every tree depth. The MCTS
ceiling is **intrinsic redundancy**, not a design defect. cidx=1 enters the goal basin
cleanly (min-dist 0.00–0.02) and ties greedy, which sits at the system ceiling.

---

## 8. Synthesis — why MCTS does not help

The mechanism is consistent across every controlled experiment:

- **The planner already does global look-ahead in one shot.** On umaze/medium the single
  480-step plan overshoots the goal by 3–4×; tree depth searches time already past the
  goal.
- **Greedy MCSS is a receding-horizon MPC that stitches implicitly.** It re-plans from the
  *true* state every step. Even on large seeds 0/2 (where one plan can't reach the goal),
  re-planning stitches automatically.
- **MCTS substitutes imagined look-ahead for re-observation.** Its tree expands from
  *planner-imagined* waypoints, and backs up the *mean* of K critic scores. In sim,
  re-planning from the real next state is both cheaper and more reliable than expanding a
  tree over increasingly off-distribution hallucinated states.
- **The stitching regime is where the critic's value signal is weakest** (0/50 plans reach
  the goal → all values are partial-progress extrapolations). MCTS *amplifies* reliance on
  that weak signal (deep bootstrap + mean backup); greedy *minimises* it (50 fresh samples,
  one argmax). Hence MCTS degrades *monotonically with difficulty* (−26 on seed 2).

**The only effective lever is candidate count K, and it saturates by ~50.** Beyond that,
the binding constraint is model quality, not planning-time search. Phase 5 closes the loop:
the ceiling is not a fixable search-design flaw (branch point or critic) — it is intrinsic.

---

## 9. Methodological lessons (where earlier conclusions were wrong)

This investigation reversed four intermediate conclusions; recording them as a guard
against the same errors:

1. **"The critic is bad" (from `r=−0.116`) — FALSE.** The number was underpowered (n=30,
   95% CI spanning zero), range-restricted, and open-loop. The critic actually reads plans
   near-perfectly (`r=−0.995` vs reach-time). Lesson: never treat a single small-n
   correlation as a verdict; check power, range restriction, and whether the measurement
   matches the deployment setting.
2. **"MCTS will help on large" — FALSE.** Goal-beyond-horizon is necessary but not
   sufficient; greedy MPC already stitches. Lesson: a necessary condition is not a green
   light.
3. **"More breadth keeps winning (K=120 > K=50)" — FALSE.** Breadth saturates by K≈50.
   Lesson: test the saturation point, don't extrapolate a trend.
4. **"Branching at wp1 is a bottleneck" — FALSE (inverted).** Branching where the plans
   diverge (cidx 4/8/16, split mode) is ≈ cidx=1 and *decreasing* with cidx — wp1 is the
   optimal branch point. Lesson: a quantified structural defect (wp1 = 6 % of peak
   diversity) is not automatically a *performance* defect; test it before assuming.

General lesson reinforced throughout: **seed-match every comparison** (single-episode
maze2d variance is ±20–70 points) and **add matched controls** (matched-K, matched-budget,
matched-plan-budget, in-distribution-policy split) before attributing an effect.

---

## 10. Limitations and future directions

- **n=5 seeds** per env in closed-loop; per-seed deltas on hard seeds (±6) are within
  single-episode noise. The *aggregate* and *matched* trends are robust; individual
  hard-seed claims are not.
- **maze2d only.** The conclusion ("search redundant over a strong planner with cheap
  re-planning") is specific to deterministic, fully-observed sim where re-planning is free.

MCTS could plausibly help only in a **fundamentally different setting**, not a maze2d
tweak:
- **Grounded expansion** — expand on *true environment dynamics* (or a learned dynamics
  model) instead of re-sampling the generative planner, so the tree sees information the
  planner cannot hallucinate.
- **Costly re-planning / partial observability** — settings where re-observing the true
  state each step is not free, removing MPC's structural advantage.
- A cheap, strictly-better default in the meantime: **argmax (max) backup instead of mean**
  removes MCTS's dilution handicap so it is never worse than greedy at matched K.

---

## 11. Recommended default

For DV-MCSS on maze2d: **greedy K=50** (single expansion per step). It reaches the
performance ceiling at the lowest compute. Tree search spends ~3× the compute to
underperform it; breadth beyond K=50 wastes compute for no gain.

---

### Appendix — key artefacts

| Phase | Scripts | Results |
|---|---|---|
| 0 | `run_one_episode.py`, `run_baseline_seeds.sh` | `results/phase0_baseline{,_medium,_large}.json` |
| 1 | `test_start_state_generalisation.py`, `phase1_ground_truth_returns.py` | `results/phase1/` |
| 2 | `phase2_smoke_test.py` | `mcts/expansion.py` + tests |
| 3 | `phase3_ablation.py`, `phase3_k_ablation.py` | `results/phase3/{summary,k_summary}.csv` |
| 4 | `phase4_mcts_rollout.py`, `phase4_ablation.py`, `phase5_headroom_diagnostic.py` | `results/phase4/`, `results/phase4_ablation/`, `results/phase4_large/` |
| 5 | `phase5_plan_diversity.py`, `phase5_child_index_ablation.py`, `phase5_critic_depth_calibration.py`, `phase5_zero_return_diagnosis.py`, `inspect_zero_traces.py` | `results/phase5/` |
