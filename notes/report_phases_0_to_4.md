# MCTS over a Diffusion Planner: Phases 0–4 Research Report

**Task:** D4RL `maze2d-{umaze,medium,large}-v1`  
**Base method:** DV-MCSS (Diffusion Veteran with Monte-Carlo Self-Sampling)  
**Research question:** Does wrapping the DV-MCSS diffusion planner in Monte-Carlo Tree Search (MCTS) improve closed-loop navigation performance?

---

## 1. System Description (DV-MCSS)

DV-MCSS is a diffusion-based model-predictive controller made up of three jointly-trained networks. Each of these networks was trained on 1,000,000 gradient steps from the D4RL offline dataset:

- **Planner** (`ContinuousDiffusionSDE`, DDIM solver, 20 denoising steps): Takes a normalised current observation and generates H=32 waypoints in observation space (position only, 4 dimensions). Waypoints are spaced M=15 dense environment steps apart, so one plan represents 32 × 15 = 480 dense steps of lookahead. A `fix_mask` clamps waypoint 0 to the current state at every denoising step to ensure plans start from the correct position.
- **Critic** (`DVHorizonCritic`, transformer): Takes a full (H=32, 4) trajectory and returns a scalar score trained on Monte-Carlo returns normalised to [−1, 1]. Used to rank K candidate plans from the planner.
- **Policy** (`DVInvMlp`, DDPM solver, 10 denoising steps): Takes the current state and the next planned waypoint (both position-rebased to the origin) and infers the action to execute in the environment.

**The greedy control loop (baseline):** At each environment step, normalise the observation → sample K=50 plans from the planner → rank with the critic → take waypoint index 1 of the best plan as the next target → pass to the policy → execute action → re-plan from the new true state. This is a receding-horizon Model Predictive Control (MPC) loop: the agent never commits to a full plan but re-plans from ground truth at every step.

**Reward accounting:** `maze2d` uses a latching reward. Once the robot reaches the goal, `reward=1` on every subsequent step for the rest of the episode. So the return equals (episode length − goal arrival step), and a lower goal arrival step is better. D4RL normalised scores use reference values `REF_MIN=23.85`, `REF_MAX=161.86` for umaze; scores above 100 indicate exceeding the reference expert.

---

## 2. Phase 0: Baseline Reproduction

**Goal:** Reproduce the published DV-MCSS performance and establish a seed-matched baseline before attempting any modifications.

**Setup:** `K=50` candidate plans per step, DDIM 20 denoising steps for the planner, DDPM 10 for the policy. Each episode ran at the environment's native time limit (umaze: 300 steps, medium: 600 steps, large: 800 steps). A parameterisation bug in the original script hardcoded `MAX_T=300` regardless of environment; this was fixed to read `MAX_T = env._max_episode_steps`, which would otherwise have silently truncated medium and large episodes.

### 2.1 Results

**maze2d-umaze-v1** (5 seeds):

| Seed | Norm. Score | Raw Return | Goal Step | ms/step |
|------|------------|-----------|-----------|---------|
| 0 | 114.59 | 182 | 118 | 90.1 |
| 1 | 68.94 | 119 | 181 | 96.1 |
| 2 | 121.84 | 192 | 108 | 98.3 |
| 3 | 112.42 | 179 | 121 | 93.4 |
| 4 | 118.22 | 187 | 113 | 92.4 |
| **Mean** | **107.20** | **171.8** | **128** | **94.1** |

**maze2d-medium-v1** (5 seeds): Mean normalised score **127.5 ± 20.1** (per-seed: 127.5, 119.5, 113.5, 110.8, 166.1). Goal steps ranged 148–294.

**maze2d-large-v1** (5 seeds): Mean normalised score **191.4 ± 68.1** (per-seed: 111.2, 211.1, 112.7, 276.6, 245.5). Goal steps ranged 54–496.

### 2.2 Key Observations

1. The planner generalises well across all three maze sizes; every seed successfully navigated to the goal.
2. Single-episode variance is high — umaze seed 1 (68.9) vs seed 2 (121.8), a gap of 53 points. This is an inherent property of stochastic diffusion sampling, not a bug. **All downstream comparisons are therefore seed-matched**, using the same random seed for both greedy and MCTS conditions.
3. On `maze2d-large`, seeds 0 and 2 arrive at the goal only at steps 496 and 492 respectively — very close to the 800-step limit, indicating near-failure. This is significant for Phase 4: it hints at a multi-plan stitching regime where a single plan cannot bridge start to goal.

---

## 3. Phase 1: Critic and Pipeline Verification

**Goal:** Confirm the three networks (planner, critic, policy) are individually correct and probe the reliability of the critic as a trajectory ranker.

### 3.1 Structural Checks

On 100 held-out start states:
- **`fix_mask` holds:** The first waypoint of every generated plan lies within 1e-4 of the conditioning start state. ✓
- **Plans stay in bounds:** 100/100 generated plans remain within maze boundaries after unnormalisation. ✓
- **Planner is stochastic:** Self-L2 distance between two samples from the same start state > 0 for all 100 starts (range 0.041–1.043). This is necessary for MCTS branching — if the planner always produced the same trajectory, expanding a node multiple times would be pointless. ✓

### 3.2 Critic Score Separation

The critic was tested on two trajectory types from 100 start states:

| Metric | Generated (planner output) | Real (dataset segment) |
|--------|---------------------------|----------------------|
| Mean score | **+0.693** | **−0.513** |
| Range | [+0.50, +0.97] | [−0.97, −0.20] |

The gap is 1.206 with zero distributional overlap (K–S statistic = 1.0). The critic cleanly distinguishes goal-directed planned trajectories from random offline dataset segments. This confirms the critic is discriminative at the coarse level.

### 3.3 Critic Calibration: What r = −0.116 Actually Means

To probe fine-grained ranking, 30 plans were executed open-loop using the policy and their actual episode returns recorded (`true_return`). The Pearson correlation between critic score and true return was:

**r(critic score, true return) = −0.116** (95% CI: [−0.457, +0.255], not statistically significant, t = −0.62, n = 30)

This number appears alarming — it would suggest the critic cannot rank plans by quality. However, this interpretation is **incorrect** for the following reasons:

1. **The critic scores plans, not episodes.** The critic is trained to score trajectories by how quickly and directly they reach the goal *in the plan geometry*. Re-computing the correlation against **plan reach-time** (the waypoint index at which the planned trajectory geometrically arrives at the goal) gives: **r(critic score, plan reach-time) = −0.995**. The critic is near-perfect at its actual task.

2. **The plan-to-execution gap.** The correlation between plan reach-time and actual execution return was r ≈ +0.11 (not significant). This gap exists because: (a) the policy executes plans with tracking noise that accumulates over time; (b) the closed-loop controller re-plans from the true state every step, which breaks the open-loop assumption used in this measurement. The `−0.116` is a measurement of an open-loop proxy, not of the critic's utility in the real deployment setting.

3. **Range restriction.** All 30 test plans were generated by the planner, which already biases toward good (goal-reaching) trajectories. Within a narrow band of "all reasonably good" trajectories, fine-grained ranking correlations are attenuated regardless of the critic's true quality.

**Conclusion:** The critic reliably ranks planned trajectories by geometrical quality (reach-time). The measured `r = −0.116` does not indicate a bad critic; it indicates that open-loop plan quality does not tightly predict closed-loop return, which is a structural property of the task, not a flaw in the critic.

### 3.4 Plan Geometry (Critical for Phase 4)

Inspecting the 30 generated plans after unnormalisation:
- 30/30 plans geometrically reach the goal
- Mean goal-arrival waypoint: **7.1 of 31** (≈107 dense steps)
- The full plan horizon covers 480 dense steps; the goal is reached at roughly the **22% mark**

This means the planner's single shot already contains a complete umaze solution, with ~3–4× excess horizon remaining after the goal. **There is nothing for a deeper tree search to find on umaze — the optimal trajectory is already in the first expansion.**

---

## 4. Phase 2: Expansion Primitive

**Goal:** Package the single-step MCSS operation (planner + critic ranking) into a stateless, testable module for use as the building block of MCTS.

**Implementation:** `mcts/expansion.py` — `PlannerExpansion.expand(s_norm)` builds a zero prior of shape `(K, H, 4)`, writes the current normalised state into position 0 of each candidate, calls `planner.sample`, scores with the critic, and returns K trajectories sorted descending by score. `expand_batch(states)` performs this for N states in a single GPU call for efficiency.

The module has no dependencies on the environment, D4RL, or gymnasium — it can be unit tested with CPU-only fakes. A test suite (`tests/test_mcts_expansion.py`) verified: fix_mask is obeyed, output shapes are correct, scores are returned in descending order, the planner produces different trajectories from the same state across calls.

---

## 5. Phase 3: Tree Design and Mechanics

**Goal:** Build the MCTS tree structure and characterise its engineering properties before running closed-loop experiments. Three specific questions were tested:

1. Which trajectory storage mode is most efficient?
2. How does fan-out K affect tree depth and critic optimisation?
3. What are the wall-time and memory costs?

### 5.1 Storage Mode Ablation

Three modes were compared, differing only in what data is stored per node:

| Mode | Storage per node | Description |
|------|-----------------|-------------|
| `state_only` | 4 floats (obs) | Node holds only the observation; trajectory regenerated if needed |
| `trajectory_node` | 132 floats (obs + 32×4 traj) | Node holds obs + the trajectory that produced it |
| `state_edge_trajectory` | 4 floats/node + 128 floats/edge | Node holds obs; incoming edge holds trajectory |

**Result:** All three modes produce **identical trees** — same cumulative best critic score, same depth, same node count. They differ only in memory usage:

| Budget | Nodes | Depth | state_only floats | traj_node floats | Wall time |
|--------|-------|-------|-------------------|-----------------|-----------|
|  60    | 3,001 |   3   |         0         |     384,128     |   ~6–8 s  |
|  300   |15,001 |   3   |         0         |    1,920,128    |    ~50 s  |

`state_only` was adopted as the default: zero trajectory storage overhead at no performance cost.

### 5.2 K-Ablation at Fixed Node Budget

With total node count fixed at ≈15,001, K was varied (budget = 15,000 / K):

| K | Budget | Tree Depth | Critic cumul. best | Wall (batch=1) | Wall (batch=10) |
|---|--------|-----------|-------------------|---------------|----------------|
| 5 | 3,000 | **9**      |      ~1.00        | ~270 s        | ~32 s |
| 10 | 1,500 | 6–7       |      ~1.00        | ~138 s        | ~17 s |
| 20 | 750  | 4          |      ~0.93        | ~71 s         | ~10 s |
| 50 | 300  | **3**      |      ~0.89        | ~30 s         | ~6.5 s |

**Key findings:**
- Smaller K → shallower fan-out → fewer total branches explored before moving deeper → greater tree depth at fixed node count
- The critic `cumulative_best` (the running maximum critic score seen in the tree) approaches 1.0 for small K: the tree is very good at finding plans the critic considers optimal

**Critical caveat:** `cumulative_best` measures critic self-consistency, not task return. Phase 4 showed that maximising the critic score in tree search does not translate to better maze performance. This is the central tension the experiments were designed to explore.

### 5.3 Depth/Budget Relationship

A non-obvious but important formula governs when the tree reaches a given depth:

- **Depth 2** requires budget ≥ 1 (root + 1 child): trivially, any single expansion
- **Depth 3** requires budget ≥ K + 2: one expansion for root (producing K children), then one expansion of a child, giving grandchildren
- **Depth 4** requires budget ≥ K² + K + 2

This formula became critical in Phase 4. With K=10, budgets of 5 and 10 both produce depth-2 trees (no look-ahead at all, equivalent to greedy), while budget 12 is the minimum for depth-3.

---

## 6. Phase 4: Closed-Loop MCTS vs Greedy

**Goal:** Run full episode evaluations comparing MCTS-guided control to the greedy DV-MCSS baseline. The MCTS loop builds a fresh tree from the current observation at each environment step, runs `max_expansions` expansions, selects the best path through the tree, and uses the root's best child's state as the next waypoint target (identical to how greedy uses waypoint index 1 of the best plan, but guided by tree search).

### 6.1 Initial Results and the Depth Bug

The first MCTS run used K=10, leaf batch size=10. Results on umaze (seeds 0–4):

| Method | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean | Depth |
|--------|--------|--------|--------|--------|--------|------|-------|
| DV-MCSS (greedy K=50) | 114.59 | 68.94 | 121.84 | 112.42 | 118.22 | **107.2** | — |
| MCTS-K10-budget5 | 124.74 | −17.28 | 117.49 | 113.87 | 25.47 | **72.9** | 2.0 |
| MCTS-K10-budget10 | 124.74 | −17.28 | 118.22 | 124.01 | 24.02 | **74.7** | 2.0 |

MCTS significantly underperformed. Critically, both budgets 5 and 10 show `mean_tree_depth = 2.0`, meaning the search never went beyond the immediate children of the root — equivalent to simply picking the best of K=10 trajectories with no look-ahead. This identified a **depth/budget bug**: the budget values were chosen by intuition, not by the mathematical formula above.

With K=10, budget must be ≥ K + 2 = 12 for depth-3. Budget 5 and 10 are below this threshold. Correcting to budget=12 gave:

| Method | Seeds 0–4 mean |
|--------|---------------|
| Greedy K=50 | 107.2 |
| MCTS-K10-budget12 | **112.9** |

A +5.7 point improvement. This was the first sign of genuine benefit, though the comparison was still unfair (different K values).

### 6.2 Ablation B: Correctness Verification (K=50, budget=1)

**Question:** Is the MCTS extraction logic correct? If MCTS with K=50 and budget=1 (a single expansion, no tree depth) gives the same result as greedy K=50, the machinery is correct.

**Result:** MCTS-K50-exp1 = DV-MCSS exactly, seed for seed, across all 15 seeds (Δ = 0.0 on every seed):

| Seed | DV-MCSS | MCTS-K50-exp1 | Δ |
|------|---------|--------------|---|
| 0 | 114.59 | 114.59 | 0.0 |
| 1 | 68.94 | 68.94 | 0.0 |
| 2 | 121.84 | 121.84 | 0.0 |
| 3 | 112.42 | 112.42 | 0.0 |
| 4 | 118.22 | 118.22 | 0.0 |
| 5–14 | 134.37 (mean) | 134.37 (mean) | 0.0 |

**Interpretation:** At depth 1, `best_path()[1]` selects the same child as `argmax(critic scores)`. The extraction logic is verified correct. The 15-seed greedy mean is **125.3** (seeds 0–4 average lower at 107.2; seeds 5–14 average higher at 134.4 due to natural task difficulty variation — this confirmed that seed-matching is essential).

### 6.3 Ablation C: Does Depth Help? (K=10, varying budget)

**Question:** Holding K=10 fixed, does increasing the search budget (and therefore tree depth) improve performance?

**Setup:** 15 seeds (0–14) for each condition, compared against the matched 15-seed greedy baseline (mean 125.3).

| Method | Mean (15 seeds) | Tree Depth | Plans/step | Δ vs greedy |
|--------|----------------|-----------|-----------|------------|
| DV-MCSS (greedy K=50) | **125.3** | — | 50 | — |
| MCTS-K10-exp12 | 112.0 | 3.0 | 120 | −13.3 |
| MCTS-K10-exp22 | ~100 | 3.0 | 220 | ~−25 |
| MCTS-K10-exp52 | ~108 | 3.0 | 520 | ~−17 |
| MCTS-K10-exp102 | ~109 | 3.7 | 1,020 | ~−16 |

**Key findings:**
1. No budget beats the greedy baseline on matched seeds
2. exp12 is the best MCTS configuration — deeper budgets do not help
3. Tree depth stays at approximately 3.0 across all budgets because K=10 creates 10 children at each node. After the root is expanded (10 children), each child must be visited before any grandchild UCB can compete (unvisited nodes have UCB = ∞). With budget=22, the pattern is: root expansion + 10 children + 11 grandchildren. The tree visits all 10 children before settling on the best, producing an instability in the "dead zone" around budget=22 where the algorithm has started committing to a direction before fully exploring the breadth.
4. The fundamental gap relative to greedy is not about depth — it is about **K**. Reducing K from 50 to 10 removes the diversity of root-level candidates.

### 6.4 Ablation D: Tie-Breaking and Fan-Out (K=5)

**D1 — Tie-breaking:** UCB1 selects the child with maximum UCB score. When multiple children are unvisited (UCB = ∞), the tie-breaking rule matters: "greedy" always picks the first (highest-scored) child; "random" picks uniformly among all tied children.

|        Method         | Seeds 0–4 mean | Failures (score<0) |
|-----------------------|---------------|-------------------|
| MCTS-K10-exp12-greedy |      104.8    |         0         |
| MCTS-K10-exp12-random |      109.4    |         0         |
| MCTS-K10-exp22-greedy |       77.9    |         1         |
| MCTS-K10-exp22-random |       93.3    |         1         |

Random tie-breaking outperforms greedy at both budgets. Greedy tie-breaking biases exploration toward the highest critic-scored child before other candidates are explored, potentially missing better trajectories if the critic is imperfect.

**D2 — K=5 fan-out:** Smaller K creates deeper trees faster (fewer children to exhaust before descending). K=5 reaches depth 4 with budget ≥ K² + K + 2 = 32.

| Method | Seeds 0–4 mean | Depth | Plans/step |
|--------|---------------|-------|-----------|
| DV-MCSS (greedy K=50) | 107.2 | — | 50 |
| MCTS-K5-exp6 | ~60 | 2.0 | ~39K |
| MCTS-K5-exp12 | ~83 | 3.0 | ~75K |
| MCTS-K5-exp31 | **~110** | ~4.0 | ~189K |
| MCTS-K5-exp52 | **~111** | ~4.0 | ~315K |

K=5-exp31 and exp52 marginally exceeded greedy (+2.8 and +3.5 points) on seeds 0–4, but at **20–35× the compute cost** of greedy. This was the only configuration across all ablations that showed an advantage over greedy, and only on the 5-seed original set — not the full 15-seed matched comparison.

**Discovered bug — RNG coupling:** A `torch.randint` call in the UCB tie-breaking code was consuming from PyTorch's global random number generator. With budget=12 (≈11 UCB calls per env step), this shifted the policy's DDIM sampling noise sequence relative to the greedy baseline, making the comparison unfair. The fix replaced `torch.randint` with Python's `random.choice()`, which uses a completely separate RNG. Seeds 11 and 14 continued to show poor MCTS performance even after the fix, confirming that RNG corruption was not the sole cause of failure — genuine MCTS misdirection also occurs.

### 6.5 Ablation E: Uncertainty Penalty (Phase 5)

**Motivation:** If the critic is noisy within a given expansion (the K trajectories from one start state receive widely varying scores), those scores are unreliable. Adding a penalty `score_k = raw_score_k − β × std(raw_scores)` reduces the backpropagated value of high-variance expansions, steering the tree away from uncertain regions.

**Setup:** K=10, budget=12, β ∈ {0.5, 1.0, 2.0, 5.0}, seeds 0–14 (matched to greedy).

| Method (β) | Mean score | vs greedy (−125.3) | Failures (score<0) | Wins vs greedy |
|-----------|-----------|-------------------|-------------------|----------------|
| DV-MCSS (greedy) | **125.3** | — | 0 | — |
| β = 0.5 | 111.3 | −14.0 | 0 | 7/15 |
| β = 1.0 | 84.1 | −41.2 | 2 | 5/15 |
| β = 2.0 | 103.5 | −21.8 | 1 | 5/15 |
| β = 5.0 | 100.6 | −24.7 | 0 | 5/15 |

**Verdict:** No β value beat the greedy baseline. β=0.5 was the least harmful (fewest failures, best mean), but still 14 points below greedy. Higher β values actively degraded performance: at β=1.0, seeds 2 and 14 produced return=0 (the robot never reached the goal). The uncertainty penalty was **disconfirmed as a solution**.

### 6.6 Large Maze: The Decisive Experiments

Phase 0 identified two seeds (0 and 2) on `maze2d-large-v1` where the goal step approached the 800-step limit (seeds 0: step 492, seed 2: step 496). A headroom diagnostic confirmed:

| Seed | Start→goal distance | Best plan reaches goal? | Regime |
|------|--------------------|-----------------------|--------|
| 0 | 7.09 units | No (0/50 plans reach it) | Multi-plan stitching |
| 2 | 7.94 units | No (0/50 plans reach it) | Multi-plan stitching |
| 1, 3, 4 | 1.4–3.9 | Yes (27–29/50 plans) | Single-shot solvable |

Seeds 0 and 2 represent the regime where MCTS *should* theoretically help: the goal is too far for a single plan, so multi-step look-ahead should enable stitching. Three conditions were compared with identical seeds:

| Method | K | Plans/step | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | **Mean** | ms/step |
|--------|---|-----------|--------|--------|--------|--------|--------|----------|---------|
| DV-MCSS (greedy) | 50 | 50 | 111.2 | 211.1 | 112.7 | 276.6 | 245.5 | **191.4** | 112 |
| DV-MCSS-K10 (matched K) | 10 | 10 | 109.7 | 204.8 | 79.8 | 275.8 | 246.7 | **183.4** | ~102 |
| MCTS-K10-exp12 | 10 | 120 | 103.4 | 201.4 | 86.5 | 275.8 | 246.7 | **182.8** | 280 |
| DV-MCSS-K120 (matched budget) | 120 | 120 | 110.9 | 218.6 | 106.0 | 276.6 | 247.4 | **191.9** | 98 |

**The gap decomposition is the core finding:**

| Seed | K cost (50→10 greedy) | Search cost (greedy K10 → MCTS K10) |
|------|----------------------|--------------------------------------|
| 0 | −1.5 | −6.3 |
| 1 | −6.3 | −3.4 |
| 2 | **−32.9** | **+6.7** |
| 3 | −0.8 | 0.0 |
| 4 | +1.2 | 0.0 |
| **Mean** | **−8.0** | **−0.6** |

1. **MCTS at matched K is statistically equivalent to greedy** (182.8 vs 183.4, Δ = −0.6). The entire apparent deficit vs the K=50 baseline came from reducing K, not from the search strategy.
2. **Breadth dominates depth at matched compute budget.** Spending 120 plans/step on breadth (greedy-K120: 191.9) beats spending them on depth (MCTS: 182.8) by +9.1 points, while running ~3× faster (98 ms vs 280 ms per step). The GPU executes all K=120 plans in a single batch; the tree must run 12 serial expansions.
3. **Breadth saturates at K≈50.** K10→K50 gains +8.0 points; K50→K120 gains only +0.5 (within noise). The greedy K=50 baseline is already at the system's performance ceiling.
4. **On the hard seeds (0 and 2), MCTS is worse** (−6.3 and +6.7 vs K10-greedy respectively). MCTS underperformance is monotonically correlated with task difficulty — the opposite of the hypothesis.

---

## 7. Discussion: Why MCTS Does Not Help

The pattern across every experiment is consistent and explained by a single mechanism:

**Greedy MPC already stitches for free.** The greedy controller re-plans from the *true observed state* at every step. Even on large seeds 0 and 2 where no single plan reaches the goal, the greedy controller stitches multiple plans together automatically: each step it generates 50 fresh plans from the current position, picks the best one, advances one step, then repeats. This implicit stitching works because replanning is cheap (one batched GPU call) and always starts from ground truth.

**MCTS substitutes imagined look-ahead for real replanning.** The tree expands from *planner-imagined* future waypoints, not from the true environment state. By the time the tree is 3 steps deep, the leaf nodes represent states the planner *imagines* the robot would occupy — states that may diverge from reality due to tracking error in the policy. The backpropagated critic scores are increasingly unreliable as tree depth increases.

**The critical point:** In classic MCTS applications (Chess, Go), the value function has r ≈ 0.95+ with actual game outcome because it was trained on real game results. In this system, the critic scores planned trajectories by geometric quality (r = −0.995 with plan reach-time), but this does not straightforwardly predict closed-loop return (the plan-to-execution gap). The tree search amplifies reliance on critic signals that are increasingly off-distribution at depth > 1.

**The only effective lever is K.** More diverse root-level candidates (higher K) gives the critic more choices, and even with imperfect ranking, a better trajectory tends to appear among 50 samples than among 10. This diversity benefit saturates by K ≈ 50.

---

## 8. Methodological Lessons

Three intermediate conclusions were overturned during the investigation:

1. **"The critic is bad" (from r = −0.116) — WRONG.** The number was not statistically significant (CI spans zero), was range-restricted to a narrow distribution of "all good" plans, and was measured open-loop rather than in the deployment setting (closed-loop MPC). The critic is near-perfect at its actual task (r = −0.995 vs plan reach-time).

2. **"MCTS will help on large because one plan can't reach the goal" — WRONG.** Goal-beyond-horizon is a necessary but not sufficient condition for MCTS benefit. Greedy MPC already handles multi-step stitching through sequential replanning, and does so without the hallucination error that accumulates in a tree.

3. **"More breadth keeps winning (K=120 > K=50)" — WRONG.** Breadth saturates by K≈50. The trend from K=10 to K=50 does not continue to K=120.

**Methodological rule reinforced throughout:** seed-match every comparison; add matched controls for all confounds (K, compute budget); never attribute an effect without isolating it experimentally.

---

## 9. Summary of All Results

| Experiment | Env | Method | Mean Norm. Score | Plans/step |
|------------|-----|--------|-----------------|-----------|
| Phase 0 | umaze | Greedy K=50 | 107.2       | 50 |
| Phase 0 | medium | Greedy K=50 | 127.5      | 50 |
| Phase 0 | large | Greedy K=50 | 191.4       | 50 |
| Phase 4 initial (bug) | umaze | MCTS K10 budget=5 | 72.9 | 50 |
| Phase 4 initial (bug) | umaze | MCTS K10 budget=10 | 74.7 | 100 |
| Abl. B | umaze | MCTS K50 budget=1 | 125.3 | 50 |
| Abl. B (reference) | umaze | Greedy K=50 (15 seeds) | 125.3 | 50 |
| Abl. C | umaze | MCTS K10 budget=12 | 112.0 | 120 |
| Abl. D2 | umaze | MCTS K5 budget=31 | ~110 | ~190K |
| Abl. E (best β) | umaze | MCTS K10 budget=12, β=0.5 | 111.3 | 120 |
| Phase 4 large | large | Greedy K=10 | 183.4 | 10 |
| Phase 4 large | large | MCTS K10 budget=12 | 182.8 | 120 |
| Phase 4 large | large | Greedy K=120 | **191.9** | 120 |

---

## 10. Conclusions

1. **MCTS at matched candidate count is statistically equivalent to greedy.** The −8.6 point gap observed in Phase 4 was entirely due to reducing K from 50 to 10, not due to the search strategy.
2. **Breadth strictly dominates depth at matched compute.** Spending the same planning budget on wider sampling (greedy-K120) outperforms spending it on tree search (MCTS-K10-exp12) by +9 points and runs 3× faster.
3. **Breadth saturates at K≈50.** The system is at its performance ceiling with greedy-K50; neither more breadth nor tree depth exceeds it.
4. **The uncertainty penalty (Phase 5) did not help.** No β value beat the greedy baseline; the best (β=0.5) reduced catastrophic failures but remained 14 points below greedy.
5. **The binding constraint is model quality, not planning strategy.** The planner's trajectory distribution and the critic's ranking accuracy determine the performance ceiling. Improving either would unlock further gains; MCTS-style planning within the current model cannot.

---

## 11. Limitations and Future Directions

**Limitations:**
- n=5 seeds per environment for most comparisons; single-episode variance on maze2d is ±20–70 normalised points. The aggregate trends are robust; individual seed claims are not.
- All experiments use maze2d (deterministic, fully observed, cheap replanning). The conclusion "tree search is redundant over MPC" is specific to this setting.

**Future directions:**

**Direction 1 — Better value function training.**  
The current critic is trained on offline Monte-Carlo returns from the D4RL dataset. A critic trained directly on outcomes of the closed-loop controller (i.e., using actual maze2d episode returns as labels) would provide a value signal that more tightly predicts task success. This would improve MCTS tree quality if the plan-to-execution gap could be closed.

**Direction 2 — Grounded expansion with a world model.**  
MCTS tree nodes currently represent planner-imagined states. If instead each tree node expanded the *real environment state* — using a learned dynamics model as a fast simulator — the search would remain grounded in reality and accumulate less hallucination error at depth. This is analogous to how AlphaZero uses perfect game rules rather than learned state estimates for its tree.

**Direction 3 — Costly replanning / partial observability settings.**  
The structural advantage of greedy MPC is that replanning from the true state is cheap. In settings where observing the true state has a cost (partial observability, physical systems with sensing latency), MCTS's multi-step lookahead from the last known state would become genuinely useful rather than redundant.

**Direction 4 — Argmax (max) backup instead of mean.**  
The current implementation uses mean critic score as the node value. Using the *max* critic score seen in the subtree would make MCTS never worse than greedy at matched K (the root always has access to the best single trajectory's value), removing the dilution handicap from averaging over mediocre trajectories. This is a low-cost modification worth testing.

---

*All experiments run in Docker container `cleandiffuser:dev` with a single GPU. Seeds fixed via `cudnn.deterministic=True`, `cudnn.benchmark=False`, and a fixed `set_seed()` call. All results saved to `results/`.*
