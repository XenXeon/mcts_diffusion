# MCTS as the Sampler for Diffusion Veteran — Findings Report (Phases B–F)

*Phase B–E: the state-value tree and the matched-compute search win (§1–§5). Phase F:
the goal-conditioned value `V(s, g)` — a tested negative result that sharpens the win (§6).*

**Project brief:** *"Modern generative RL techniques provide powerful tools for robot
autonomy based on imagining how environments might unfold. Unfortunately, poor decisions
early-on in these imagined rollouts can significantly limit the final quality of the chosen
course of action. This project will take an off-the-shelf generative RL codebase and
integrate Monte-Carlo Tree Search to allow it to efficiently explore multiple potential
rollouts in parallel."*

**Codebase:** Diffusion Veteran (DV). Stock sampler = MCSS (sample K plans → trajectory-
critic argmax → execute first step → replan). This work replaces MCSS with **MCTS as the
sampler**, reusing DV's trained planner (diffusion trajectory generator) and inverse-
dynamics policy unchanged, and retraining **only the critic** into a form a tree can use.

**Headline result (antmaze-large-diverse-v2, closed loop, paired n=150, seeds 0–2):**

> **MCTS look-ahead beats MCSS by +11.3 pp at matched compute and equal wall time**
> (83.3% vs 72.0% reach at 272 candidates/step; exact McNemar **p = 0.021** on 150
> scenario-paired rollouts). The mechanism is visible in the paired data: the search
> **fixes 79% of greedy's failures** (33/42) at a roughly **constant ~15% break rate**
> (16/108 of greedy's successes), so the net gain concentrates where greedy struggles
> (per-seed: +22 pp where MCSS=64%, +14 pp at 70%, −2 pp at 82% — a ceiling effect).
> The win is the *structure*, not the candidate budget: flat best-of-N scaling actually
> *backfires* (MCSS k50→k272 = −7.3 pp, critic over-exploitation, §5.3), while the same
> compute organised as a value-guided tree scales positively. This validates the project
> brief: deeper exploration of imagined rollouts rescues the poor early commitments that
> cap greedy selection.

A ring of **characterised negatives** (§5.3, §6) sharpens the claim by ruling out the
obvious "fixes": more candidate budget under flat argmax backfires; goal-conditioning the
search value does **not** help despite the value being sound (§6 + D2); deeper branching
hurts (§5.3 child-index sweep); the value cannot stitch off-manifold but does not need to
(MPC replanning stitches implicitly, §3). The lever is the **search structure riding a
simple goal-agnostic geometry value** — not goal-awareness, data volume, or depth. Honest
ceiling: the best search config (83.3%) only marginally beats the *cheap* MCSS k=50
baseline (79.3%, +4 pp, n.s.) — antmaze-large-diverse is near-saturated for this planner,
so the rigorous result is the matched-compute contrast, not a large absolute gain.

> **⚠ Update (2026-06-24) — two findings sharpen and partly revise the headline above:**
> **(1) At n=500 (10 seeds) the matched-compute win does not survive high power.** The
> +11.3 pp (p=0.021, n=150) becomes **+4.2 pp (p=0.12)** for k272→b16, and the tree only
> *ties* the cheap k50 baseline (78.8 vs 79.0, p=1.0); the flat-scaling backfire survives
> in sign (k50→k272 −4.0 pp). Root cause, found by diffing the per-seed JSONs: the diffusion
> planner draw is **unseeded**, so the McNemar test pairs only the goal while the dominant
> variance (planner sampling + start jitter, ~±6 rollouts per config — the effect sign even
> *flips* on identical seed-0 scenarios) is unpaired. The n=150 gap was small-sample + noise
> inflation. The robust *qualitative* claim (flat best-of-N backfires; the tree avoids that
> degradation) stands; the large, significant absolute gain does not.
> **(2) A failure-attribution study (§7) shows FLAT selection is saturated — but not that
> search can't help.** Measured *paired*, a **perfect** geodesic ranker that picks the
> closest-endpoint candidate each step nets only **+0.7 pp** (oracle 78.7% vs critic 78.0%;
> fixes 23 / breaks 22), and the DV critic already picks the geodesic-best candidate. That
> bounds *flat best-of-N* selection only. Whether *structured* MCTS look-ahead driven by an
> *accurate* value beats ~78% is the **open** test (§7.5, geodesic-in-the-tree, ⏳) — so the
> §6 `V(s, g)` line is **not** closed and "near-saturated" applies to flat selection, not
> search. (An earlier draft's "+15 pp / 93% ceiling" was a fixes-only artifact that ignored
> the breaks — corrected in §7.2.)

This report records what was built, what was measured, what was wrong along the way, and
the data-backed hypothesis for the next improvement (goal-conditioned value).

---

## 1. Why the stock MCSS critic cannot drive a tree

DV's critic (`DVHorizonCritic`) scores a **whole (H, D) trajectory → one scalar**
(return-to-go of the entire plan, read from transformer token 0). Three code facts
(`cleandiffuser/utils/building_blocks.py:210`, `mcts/expansion.py`,
`mcts/tree.py:_backprop`) imply a structural dead end for search:

1. The value does **full lookahead at ply 1** — a depth-3 node's score already contains
   everything its parent's score contains. Values at different depths are not composable.
2. The Phase-3/4 tree backed up the **mean** of these full-plan scores — semantically
   meaningless across depths.
3. Consequently the Phase-4 measurement found **MCTS ≈ MCSS at matched K**: the tree had
   nothing to compose. A full-trajectory critic structurally cannot benefit from depth.

**Design conclusion:** the tree needs a value that is a function of a *single state* —
`V(s) → return-to-go from s` — so that node values depend on *where you are*, depth means
*progress*, and branches can be compared mid-route.

## 2. Phase B — the retrained critic `V(s)` (`mcts/value_net.py`)

**Supervision is identical to the MCSS critic's, keyed per-state.** The DV dataset already
computes a per-timestep discounted return-to-go `seq_val[p,t]` over normalised states
`seq_obs[p,t]`; the MCSS critic regresses `(whole trajectory → seq_val[p,start])`, the new
critic regresses `(seq_obs[p,start] → seq_val[p,start])` over the same `dataset.indices`.
Only the input representation changes, so the two critics are directly comparable.
Target config matches the pipeline exactly: `discount=1.0, continous_reward_at_done=True,
reward_tune="iql", center_mapping=True` ⇒ `seq_val` = normalised **negative time-to-
terminus** in [−1, 1] (1 = at a terminus; antmaze min raw return −867).

Architecture: MLP `obs_dim → [256]×3 (SiLU, LayerNorm) → 1`, 133k params (vs the
transformer trajectory critic) — cheap enough to score every tree node.

**Training** (`scripts/train_state_value.py`):

| env | steps | val_mse | val_corr | note |
|---|---|---|---|---|
| maze2d-large-v1 | 200k | 0.1455 | 0.759 | plateau by ~30k; train≈val (no overfit) |
| antmaze-large-diverse-v2 | 1M | 0.0405 | 0.809 | train_mse→0.0000 = memorised; 50k is plenty |

The ~0.76 maze2d ceiling is label noise, not underfitting: targets are *behaviour-policy*
return-to-go (the data wanders to random goals), so a state has intrinsically noisy labels.

**Offline selection check** (`scripts/eval_state_value.py`, K=50 plans/start, 5 seeds):
on maze2d-large, selecting plans by `V(endpoint)` exactly matched the MCSS critic
(3/5 vs 3/5, zero regressions) — the state-value head is a sound drop-in selector.
Selecting by `max_h V(traj_h)` was worse (2/5): **endpoint value, not best-state value**,
is the right plan score. On antmaze, `V(endpoint)` correlated ~0 with goal-closeness
(per-seed corr 0.07, −0.08, −0.27, 0.03, 0.01) — the value is **goal-agnostic** (§5).

## 3. The closed-loop correction (a methodology lesson)

Early diagnostics in this project were **single-shot**: "from a standing start, does any of
K full plans reach the goal?" By that metric maze2d-large seeds 0/2 looked unsolvable
(0/50 plans reach) and antmaze looked broken (0–7 of 50). Both conclusions were **wrong**,
because DV never executes a whole plan — it **replans every step** (MPC) and executes only
the first waypoint. Verified closed-loop ground truth from the DV pipeline logs:

| env | DV MCSS closed loop | implication |
|---|---|---|
| maze2d-large-v1 | 201.4 norm, **all 50 envs reach** | saturated; no headroom for any sampler |
| antmaze-large-diverse-v2 | **76.9 ± 1.3** (n=1000) | ~23% fail — the only real headroom |

Single-shot reach is *not* a proxy for closed-loop success: a goal-blind planner that never
one-shots a 41-unit goal still reaches 77% of the time by repeatedly taking locally good
steps through the maze's corridors. **All sampler comparisons below are closed-loop.**
The remaining ~23% MCSS failures are *wrong early turns* — greedy MPC commits to a
locally-plausible corridor and cannot take it back — precisely the failure mode the project
brief targets, and the right testbed for look-ahead.

## 4. Phase C — the MCTS sampler (`mcts/value_forest.py`, `mcts/mcts_loop.py`)

A forest of **M lockstep trees** (one per parallel env):

- **Expansion:** from a node's state, the DV planner (unchanged, start-state inpainting
  only) generates `k_mcts` candidate continuations; each child = the plan's state at
  `child_index` (default 1 waypoint = stride dense steps ahead); each child's prior =
  `V(child_state)`.
- **Selection:** UCB descent, `Q + c·sqrt(ln(N_parent+1)/(N_child+1))` — the (N+1)
  smoothing lets the value prior steer best-first instead of forcing every sibling open.
- **Backup: MAX, not mean.** An expanded node's value = max over its children — the best
  continuation *overrides* the prior up or down. A child whose first step looks mediocre
  but leads to a high-value region is rewarded; that is the look-ahead the brief asks for.
- **Action extraction:** the root child with the highest backed-up value; execute its
  *first* waypoint (one env step), then replan. Same cadence, normaliser, and inverse-
  dynamics policy as MCSS — the sampler is the only difference.
- **Parallelism:** all M trees expand in lockstep; each round gathers one leaf per tree and
  runs **one batched planner+value call** of shape `(M·k_mcts, H, obs_dim)`. A budget-B
  search costs B+1 batched calls per env step (not M·B separate calls). At matched
  candidates/step this makes MCTS wall-time ≈ MCSS wall-time (5579s vs 5896s below).

The tree bookkeeping is torch-free and unit-tested (8/8 in `tests/test_value_forest.py`,
covering selection, max-backup override in both directions, extraction, error paths).
Compute accounting: candidates/step = `k_mcts × (budget+1)` for MCTS, `k_mcss` for MCSS;
identical DDIM steps per candidate (20) and per-step policy cost (10) for both.

## 5. Phase E — closed-loop results

### 5.1 Preliminary sweep (n=25, single seed) — now superseded by §5.2

All preliminary cells: antmaze-large-diverse-v2, `n=25` rollouts each, **same seed 0**,
full 1000-step episodes, `k_mcts=16`, `child_index=1`, `c=√2`, run on the original
harness (async vector env, aggregate-only logging — the exact code diff vs the
confirmatory harness is recorded in `notes/harness_changelog.md`; the sampler, search,
and checkpoints are identical between the two).
Harness validation: this harness's MCSS k=50 measured **76.0%** (and 84.0% in an earlier
nominally-identical replicate — i.e. ±2 rollouts run-to-run), bracketing the pipeline's
**76.9%** — the loop faithfully reproduces DV's baseline.

| candidates/step | MCSS reach% | MCTS reach% | wall (s) MCSS / MCTS |
|---|---|---|---|
| 50  | 76.0 ± 8.5 | — | 1103 / — |
| 80  | — | 60.0 ± 9.8 (budget 4) | — / 1697 |
| 144 | 80.0 ± 8.0 (k=144) | 80.0 ± 8.0 (budget 8) | 3070 / 3030 |
| 272 | 84.0 ± 7.3 (k=272) | **96.0 ± 3.9 (budget 16)** | 5896 / 5579 |

maze2d-large sanity (n=12): MCSS 100%, MCTS 100% — the search does not break a solved
task. (The maze2d "norm −2.1" is a metric artifact: success is clipped to {0,1} before
`get_normalized_score`, which for maze2d expects raw return; **reach% is the metric**.)

**Preliminary findings (qualitative trends; absolute levels superseded by §5.2):**

1. **Look-ahead wins at matched compute** (96 vs 84 at 272 cand/step, comparable wall
   time) — confirmed at scale with the gap essentially unchanged (+11.3 pp, §5.2),
   though both absolute levels proved draw-optimistic. The win is *not* candidate
   count: MCSS saturates (76→80→84 for 5.4× compute — consistent with the Phase-0
   finding that K saturates ≈50), while the same candidates organised as a tree keep
   paying (60→80→96).
2. **Budget monotonicity.** Three ordered points (60 → 80 → 96) on one scenario set —
   evidence that *depth of search*, not sampling volume, drives the gain. (Measured at
   n=25 on a single scenario set; the trend, not the levels, is the finding. A re-run
   of b4/b8 on the confirmatory harness is the planned follow-up.)
3. **Shallow search is worse than greedy** (budget 4: 60% < 76%). Mechanism (hypothesis):
   at low budget the decision rests on 1-ply `V` estimates of states one waypoint ahead —
   noisier than the trajectory critic's whole-plan score. Only with enough depth does
   composed look-ahead overcome the noisier per-state value. Practical rule: **the tree
   must be given enough budget to out-think the one-shot critic it replaced.**
4. **Why a goal-agnostic value supports look-ahead at all** (correcting an earlier
   prediction in `mcts_sampler_design.md`): `V` = time-to-*some*-terminus carries
   directional signal through maze geometry — dead ends and backtracks have poor
   time-to-terminus, so deeper search prunes exactly the wrong-early-turn corridors that
   sink greedy MPC. Blind-but-deep beats sighted-but-greedy on this map.

### 5.2 Confirmatory scale-up (paired n=150, seeds 0–2) — the headline

Design: the two headline arms only — MCSS k=272 vs MCTS budget=16, k_mcts=16 (both 272
candidates/step) — at `n=50` envs per seed, seeds {0, 1, 2}, full episodes, on the
instrumented harness (`mcts/mcts_loop.py`): every rollout logs success, first-reach
step, start, and goal, and the two arms of each seed see the **same 50 scenarios**
(per-index goal identity is machine-verified by `scripts/collate_mcts.py` before any
test is computed). The right statistic is then exact McNemar on the discordant pairs:
**fixes** = scenarios MCSS missed but MCTS reached; **breaks** = the reverse. Wall time
is matched by construction (11.1–11.5 ks per cell, both arms).

| seed | MCSS k272 | MCTS b16 | Δ (pp) | fixes (of MCSS misses) | breaks (of MCSS successes) | exact p |
|---|---|---|---|---|---|---|
| 0 | 64.0 | 86.0 | +22.0 | 16/18 (89%) | 5/32 (16%) | 0.027 |
| 1 | 82.0 | 80.0 | −2.0 | 5/9 (56%) | 6/41 (15%) | 1.00 |
| 2 | 70.0 | 84.0 | +14.0 | 12/15 (80%) | 5/35 (14%) | 0.14 |
| **pooled** | **72.0 ± 3.7** | **83.3 ± 3.0** | **+11.3** | **33/42 (79%)** | **16/108 (15%)** | **0.021** |

**Confirmatory findings:**

1. **The headline replicates and is now significant.** +11.3 pp at matched compute,
   exact McNemar p = 0.021 over 150 scenario-paired rollouts — the preliminary +12 pp
   effect size was accurate; only the absolute levels (84/96) were optimistic.
2. **Heterogeneity: the gain is monotone in scenario-set hardness.** Where greedy
   struggles, look-ahead pays most (+22 pp at MCSS=64%); where greedy is near its
   ceiling there is little left to fix and the gap closes (−2 pp at MCSS=82%, exact
   p = 1.0 — a tie, one discordant env of margin). The per-seed spread (64→82% MCSS on
   identical configs) is scenario-set variance, and is itself the explanation for the
   preliminary cells' optimism.
3. **Break-rate mechanism.** The fix and break rates decompose the net gain:
   `net = fix_rate × misses − break_rate × successes`. The fix rate is high (79%
   pooled, 89% on the hardest set) while the break rate is **roughly constant at
   ~15%** across all three seeds (16/15/14%) — the search occasionally wanders off a
   scenario greedy would solve, plausibly because the goal-agnostic `V(s)` lets a deep
   branch toward *some* efficient corridor outscore the corridor toward *today's*
   goal. This *motivated* the goal-conditioned value experiment (§6) — whose job was to
   cut the 15% break rate by giving the search an explicit direction. **§6 reports that
   it did not: `V(s, g)` did not improve closed-loop reach, and the break rate did not
   fall.** The break rate is a real cost, but goal-conditioning is not its cure.
4. **Old-vs-new reconciliation (no harness effect).** Both arms moved down together
   from the preliminary cells (MCSS 84→72.0, MCTS 96→83.3; z = 1.14 and 1.66 — within
   sampling noise) while the gap was preserved — the signature of an easier preliminary
   scenario draw plus n=25 noise, not of a code change. The only behaviour-relevant
   harness difference (async → sync vector env, which changes the realized scenario
   draw but not the dynamics) and the verbatim old evaluation loop are recorded in
   `notes/harness_changelog.md`.

### 5.3 Full-grid ablations at scale (paired n=150, the compute-and-structure picture)

Re-running the budget and candidate-count controls on the confirmatory harness (the
§5.1 follow-up), all paired against the same scenarios, gives the complete picture of how
inference compute is best spent. Pooled reach% (3 seeds × n=50):

| arm | cand/step | reach% | wall (ks)/seed | vs cheap k50 (paired p) |
|---|---|---|---|---|
| MCSS k=50  |  50 | **79.3** | 2.1 | — (the cheap baseline) |
| MCSS k=272 | 272 | **72.0** | 11.2 | −7.3 pp (p = 0.18) |
| MCTS b4    |  80 | 78.0 | 3.4 | −1.3 pp (p = 0.89) |
| MCTS b8    | 144 | 78.7 | 6.0 | −0.7 pp (p = 1.00) |
| MCTS b16   | 272 | **83.3** | 11.1 | +4.0 pp (p = 0.44) |

**Findings:**

1. **Flat candidate-scaling backfires (the sharpest negative).** MCSS *degrades* from 79.3%
   at k=50 to 72.0% at k=272 (−7.3 pp, paired; down on 2 of 3 seeds). Argmax over 272 critic
   scores selects the most *over-estimated* plan more reliably than argmax over 50 — the
   optimizer's curse / critic over-exploitation, measured cleanly on paired scenarios. The
   preliminary "MCSS saturates upward (76→80→84)" was n=25 noise; at scale the curve bends
   *down*. This is the control that makes the headline a *structure* result: the same 272
   candidates organised as a tree reach 83.3% while flat argmax over them reaches 72.0%.
2. **Budget curve is mildly monotone** (b4 78.0 → b8 78.7 → b16 83.3; b4→b16 paired
   p = 0.31). Depth helps, but the gain is concentrated at b16 and is not individually
   significant — the matched-compute contrast vs k272 (§5.2) is where the significance lives.
3. **The honest ceiling.** The best search config (b16, 83.3%) beats the *cheap* k=50
   baseline by only +4.0 pp (p = 0.44, n.s.). antmaze-large-diverse is near-saturated for
   this planner: the rigorous, significant result is the matched-compute contrast
   (tree vs flat-at-272), not a large absolute gain over the cheapest sampler.
4. **Decomposition of the matched-compute win.** Of b16's 33 fixes over k272, 26 are
   scenarios k50 already solved (k272's self-inflicted damage *undone* by the tree), and
   only 7 are genuinely greedy-hard (missed by both MCSS configs). The greedy-hard core is
   small — 9/150 (6%) — and the tree cracks most of it (b4 solves 9/9, b16 7/9). So the
   tree's value is *both* avoiding the flat-argmax over-exploitation *and* extending
   capability into the small hard core.
5. **Child-index (segment length) — far branching hurts** (seed 0, n=50): b16 with the tree
   branching `child_index` waypoints ahead gives L1 = 86%, L2 = 78%, L4 = 74% — monotone
   *down*. Realized tree depth is ~2.5–3.0 regardless of L (so L multiplies the seen horizon:
   ~70 / ~125 / ~260 dense steps), and the wins on the greedy-hard core are *preserved*
   (L4 solves 3/3) while losses on solid-greedy scenarios *double*. With a goal-agnostic
   value, the discriminative signal decays with distance (far out, many corridors look
   alike) and the max-backup amplifies the wider, noisier spread of far-state values —
   so seeing farther means committing harder to confidently-wrong distant regions.
   **`child_index = 1` (fine-grained branching) is optimal**, surviving the engine change
   from the Phase-3/4 trajectory critic to the state-value tree.

**Synthesis (§5.2 + §5.3).** *Search structure determines whether imagination compute helps
or hurts.* Scaling a diffusion planner's candidate count under greedy argmax backfires
(−7.3 pp); organising the identical compute as fine-grained (`L=1`), adequately-deep
(budget 16) value-guided look-ahead scales positively and beats the candidate-matched flat
control by +11.3 pp (p = 0.021). The remaining question — whether a *better-informed* value
could push past the goal-agnostic `V(s)` — is §6.

## 6. The goal-conditioned value `V(s, g)`: a tested negative result

The §5.2 break-rate analysis suggested the search wanders because `V(s)` is goal-blind: it
knows *how far some terminus is*, not *where today's goal is* (corr ≈ 0 between `V(endpoint)`
and goal-closeness on antmaze, §2). The natural fix is a goal-conditioned value `V(s, g)`,
giving look-ahead an explicit direction. We built it, validated it offline, and ran it
closed-loop. **It did not help** — a negative result, but a *well-characterised* one: the
value is sound; goal-*targeting* is simply the wrong objective for this MPC search.

### 6.1 Construction (planner/policy untouched; only the critic is new)

- **Goal relabelling.** Each dataset trajectory's future states relabel its earlier states:
  for `(p, t)` sample a goal index `t' ≥ t` in the same trajectory; input `(seq_obs[p,t],
  g = xy of seq_obs[p,t'])`, target `−(t'−t)` on the *identical* pipeline affine as `V(s)`
  (so `target(t'=t) = 1.0` exactly, on one scale with the planner-endpoint values).
  Mixture 70 % future-state / 20 % terminus / 10 % current-state goals (the last pins the
  zero point). `mcts/relabel.py`, `mcts/value_scale.py`.
- **Expectile loss τ = 0.9, not MSE.** Diverse-data trajectories wander, so the relabelled
  times have large spread; MSE regresses to the *mean* wandering time — biasing exactly the
  child-ranking the tree performs. Expectile on the same exact MC targets approximates the
  min-time over in-support behaviour (the discount-1.0 analog of IQL's expectile).
- **Ensemble of 5** with `min`-pessimism at inference (epistemic guard; orthogonal to the
  expectile's aleatoric job). `mcts/value_net.py:DVStateValueEnsemble`.
- **Goal at inference** is the env's real target, normalised by the state normaliser's xy
  statistics through the single shared `normalize_goal_xy` helper — identical to training.

### 6.2 Three diagnostics, run before any closed-loop rollout

- **Data-efficiency surprise — more relabelling data *hurts*.** The DV dataset
  (`learn_policy=False`) keeps only the 106 terminus-reaching trajectories; relabelling does
  not need them, so we also trained on the ~893 timeout (goal-failing) trajectories
  (`--full-data`). It made the critic *worse*: val_corr 0.905 → 0.545, far-zone calibration
  collapsing (D4 bias −342 steps at 400–700, −580 at 700+). The timeout trajectories are
  *wanderers*, and MC's target is *behavioural* time-to-go — a biased geodesic estimator that
  more wandering data only degrades (steps-per-cell jumps 35 → 74, the wandering signature).
  Data *quality* beat quantity; the terminus-only critic (val_corr 0.905 @ 36 k, best-val)
  is the one carried forward.
- **D1 — compass resolution, connectivity-stratified.** Using a dev-only BFS oracle to split
  query pairs into *coverable* (some single trajectory connects `s` and `g`) vs *stitched*
  (none does), `V(s, g)` **resolves the deployment goal** — the eval corner — at ~3-cell
  granularity (0.94 ranking accuracy, beating `V(s)`'s 0.91), but **fails on generic
  stitched pairs** (0.73–0.78, worse than `V(s)`), the textbook limit of MC relabelling: it
  learns the data manifold and does not extrapolate across trajectory boundaries.
- **D2 — exploitability on planner-generated states (the decisive one).** The tree scores
  *planner-imagined* states, not dataset states, so the worry was that the goal channel
  assigns spurious closeness to hallucinations and the max-backup commits to them. **D2
  refutes this.** Of 15 k generated chunks only 1.3 % are physically invalid (in-wall /
  unreachable); among the top-5 % the tree would pick, `V(s, g)` is *less* invalid than the
  pool (enrichment 0.40×, vs `V(s)` 1.00×) — it *avoids* hallucinations. And it is
  well-calibrated off-manifold: predicted-vs-oracle-geodesic MAE is 27 steps on both real and
  generated states (gap +0), with a slightly *conservative* bias. **The value is sound.**

### 6.3 Closed-loop verdict (paired n=150, vs the `V(s)` tree)

| | MCTS+V(s) (b16) | MCTS+V(s,g) (b16-sgP) | Δ | exact p |
|---|---|---|---|---|
| seed 0 | 86.0 | 72.0 | −14.0 | — |
| seed 1 | 80.0 | 80.0 | 0.0 | — |
| seed 2 | 84.0 | 78.0 | −6.0 | — |
| **pooled** | **83.3** | **76.7** | **−6.6** | **0.22** |

`V(s, g)` never beats `V(s)` on any seed (fixes 22 / breaks 32 pooled); it also ties the
cheap k=50 (79.3) and beats only the flat k272 (+4.7 pp, n.s.). The underperformance is mild
and not individually significant (p = 0.22), but it is *consistent* across three independent
scenario sets — **not** a seed-0 fluke, and decisively **no improvement** over the
goal-agnostic value.

### 6.4 Why a *sound* value does not help — and why neither escalation is indicated

The reconciliation is the finding. D2 shows `V(s, g)` is well-calibrated and non-exploitable,
so this is not a bad value — **goal-targeting is the wrong objective for this search**. Under
MPC replanning with a goal-agnostic planner, the tree needs the best *next step*, and
**robust local progress** — what `V(s)` supplies by riding maze geometry ("am I advancing
through a corridor") — is a more robust next-step criterion than **greedy global goal-
targeting** (`V(s, g)`'s corner-distance). The agent re-observes the true state every step,
so accurate long-range corner-direction (which D2 confirms `V(s, g)` has) buys nothing that
geometry-progress does not already give, while mildly over-committing toward the corner. This
is the project's MPC theme at the value level: *robust-local beats greedy-global when you
replan every step.*

Consequently **neither pre-registered escalation is warranted**, and the diagnostics say so:

- **A feasibility gate** (discard off-manifold chunks before scoring) targets exploitation —
  but D2 found almost nothing to gate (1.3 % invalid, and `V(s, g)` already avoids it).
- **IQL-u** (TD-bootstrapped value) targets weak off-manifold calibration and stitching — but
  D2 shows the calibration is *fine*, and `V(s)` proves stitching is not the closed-loop
  bottleneck (it also cannot stitch, yet wins). The pre-registered "v_sg ≤ v_s ⇒ IQL-u"
  trigger *assumed* a weak value; D2 falsified that premise, so following it would build
  machinery to fix a problem that is not the cause. (This is precisely what the
  diagnostic-before-escalation discipline is for — D2 earned its keep by preventing it.)

**One untested variant**, left as future work: a *combined* `V(s) + λ·V(s, g)` (geometry for
robustness, goal as a tie-breaker). `V(s, g)` *alone* shows no signal to build on, so this is
speculative rather than indicated.

## 6′. Superseded hypothesis (kept for the record)

The original pre-registration predicted, in order of confidence, that `V(s, g)` would
(i) cut the ~15 % break rate below 15 %, (ii) improve the budget-4/8 cells most (direction
compensating for shallow depth), and (iii) lift even MCSS when re-ranked by `V(s_end, g)`.
The break-rate prediction was the headline target and the directly measurable one. **None
held**: the break rate rose (32 vs 16 over the `V(s)` baseline, §6.3), and closed-loop reach
did not improve. The diagnosis (§6.4) — that the value is *sound* but goal-targeting is the
wrong objective under MPC — is the correction. The mechanistic prediction in
`mcts_sampler_design.md` (that goal-blindness *caps* the search) is likewise overturned:
goal information was *added* and the search did not improve, so goal-blindness was not the
cap. The cap, to the extent there is one, is the near-saturation of the env (§5.3) and the
inherent quality of the planner/policy, not the value's goal-awareness.

## 7. Failure attribution — is the critic the lever? (the oracle counterfactual)

§5–§6 measured what helps closed-loop; this section asks the dual question: of MCSS's
~22% failures, how many are the *critic's* to fix, and what caps the rest? A failure-
instrumentation harness (`mcts/instrument.py`; torch-free analysis in
`mcts/failure_modes.py`; Rule-1 dev-only oracle) logs, per failed rollout, the executed
path, body-state (torso height/uprightness/speed), the per-step BFS-geodesic distance to
goal, and the full 50-candidate pool. Run on the instrumented MCSS baseline
(3 seeds × n=50, **78.0% reach, 33/150 fail**; the planner draw is torch-seeded here for a
reproducible critic↔oracle comparison).

### 7.1 Shape attribution fails; the oracle *is* the attribution (a methodology lesson)

The obvious approach — classify each failure by trajectory shape (wrong-turn / fell-over /
timeout / no-plan) and call the wrong-turns "critic-fixable" — does **not** work, and the
data shows why. A shape classifier labelled **all 33** failures `FELL_OVER`: the Ant is
genuinely tilted/stalled at the end of a failed episode. But that same end-of-episode
collapse appears in failures a better ranker *fixes* and ones it *cannot* — it is a
**symptom** (downstream of an earlier ranking error that steered the ant into trouble), not
an independent execution cause. No pose/shape threshold separates "fell from bad luck" from
"fell because ranking chose a bad corridor"; only a **counterfactual** can. (Two recalibration
passes confirmed this: tuning thresholds moved the classifier from 0% to 100% `FELL_OVER`
without ever tracking the truth — see `notes/instrumentation.md`.)

That counterfactual is the **oracle-V** probe (Rule-1 dev-only, never reportable): re-rank
the *same* 50 planner candidates each step by true BFS-geodesic-to-goal instead of the DV
critic, keeping the planner and inverse-dynamics policy unchanged. "Oracle solves it" ≡ "a
better critic solves it" *by construction*. So attribution comes from the oracle, and the
shape modes are kept only as description (the analyzer reports them descriptively and flags,
via a mode×oracle cross-tab, that `FELL_OVER` splits across both outcomes).

### 7.2 The oracle re-rank, measured PAIRED: **flat** selection is saturated

⚠ An earlier draft of this section reported "+15.3 pp recoverable, ceiling 93.3%" from a
**fixes-only** count (oracle solves 23 of the critic's 33 failures). That was wrong — it
ignored the **breaks**. Measured paired across all 150 scenarios (the oracle run's own reach,
fixes *and* breaks netted):

| sampler (same planner + policy + candidates) | reach | paired vs critic |
|---|---|---|
| MCSS (DV trajectory critic) | 78.0% | — |
| **oracle-V** (true-geodesic ranking over the same 50 candidates) | **78.7%** | fixes 23 / **breaks 22** / **net +0.7 pp** |

**A *perfect* goal-distance ranker nets essentially zero (+0.7 pp, breaks/fixes = 0.96).** It
reshuffles *which* 23 scenarios are solved while breaking 22 others — the exact unpaired-noise
trap as the §5.2 n=500 result (a different but equally-good rollout, not a better one). The
"23 fixes" were never free; counting them without the breaks inflated the ceiling by the full
amount. **Ranking the candidates is not where the headroom is.**

### 7.3 Triangulation — the DV critic already ranks about as well as the geodesic

Two independent checks (`scripts/plot_candidates.py`, `scripts/diag_wall_blindness.py`) agree:

- **Candidate gap ≈ 0.** On every failed-episode decision, the critic's chosen candidate's
  endpoint is at (or within <2 cells of) the geodesically-closest endpoint available
  (mis-rank rate 0%, mean gap 0.0). The critic's *top* pick already matches the oracle's — the
  DV return-to-go is strongly anti-correlated with endpoint distance — which is why the two
  picks coincide in the cloud panels. (Per-decision Spearman of the *full* 50-way order is only
  −0.23, i.e. the tail order is noisy, but only the top pick is executed, and it is fine.)
- **`V(s, g)` tracks the geodesic well on average, but has LOCALIZED wall-blind spots.**
  `corr(value, geodesic) = −0.945` on dataset states and the *global* detour-stratified
  over-optimism gap is only −0.05 — but those are averages, and the per-cell error map shows
  real **over-optimistic (red) regions** (e.g. behind walls near the start corridor and a
  top-right pocket) alongside over-*conservative* (blue) cells that cancel them in the mean.
  So the value is *not* uniformly wall-blind, but it is locally wrong exactly where it must
  extrapolate across an un-traversed wall (the D1 stitched-pair limit, made spatial). Those
  local errors are precisely what could mis-guide selection at specific states — i.e. the
  value is *not accurate enough to trust point-for-point*, even if its global ranking is good.

### 7.4 What is and is **not** shown — flat selection vs structured search

Be careful about the scope of the oracle result. What it shows:

- **FLAT selection is saturated.** A perfect geodesic ranker, picking the closest-endpoint
  candidate each step (MCSS-with-a-perfect-critic), nets ~0 (§7.2), and the DV critic already
  picks the geodesic-best endpoint (§7.3). So no *flat best-of-N* re-ranking — learned or
  perfect — beats ~78%.

What it does **not** show (open questions, not conclusions):

- **It does NOT show structured search can't help.** The oracle test is flat; MCTS instead
  looks ahead over *intermediate* states and backs up, so it can prefer a first step that
  *leads* to a reliably-reachable region rather than greedily chasing the nearest endpoint —
  exactly the project brief. The MCTS results so far (§5 tree ≈79% with goal-*agnostic* `V(s)`;
  §6 `V(s, g)` ≈77% with the *imperfect* learned value) used values that are not accurate
  point-for-point (§7.3), so they do **not** bound what a tree with an *accurate* goal-value
  could do. That combination is untested.
- **It does NOT close the `V(s, g)` line.** The right test is `V(s, g)` (or the geodesic) used
  *in the tree*, not as a greedy ranker. §7.5 runs that with a perfect value to find the ceiling.
- **The task is solvable from this start/goal.** antmaze-large-diverse has a near-fixed
  start→far-corner, and ~78% of stochastic rollouts succeed, so the ~22% failures are *unlucky*
  (tip-over / wander / get-stuck), not structurally impossible — precisely the regime look-ahead
  is meant to rescue. "Near-saturated" is only established for *flat* selection.

**Methodology lesson (twice now): attribute by the PAIRED counterfactual, never by fixes
alone.** And: a *flat*-selection probe bounds *flat* selection only — do not generalise it to
search.

### 7.5 Open test — the attribution ladder (`scripts/diag_oracle_tree.py` + `collate_mcts.py`)

The decisive experiment runs the same MCTS forest but scores tree children by the **true BFS
geodesic** (`value_mode="oracle"`, Rule-1 dev-only; mapped onto the learned-value `[−1,1]` scale
so `c_ucb` carries, off-graph children → −1). To avoid the trap that just bit the flat oracle —
ambiguous net, no significance, wrong baseline — it is **not** scored ad-hoc; it emits a
`collate_mcts`-compatible JSON and is read as a **four-arm ladder**, each rung isolating one
factor, with `collate_mcts` doing the **goal-verified, exact-McNemar (per-seed + pooled)**
comparison (matched compute among the three trees):

| rung (baseline → treatment) | isolates | compute |
|---|---|---|
| `k50 → b16` | flat selection → structured tree | 50 → 272 cand/step |
| `b16 → b16sgP` | goal-conditioning the tree value | matched (272) |
| **`b16sgP → b16orc`** | **value ACCURACY** (learned `V(s, g)` → perfect geodesic) | matched (272) |

The **`b16sgP → b16orc`** rung is the one that answers "does an *accurate* value inside the tree
help" — both are goal-aware trees at the same compute; only the value's accuracy differs.

- **pooled net > 0, significant** ⇒ structured search + an accurate value *is* a lever — the job
  becomes training `V(s, g)` toward geodesic accuracy on planner-reachable states (§7.3).
- **pooled net ≈ 0 (n.s.)** ⇒ endpoint-geodesic look-ahead cannot beat ~78%, and *then* the
  saturation claim extends to (this form of) search.

**Read the pooled exact-p, not the raw net** (a +5 net over ~45 discordant pairs is p ≈ 0.5 — the
n=500 lesson). **Two interpretation caveats (F4):** (i) the geodesic value also *gates* off-graph
children (→ −1), so a positive result motivates value accuracy **and** a feasibility gate, not
value alone; (ii) it scores the *endpoint* geodesic, not 25-step *segment* feasibility, so a null
bounds endpoint-lookahead only — a segment-feasible value stays untested.

**RESULT (3 seeds, n=150, all pairs n.s.):** `b16` (V(s)) 83.3 → `b16-orc` 82.0, **net −1.3 pp,
p = 0.89**; `b16-sgP` (V(s,g)) 76.7 → `b16-orc` 82.0, **+5.3 pp, p = 0.32**. So a **perfect value
in the tree only matches the goal-agnostic V(s)** — value accuracy is **not** the lever at this
tree config — while it does beat the *learned* `V(s, g)` (confirming that value was suboptimal,
but fixing it only reaches V(s) level). Everything sits in a 76–83% cluster.

**Implication and next test (§7.6).** The cap at ~82% is therefore **not in the value**. Either
the **tree structure** is limiting (realized depth only ~2.5–3 even at budget 16 — shallow-and-
broad, redundant with the planner's full-H ply-1 look-ahead, §1), or the cap is below the search
layer (planner can't propose / policy tips the Ant). This is **not** pre-judged: holding the value
*perfect* (oracle), we sweep the tree itself — `child_index` first (the matched-compute knob; with
the perfect value, L>1 may finally pay where it hurt with the noisy V(s)), then `c_ucb`/budget.
If **no** oracle-tree config clears ~82% → the cap is execution/proposal (policy/planner), and the
sampler is genuinely saturated; if one does → the tree *was* the limit, and the job is to optimise
the search (then train a value to feed it). `scripts/diag_oracle_tree.py --child-index {1,2,4}`.

⚠ Rule-1: the `orc` arm is privileged/diagnostic and must never appear in a reported results
table as achievable; it is a ceiling probe, not a sampler.

### 7.6 RESULT — the cap is below the sampler: the Ant's locomotion (CLOSED)

§7.5 left two candidates for the ~82 % cap: the tree structure, or the execution/proposal
layer. The execution layer is now established as the cap, by a converging set of probes
(`scripts/diag_oracle_flat.py`, `scripts/diag_fall_geometry.py`; all Rule-1, seed 0, n=50):

1. **The failures are 100 % physical topples.** Every failed rollout ends with the Ant tilted
   and motionless (uprightness ≈ −0.91, torso height ≈ 0.26, planar speed ≈ 0.005), not lost or
   off-route. Fall rate *scales with selection aggressiveness* (oracle 20 % / critic 22 % / fs2
   24 % / fsf 28 %) — bolder plans tip more.
2. **Every execution-aware selector is null or negative.** Holding the value *perfect* (geodesic)
   and changing only *which* goalward candidate to take: `stbU` (most-upright) 80 % — and the
   planner predicts **all** plans upright (candidate spread 0.038, the orientation channel is
   empty); `stbD` (gentlest) 0 % (degenerate creep/oscillation); `stbA` (smoothest) 80 %; `gnt`
   (drop the biggest-displacement lunges) 64–84 % non-monotone **noise** (all McNemar p ≥ 0.06);
   `smt` (cap the commanded turn 20–120°) 64–74 %, **all ≤ baseline**.
3. **Fall-geometry refutes the wall hypothesis and demotes the turn one to a symptom.** Wall
   clearance at the topple is normal (onset/baseline ratio 0.90–1.16); falls do **not** cluster
   near walls. The sharpest commanded turn entering a stall is 130–170° (a near-reversal) vs a
   ~50° moving baseline — a real correlation — **but** forbidding those turns (`smt`) does not
   reduce topples (it trades them for creep), so the near-reversal is *downstream* of the ant
   already destabilising, not the cause.

**Mechanism and closure.** The diffusion planner is a *trajectory prior*: trained on upright,
successful data, it only ever imagines upright futures (hence spread 0.038), so it structurally
**cannot foresee that executing a given waypoint topples the Ant**. Therefore no plan-space
method — flat selection (§7.2), structured search (§7.5), or any value, learned or perfect —
can avoid the falls, because none of the imagined candidates predicts falling. The ~80–83 %
cluster across *all* samplers and values is the signature of a cap **below the sampler**: the
**DV inverse-dynamics policy tipping the Ant** during low-level control. This closes the §7.5
open test in the *execution* branch (not the tree-structure branch): the sampler is genuinely
saturated on antmaze-large because the residual headroom is locomotion, not planning.

**The one remaining lever (different thesis).** Only a **forward dynamics / world model** that
predicts falls-after-action could push past this — a privileged true-simulator 1-step lookahead
is the clean ceiling test ("are the falls avoidable with foresight at all?"), a learned
fall-model the deployable version. That is model-based RL, a scoping decision for the supervisor,
not a sampler change. See [findings_summary.md](findings_summary.md) §4–§6 for the
progression-report consolidation and the locomotion-confound critique (the case for moving the
*planning* claim to a non-falling, headroom-bearing benchmark — the OGBench pivot).

## 8. Artifact map

| Artifact | Role |
|---|---|
| `mcts/instrument.py` | Tier-0/2 traced MCSS rollout (body-state, candidate pool, BFS distance) + oracle-V re-rank (§7) |
| `mcts/failure_modes.py` | torch-free failure classifier (descriptive); 31 unit tests |
| `scripts/run_instrumentation.py`, `scripts/analyze_failures.py`, `scripts/plot_failures.py` | run the trace, oracle-attribution + immune-set evidence, per-episode figures (§7) |
| `scripts/plot_candidates.py` | Tier-3: 50-candidate dispersion + critic-vs-oracle pick + per-decision Spearman / mis-rank rate (§7.3: critic already picks the geodesic-best) |
| `scripts/diag_wall_blindness.py` | `V(s,g)` vs true geodesic — corr −0.945 global but localized over-optimistic (wall-blind) cells (§7.3) |
| `scripts/diag_oracle_tree.py` | **§7.5 open test**: MCTS with the true geodesic as the tree value (Rule-1) — does structured look-ahead with a perfect value beat ~78%, measured paired |
| `notes/instrumentation.md` | the §7 runbook, Rule-1 firewall, and the shape-attribution post-mortem |
| `mcts/value_net.py` | `DVStateValue` MLP `V(s)` + loader |
| `scripts/train_state_value.py` | trains `V(s)` on the per-state DV target |
| `scripts/eval_state_value.py` | offline selector comparison (mcss / v_end / v_max) |
| `mcts/value_forest.py` | torch-free lockstep forest; max-backup; 8/8 tests |
| `mcts/mcts_loop.py` | shared closed-loop harness; per-rollout success/goal logging |
| `mcts/specs.py` | shared env-family constants (single source of truth) |
| `scripts/run_mcts_compare.py` | CLI; `--value-mode {v_s,v_sg,v_sg_pess}`; `--out` JSON (ckpt + commit + value/gate metadata) |
| `scripts/collate_mcts.py` | candidates-vs-reach table + per-seed and **pooled** exact McNemar (canonical baseline direction) |
| `mcts/value_net.py:DVStateValueEnsemble` | goal-conditioned `V(s,g)` ensemble + `min`/`mean−βstd` pessimism (§6) |
| `scripts/train_state_value.py --goal-conditioned` | trains `V(s,g)` (relabelling, expectile, ensemble; `--full-data` for the data-efficiency ablation) |
| `mcts/relabel.py`, `mcts/value_scale.py`, `mcts/coverage.py`, `mcts/maze_oracle.py` | relabelling mixture, shared affine, connectivity stratum, dev-only BFS oracle |
| `scripts/diag_d1_compass.py`, `diag_d4_calibration.py`, `diag_d2_exploitability.py` | the §6.2 diagnostics (compass / calibration / exploitability) |
| `notes/mcts_sampler_design.md` | design rationale + verified code facts |
| `notes/harness_changelog.md` | exact old-vs-new harness diff; verbatim old eval loop |
| `results/mcss_antmaze_*.json`, `results/mcts_antmaze_b*.json` | §5.1 preliminary cells (aggregates only) |
| `results/scale_mcss_k{50,272}_s*.json`, `results/scale_mcts_b{4,8,16}_s*.json` | **§5.2/§5.3 grid cells** (per-rollout vectors) |
| `results/scale_mcts_b16sgP_s{0,1,2}.json` | **§6.3 V(s,g) cells**; `results/d{1,2,4}_*.json` the diagnostics |

**Reproduction — headline grid (≈3.1 h per cell on the eval GPU; ~19 h total):**
```bash
for s in 0 1 2; do
    python scripts/run_mcts_compare.py --env antmaze-large-diverse-v2 --method mcss \
        --n-envs 50 --n-episodes 1 --seed "$s" --k-mcss 272 \
        --out "results/scale_mcss_k272_s${s}.json"
    python scripts/run_mcts_compare.py --env antmaze-large-diverse-v2 --method mcts \
        --n-envs 50 --n-episodes 1 --seed "$s" --budget 16 --k-mcts 16 \
        --out "results/scale_mcts_b16_s${s}.json"
done
python scripts/collate_mcts.py results/scale_*.json   # per-seed + pooled McNemar
```
(The §5.1 preliminary cells used n=25, seed 0, and the pre-changelog harness; their
exact scenario draws are unrecoverable — see `notes/harness_changelog.md`.)

**Reproduction — V(s,g) arm (§6):**
```bash
python scripts/train_state_value.py --env antmaze-large-diverse-v2 \
    --goal-conditioned --ensemble 5 --loss expectile --tau 0.9      # -> *_best.pt
python scripts/diag_d1_compass.py      --env antmaze-large-diverse-v2 --sg-ckpt state_value_sg_ckpt_best.pt
python scripts/diag_d2_exploitability.py --env antmaze-large-diverse-v2 --sg-ckpt state_value_sg_ckpt_best.pt
for s in 0 1 2; do
    python scripts/run_mcts_compare.py --env antmaze-large-diverse-v2 --method mcts \
        --n-envs 50 --n-episodes 1 --seed "$s" --budget 16 --k-mcts 16 \
        --value-mode v_sg_pess --sg-ckpt state_value_sg_ckpt_best.pt \
        --out "results/scale_mcts_b16sgP_s${s}.json"
done
python scripts/collate_mcts.py results/scale_*.json   # b16->b16-sgP pooled McNemar
```

**Reproduction — failure attribution / oracle ceiling (§7; ~0.6 h/seed, CPU analysis):**
```bash
python scripts/run_instrumentation.py --env antmaze-large-diverse-v2 \
    --seeds 0 1 2 --n-envs 50 --value-source both      # critic (Tier-0) + oracle (Tier-2)
python scripts/analyze_failures.py --in-dir results/instr --out results/instr/summary.json
python scripts/plot_failures.py    --in-dir results/instr --out-dir results/instr/figs
```

## 9. One-paragraph summary

DV's stock MCSS sampler picks the best of K imagined plans with a whole-trajectory critic —
a structure that cannot benefit from tree search, because its value does all its lookahead
at ply 1. Retraining the critic as a per-state value `V(s)` (same data, same target, input
changed from trajectory to state) enables a batched, max-backup MCTS that explores
continuations in parallel before committing to a step. Closed-loop on antmaze-large-diverse
— the one tested setting with real headroom (MCSS 76.9%) — the confirmatory grid (150
scenario-paired rollouts, 3 seeds, matched 272 candidates/step and equal wall time) gives
**MCTS 83.3% vs MCSS 72.0%: +11.3 pp, exact McNemar p = 0.021**. The result is a property
of the *search structure*, not the candidate budget: flat best-of-N scaling backfires
(MCSS k50→k272 = −7.3 pp, critic over-exploitation under argmax), while the identical
compute organised as a fine-grained (`L=1`), adequately-deep (budget 16) value-guided tree
scales positively. A ring of characterised negatives rules out the obvious alternatives and
sharpens the claim: goal-conditioning the value does **not** help (`V(s,g)` 76.7 % ≤ `V(s)`
83.3 %, p = 0.22) even though the goal-conditioned value is *sound* — well-calibrated and
non-exploitable on planner-generated states (D2) — because under MPC replanning *robust
local progress* (the goal-agnostic `V(s)` riding maze geometry) beats *greedy global goal-
targeting*; more relabelling data hurts (wandering-trajectory noise); deeper branching hurts
(value discrimination decays with distance); and the value cannot stitch off-manifold but
does not need to, since re-planning stitches implicitly. The honest ceiling is that the best
search config beats the *cheap* MCSS k=50 by only +4 pp (n.s.) on this near-saturated env —
so the rigorous contribution is the matched-compute, paired, mechanistically-explained
demonstration that **search structure, not sampling volume or value goal-awareness, is what
converts a diffusion planner's inference compute into closed-loop performance.**
*Two later results revise this (see the Update note above and §7): at n=500 the
matched-compute gap shrinks to +4.2 pp (n.s.) once the unseeded planner draw is accounted
for, so the durable claim is the **qualitative** one (flat best-of-N backfires; a tree
avoids that degradation); and a Rule-1 oracle-V probe, measured **paired**, shows that even a
**perfect** geodesic ranker that picks the closest-endpoint candidate each step nets only **+0.7 pp**
(oracle 78.7% vs 78.0%; fixes 23 / breaks 22) — so **flat best-of-N selection is saturated**
(the DV critic already picks the geodesic-best candidate). This bounds *flat* selection only:
whether *structured* MCTS look-ahead driven by an *accurate* goal-value beats ~78% is the open
test (§7.5). `V(s, g)` tracks the geodesic on average but has localized wall-blind spots, so it
is not yet accurate enough to feed a tree; and since the task is solvable from this near-fixed
start/goal (~78% of stochastic rollouts succeed), the ~22% are unlucky tip-over/wander/stuck
rollouts — the regime look-ahead is meant to rescue — not a proven dead end.*
