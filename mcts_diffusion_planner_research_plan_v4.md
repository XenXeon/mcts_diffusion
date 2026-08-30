# MCTS over diffusion trajectory chunks for sparse-reward long-horizon control
## Research plan v5.1 — D4RL rescope, after Round-5 audit

This document is the implementation spec. If a formula, gate, or rule is not written
here, assume it will not be in the code. Changes from v4/v5 are tracked in the
cumulative traceability table (Section 12).

**v5 rescope (2026-06-13, project decision):** D4RL only — no OGBench, no env or repo
migration, no MCTD-family same-config comparisons (their published numbers may be
cited as context). All v4 stages survive but are grounded in the existing, validated
DV/cleandiffuser stack: the frozen DV planner and inverse-dynamics policy are reused;
the build is the **goal-conditioned critic, the hardened judge, and the pipeline that
puts them under the existing tree** — completing the 2×2 mechanism experiment whose
naive column is already measured.

---

## 0. Phase-1 results (completed — the measured premises of this plan)

All on antmaze-large-diverse-v2, closed loop, n=150 scenario-paired rollouts
(3 seeds × 50; per-index goal identity machine-verified), matched compute where
stated. Full record: `notes/writeup_mcts_sampler.md`; harness provenance:
`notes/harness_changelog.md`.

| # | Finding | Number |
|---|---|---|
| P1 | Flat candidate-scaling under critic-argmax **backfires** (optimizer's curse / critic over-exploitation) | MCSS k50→k272: **−7.3 pp** (79.3→72.0, paired p=0.18) |
| P2 | The same compute organised as value-guided tree search scales positively | MCTS b4→b16: +5.3 pp (78.0→83.3) |
| P3 | At matched compute (272 cand/step) the divergence is significant | tree 83.3 vs flat 72.0: **+11.3 pp, exact McNemar p=0.021** |
| P4 | vs the best cheap baseline the gain is not yet established | b16 vs k50: +4.0 pp, p=0.44 |
| P5 | Goal-agnostic V(s) caps useful look-ahead at ~1 waypoint; far branching adds wandering, not capability | child-index L1/L2/L4 = 86/78/74% (seed 0); fixes constant, breaks double; realized tree depth ~2.5–3.0 throughout |
| P6 | Search fixes most genuinely greedy-hard scenarios but breaks ~15% of solved ones | fixes 33/42 (79%); breaks 16/108 (15%, constant across seeds); greedy-hard core = 9/150 (6%) |
| P7 | DV baseline reproduced; failures outside the hard core are largely stochastic per run | harness k50 = 79.3±3.3 vs published 76.9±1.3; miss-sets across configs ≈ independent |

P1 is the empirical case for the hardened judge (Stage 1). P5 is the empirical case
for the goal-conditioned critic (Stage 0) and pins the chunk-length question. P3/P4
define what remains to be shown: convert the matched-compute win into a win over the
best baseline by removing the two measured costs (P5 wandering, P6 break rate).

---

## 1. Headline claims, benchmarks, and rules of evidence

**H1 (headline, rescoped):** On D4RL antmaze, tree search over diffusion-generated
plan segments, scored by a hardened goal-conditioned value, (a) beats flat best-of-N
at matched per-decision compute, and (b) converts inference-compute scaling from
negative (P1) to positive — with the hardened tree beating the **best** flat
configuration, not merely the matched-compute one (the gap P4 leaves open).

**H2 (mechanism):** The improvement is attributable to the *search* and the *judge*
separately and in combination — established by the 2×2 in Stage 3, of which the naive
column is already measured (P1–P4).

**Benchmarks.**

| Tier | Task | Purpose |
|---|---|---|
| Primary | D4RL antmaze-large-diverse-v2 | All Phase-1 cells exist here; headroom measured (best arm 83.3%, ceiling ~94% after removing the stochastic-failure floor) |
| Secondary | D4RL antmaze-large-play-v2 | Same maze, different data distribution — one transfer row, run once at the end |
| Sanity | D4RL maze2d-large-v1 | Saturated; regression check only (never used to look for gains) |

**Comparison set:** DV/MCSS (k50 and k272, done), best-of-N with the hardened judge
(the BoN(hard) cell — the critical mechanism control), MCTS with naive value (done),
MCTS hardened. External numbers (MCTD family, Diffuser, DD, IQL) cited from
publications as context only — no same-config claim is made.

**Rules of evidence (binding):**
1. **Oracle discipline.** Maze geometry oracles (BFS geodesics, wall checks — adapt
   `BFSValue` from `scripts/phase6_stage0_oracle.py`) are development/diagnostic
   only. They never appear in any system whose numbers reach the results table, nor
   in training data of any learned component (including classifier negatives).
2. **Matched per-decision compute** for all search-vs-sampling comparisons: equal
   planner candidates/step (272 for headline cells) and identical DDIM steps;
   wall-clock per cell reported.
3. **Paired evaluation protocol** (replaces v4's bootstrap-over-seeds): n=150
   scenario-paired rollouts per cell (3 seeds × 50 envs, n_episodes=1), per-rollout
   `success`/`reach_step`/`starts`/`goals` vectors, per-index goal identity verified
   by `scripts/collate_mcts.py` before any test; exact McNemar on discordant pairs;
   binomial SEM error bars. New cells MUST reuse seeds {0,1,2} at n_envs=50 so they
   pair against all existing Phase-1 cells for free.
4. **Tuning:** hardened-judge knobs (gate threshold, pessimism form) tuned on
   held-out seed 3 (n=20 cells, ≤4 candidates per knob); never on seeds 0–2.
   Oracle-based proxies are development-only and disclosed.
5. Hardware: the existing eval GPU (272-cand cell ≈ 3.1 h / 50 episodes). Compute
   gates in Section 6 are stated for this box.

---

## 2. Conventions and scoring mathematics

**Frozen components (reused, never retrained):** DV planner (DiT1d, H=40 waypoints,
stride T=25 dense steps, DDIM 20, start-state inpainting) and DV inverse-dynamics
policy — both validated through every Phase-1 cell. The executor question (v4 §5) is
closed by P7: closed-loop tracking is demonstrably sufficient; π_lo is out of scope.

**Reward/served value scale.** The existing pipeline scale is kept for the drop-in
critic: normalised negative time-to-go in [−1, 1] (1 = at goal), min-max constants
from the DV dataset (`TARGET_CFG` in `mcts/specs.py`). The v4 u=(1−γ)V
parameterization and steps-space arithmetic are adopted **only if** the IQL-u upgrade
path triggers (Section 3); with MC targets at discount 1.0 the [−1,1] scale is
already order-1 and depth-compositional.

**Tree mechanics (existing engine `mcts/value_forest.py`, kept):** batched full-wave
expansion (k=16 children per selected leaf in one planner call — v4's burst widening,
already implemented), UCB selection with (N+1) smoothing, MAX backup, root action =
best child value, replan every step. Realized-depth logging is built in.

**Additions to the engine for the hardened arm (the only tree changes):**
- **Pessimistic value:** V_pess(s, g) = min over the K-member ensemble (ablation:
  mean − β·std).
- **Feasibility hard gate:** children whose generating segment fails the Stage-1
  classifier (min pairwise score over consecutive waypoints below threshold) are
  discarded before scoring and never enter the tree. No per-branch soft penalty in
  the headline (soft-gate λ ablation only).
- **Point-to-segment goal check:** for each consecutive waypoint pair of a candidate
  segment, min distance from the goal to the segment; if ≤ goal radius the child is
  terminal with value 1 (on-scale maximum). A 25-step hop can step clean over the
  goal radius; endpoint-only checks are forbidden.
- **Goal plumbing:** the per-env goal is already captured by the harness
  (`run_episodes` logs it); the Sampler passes it to the value call, normalised with
  the state normaliser's xy statistics.
- **Gate placement symmetry:** in the BoN(hard) arm the gate is
  **discard-before-rank**, mirroring the tree's discard-before-attach, so both
  arms rank over the same effective candidate distribution.
- **c_ucb:** the value range is unchanged ([−1,1], terminal=1.0 within range), so
  c=√2 carries over; if hardened tree cells underperform expectations, spot-check
  c on the seed-3 held-out budget before concluding anything about the value.

**Chunk length / child_index:** locked at L=1 for the V(s) platform (P5). After the
goal-conditioned critic passes D1, the **pre-registered interaction cell** re-runs
L=4 (b16, seed 0, n=50): if goal conditioning shrinks or flips the L4 penalty
(74% → ≥86%), the value-information-decay mechanism is confirmed causally.

---

## 3. Stage 0 — Goal-conditioned critic V(s, g) and diagnostics

**Step (a) — MC-relabelled V(s, g) (the build default; extends
`scripts/train_state_value.py`):**
- **Relabeling (within-trajectory, exact targets — no TD needed):** for valid
  (path p, time t), sample goal index t′ ≥ t within the same trajectory;
  input (seq_obs[p,t], goal = xy of seq_obs[p,t′]); target = −(t′−t) raw steps
  mapped through the **identical pipeline affine** (the same raw-steps→[−1,1]
  transform the V(s) targets used — do NOT refit min-max on the relabelled
  distribution; targets for offsets beyond the original range clip at −1). This
  makes target(t′=t) = 1.0 **exactly**, and therefore identical to the
  point-to-segment terminal-check value in §2 — terminal children and pinned
  zero-distance targets sit on one scale inside the same MAX backup (asserted in
  code). Mixture: 70% future-state goals (geometric over offsets, matching the
  deployment horizon distribution), 20% terminus goals (t′ = terminus index from
  `seq_tml`; note the dataset at learn_policy=False already excludes
  timeout-truncated paths, so every stored path has a true terminus — if the
  dataset config ever changes, truncation ends are still valid terminus goals
  with target −(T−t), no special handling), 10% current-state goals (t′ = t,
  target exactly 1.0 — pins the zero point).
- **Loss: expectile regression, τ = 0.9** (NOT MSE). Diverse-data trajectories
  wander: for a neighbourhood of (s, g) the relabelled times have large spread,
  and MSE converges to the *mean* wandering behaviour — biasing exactly the
  ranking the tree performs. Expectile on the same exact MC targets approximates
  min-time over in-support behaviour — the discount-1.0 analog of what IQL's
  expectile does — with no TD and no scale change. **Pre-registered training
  ablation:** MSE vs expectile τ∈{0.7, 0.9}, selected by D1/D4 on the path-level
  validation split. (Orthogonal to ensemble-min at inference: expectile addresses
  aleatoric behaviour spread; the ensemble addresses epistemic error.)
- Goal input = raw xy normalised by the state normaliser's dims [0:2] statistics.
- Architecture: `DVStateValue` widened to obs_dim+2 input; **ensemble of 5**
  (independent seeds/inits) saved as one checkpoint; loader returns the ensemble.
- Training: same optimiser/schedule as V(s); 200k steps; path-level val split;
  log val loss/val_corr per member + ensemble-min calibration.

**Step (b) — IQL-u upgrade (trigger pre-registered on the STITCHED stratum):** the
within-trajectory relabeling of step (a) cannot, by construction, produce any
(s, g) pair that no single dataset trajectory connects — and TD bootstrapping is
precisely the mechanism that repairs that regime. Therefore step (b) triggers when
**D1's stitched stratum** fails its gate, not only the overall Δ\*. Spec as v4 §3 —
u=(1−γ)V, expectile τ=0.9 (retune list τ∈{0.7,0.8,0.9}), n-step=5, termination
mask, γ ≥ 1−0.7/d_max with d_max from the BFS oracle (provisional γ=0.999; restate
after `scripts/measure_dmax.py`). The quasimetric parallel track is dropped at this
scope; contrastive distances remain the named contingency if (a) and (b) both fail.

**Diagnostics (build BEFORE training anything — item-1 ordering: the connectivity
stratum changes what step (a)'s pass/fail even means):**
- **D1 — Compass resolution, connectivity-stratified.** State pairs binned by
  proprioception (torso height, speed, uprightness) at controlled geodesic gaps Δ
  (BFS oracle, dev-only); Δ\* = smallest Δ with ≥80% ranking accuracy in the far
  zone (>400 steps). Reported in THREE strata:
  1. **Within-trajectory-coverable** pairs (some dataset trajectory passes within
     ε of both s and g — the direct test "could relabeling have produced this
     pair"; ε = goal radius; this is cleaner than a geodesic-span proxy and is the
     primary stratum split),
  2. **Stitched** pairs (no single trajectory covers both — pure extrapolation
     for step (a)),
  3. **(state, eval-corner-goal)** queries specifically — the deployed
     distribution.
  **Hard gate (only one): stitched-stratum + corner-stratum Δ\* ≤ 100 dense steps
  unlocks cell D (L4).** For L1 cells D1 is **informative, never blocking**:
  Phase-1 ran the naive L1 tree to 83.3% with a critic that almost certainly
  fails far-zone 25-step resolution — at realized depth ~2.5–3 with MAX backup,
  root children are compared on subtree extrema ~75+ steps apart, converting
  near-goal sharpness into far-zone discrimination. D1-on-L1 predicts how much of
  the win is depth-conversion vs direct ranking; it cannot self-inflict a stop on
  cells that demonstrably run.
  Run on BOTH V(s) (the measured Phase-1 baseline for the decay story) and V(s,g).
- **D2 — Exploitability.** (b-realistic, primary): a fixed pool of ≥10k generated
  segments; wall-clip enrichment in the top 5% by V_pess vs base rate (oracle wall
  check, evaluation-only). (c): V MAE on generated vs dataset states. Run
  before/after the Stage-1 gate **on the same fixed pool** — the P1 finding
  predicts large enrichment for the naive critic.
- **D4 — No-oracle calibration.** On relabelled (state, future-goal) pairs:
  predicted vs empirical step distance, calibration curve + MAE by distance band.
  **Known blind spot (disclosed):** D4 tests within-trajectory pairs by
  construction and can only validate the regime step (a) trained on; the stitched
  regime is covered exclusively by D1's strata 2–3. D4 is the no-oracle diagnostic
  that travels; D1-stitched is the one that protects deployment.
- (v4's D3 chunk-length window is replaced by the locked L=1 + the pre-registered
  L4 interaction cell.)

---

## 4. Stage 1 — Hardened judge

v4 §4 essentially verbatim, on D4RL data at stride 25:
- **Feasibility classifier** D(s_t, s_{t+25}) on stride-25 transition pairs.
  Positives: real stride transitions. Negatives (corruptions of real data only —
  Rule 1): temporally shuffled pairs, **displacement-matched cross-trajectory pairs**
  (negative displacement distribution matched to positives'), Gaussian-noised
  endpoints. No oracle-labelled, no diffusion-generated negatives.
- **Segment aggregation:** D applied to every consecutive waypoint pair; reject the
  segment if the **minimum** pairwise score is below threshold.
- **Calibration:** threshold at ≤5% false rejection on held-out real transitions;
  stratified check on long successful trajectories within +2 points.
- **Gate:** D2(b) through the full stack — top-5% wall-clip enrichment ≈ base rate.

---

## 5. Stage 2 — Pipeline integration (the build that wires it together)

File-level work items, in order:
1. `scripts/train_state_value.py` — add `--goal-conditioned` (relabeling mixture per
   §3a), `--ensemble N`; checkpoint `state_value_sg_ckpt_{step}.pt`.
2. `mcts/value_net.py` — `DVStateValueEnsemble` (stack of DVStateValue, min/mean−βstd
   reduction) + loader; goal-input concat handled in the net.
3. `scripts/train_feasibility.py` + `mcts/judge.py` — Stage-1 classifier (train +
   batched inference, min-pairwise segment aggregation).
4. `mcts/mcts_loop.py` — Sampler gains `value_mode ∈ {v_s, v_sg, v_sg_pess}` and
   `gate ∈ {none, hard}`; expand_fn: goal-conditioned value call, gate filter before
   child attach, point-to-segment terminal check; MCSS arm gains the same re-ranker
   options (for the BoN(hard) cell). Per-step gate-rejection fraction logged.
5. `scripts/run_mcts_compare.py` — flags through to Sampler; JSON records
   value/gate config (collate label suffixes, e.g. `b16-sgP`, `k272-sgP`).
6. `scripts/collate_mcts.py` — label parsing for the new variants (everything else
   already works: pairing, McNemar, depth).
7. D1/D2/D4 diagnostic scripts (§3) + `scripts/measure_dmax.py` (BFS oracle port).
Torch-free parts unit-tested locally as established practice; GPU smoke
(`--n-envs 4 --max-steps 50`) before any full cell.

---

## 6. Stage 3 — The 2×2 and the headline cells

All cells: antmaze-large-diverse-v2, 272 candidates/step, seeds {0,1,2} × n=50,
full episodes — pairing against every Phase-1 cell is automatic.

| | naive value (DV critic / V(s)) | hardened (V_pess(s,g) + gate) |
|---|---|---|
| **BoN / flat (MCSS k272)** | **72.0% — DONE (P1)** | cell A — to run |
| **Tree (MCTS b16)** | **83.3% — DONE (P3)** | cell B — to run |

Plus: cell C — BoN(hard) at k=50 (does hardening rescue the *cheap* baseline?);
cell D — the pre-registered L4 interaction cell (§2).

**Attribution ladder (the hardened column bundles three changes — goal-cond,
ensemble-min, gate — which the 2×2 alone cannot separate):**
- Seed-0 ladder, always run (~6 GPU-h at measured wall): {V(s)} (done, 86%) →
  {V(s) + gate} (gate-only rung) → {V(s,g), no gate, no ensemble} →
  {V_pess(s,g) + gate} (= cell B seed 0). One change per rung.
- **Cell E — tree, V(s,g), no gate, conditional-MANDATORY at 3 seeds:** runs
  whenever B beats A or the P3 cell significantly (it apportions the win between
  goal-conditioning and the gate at headline n).
- **Hardened-slope cell:** hardened tree at b4, seed 0 (~1 GPU-h) — gives the
  hardened-tree compute curve two points against P2's naive curve, so
  "negative→positive scaling" is a slope claim, not an endpoints claim.

**Power pre-registration for H1(b) (B vs k50):** P4 measured +4.0 pp at p=0.44
with this exact protocol; n=150 resolves ~8–10 pp at the observed discordance
rates. **Pre-registered now:** if at n=150 the B−k50 point estimate is ≥ +5 pp
with p > 0.05, both cells extend to seeds {0–5} (n=300) and the pooled test is
reported as confirmatory (not exploratory). Budgeted: ~12 GPU-h at measured wall
(3×(3.1+0.6) h).

**Compute:** cells A+B ≈ 19 GPU-h, C ≈ 2, D ≈ 3, ladder ≈ 6, slope ≈ 1,
E (conditional) ≈ 9, power extension (conditional) ≈ 12. Critic + classifier
training ≪ 1 each. Base ≈ 31 GPU-h; worst case ≈ 52 GPU-h.

**Per-decision diagnostics (logged):** gate rejection fraction, ensemble
disagreement at chosen children, realized depth (built), decision-flip rate
vs the naive arm on shared scenarios.

**Decision rule (pre-registered):**
- B > A beyond paired CI **and** B > k50 (P4 closed, extension protocol above) →
  H1 supported; H2 attribution from the ladder + cell E.
- A ≈ 72 but B ≫ 83.3 → the judge needs the tree (compute conversion story).
- A ≫ 72 with B ≈ A → "the judge, not the search" — report honestly; the P3
  matched-compute result remains the search contribution. **Written now, not
  negotiated later:** if hardened flat scaling turns positive (A ≥ C), part of
  the honest headline becomes "the judge repairs flat scaling", and the
  tree-specific claim rests entirely on B vs A.
- Break-rate target: hardened tree breaks < 15% baseline (P6); fix-retention:
  greedy-hard core wins held (≥6/9 pooled).

---

## 7. Out of scope (v5)

Hierarchical/subgoal trees (v4 §7), OGBench giant/stitch, MCTD same-config
comparisons, π_lo executor, quasimetric critic track. Each is future work; none
blocks the H1/H2 claims at this scope.

---

## 8. Timeline (from 2026-06-13)

| Week | Work | Exit |
|---|---|---|
| 1 | **Diagnostics first** (§5 item 7: d_max + connectivity-stratified D1 + D4 harness), then §5 items 1–2 (V(s,g) training, MSE-vs-expectile ablation on val split) | D1 three-strata report on V(s) and V(s,g); calibration curve; step-(b) trigger evaluated on the stitched stratum |
| 2 | §5 items 3–6 (judge + pipeline wiring) + D2 pre/post gate on the fixed pool | judge gate (≤5% FR, enrichment ≈ base) |
| 3 | Cells A, B (2×2 hardened column), C, seed-0 attribution ladder, hardened-b4 slope cell | H2 decision rule + ladder read |
| 4 | Cell D (L4 interaction), conditional E, conditional power extension (B & k50 → seeds 0–5), antmaze-large-play row, ablations (pessimism form, soft gate) | results table complete |
| 5+ | Writing; maze2d sanity re-run with final config | dissertation chapter |

| Failure mode | Detector | Contingency |
|---|---|---|
| V(s,g) compass flat far out | D1 Δ\* | IQL-u upgrade (§3b) → contrastive |
| Gate kills needed behaviour | stratified false-rejection | re-mine negatives; threshold; soft-gate ablation |
| Search exploits V(s,g) too | D2(b) enrichment; break rate vs P6 | stronger pessimism (β sweep, seed-3 held-out) |
| Hardened cells ≈ naive | decision rule §6 | report mechanism honestly; P1–P3 stand alone |
| Stitched (s,g) queries unanchored (within-trajectory relabeling cannot cover them; D4 structurally blind) | **D1 strata 2–3** (stitched + eval-corner) — the only detectors that can fire for this failure | step (b) IQL-u (TD bootstrapping repairs exactly this regime) → contrastive |
| Goal-input OOD (eval corner vs data termini) | D1 stratum 3 + terminus-goal coverage map near (33,25) | corner-weighted relabeling; step (b) |

---

## 12. Traceability table (cumulative — this table travels with the document;
auditing it is the mechanism that catches silent regressions, so it is never
moved to git history)

**Round 5 audit (v5 → v5.1):**

| Item | Section(s) | Status |
|---|---|---|
| R5.1 — Stitching blind spot: connectivity-stratified D1 (within-coverable / stitched / eval-corner strata); step-(b) trigger pre-registered on the stitched stratum; D4 blindness disclosed; risk-register detector corrected | §3, §8 | INC (operationalization refined: direct two-point trajectory-coverage check at ε = goal radius, rather than the geodesic-vs-max-span proxy — it tests "could relabeling have produced this pair" exactly) |
| R5.2 — Expectile (τ=0.9) replaces MSE on the MC-relabelled targets; MSE-vs-expectile pre-registered as training ablation on the val split | §3a | INC |
| R5.3 — D1 non-blocking for L1 cells; the only hard gate is the L4 unlock (stitched+corner Δ\* ≤ 100); depth-conversion rationale recorded | §3 | INC |
| R5.4 — Attribution ladder: seed-0 rungs {V(s)+gate} and {V(s,g) plain}; cell E promoted to conditional-mandatory at 3 seeds | §6 | INC |
| R5.5 — H1(b) power extension pre-registered (B−k50 ≥ +5pp with p>0.05 at n=150 ⇒ seeds 0–5, pooled confirmatory); budgeted ~12 GPU-h at measured wall | §6 | INC (cost restated from measured wall times: 3×(3.1+0.6) h) |
| R5.6 — Hardened-b4 slope cell (seed 0); "judge repairs flat scaling" decision-rule sentence written now | §6 | INC |
| R5.7a — Scale consistency: relabelled targets pass through the identical pipeline affine (no min-max refit); target(t′=t)=1.0 ≡ terminal-check value, asserted in code | §3a, §2 | INC |
| R5.7b — Timeout-truncation termini rule stated | §3a | MOD (vacuous for the current dataset: DV_D4RLAntmazeSeqDataset at learn_policy=False excludes timeout-only paths — every stored path has a true terminus; rule stated in case dataset config changes) |
| R5.7c — c_ucb: range unchanged ⇒ c=√2 carries; seed-3 spot-check rule | §2 | INC |
| R5.7d — BoN(hard) gate placement: discard-before-rank, mirroring the tree | §2 | INC |
| R5.7e — D2 pre/post-gate on the same fixed 10k pool | §3 | INC |
| R5.8 — Cumulative traceability restored in-document (Rounds 2–3 re-inlined below) | §12 | INC |
| R5.9 — P5 claim-language discipline: single-seed L-sweep numbers stay labelled (seed 0) and motivate cell D; not claim-language until D runs | §0, §2 | INC (verified already compliant; noted for the writing phase) |

**Round 4 (v5 rescope):**

| Item | Section(s) | Status |
|---|---|---|
| R4.1 — D4RL-only; OGBench/MCTD/giant/stitch dropped | §1, §7 | INC (user decision 2026-06-13) |
| R4.2 — Phase-1 results P1–P7 added as measured premises | §0 | INC |
| R4.3 — Frozen DV planner/policy replace v4 generator spec (§6) and executor stage (§5) | §2 | INC (P7 closes the executor gate) |
| R4.4 — Paired-scenario protocol replaces bootstrap-over-seeds | §1 Rule 3 | INC (stronger for fixed-config cells; n=150 pooling justified by P7 stochasticity) |
| R4.5 — Critic: MC-relabelled V(s,g) first; IQL-u as triggered upgrade; quasimetric dropped | §3 | MOD (v4's IQL-u demoted to contingency — MC targets are exact under relabeling, discount 1.0, and drop into the validated [−1,1] pipeline scale; revisit if D1 fails) |
| R4.6 — D3 replaced by locked L=1 + pre-registered L4 interaction cell | §2, §3 | MOD (P5 measured the window question closed-loop) |
| R4.7 — 2×2 naive column = Phase-1 cells; only hardened column to run | §6 | INC |
| R4.8 — Burst widening: already implemented in value_forest (full-wave k=16) | §2 | INC (no engine change needed) |
| R4.9 — u-units/steps-space arithmetic deferred to the IQL-u branch | §2 | MOD (unnecessary at discount 1.0 on [−1,1] MC scale) |
| R4.10 — Goal-OOD risk: corner-coverage check added to D4 + risk register | §3, §8 | INC (v4 §6 risk, made concrete for the D4RL eval corner) |

**Round 2 review (v3 → v4, seven items):**

| Item | Section(s) | Status |
|---|---|---|
| R2.1 — 2×2 attribution {BoN, tree} × {naive, hardened} | §6 | INC |
| R2.2 — Headline = DV (then MCTD family too); oracle discipline | §1 | INC (superseded in part by R3.1, then R4.1) |
| R2.3 — Compute budget, fast sampler, batching, tree reuse, gate | §6 | INC (batching corrected per R3.4; superseded by frozen-DV stack per R4.3) |
| R2.4 — Steps-space scoring, per-waypoint goal check, normalization, subgoal conversion | §2 | INC (goal check upgraded to point-to-segment per R3.7f; steps-space deferred per R4.9) |
| R2.5 — Proprioception-matched D1; probes (a)/(b); generalization gap | §3 | INC (connectivity strata added per R5.1) |
| R2.6 — Transition classifier; stratified false-rejection; per-maze pessimism via held-out; chunk-length window | §4, §9, §3 | INC (oracle tuning demoted to dev proxy; window replaced per R4.6) |
| R2.7 — Critic-family fallback; evaluation protocol; realistic timeline | §3, §1, §10 | INC (fallback re-scoped per R4.5/R5.1; protocol upgraded per R4.4) |

**Round 3 audit (v4):**

| Item | Section(s) | Status |
|---|---|---|
| R3.1 — OGBench large/giant + stitch co-primary; D4RL secondary with ceiling warning | §1 | INC, then superseded by R4.1 (D4RL-only; the ceiling warning became measured fact P4) |
| R3.2 — Relabeling: 10% current-state goals (target pinned); termination mask; truncated n-step | §3 | INC (carried into v5 §3a/§3b) |
| R3.3 — Hierarchy = expected path for giant, firm schedule | §7, §10 | Superseded by R4.1 (out of scope) |
| R3.4 — Burst widening (W=16) mandatory; virtual-loss parallelism optional | §2, §6 | INC; discharged by R4.8 (already implemented in value_forest) |
| R3.5a — Normalized u = (1−γ)V parameterization | §2, §3 | Deferred to the IQL-u branch per R4.9 |
| R3.5b — D4 no-oracle calibration diagnostic | §3 | INC (blind spot disclosed per R5.1) |
| R3.5c — Quasimetric critic in parallel | §3, §10 | Dropped at v5 scope per R4.5 |
| R3.6 — Baseline track weeks 2–4 | §8, §10 | Superseded: DV row done (P7); MCTD dropped per R4.1 |
| R3.7a–i, R3 process | various | INC as in v4; surviving items carried forward unchanged |
