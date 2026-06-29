# MCTS-as-Sampler for Diffusion Veteran — Consolidated Findings (for a progression report)

*Standalone summary of the whole DV/antmaze investigation, written so another agent (or a
supervisor) can read it cold and produce a progress report. Numbers are the authoritative
ones from `results/` as of 2026-06-28. Companion technical report with full methodology:
[writeup_mcts_sampler.md](writeup_mcts_sampler.md). Privileged "oracle/orc/fs/gnt/smt"
numbers are Rule-1 DIAGNOSTIC-ONLY ceiling probes — never citable as achievable results.*

---

## 0. TL;DR (the one paragraph)

We integrated **MCTS as the sampler** into the Diffusion Veteran (DV) generative-RL planner,
replacing its stock MCSS (best-of-K) sampler while reusing DV's trained diffusion planner and
inverse-dynamics policy unchanged and retraining only the critic into a tree-usable per-state
value `V(s)`. On the one D4RL setting with real headroom (antmaze-large-diverse-v2, closed
loop), the **qualitative** contribution holds: **scaling a diffusion planner's candidate
budget under flat best-of-N argmax *backfires* (critic over-exploitation, −4 pp), while the
same compute organised as a value-guided tree avoids that degradation**. But at full
statistical power (n=500, 10 seeds) the tree only **ties the cheap MCSS baseline (79.0 % vs
78.8 %)** and the matched-compute win shrinks to **+4.2 pp (p = 0.12, n.s.)**. A long ladder
of ceiling probes then established **why**: the residual ~20 % failures are **not** a
selection, value, or look-ahead problem — they are **physical topples of the Ant during
execution**. A *perfect* goal-distance ranker nets ~0; a *perfect* value inside the tree
reaches only ~82 %; and every execution-aware selector we built (stability, gentle, smooth)
is null or negative. **The ceiling on this env is the low-level locomotion policy, which sits
below the sampler and which no plan-space method can reach.** The clean next step is therefore
either (a) a different benchmark whose headroom is *planning*, not locomotion (the OGBench
pivot), or (b) a forward dynamics / world model that can foresee falls (a different,
model-based thesis).

---

## 1. What was built (the integration)

| Component | What it is | Status |
|---|---|---|
| **MCTS sampler** | Torch-free forest of M lockstep trees, UCB descent, **max-backup**, batched `(M·k, H, D)` planner+value call per round; extract best root child, execute its first waypoint, replan (MPC). `mcts/value_forest.py`, `mcts/mcts_loop.py` | Done, 8/8 unit tests |
| **State value `V(s)`** | MLP `obs→[256]×3→1`; same DV return-to-go target as the MCSS critic, keyed per-state so the tree can compose depth. `mcts/value_net.py` | Done, val_corr 0.809 (antmaze) |
| **Goal value `V(s,g)`** | Relabelled, expectile-τ0.9, 5-ensemble + min-pessimism. `mcts/relabel.py`, `mcts/value_scale.py` | Done, val_corr 0.905 |
| **Paired evaluation** | Goal-verified per-scenario McNemar (per-seed + pooled). `scripts/collate_mcts.py` | Done |
| **Failure instrumentation** | Tier-0 traced rollout (body pose, candidate pool, BFS distance), torch-free classifier, Rule-1 oracle re-rank. `mcts/instrument.py`, `mcts/failure_modes.py` | Done, 31 tests |
| **Execution-layer probes** | Flat selectors by true geodesic + stability / gentle / smooth; fall-geometry diagnostic. `scripts/diag_oracle_flat.py`, `scripts/diag_fall_geometry.py` | Done |

The brief: *"integrate MCTS to efficiently explore multiple potential rollouts in parallel"* so
that *"poor early decisions in imagined rollouts"* stop capping the final plan. The build does
exactly that; the science below is whether it moves the needle and, where it doesn't, why.

---

## 2. Headline results (reportable, paired, closed-loop antmaze-large-diverse-v2)

**DV MCSS reference (pipeline, n=1000): 76.9 %.** Our harness reproduces it (k50 ≈ 77–79 %).

### 2.1 At full power — n=500, 10 seeds (the number to quote)

| arm | cand/step | reach % | paired vs … | Δ | exact p |
|---|---|---|---|---|---|
| MCSS k50 (cheap baseline) | 50 | **78.8** | — | — | — |
| MCSS k272 (flat-scaled) | 272 | **74.8** | k50 | −4.0 pp | 0.15 |
| **MCTS b16** | 272 | **79.0** | k272 | **+4.2 pp** | **0.12** |
| MCTS b16 | 272 | 79.0 | k50 | +0.2 pp | 1.00 |

**Read:** the tree **ties** the cheap baseline and **beats flat-scaled MCSS by +4.2 pp, but
not significantly**. The durable, sign-robust claim is the **contrast**: flat best-of-N
*degrades* with more candidates (optimizer's curse on the critic), the tree does not.

### 2.2 At n=150, 3 seeds (the original headline — small-sample optimistic)

`MCSS k272 72.0 → MCTS b16 83.3, +11.3 pp, exact McNemar p = 0.021`; fixes 33/42 (79 %),
breaks 16/108 (15 %). **Why it shrank at n=500:** the diffusion planner draw is *unseeded*, so
McNemar pairs only the goal while the dominant variance (planner sampling + start jitter,
±6 rollouts/config) is unpaired — n=150 was noise-inflated. The effect *sign* is robust; the
large significant *magnitude* was not.

### 2.3 Ablations (all paired, n=150 unless noted)

- **Depth helps mildly:** b4 78.0 → b8 78.7 → b16 83.3 (b4→b16 p = 0.31).
- **Fine branching is best:** child_index L1 86 % > L2 78 % > L4 74 % (seed 0) — with a goal-
  agnostic value, looking farther per branch commits harder to confidently-wrong distant cells.
- **Goal-conditioning does NOT help:** `V(s,g)` b16-sgP **76.7 % ≤ V(s) 83.3 %** (p = 0.22),
  consistent across all 3 seeds — even though `V(s,g)` is *sound* (D2: non-exploitable,
  enrichment 0.40×, geodesic MAE 27 steps). Diagnosis: under MPC replanning, **robust local
  progress (V(s) riding maze geometry) beats greedy global goal-targeting**.

---

## 3. The ceiling investigation — why ~80 %? (ladder of probes)

Each rung holds something fixed and perfects another, to localise the cap. **All "oracle"
numbers are Rule-1 privileged ceiling probes (true BFS geodesic), never reportable.**

| probe (Rule-1 unless noted) | question | result | verdict |
|---|---|---|---|
| **Oracle flat re-rank** | can a *perfect ranker* over the 50 candidates beat the critic? | 78.7 % vs critic 78.0 %, fixes 23 / **breaks 22**, **net +0.7 pp** | **flat selection saturated** |
| Candidate-gap check (reportable) | does the DV critic mis-rank vs geodesic? | mis-rank 0 %, mean gap 0.0 cells | critic already picks geodesic-best |
| **Oracle-in-the-tree** | does a *perfect value inside MCTS* beat ~78 %? | b16-orc **82.0 %** ≈ V(s) b16 83.3 % (−1.3, p = 0.89); beats V(s,g) +5.3 (p = 0.32) | **value accuracy is NOT the lever** |
| Failure-mode trace (reportable) | what do the ~20 % failures *do*? | **100 % physical topples** — uprightness −0.91, torso height 0.26, motionless | the failure is *execution*, not routing |
| Fall rate vs selection aggressiveness | does picking bolder plans fall more? | orc 20 % / critic 22 % / fs2 24 % / fsf 28 % | aggressive selection → more falls |

### 3.1 Execution-aware selectors — every one is null or negative (seed 0, n=50, Rule-1)

Having localised the failure to topples, we tried to *select around* them:

| selector | idea | reach % | verdict |
|---|---|---|---|
| `stbU` (upright) | prefer plan predicted most upright | 80 % | = endpoint; **planner predicts ALL plans upright** (candidate spread 0.038 → orientation channel empty) |
| `stbD` (min displacement) | gentlest step | **0 %** | degenerate **creep/oscillation** (moves a lot, no net progress) |
| `stbA` (min angular vel) | smoothest | 80 % | no effect |
| `gnt` (drop biggest-displacement lunges) | progress minus the lunge | 64–84 %, non-monotone | **noise** (all McNemar p ≥ 0.06; gnt30 = 84 % is top of the noise band, not a win) |
| `smt` (cap commanded turn 20–120°) | forbid the sharp turn | 64–74 %, all ≤ baseline | **negative**; tightening trades topples for creep |

### 3.2 Fall-geometry diagnostic — your two physical hypotheses, tested on the logs

`scripts/diag_fall_geometry.py` (no GPU; reads existing failure traces). At the stall onset vs
the ant's own moving baseline:

- **H-wall (falls cluster near walls): REFUTED.** Wall clearance at the topple is normal
  (onset/baseline ratio 0.90–1.16 ≈ 1 across all tags). A wall-avoidance / danger-zone penalty
  would target the wrong thing — **not built, by evidence.**
- **H-turn (falls follow a sharp steer): real correlation, but a SYMPTOM.** The sharpest
  commanded turn entering a stall is 130–170° (a near-reversal) vs a ~50° moving baseline.
  **But** the `smt` selector that forbids those turns does **not** reduce topples (even smt20
  still topples; it just creeps instead) — so the near-reversal is *downstream* of the ant
  already destabilising, not the cause. Selection cannot reach the fall.

---

## 4. Conclusion (what is established)

1. **The MCTS integration works and is at least as good as MCSS**, and is **more robust to
   flat-scaling degradation** — the defensible methodological contribution. At matched compute
   the tree avoids the −4 pp backfire that flat best-of-N suffers; the gap is +4.2 pp (n.s. at
   n=500).
2. **Ranking/selection is saturated** on this env: a *perfect* flat ranker nets +0.7 pp and the
   DV critic already picks the geodesic-best candidate.
3. **Value accuracy is not the lever**: a *perfect* value inside the tree reaches only ~82 %.
4. **The cap is the low-level locomotion policy.** The residual ~20 % are physical topples of
   the Ant during execution. The diffusion planner is a *trajectory prior* — it only dreams
   upright, successful futures (hence the empty orientation channel, spread 0.038), so it
   **cannot foresee execution falls**, and therefore **no plan-space selection, value, or
   look-ahead can avoid them.** This is consistent across the oracle ceilings, the 100 %-topple
   trace, and the null stability/gentle/smooth selectors.
5. **The only untested lever for *this* ceiling is a forward dynamics / world model** that
   predicts whether an action topples the ant (privileged true-simulator 1-step lookahead is
   the clean ceiling test; a learned fall-model is the deployable version). That is a different,
   model-based research direction.

---

## 5. Alignment with the dissertation thesis — what may be overlooked

**Thesis: "Integrating MCTS into long-horizon planning for diffusion planners."** Honest audit
of coverage and gaps, for the supervisor conversation:

### Covered
- ✅ MCTS sampler integrated into an off-the-shelf diffusion-RL codebase (the brief, literally).
- ✅ Matched-compute, paired comparison vs the stock sampler, with a mechanism (flat-scaling
  backfire vs tree robustness).
- ✅ Budget/depth and segment-length (child_index) sweeps.
- ✅ A retrained tree-usable value, plus a goal-conditioned variant, each characterised.
- ✅ A full ceiling/attribution analysis localising the residual failure.

### Gaps / overlooked (ordered by importance)
1. **The locomotion confound is the headline limitation, and it is structural to the env
   choice.** antmaze-large has headroom, but the headroom turned out to be **Ant locomotion
   (topples)**, not **planning**. maze2d-large (a non-falling point mass) has **no** headroom
   (MCSS already 100 %). So with these two D4RL envs we can isolate planning quality *or* have
   headroom, **never both**. To actually demonstrate "MCTS helps *long-horizon planning*" we need
   **a long-horizon benchmark with planning headroom and no locomotion confound** — e.g. the
   **OGBench** pointmaze/antmaze-giant family (already the planned pivot; this DV/antmaze work is
   the frozen baseline row). **This is the single most important thing to raise.**
2. **The world-model / sim-lookahead lever** for the antmaze ceiling is identified but unbuilt.
   It's the only thing that could push past ~80 % here, but it changes the thesis from
   *search-as-sampler* to *model-based*. Frame as future work / a scoping question for the
   supervisor.
3. **Firming the matched-compute headline.** The n=500 wash is a *pairing* artifact (unseeded
   planner draw), not necessarily a null effect. Seeding the diffusion draw and/or adding seeds
   could recover significance on the +4 pp tree-vs-flat contrast — cheap, optional, only worth it
   if the supervisor wants a significant absolute number rather than the qualitative claim.
4. **Open-loop execution horizon** (commit to >1 waypoint before replanning) is untested, but
   §6.4 predicts it would *hurt* (MPC replanning is the thing that works). Low priority.
5. **Single env / single start-goal.** antmaze-large-diverse has a near-fixed far-corner goal.
   Other variants (play, medium, ultra) untested. Breadth is better spent on the OGBench pivot
   than on more D4RL antmaze variants.

---

## 6. Presentation assets — PRODUCED (in `results/figs/`)

Regenerate any of these with `python scripts/make_report_figures.py` (local, no GPU). They are
report-grade PNGs at 150 dpi:

| file | what it shows |
|---|---|
| `fig1_matched_compute.png` | **headline.** n=500 bars: MCSS k50 78.8 / k272 74.8 / MCTS b16 79.0; the flat-scaling backfire and the matched-compute tree edge (+4.2 pp, n.s.) annotated. |
| `fig2_compute_scaling.png` | **the mechanism.** Same compute, opposite slope: flat best-of-N slopes *down* with candidates, the tree does not (n=150 full sweep). |
| `fig3_ceiling_cluster.png` | **the conclusion.** Every sampler — and a *perfect* value — saturates at ~76-83%; callout: the cap is locomotion (100% topples), not selection. |
| `fig4_topple_anatomy.png` | **the cause.** One failed episode: BFS-distance falls 14→3 cells, then uprightness collapses 1→−1, torso height drops to 0.26, speed dies — the Ant marches up then tips and lies motionless 570 steps. |
| `fig5_fall_geometry.png` | **the physical hypotheses.** H-wall refuted (clearance at topple ≈ normal), H-turn is a symptom (near-reversal precedes the stall but capping it doesn't help). |
| `anim/anim_instr_mcss_critic_s0e10.gif` | **topple GIF** (slides): the same env 10 marching into a fall, with candidate cloud + BFS curve. |

**One asset still needs a single GPU run — the "money shot" (MCTS rescues MCSS):**
```bash
# on the GPU box (~0.6 h k50 + ~0.6 h b16):
python scripts/run_compare_trace.py --env antmaze-large-diverse-v2 --seed 0 --n-envs 50
#   -> prints the FIX env indices (MCSS fail, MCTS reach) + writes results/instr/cmp_s0*.{npz,json}
# then locally (render-tested end-to-end):
python scripts/animate_compare.py --seed 0 --env-idx <a FIX idx> --gif
#   -> results/figs/anim/cmp_s0e<idx>.png  (side-by-side: MCSS path | MCTS path | shared BFS curve)
```
`run_episodes(trace=True)` (added, default-off so it can't affect the headline runs) logs the
per-step executed path; `animate_compare.py` renders the side-by-side. This is the visual proof
of "look-ahead rescues the wrong early turn" (the brief). Headline reproduction commands:
`writeup_mcts_sampler.md` §8.

---

## 7. Pointers

- Full methodology, every ablation, the V(s,g) negative, the oracle ladder: `writeup_mcts_sampler.md`.
- Instrumentation runbook + Rule-1 firewall + shape-attribution post-mortem: `instrumentation.md`.
- Memory: `project_mcts_results.md` (results), `project_dv_pipeline.md` (architecture),
  `project_plan_v4.md` (the OGBench pivot).
- Results JSON: `results/s10_*` (n=500 headline), `results/scale_*` (n=150 grid + V(s,g) + oracle),
  `results/instr/*` (failure traces + flat-selector logs).
