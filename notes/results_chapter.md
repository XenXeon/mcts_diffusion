# Results and Analysis (dissertation chapter draft)

Draft of the results chapter, 2026-07-11. Companion to `methodology_report.md`
(the methodology chapter): that document defines the systems, algorithms, and
protocols; this one presents the findings as an argument. Every number is
traceable to a results JSON or a recorded gate output (file names in-line;
provenance notes where raw vectors were lost). Numbers are DV-normalized scores
unless stated; kitchen scores are cumulative subtask completions normalized to
0–100 (25 points per subtask).

**The chapter's argument in one paragraph.** We set out to make Monte-Carlo tree
search beat a state-of-the-art diffusion planner's flat best-of-K inference
(MCSS), and instead established, through controlled single-variable experiments,
the conditions under which search can and cannot pay: tree search over diffusion
plans is harmful under a full-sequence backbone (its expansion is not a faithful
conditional generation, and its backup amplifies evaluator error on stitched
plans), flips to a confirmed, replicated gain under a Diffusion-Forcing backbone
(exact prefix conditioning), grows as the flat baseline weakens (a monotone
headroom curve across three backbones and, independently, across three guidance
strengths), and terminates — for every learned-value method, including the
baseline itself — at the dataset's demonstration ceiling. A per-token noise-aware
value model, used as classifier guidance on the frozen causal planner, lifts the
flat baseline with zero physical cost and composes only partially with search:
guidance and composition are partial substitutes converging on the same landing
point. The contribution is not a new state of the art; it is a measured law of
when structured search helps a diffusion planner, with each clause carrying its
own controlled experiment.

---

## 1. Protocol summary

All comparisons are **paired**: MCSS and tree arms share environment instances,
start/goal draws, the frozen evaluator (the DV trajectory critic), and the frozen
inverse-dynamics policy — the planner/search mechanism is the only manipulated
variable within any comparison. Statistical standard: pooled paired t across
seeds with per-rollout differences (threshold t ≥ 2.0 declared in advance;
kitchen adds a sign test because scores are quantized in 25-point subtask units).
Multi-seed claims: maze2d DF, 5 seeds (n=125 paired rollouts); kitchen DF, 4
seeds (n=100). Single-seed arms (n=25) are reported as directional and labeled
so. Baselines were reproduced before any treatment arm was interpreted:
maze2d-large DV-MCSS k50 199.4 / k256 201.2 at the critic's selection ceiling;
kitchen-mixed DV-MCSS 75.0 vs the DV paper's published 73.6.

**Measurement note (start-matching).** The maze2d camping score is strongly
start-state dependent, so every maze2d difference reported in this chapter is
*start-matched*: the compared arms are differenced per-rollout on an identical
`starts` array, asserted programmatically before differencing. The start-matched
DV-MCSS baselines are **k50 = 199.4** (root width) and **k256 = 201.2**
(compute-matched to the tree's ≈290 planner draws), both at seeds 0–9, n=250.
Earlier drafts of this work quoted a baseline of 204.9, a valid measurement on a
different 150-rollout start set that was mistakenly used as the comparator for
arms run on other starts; the corrected figures appear throughout, and the full
audit is in `notes/maze2d_startmatched_correction.md`. No sign or conclusion
changed; the naive-tree result strengthened.

## 2. The negative result: naive tree search loses on the DV backbone

On maze2d-large, a UCB tree with the DV planner as its expansion operator and
MAX backup loses to its start-matched flat baseline on **every one of ten seeds**:
pooled **−5.05, seed-t = −10.57 (n=250)**. The explanation is not "search doesn't
work"; it decomposes into three measured defects, each with a fix that was
implemented and priced (methodology §3–§5):

| defect | measurement | fix | recovered |
|---|---|---|---|
| ill-posed per-state values | V(s) val corr 0.39–0.76 (target depends on unseen future) | plan-value distillation V̂(s), corr 0.74→0.98 | +35 closed-loop |
| incomparable node scores | leaf windows start at different depths | composed-window critic scoring | ~+1.5 |
| winner's curse on stitched plans | seam off-manifold for the critic (MSE ratio ~10⁴, directional) ; MAX backup selects hallucinated depth | top-3 tempered backup | **+4.54 (roll-t 5.00, 3/3 seeds)** |

With every fix installed, the DV tree reaches **202.2 — exact parity with its
root-width flat baseline (+0.25, t = 0.33, n.s.) and −2.16 (paired t = −2.60)
below the compute-matched k256 baseline**: no win under either comparator, at
roughly six times the compute. Width saturates (k50 199.4 → k256 201.2, +1.8 for
5× the samples), so the deficit is not sampling volume. The residual is
structural: DV-MCSS already operates at its critic's selection ceiling —
**search had nothing left to buy from the evaluator**. The prefix-inpainting
variant (clamping the search prefix into the frozen full-sequence denoiser —
replacement conditioning on an input configuration the model never saw) scores
**182.1, a start-matched −18.5 (t = −3.02)**, the chapter's
cleanest demonstration that *unfaithful conditioning, not search, is the poison*:
the glue seam had been the critic's accidental defense, and "fixing" the seam
with fake conditioning removed the defense while keeping the lie.

**Interpretive insight (Phase 1).** MCSS-with-a-trajectory-critic is itself a
value comparison, not a "flat vs structured" contrast: the whole-trajectory
critic is a better value function than any per-state V(s) a tree can bootstrap,
because its input determines its target. Naive MCTS lost the value comparison,
not the search comparison.

## 3. The flip: search helps the Diffusion-Forcing backbone (confirmed, 5 seeds)

Replacing the planner with a causal Diffusion-Forcing transformer (per-token
noise training; tree expansion becomes *exact* conditional generation on the
search prefix) flips the result. maze2d-large, identical protocol:

| arm | DV backbone (seeds 0–2) | DF backbone (seeds 0–4) |
|---|---|---|
| flat MCSS, root width k50 | 202.0 | 183.4 |
| flat MCSS, compute-matched k256 | 204.4 | — |
| tree (top-3 backup) | 202.2 (**+0.25 n.s.** vs k50; **−2.16, t = −2.60** vs k256) | **192.4 (+9.04, paired t = 3.90, n=125)** |
| MAX-vs-top-3 gap (curse signature) | +4.54 (roll-t 5.00) | +1.8 (n.s.) |

(Baselines here are restricted to the tree arms' own seeds. Across the full
seeds 0–9 pool the flat baselines are k50 199.4 / k256 201.2; the width control
is computed on that pool.)

Per-seed differences: +9.4 / +3.9 / +10.3 / +5.9 / +15.7 — positive on every
seed. Two mechanisms, both evidenced. (i) *Faithful expansion → trustworthy
composite scores*: the expansion-fidelity triptych maps monotonically onto tree
performance (glue −4.3 < fake conditioning −18.5 < exact conditioning +9), and
the winner's-curse signature vanishes on DF (MAX ≈ top-3) — with true
conditional samples there is no hallucinated depth for MAX to promote.
(ii) *Headroom*: DF-MCSS at 183.4 leaves ~18 points below the critic ceiling;
the tree recovers roughly half of it. The absolute ordering DF < DV is the price of
causality (a left-to-right sampler commits without seeing its own future; the DF
paper's own maze2d-large figure, 159.0, sits far below ours) — and that price is
exactly what the tree partially refunds.

## 4. The headroom curve: three backbones, one line

A shortcut-forcing backbone (Diffusion Forcing + Shortcut Models' dyadic
self-consistency; 8 joint Euler sweeps, ~5.8× faster end-to-end) provides a
third, weaker point on the same axis. maze2d-large, seeds 0–4, n=125:

| backbone | flat MCSS | tree (top-3) | tree − flat |
|---|---|---|---|
| DV (full-sequence) ‡ | 204.4 | 202.2 | −2.16 (t=−2.60) |
| DF (causal pyramid) | 183.4 | 192.4 | +9.0 (t=3.90) |
| shortcut-forcing (8 sweeps) | 148.3 | 185.5 | +37.2 (t=6.58) |

‡ DV row is start-matched against the compute-matched k256 baseline on the tree
arm's own seeds (0–2); against the root-width k50 baseline the same tree ties
(+0.25, n.s.). See the §2 measurement note.

As the flat baseline falls 204 → 183 → 148, the tree's gain grows monotonically
while its landing point barely moves (202 → 192 → 185): **the tree is a partial
equalizer across backbone quality**. A caution ride-along: the DV critic scores
shortcut plans *above* the 52-sweep DF planner's open-loop (+0.055 vs −0.003),
yet shortcut-MCSS executes worst (148.3) — the hop-tail the gate flagged (p99
0.62 vs 0.38 real) materializing as un-followable plans. *Few-step samplers can
look good to a value model and be worse for control.*

## 5. Kitchen: the dichotomy replicates on a second environment

FrankaKitchen (kitchen-mixed-v0) differs from maze2d in observation space (60-D
manipulation vs 4-D nav), metric (4 discrete subtasks vs continuous return),
window coverage (H·stride ≈ 124 of 280 steps — depth carries information), and
data regime (sub-optimal mixed demonstrations). After reproducing the DV
baseline (75.0 ≈ published 73.6; SEM ≈ 0 — every rollout completes exactly 3 of
4 subtasks; width flat across k150/k300/k600 = 75.0/74.5/75.0):

| arm (tree = r150/b15/k16/top-3, critic @200k) | score |
|---|---|
| DV-MCSS | 75.0 |
| DV-tree | 74.0 (−0.5, t=−0.57, n=50 — null) |
| DF-MCSS k150 / k600 | 60.5 / 57.0 (width flat) |
| DF-tree, seeds 0–3 | 68 / 71 / 73 / 68 |
| **DF-tree − DF-MCSS, pooled n=100** | **+10.5, paired t = 5.47** (sign test 48W/10L/42T, p ≈ 4×10⁻⁷) |

The DF backbone passed its fidelity gate more cleanly than maze2d's (critic
gen-vs-real gap 0.0004 vs 0.065; generated hop statistics indistinguishable from
data). Because kitchen scores are quantized, the paired differences read directly
as subtasks: **the tree completes one extra subtask in 43% of rollouts** (two
extra in 5%). Two controls come free: *compute* — flat DF-MCSS at k600 spends
more planner windows (600) than the tree (~390) and scores 13 points below it,
so the gain is composition, not sampling volume; *backbone* — the DV-tree null
on the same environment and tree code rules out "kitchen just rewards trees."
The maze2d dichotomy therefore replicates exactly: **the deciding variable is
the backbone's conditioning faithfulness, not the environment.**

## 6. Per-token noise-aware classifier guidance

The remaining gap (DF-tree 70.0 < DV-MCSS 75.0) pointed at the evaluator: the DV
critic's 0.0004 gen-vs-real gap means it is saturated on shape-realism and
cannot rank task completion among faithful samples. The response was a
generation-side lever: a **per-token noise-aware value model V(x, k)** — the
diffusion-forcing property applied to the value function — used as classifier
guidance on the frozen DF sampler (eps-shift, ε ← ε − w·√(1−ᾱ[k])·∇ₓV,
self-annealing per token as it denoises).

Positioning (novelty claim): classical classifier guidance (Diffuser; the DV
codebase's own `cg` path) trains its classifier at *one trajectory-level noise
per window*. Under Diffusion Forcing there is no single "noise level of the
trajectory" — history rides clean while the far future is noise — so the value
model must condition at **token-level noise resolution**, and is trained on the
sampler's actual query distribution (pyramid-schedule rows with clean-history
prefixes, mixed with uniform coverage). This pushes the reward model from
trajectory-level to token-level resolution, letting guidance weight each
timestep's contribution by how reliable it currently is. First-order novelty
check: arXiv:2405.20555 (Diffusion Actor-Critic) applies Q-gradients inside
policy *training* and trains no noise-conditioned value model; no per-token
noise-aware guidance for planning was found. CFG was deliberately not run
(§7 retires it on principle), and the noise critic was deliberately not used as
the in-tree node value, keeping the DV critic the constant evaluator across all
arms.

**Open-loop gates.** Held-out sched-pattern correlation **0.915** (peak at 80k
of 200k steps — the value-model overfit pattern recurs; the best checkpoint is
deployed). The strength sweep is a clean monotone pass with **no physical cost
through w=8**: hop p99 stays at real-data level (1.03–1.11 vs real 1.09) while
the *independent* DV critic's score of generations rises near-linearly, reaching
+0.064 *above real data* at w=8 (0.294 vs 0.231) — an evaluator that took no
part in guidance confirming the shift.

**Closed loop (kitchen-mixed, seed 0, n=25/arm; single-seed, directional).**

| guidance | flat MCSS | tree (top-3) | tree − flat |
|---|---|---|---|
| none | 60.0 | 68.0 | +8.0 |
| w = 4 | 64.0 | 70.0 | +6.0 |
| w = 8 | 66.0 | 70.0 | +4.0 |

Guidance dose-responds on the flat baseline (specifically eliminating 1-task
failure rollouts), but the tree's landing point is **pinned at 70.0 across three
guidance strengths** while its gain shrinks monotonically (+8 → +6 → +4). This
is the headroom law's third appearance, now *within one environment*: guidance
lifts the pool, search composes over the pool, and the two partially substitute
because both surface plans from the same manifold. Guidance costs ~1.9× wall
(one critic forward+backward per denoising sweep). (Provenance: the w=4 tree
run's raw per-rollout vectors were overwritten by a filename collision; its
summary statistics — 64.0 ± 2.5 / 70.0 ± 2.0, +6.0 paired — are recorded in the
runbook and here.)

**Seed replication of the flat guidance lift (2026-07-11).** Flat-only guided
arms were replicated across seeds and paired against the unguided arms on
matched environment instances (same seed → same reset sequence; the analysis
tool pairs by the JSON's internal cg_w field): pooled flat DF-MCSS is 59.8
(n=150) unguided → 62.0 (w4, n=50) → 64.3 (w8, n=100). Paired: **w=8 is
confirmed at +4.67, paired t = 2.22, n = 75, 3 seeds** (24W/12L/39T — one
extra subtask in a third of matched environments), clearing the pre-declared
t ≥ 2 bar; **w=4 does not separate from noise** at the replicated sample
(+1.50, t = 0.46, n = 50) — the seed-0 w4 reading was optimistic. The
guidance claim is therefore stated at w=8 only, and the dose-response is
better described as "confirmed lift at sufficient strength" than as a
three-point curve.

## 7. The boundary: the demonstration ceiling

A census of **all 750 kitchen rollouts** recorded across every configuration —
DV-MCSS (200/200 rollouts at exactly 75.0), DV width scans, DV-tree, DF-MCSS to
k=600, four DF-tree seeds, and all guided arms — contains **not a single
4-subtask completion**. Kitchen-mixed's defining property is that its
demonstrations never solve all four tasks; consequently every *learned*
component in the stack (planner, inverse-dynamics policy, DV critic, noise
critic) carries labels and targets capped at the 3-task ceiling. A learned value
cannot prefer a 4-task plan — such a plan lies outside its label range — so
**search steered by learned values cannot pass the dataset's demonstration
ceiling.** DV-MCSS's 75.0 is that ceiling; the DF stack's levers closed
two-thirds of the gap to it (60 → 70) and stopped, as they must.

**The data-side premise, verified (2026-07-11).** Per-trajectory maxima of
simultaneously-solved goal subtasks in the raw datasets (613 demonstrations
each — the same corpus, relabeled per split's goal set): kitchen-**mixed**
{0: 1, 1: 35, 2: 312, 3: 265} — **no demonstration ever reaches 4**;
kitchen-**partial** {1: 87, 2: 259, 3: 248, 4: 19} — 3.1% of demonstrations
reach all four, confirming the environment supports the 4th task when the
data teaches it. (Measurement note: the dataset's reward field is dense —
per-step count of currently-solved goal subtasks — while the environment pays
sparse completion events at evaluation; the ceiling statistic is the
per-trajectory max of the dense field.)

**A fourth method family obeys the same wall.** The DV pipeline's *original*
trajectory-level classifier guidance (`guidance_type=cg`: guided generation +
selection by the classifier's own log-probability over 150 candidates) was run
as a pre-registered test — prediction: ≤ 75, since it optimizes the same
label-capped returns. At w=2.0 it scores **74.75 ± 0.25** with per-rollout
distribution {50: 1, 75: 99}: parity with DV-MCSS, and 100 more census
rollouts with the 100-bin still empty (850+ rollouts, four method families,
zero 4-task completions). [Remaining w rows of the scan pending; expected to
sit in the same 73–75 band.]

**The generation-level census: where exactly the ceiling binds (2026-07-11).**
With the grounded checker (below, §7.1) as a non-learned scorer of *imagined*
windows, the boundary can be located inside the generator itself. Conditioning
the frozen DF planner on real dataset states and sampling 150 windows per
state (9,600 windows per condition): from **3-done** states — the dataset's
best-mode contexts — **every single window scores exactly 3** (0/9,600 reach
4; per-state max = 3 for all 64 states), and per-token guidance at w=8 changes
*nothing* (an identical 0/9,600 — a learned guide cannot summon what its
labels never contained, the label-cap mechanism's third independent
confirmation). But from **2-done** states, 28.45% of windows imagine the
*demonstrated* 2→3 transition (validating the diagnostic), and **24 windows
across 7 of 64 states imagine four-solved — above the demonstration ceiling.**
The generator, unlike every learned value, is therefore *not strictly capped
by the data*: it can compose undemonstrated completions, but only from
contexts with rich continuation diversity (2-done), never from the 3-done
manifold, which is a dead end in the demonstrations (trajectories that reach
3 stop there) and hence a dead end in the model's imagination. Caveats of
record: the rate is 0.25%, and the union-over-steps scoring counts transient
near-goal passes, so these imaginations may be unexecutable grazes — the
grounded closed-loop arm (which prefers such windows whenever sampled at
2-done steps during a rollout) is the test.

**Grounded closed loop (seed 0, n=25/arm): the crossing fails, and the failure
is diagnostic.** Grounded-valued selection scores *below* its critic-valued
counterparts — flat 55.0 ± 2.8 (vs 60.0–60.5) and tree 63.0 ± 3.2 (vs
68.0–70.0), with the tree's +8.0 gain over its own flat arm preserved — and
no rollout completed the 4th subtask. The mechanism is the dissertation's
second imagination/execution dissociation (the first was the shortcut
backbone, §4): the grounded value ranks plans by *imagined touched-count*,
which rewards exactly the transient near-goal grazes the union scoring
admits; preferring a grazing window over the critic's data-like, executable
one diverts the executed first step toward completions that never get cashed,
and the coarse 5-level score invites preference-dithering across replans.
The learned critic's conservatism — scoring shape-realism rather than
optimistic achievement — turns out to be load-bearing for execution. The
boundary is therefore enforced **twice, independently**: in the values (the
label cap — no learned evaluator can prefer a 4-task plan) and in the
executable data manifold (the one evaluator exempt from the cap finds the
generator's rare beyond-ceiling imaginations, and they do not survive
execution; the 3-done bottleneck offers it nothing at all). Within the
offline problem as posed, every layer — values, guidance at both noise
resolutions, search, and grounded selection over the generator's own
beyond-ceiling imaginations — has now been tested against the ceiling, and
the ceiling held. What would change the answer changes the problem: complete
demonstrations (kitchen-partial lifts the ceiling to 94 by construction),
online interaction, or synthetic data surgery (hindsight-stitching 3-done →
4th-task sequences across demonstrations) — the future-work chapter's
opening list.

**Extraction reliability vs demonstration frequency** — the ceiling story in
one contrast, using only these distributions and published numbers. On mixed,
the best demonstrated outcome (3-of-4) appears in 43% of demonstrations
(265/613), and the deployed DV stack achieves it in **100% of rollouts** —
selection-based inference extracts the dataset's best mode with certainty,
which is why DV sits *exactly at* the ceiling. On partial, the best mode
(4-of-4) appears in only 3.1% of demonstrations, and DV converts ~76% of
rollouts to it (published 94.0 ≈ 3.76 subtasks). The DV paper's own
mixed-vs-partial contrast (73.6 vs 94.0 — identical environment and
architecture, different data) is therefore the *intervention* form of this
chapter's boundary claim: move the demonstration ceiling and the method's
score moves with it; hold it fixed and no learned-value lever passes it.

The census retires the untried levers on principle rather than by (unaffordable)
exhaustion: CFG-DF conditions generation on the same capped returns; deeper
trees, stronger backbones, and larger w optimize the same capped values. The one
mechanism that can even *express* a preference for the 4th subtask is a
**grounded subtask checker** — computed from state, not learned, hence not
label-capped — retained as an explicitly speculative extension (the policy would
still need to execute a never-demonstrated context).

## 8. Synthesis: the law and its evidence

**Structured search pays exactly where its expansion is a faithful conditional
generation and the flat baseline leaves evaluator headroom — and no learned-value
method passes the data's demonstration ceiling.** Clause by clause:

| clause | evidence |
|---|---|
| faithful expansion is necessary | triptych: glue −4.3 / inpaint −18.5 / DF +9.0..10.5 (2 envs) |
| comparable node values are necessary | composed-window scoring; V(s) posedness ladder (corr 0.39→0.98) |
| tempered backup is necessary under evaluator noise | winner's curse −4.29 (seed-t −4.75), top-3 +4.54 (roll-t 5.00); curse signature absent on DF |
| the gain scales with flat-baseline headroom | backbone curve −2.16/+9.0/+37.2; within-kitchen curve +8/+6/+4 as flat rises 60→66 |
| the tree is an equalizer, not an amplifier | landing points 202/192/185 (maze2d) and 68/70/70 (kitchen) vs flat spans of 57 and 6 points |
| guidance and composition partially substitute | tree gain shrinks monotonically under guidance; same landing point |
| the ceiling is the data's, not the search's | 750-rollout census, zero 4-task completions, every method incl. the DV baseline |

Cost accounting (honesty row): DV-tree = parity at ~6× compute; DF-tree ≈ 2.5×
MCSS wall per seed; guidance ×1.9 on top; shortcut recovers ~5.8× where its
quality suffices. Search is never the cheap option — it is the option that
converts a weak sampler's budget into most of a strong sampler's result.

## 9. Limitations and threats

1. **CG arm status after seed replication**: the flat guidance lift is
   confirmed at w=8 (+4.67, paired t=2.22, n=75, 3 seeds) but NOT at w=4
   (+1.50, t=0.46) — quote the w=8 claim only. The guided-*tree* arms (the
   pin) remain single-seed/directional; the +10.5 unguided tree gain and the
   maze2d +9.04 are the fully confirmed search claims.
2. **maze2d compute-matched control — RESOLVED.** The start-matched k256 MCSS
   *is* the k≈290 compute-matched baseline: the DV critic-tree loses −2.16
   (t=−2.60) to it and width saturates (k50 199.4 → k256 201.2), so the tree
   reaches the ceiling by efficient sampling, not composition beyond it —
   matching kitchen's k600 control (§7). No longer an open item.
3. **The w=4 guided-tree raw vectors were lost** to a filename collision
   (summary statistics preserved); the pin claim rests on three landing points,
   two with full vectors.
4. **Antmaze is calibration only** (locomotion-capped; DF-MCSS 44% reach) — the
   law is tested on two environments, not three.
5. **Evaluator caveats carry over from methodology** §9: the DV base critic is
   near-memorized (unquantified holdout), so absolute magnitudes of critic-side
   MSE ratios are directional only and are quoted as such; all within-backbone
   comparisons share the evaluator, so orderings are unaffected.
6. **The demonstration-ceiling argument is now verified on both sides** (raw
   mixed data max = 3-of-4 across all 613 demos; 850+ rollout census with an
   empty 100-bin across four method families). Residual caveat: it is an
   empirical-plus-mechanism claim, not a theorem — a non-learned evaluator
   (grounded subtask checker) with strong policy generalization could in
   principle exceed it; no learned-value method can.
7. **Shortcut-on-kitchen and the grounded checker are designed but unrun** —
   reported as such, not as results.

## 10. Figure and table inventory (for the document)

Figures F1–F6 are GENERATED by `scripts/make_figures.py` into `figures/` as
vector PDF (for LaTeX `\includegraphics`) + PNG preview, from the per-rollout
`results/*.json` vectors (kitchen arms, maze2d DF/shortcut 5-seed pools) plus
the documented DV maze2d baselines (tagged `[DOC]` in the script with their
methodology-§7.5 source). Palette follows the validated data-viz defaults
(blue/aqua categorical, blue↔red diverging, blue sequential ramp for the
ordinal census); re-run the script to regenerate after any results change.

1. **F1 — the mechanism triptych** `fig1_expansion_triptych` (bar: glue −4.3 /
   inpaint −18.5 / DF +9.0 vs each start-matched flat baseline, maze2d):
   unfaithful conditioning is the poison.
2. **F2 — winner's curse** `fig2_winners_curse` (MAX vs top-3 on DV vs DF;
   +4.54 roll-t 5.00 vs +1.8 n.s.): backup temper needed exactly when expansion
   lies.
3. **F3 — backbone headroom curve** `fig3_headroom_curve` (flat vs tree,
   shortcut/DF/DV; gaps +37 / +9 / −2.2, DV row on the start-matched k256
   baseline 201.2): the equalizer.
4. **F4 — kitchen 2×2** `fig4_kitchen_2x2` (DV/DF × flat/tree, paired SEM;
   DV null −0.5, DF +10.5 t=5.47): the dichotomy replicates.
5. **F5 — within-kitchen guidance curve** `fig5_guidance_pin` (flat 60/64/66
   vs tree 68/70/70 across w=0/4/8): the pin; guidance and search as partial
   substitutes.
6. **F6 — the census** `fig6_census` (stacked per-rollout subtask-level shares,
   7 arms; the darkest "4 (100)" bin never appears): the demonstration ceiling.
7. **T1 — protocol table** — §11 below.
8. **T2 — the law table** (§8 above) — prose table ready to typeset.

## 11. T1 — Experimental protocol

All closed-loop arms below run the frozen DV stack (critic + diffusion
inverse-dynamics policy); only the planner/expansion/value is varied within a
comparison. Rollouts are paired: `--method both` runs MCSS and the tree on the
same environment instances with a shared RNG stream, so MCSS-vs-tree deltas are
within-instance. "n" is rollouts per arm (n_envs × n_episodes). Wall times are
per seed on the project's single-GPU box (the full `both` run = MCSS arm +
tree arm). Tree config throughout: k_mcts 16, child_index 1, top-m backup 3,
UCB c=√2; maze2d k_root/k_mcss 50, kitchen 150. Critic checkpoint: maze2d 1M,
kitchen 200k (the DV-config value; 1M overfits — methodology §8.2).

| # | Env | Backbone | Arms (paired) | value / guidance | k | budget | seeds | n/arm | wall/seed (MCSS→tree) | result | status |
|---|-----|----------|---------------|------------------|---|--------|-------|-------|----------------------|--------|--------|
| A0 | maze2d-large | DV (full-seq) | MCSS k50; MCSS k256 (width control) | critic | 50/256 | — | s0–9 | 25 | — | 199.4 / 201.2 (+1.8 for 5× samples — saturated) | confirmed |
| A1 | maze2d-large | DV (full-seq) | MCSS; tree glue MAX | critic | 50 | 15 | s0–2 | 25 | — | 199.4 / 197.7 (**−4.29, seed-t −4.75**) | confirmed |
| A2 | maze2d-large | DV | MCSS; tree glue top-3 | critic | 50 | 15 | s0–2 | 25 | — | 199.4 / 202.2 (**+0.25, t=0.33 — parity**; −2.16, t=−2.60 vs k256) | confirmed |
| A2b | maze2d-large | DV | MCSS; naive tree, original cfg | critic | 50 | 15 | s0–9 | 25 | — | 199.4 / 194.3 (**−5.05, seed-t −10.57, n=250**) | confirmed |
| A3 | maze2d-large | DV | tree inpaint MAX | critic | 50 | 15 | s0 | 25 | — | 182.1 (**−18.53, t=−3.02**) | established |
| A4 | maze2d-large | DV | tree V̂(s) plan-value | distilled V(s) | 50 | 15 | s0–2 | 25 | — | 202.5 (+0.52, t=0.30 — parity) | established |
| **B1** | maze2d-large | **DF** (causal) | MCSS; tree top-3 | critic | 50 | 15 | **s0–4** | 25 | 21 min → 121 min | **183.4 / 192.4 (+9.04, t=3.90, n=125)** | **confirmed** |
| B2 | maze2d-large | shortcut (8-sweep) | MCSS; tree top-3 | critic | 50 | 15 | s0–4 | 25 | 3.7 min → 21 min | 148.3 / 185.5 (+37.2, t=6.58, n=125) | confirmed |
| C1 | kitchen-mixed | DV | MCSS (paper repro) | critic | 150 | 15 | s0 | 200 | 33 min | 75.0 (≈ paper 73.6) | confirmed |
| C2 | kitchen-mixed | DV | MCSS; tree top-3 | critic | 150 | 15 | s0 | 50 | 8 min → 23 min | 74.5 / 74.0 (−0.5, null) | confirmed |
| **D1** | kitchen-mixed | **DF** | MCSS; tree top-3 | critic | 150 | 15 | **s0–3** | 25 | 23 min → 58 min | **59.5 / 70.0 (+10.5, t=5.47, n=100)** | **confirmed** |
| D2 | kitchen-mixed | DF | MCSS width control | critic | 600 | — | s0 | 50 | 248 min | 57.0 (< tree, more windows) | confirmed |
| **E1** | kitchen-mixed | DF | MCSS +CG (w=8) | per-token noise CG | 150 | 15 | **s0–2** | 25 | 47 min | **64.3 (+4.67 vs unguided, t=2.22, n=75)** | **confirmed** |
| E2 | kitchen-mixed | DF | MCSS+CG; tree+CG (w=4,8) | per-token noise CG | 150 | 15 | s0 | 25 | 46→118 min | flat 64/66, tree pinned 70 | directional |
| F1 | kitchen-mixed | DF | MCSS; tree, grounded value | grounded + 0.25·critic | 150 | 15 | s0 | 25 | 22 min → 57 min | 55.0 / 63.0 (below critic; no 4-task) | single-seed |
| G1 | kitchen-mixed | DV | MCSS, orig. classifier guidance | trajectory-level CG | 150 | 15 | s0 | 100 | ~40 min | 74.75 (pre-registered ≤75) | confirmed |

Backbone-fidelity gates (open-loop, `scripts/check_df_ckpt.py`) preceded every
DF/shortcut closed-loop arm; the demonstration-ceiling verification
(`scripts/check_kitchen_ceiling.py`) and the grounded go/no-go
(`scripts/check_grounded_pool.py`, 9,600 windows/condition) are the open-loop
diagnostics behind §7. Note the compute asymmetry the law rests on: on maze2d
the DF tree costs ~6× its MCSS arm (121 vs 21 min) for +9; the shortcut
backbone delivers a comparable relative gain at ~6× lower absolute cost
(21 vs 121 min for the tree), which is its purpose.
