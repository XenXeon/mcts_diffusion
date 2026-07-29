# Integrating Monte-Carlo Tree Search into a Diffusion Planner: Methodology Report

*Dissertation working document — complete record of the investigation, every design
decision with its rationale, algorithms/pseudocode, results, and threats to validity.
Covers the full arc up to 2026-07-06 (Diffusion Forcing backbone, first tree win).
Companion records: `notes/PROJECT_HANDOFF.md` (stale past §7), `notes/findings_summary.md`
(antmaze arc), `notes/value_lever_findings.md` (value-lever arc, §5b–5d).*

> Citation details (authors/venues) were written from memory and should be verified
> against the actual papers before submission. arXiv IDs are given where confident.

---

## 1. Problem statement and how the research question evolved

**Starting question:** can Monte-Carlo Tree Search improve a state-of-the-art diffusion
planner, replacing its flat best-of-K sampling with structured look-ahead search?

**Base system:** Diffusion Veteran (DV) [Lu et al., ICLR 2025], the strongest published
diffusion planner on D4RL [Fu et al., 2020, arXiv:2004.07219]. DV's empirical study
found that the best inference pattern is **MCSS** (Monte-Carlo sampling + selection):
sample K trajectory plans, rank them with a learned whole-trajectory critic, execute
the best plan's first step, replan (model-predictive control). Notably, DV found
sample-and-rank *beats* classifier guidance [Janner et al., ICML 2022, arXiv:2205.09991;
Ajay et al., ICLR 2023, arXiv:2211.15657] — a finding this project relies on repeatedly.

**Final question (after the investigation reshaped it):** *under what conditions does
tree search help a diffusion planner at all?* The answer, established causally across
~15 controlled experiment arms: search helps **iff (a) node values are well-posed, (b)
all nodes are scored on comparable windows, (c) backups are robust to evaluator noise,
(d) expansion is a faithful conditional generation, and (e) the flat baseline has not
already saturated the evaluator's ceiling.** On the frozen DV planner (a)–(c) are
fixable and buy parity at ~6× compute; (d) is impossible without retraining the
planner — which is what the Diffusion Forcing backbone (§7) finally provides, producing
the first tree-over-flat win of the project.

---

## 2. The DV pipeline (all components frozen unless stated)

Three independently trained networks (no joint training — three losses, three
optimizers, zero coupling; verified in `pipelines/veteran_d4rl_maze2d.py`):

| component | architecture | input → output |
|---|---|---|
| planner | DiT1d (full-attention transformer denoiser), d_model 256, depth 2 (maze2d) / 8 (antmaze) | noise → (H, D) window of stride-spaced future states, conditioned on row 0 = current state via replacement (fix-mask) |
| critic | DVHorizonCritic (transformer), d_model 256, depth 2 | (H, D) window → scalar value = normalized discounted return of the window |
| policy | DVInvMlp (inverse dynamics MLP) | (s_t, s_{t+1}) waypoint pair → action |

Geometry: maze2d H=32, stride 15 (window spans 480 dense steps ≈ whole episode);
antmaze H=40, stride 25 (spans 1000 = whole episode); kitchen H=32, stride 4 (spans
128 of 280 — the only family whose window does NOT cover the horizon).

Value target (`seq_val`): per-timestep discounted return from IQL-tuned, padded rewards
[Kostrikov et al., 2021, arXiv:2110.06169], min-max normalized to [−1, 1]. With maze2d's
discount 1.0 this is a normalized negative time-to-terminus.

**MCSS (baseline) pseudocode:**

```
every env step (MPC loop):
    prior[:, 0] = s0;  sample K windows ~ planner(prior)      # one batched call
    scores = critic(windows)                                   # (K,)
    best   = windows[argmax scores]
    a      = inv_dynamics(s0, best[1])                         # first waypoint only
    execute a; observe s0'; repeat                             # full replan each step
```

---

## 3. Phase 1 — why the naive tree loses: it is a VALUE comparison

The first MCTS implementation (per-state value network V(s) guiding UCT selection
[Kocsis & Szepesvári, 2006]) lost catastrophically: **167.3 vs a paired MCSS 202.7,
a −35.4 point deficit at p < 1e-3** (maze2d-large, DV camping score). The key
diagnostic insight of the project:

> MCSS-vs-MCTS was never flat-vs-structured. It is a comparison of *evaluators*.
> MCSS uses the whole-trajectory critic (well-posed target, dynamics-aware, evaluates
> the actual decision object). The tree needs per-state values, and every per-state
> evaluator available was worse: learned V(s) (ill-posed target, §5) > oracle geodesic
> (dynamics-blind — ignores velocity, so it is wrong exactly at decision points).

Supporting evidence: an oracle BFS-geodesic re-ranker *underperforms* the learned
critic on both maze2d and antmaze; antmaze was additionally shown to be
locomotion-capped (the Ant topples; n=500-node trees tie k=50 sampling; fall-geometry
analysis refuted wall-collision explanations) — so antmaze headroom measures
locomotion, not planning, and it was demoted to a calibration environment.

**Decision:** put the *critic itself* in the tree, and separately attack the V(s)
target (§5).

---

## 4. Phase 2 — critic-in-tree: window consistency, width/depth, winner's curse

### 4.1 Composed-window scoring (why naive critic-in-tree is biased)

A naive critic-in-tree scores each node's continuation on the continuation's own
time-shifted window [t_node, t_node+H). On progress/camping tasks, later windows score
systematically higher (more of the window is near the goal), so max-backup rewards
*visits*, not merit — a window-shift bias worth ~1.5 points. Fix (`mcts/window.py`):
every node is scored on the SAME window anchored at the real state s0:

```
composed_window(node) = (prefix ++ continuation)[0:H]     # prefix = search-chosen
critic_value(node)    = critic(composed_window(node))     #   waypoints s0..node
```

All backups then compare like with like. This "glue" expansion is the established
DV-backbone tree.

### 4.2 The tree algorithm (ValueForest, `mcts/value_forest.py` + `mcts/mcts_loop.py`)

```
MCTS-PLAN(s0, budget B, widths k_root/k, chunk L, backup top_m, c_ucb):
    root ← node(state=s0, prefix=∅)
    EXPAND(root, k_root)                        # root round: wide, its children
                                                #   are what MCSS's pool competes with
    repeat B times (one batched round across all M parallel envs):
        leaf ← descend by UCB:  argmax_child  v(child) + c_ucb * sqrt(ln(N_parent+1)/(N_child+1))
        EXPAND(leaf, k)
        BACKUP(leaf → root):  v(node) ← mean of top_m child values   # top_m=1 ⇒ MAX
    execute inv_dynamics(s0, first_waypoint(best root child)); replan next env step

EXPAND(node, k):                                # the search's only GPU touch-point
    sample k continuations from node.state      #   (one batched planner call)
    children ← continuation[L] for each         # chunk of L waypoints per edge
    v(child) ← critic(composed_window(child))   # SHARED [s0, s0+H) window
    (optional) junction filter: v ← −10 if first hop > dataset p99 step size
```

Design decisions and why:
- **Replan every env step (MPC kept):** inverse dynamics is imperfect; the reached
  state is not the planned state. Verified empirically throughout.
- **`k_root` ≥ k:** the executed action is chosen among *root* children, so root width
  is what competes with MCSS's per-step pool; deep rounds only refine that ranking.
- **top-m backup:** see §4.4. **Junction filter:** cheap feasibility rail (reject
  children whose first hop exceeds the dataset's p99 stride-displacement); never
  decisive on maze2d, useful for the antmaze-DF teleport tail (§7.5).
- **Batched forest:** all M evaluation envs run one tree each; each round batches all
  M expansions into one planner call (25 trees × 16 samples per round).

### 4.3 Width/depth decomposition (why depth could not pay on DV)

maze2d-large, seeds 0–2 paired:
- **Width saturates:** on start-matched draws (seeds 0–9, n=250 each),
  MCSS k50 199.4 → k256 201.2 — **+1.8 for 5× the samples**. The plateau is the
  *critic's own selection ceiling* — beyond it, better candidates exist but the
  evaluator cannot identify them.
- **Depth on a narrow root** recovered only +1.4 of the +6 width would buy: deep
  compute on this env is *inefficient sampling*, never look-ahead — because the
  H-window already covers the decision-relevant horizon (nothing beyond it to see).

### 4.4 The winner's curse, quantified and fixed

With a root pool that is a strict SUPERSET of MCSS's (k_root=50), the MAX-backup tree
still lost **−4.29 to its start-matched flat baseline (seed-t −4.75, 3 seeds, n=75)**.
Pure search harm. Mechanism: max-backup over noisy critic scores of *stitched
composites* promotes the most over-rated child — the optimizer's curse
[Smith & Winkler, 2006]. Supporting measurement (§5, Lever A): the critic's MSE is
0.0766 on stitched windows vs ~1.2e-6 on dataset windows. The ratio is large but its
*absolute magnitude* is not interpretable — the base critic is near-memorised on its
training data with no held-out split (§9.6) — so it is quoted as directional evidence
that the seam is off-manifold, not as a calibrated error ratio. The fix is
statistical, not learned: **top-m backup** (v = mean of m best children): max→top-3
recovered **+4.54 (roll-t 5.00, 3/3 seeds, n=75)** → parity with the flat baseline at
~6× wall clock. This figure is unaffected by the start-matching correction, since it
differences two arms run on identical starts.

---

## 5. Phase 3 — the three value levers (can better values beat MCSS?)

### Lever A — stitch-aware critic fine-tuning: mechanism-rich negative
Fine-tune the critic on stitched windows with EXACT labels via the segment identity
`V_A[sa] − γⁿ V_A[sa+n] + γⁿ V_B[sb]` on the dataset's own value recursion
(`mcts/stitch.py`; label replica matched `ds.seq_val` to 0.00e+00). Result: stitched
MSE 0.0766 → 0.0471 (−38%, plateau ~10k steps). The floor is **aleatoric** — a stitched
label imports path B's return beyond the visible window; no critic can remove that
variance, only the bias. Closed loop (start-matched, seed 0, n=25): the fine-tuned
critic is a **null on the flat baseline** (200.78 vs matched 200.61, +0.16, t=0.12)
and **adds nothing over top-m backup** (201.92 vs 202.26, −0.34, t=−0.64). Neither
arm separates from its control. **Design rule: the curse is variance-driven →
aggregate (top-m), don't retrain the evaluator on the query distribution — the
retrained evaluator buys nothing the cheap statistical fix does not.**
*(Correction of record: earlier drafts reported this arm as actively "hurting" the
baseline, 204.9 → 200.8. That was the cross-start offset, not an effect; on matched
starts it is a null. The design rule is unchanged, but it now rests on "adds
nothing" rather than "does harm".)*

### Lever B — distilled plan-value V̂(s): the posedness falsification (decisive positive)
Hypothesis test: was the V(s) *network* weak, or its *target*? Behaviour-return V(s)
is ill-posed — the same state appears in many trajectories with wildly different
returns (SNR ceiling). The plan-value target V̂(s) = critic score of the best of K=16
planner samples from s is a *deterministic function of the state* given the frozen
planner+critic (offline AlphaZero-style distillation [Silver et al., 2017/2018]).
Same MLP, same optimizer, same states, target swapped:

| env | behaviour-return corr | plan-value corr |
|---|---|---|
| maze2d-large | 0.742 | **0.9805** |
| maze2d-medium | 0.636 | **0.9572** |
| maze2d-umaze | 0.390 | **0.9430** |
| antmaze-large-diverse | 0.874 | **0.9803** |
| antmaze-medium-diverse | 0.513 | **0.8999** |

Closed loop: the same tree that scored **167.3** with behaviour-return V(s) scores
**202.5 / 202.4 / 202.6** (seeds 0/1/2) with plan-value V(s). **The entire 35-point
gap was target posedness.** Transfer: medium tree 151.6 vs paired MCSS 156.6 (n.s.),
umaze 129.2 vs 138.4 (n.s.) — same signature everywhere: a well-posed V(s) buys
parity-minus-a-few-points; the critic's direct advantage supplies the rest.

### Lever C — junction feasibility filter: built, free, kept as a rail (see §4.2, §7.5).

**Phase-3 conclusion:** every value lever on the frozen DV planner ends at the same
place — parity with MCSS, never a win, because DV-MCSS already operates at the
critic's selection ceiling. The remaining suspect was the *expansion mechanism itself*.

---

## 6. Phase 4 — prefix-inpainting: the negative result that motivated Diffusion Forcing

**Idea** (from Diffusion Forcing, brought in via TA discussion): glue-expansion's seam
— continuation sampled from the leaf state alone, concatenated onto the prefix — is
exactly where the critic's off-manifold error lives. Condition the *frozen* DV planner
on the whole prefix instead, by clamping prefix rows into the denoiser at every
diffusion step (the same replacement conditioning DV already uses for row 0; the
standard inpainting trick, cf. RePaint [Lugmayr et al., CVPR 2022]).

**Result:** r50-MAX inpaint **182.1** vs glue **198.1** — paired per-env t = −2.84,
p ≈ 0.009 (identical starts/goals). Verified three independent ways: a full static
audit (9/9 invariants, root expansion bit-identical to glue), a mechanical diagnostic
(`clamp_err = 0.00e+00` — the clamp works perfectly), and the paired closed loop.

**Diagnosis** (`scripts/diag_inpaint.py`): generated steps are physically normal, but
critic scores at prefix depth j≥4 are *inflated* relative to glue (0.145 vs 0.069 at
j=4; 0.156 vs 0.112 at j=8) while glue's scores *fall* with depth. Reading:

> **The seam was the critic's accidental defense against imagined depth.** Seam-free
> windows look like pristine data and score high while being no more executable, so
> MAX-backup trusts hallucinated depth more — the curse amplified, −16 points.

Root cause: clamping d+1 clean rows among noisy ones is a **mixed-noise-level input
the DV planner never saw in training** (it was trained with row-0 clamping only).
Replacement conditioning does not extend beyond its training configuration; no
inference-time trick fixes this. The planner must be *trained* for mixed noise levels
— which is precisely Diffusion Forcing's training objective.

---

## 7. Phase 5 — the Diffusion Forcing backbone

### 7.1 What Diffusion Forcing is
Diffusion Forcing (DF) [Chen et al., NeurIPS 2024, arXiv:2407.01392] trains a *causal*
sequence diffusion model in which **every token carries an independent noise level
k ∈ {0..K}** (noising-as-masking: k=0 = clean/unmasked, k=K = pure noise). Trained to
denoise arbitrary per-token noise configurations, the model supports, at sampling
time, any 2D scheduling matrix over (denoising sweep × token position) — including
"clean history + noisy future," i.e. **exact conditional generation from a prefix**,
the capability tree search needs and full-sequence diffusion fundamentally lacks.

### 7.2 Why our integration deliberately differs from the paper

| aspect | DF paper | this project | rationale |
|---|---|---|---|
| backbone | RNN (GRU latent); transformer only sketched (their App. B.1) | causal transformer (`CausalDFDiT`) | RNN sampling needs M×T ≈ 2,600 sequential cell calls per plan (closed-loop ≈ 10 h/arm); the transformer does one parallel forward per sweep (~52). Also architecturally closest to DV's DiT1d → fairer comparison. Transformer-DF later validated at scale by DF follow-up work (DFoT / History-Guided video diffusion). |
| tokens | [action, reward, next-obs] tuples | state-only stride-spaced waypoints (DV's exact planner windows) | keeps the frozen DV critic + inverse-dynamics policy usable — the planner is the ONLY swapped component, so every DF arm shares the evaluator and executor with every DV arm. |
| plan selection | classifier guidance + Monte Carlo Guidance (MCG) | sample-and-rank with the DV critic; NO guidance | DV's own study: sample-and-rank beats guidance. Also keeps selection machinery identical across arms. (MCG's principle — average over futures rather than max — independently corroborates our top-m design rule.) |
| training budget | (theirs) | K=20 discrete levels (cosine ᾱ [Nichol & Dhariwal, 2021], ᾱ₀=1 exactly), 400k steps, EMA 0.999 | GPU-time economy; flagged as the two legitimate quality levers (K=50, 800k steps) if the backbone ever needs strengthening. |

The fair description: *a Diffusion-Forcing-style causal planner (paper Algorithms 1–2,
transformer variant), adapted to DV's planning interface* — not a port of their
pipeline. Their full pipeline scored 159.0 on maze2d-large (their Table 1, with MCG);
adopting it wholesale was never the goal.

### 7.3 DF training (Algorithm 1, adapted — `mcts/df_model.py`, `scripts/train_df_planner.py`)

```
given: dataset windows x0 ∈ R^(B×T×D)      # byte-identical distribution to DV planner training
repeat:
    k   ~ Uniform{0..K}  independently per token          # (B, T)
    eps ~ N(0, I)
    x_k = sqrt(ᾱ[k]) · x0 + sqrt(1−ᾱ[k]) · eps            # per-token noising
    ê   = CausalDFDiT(x_k, k)       # causal attention: token t sees tokens ≤ t only;
                                    # per-token adaLN conditioning on k (vs DiT's global t)
    L   = MSE(ê, eps) over tokens with k ≥ 1              # k=0 tokens: eps unidentifiable
                                    # from a clean input — they train the CONTEXT pathway
                                    # (clean history among noisy tokens) but not the loss
    AdamW step; EMA update
```

Training-loss caveat for the write-up: DF's eps-loss (~0.36 on maze2d) is **not
comparable** to DV's (~0.04). DV denoises with full bidirectional attention at one
shared level (32 correlated views of the trajectory); DF predicts causally on
random-walk data whose future is genuinely multimodal given the past — part of the
loss is irreducible conditional entropy, not model error. Sample quality is the
metric: generated hop statistics matched real data exactly (0.1586 vs 0.1596), and
DF samples scored within 0.065 of *real dataset windows* under the DV critic.

### 7.4 DF sampling and DF-in-tree inference (Algorithm 2, adapted)

```
DF-SAMPLE(history rows h[0:ℓ], total window T, K levels):
    build pyramid matrix  Ksched[m, t] = clip(m − slope·(T−1−t), 0, K)
        # column-anchored: rows = sweeps; early tokens denoise first, the far future
        # stays noisier ("causal uncertainty"); one matrix serves a whole batch of
        # nodes with different history lengths — history columns forced to level 0
    x ← N(0, I);  x[0:ℓ] ← h  (level 0 throughout — clean history is IN-distribution
                               by training: this is exact conditioning, no clamp hack)
    for each sweep m (top row → bottom, ~52 parallel net forwards):
        ê ← net_EMA(x, k_prev)                      # ONE forward for all T tokens
        for tokens where k_new < k_prev:            # deterministic DDIM jump
            x0̂ ← (x − sqrt(1−ᾱ[k_prev])·ê) / sqrt(ᾱ[k_prev]),  clipped
            x  ← sqrt(ᾱ[k_new])·x0̂ + sqrt(1−ᾱ[k_new])·ê       [Song et al., 2021]
        re-assert history rows
    return x                                        # (T, D); rows [0:ℓ] = history exactly

DF-MCSS:      windows = DF-SAMPLE(history=[s0], T=H) × K;  rank with DV critic; execute; replan.
DF-MCTS:      identical tree to §4.2, except EXPAND(node, k) generates k windows via
              DF-SAMPLE(history = node.prefix ++ node.state) — the window IS the
              composed [s0, s0+H) window, natively; critic scores it directly.
              (harness: --df-ckpt swaps the backbone for BOTH arms; DV critic + policy
              + selection + env loop are shared, battle-tested code across all arms.)
```

### 7.5 Results (maze2d-large, identical starts/goals; DF tree win CONFIRMED, 5 seeds)

All DV-backbone figures below are **start-matched**: every tree arm is differenced
per-rollout against an MCSS arm run on the identical `starts` array (asserted
programmatically, see `notes/maze2d_startmatched_correction.md`). Deltas are quoted
against the root-width baseline k50; the compute-matched comparison against k256 is
given separately. Because the DV tree arms ran at seeds 0–2 while the baselines ran
at seeds 0–9, **every delta is taken against the baseline restricted to that arm's own
seeds**; both baseline means are shown so the arithmetic is followable.

| arm (tree = r50 / b15 / k16 / L1) | DV backbone | DF backbone |
|---|---|---|
| MCSS flat, root width (k50) † | 202.0 (s0–2) / 199.4 (s0–9) | 182.7 |
| MCSS flat, compute-matched (k256) † | 204.4 (s0–2) / 201.2 (s0–9) | — |
| tree, MAX backup (s0–2) | 197.7 (**−4.29 vs k50, seed-t −4.75**) | **190.4 (+7.6, paired t=1.35)** |
| tree, top-3 backup (s0–2) | 202.2 (**+0.25 vs k50, t=0.33 — parity**; **−2.16 vs k256, t=−2.60**) | **192.2 (+9.4, paired t=1.53)** |
| naive tree, original cfg (s0–9) | 194.3 (**−5.05 vs k50, seed-t −10.57, n=250**) | — |
| inpaint tree, MAX (fake cond., s0) | 182.1 (**−18.53, t=−3.02**) | — |
| MAX vs top-3 gap (curse signature) | +4.54, roll-t 5.00 | **+1.8, t=0.97 n.s.** |

**† Measurement note — start-matching and the DV baseline.** The maze2d camping
score is strongly start-state dependent (single-episode variance is large), so only
*start-matched* comparisons — tree seed *i* paired per-rollout with the MCSS
baseline at the same seed — are valid. The tree draws ≈290 planner samples (root 50
+ budget 15 × k 16), so its fair flat comparator is a wide MCSS, not k50; both are
reported. Start-matched, the DV critic-tree **ties** the root-width baseline
(+0.25, t=0.33) and loses to the compute-matched baseline by **−2.16 (paired
t=−2.60)** — confirming *no DV critic-tree win* under either comparator. This
start-matched k256 measurement **is** the compute-matched control formerly listed as
an open item (§9): the DV tree reaches the ceiling by efficient sampling and does not
compose beyond it. Width saturates on matched starts (k50 199.4 → k256 201.2 = +1.8
for 5× the samples), so the deficit is not sampling volume.

Earlier drafts quoted a DV-MCSS baseline of 204.9 and deltas of −6.8 / −2.6 / −22.8.
That baseline is a *valid* measurement on its own 150-rollout start set, but it was
used as the comparator for tree arms run on a different start set, importing a ~4.3
point offset into each delta. The figures above are the corrected, start-matched
replacements; the qualitative conclusions are unchanged and, for the naive tree,
strengthened — on a clean single-configuration 10-seed pool the naive critic-tree
loses **−5.05 (seed-t −10.57, n=250), negative on every one of ten seeds**, far
stronger evidence than the original single-seed figure. The tempered-backup figure is
unaffected by the correction — it differences two arms on identical starts, so the
baseline cancels. Full audit: `notes/maze2d_startmatched_correction.md`.

**Confirmation (2026-07-07, seeds 0–4, n=125 paired).** The seed-0 DF tree result
replicated across five seeds: pooled tree−MCSS = **+9.04 at paired t=3.90** (per-seed
+9.4 / +3.9 / +10.3 / +5.9 / +15.7 — positive on every seed; DF-tree 192.4 vs DF-MCSS
183.4). This clears the project's 3-seed pooled standard: **"tree search flips from
harmful to helpful once expansion is a faithful conditional generation" is an
established result, not a single-seed signal.**

**Why DF helps MCTS (two mechanisms, both evidenced):**
1. *Faithful expansion → trustworthy composite scores.* The expansion-fidelity
   triptych maps monotonically onto tree performance: seam-glue −4.3 < fake
   conditioning −18.5 < exact conditioning +8..9 (each vs its own start-matched flat
   baseline). And
   the winner's-curse signature vanishes (MAX≈top-3 on DF) — with true conditional
   samples, MAX-backup no longer promotes hallucinated depth. Search's failure on DV
   was never search per se; it was lying node evaluations.
2. *Headroom.* DV-MCSS saturates the critic ceiling (~201 on matched starts; width
   buys only +1.8 for 5× the samples) — search had nothing left to buy. DF-MCSS at
   182.7 leaves ~18 points of headroom; the tree recovers roughly half of it. Corollary (falsifiable): improving the DF backbone should *shrink* the tree
   gain — backbone quality becomes the x-axis of a "when does search help" curve.

**Why DF performs worse than DV (expected, not a defect):**
1. *Causality costs information.* A full-sequence denoiser resolves every token
   jointly — the first waypoint is generated already consistent with where the plan
   ends. A causal model commits left-to-right, blind to its own future; on multimodal
   random-walk data, fewer of K samples land on globally good plans. This is the price
   of the exact property (history conditioning) that makes tree search possible — and
   the tree gain is precisely search *giving back* what full-sequence coherence
   provided. External consistency: the DF paper's own maze2d-large (159.0, their
   inference scheme) sits far below DV's operating point; our −22 gap is comparatively
   small. 2. *Training budget:* 400k steps vs DV's 1M; K=20 levels; EMA 0.999 vs 0.9999.

Antmaze DF: backbone-limited (DF-MCSS reach 44% vs DV 76.9%; generated hops 1.8× real
with a teleport tail, p99 0.88; prefix conditioning itself exact, hist_err = 0).
Antmaze remains locomotion-capped regardless (§3), so its role is calibration only —
no retraining planned; `--junction-filter` recommended for any antmaze DF tree run.

### 7.6 The shortcut-forcing backbone and the headroom curve (2026-07-07)

To probe the headroom mechanism (§7.5, mechanism 2) *within* the DF family — and to buy
cheaper inference for the kitchen study — we trained a **shortcut-forcing** planner:
Diffusion Forcing's per-token noise combined with Shortcut Models [Frans et al., 2024,
arXiv:2410.12557] (the "shortcut forcing" objective of the Dreamer 4 world model
[Hafner, Yan & Lillicrap, 2025, arXiv:2509.24527]). The network conditions on the step
size d as well as time t, trained with a flow-matching base case at d=0 and an EMA-target
self-consistency bootstrap on a **dyadic (power-of-two) step grid**, so it samples in a few
*joint* Euler steps (`sweeps` ∈ {1,2,4,8,…}) rather than the pyramid's ~52 sequential sweeps
(`mcts/shortcut_df.py`). At 8 sweeps it samples ~5.8× faster end-to-end, and the DV critic
scores its windows *above* the 52-sweep DF planner's open-loop (gen 0.055 vs −0.003) — but
its plans carry a heavier physical tail (hop p99 0.62 vs 0.38 real; prefix seam 0.28 vs DF's
~0.19), the antmaze-DF teleport signature in milder form.

Closed loop (maze2d-large, seeds 0–4, n=125): **DF-MCSS-shortcut 148.3, tree 185.5 (+37.2,
paired t=6.58)**. Read alongside the DV and DF rows, this is a third point on a **monotone
headroom curve** — as the flat baseline weakens, the tree gain grows while the tree's landing
point barely moves:

| backbone | flat MCSS | tree (top-3) | tree − flat |
|---|---|---|---|
| DV (full-sequence, 52 steps) ‡ | 201.2 (k256, compute-matched) | 202.2 | **−2.16 (t=−2.60)** |
| DF (causal pyramid, 52 sweeps) | 183.4 | 192.4 | +9.0 |
| shortcut-forcing (8 joint sweeps) | 148.3 | 185.5 | +37.2 |

‡ DV row is start-matched throughout (§7.5 †): the compute-matched flat baseline is
k256 = 201.2 and the root-width baseline is k50 = 199.4, both at seeds 0–9. Against
the compute-matched baseline the DV tree loses −2.16 (t=−2.60); against root width it
ties (+0.25, n.s.). Either way there is no DV tree win, and the equaliser trend (gain
shrinks +37 → +9 → −2 as flat quality rises) is unchanged by the correction.

The tree is a **partial equaliser across backbone quality**: it recovers most of what a
weaker flat sampler discards (landing 202→192→185 as flat falls 201→183→148). This is the
quantitative form of the law's final clause — *search pays in proportion to the evaluator
headroom the flat baseline leaves* — and a second, independent backbone confirming that
structured search helps a faithful-conditioning (DF-family) planner.

Two caveats of record. (i) This is **not a better maze2d planner**: shortcut+tree (185.5) is
below DF+tree (192.4) and merely ties DF+MCSS (183.4) at equal wall-clock — its value is the
headroom datapoint and cheap iteration, not SOTA. (ii) A mechanistic point worth stating: the
critic *prefers* shortcut plans open-loop, yet they *execute* worst flat (the hop tail the
inverse-dynamics policy cannot follow) — **few-step planners can look good to a value model
and be worse for control**, a caution for Dreamer-style fast samplers used in MPC. The kitchen
study uses shortcut for rapid iteration but reports final numbers on the full DF planner.

---

## 8. Phase 6 — FrankaKitchen: the dichotomy replicates on a second environment (2026-07-08)

### 8.1 Why kitchen is the designed test

Kitchen-mixed-v0 is the environment the law's final clause was written for, on three
counts. (i) *Depth carries information:* the planning window (H=32, stride 4 ≈ 124 of
280 dense steps) cannot cover the episode horizon, so multi-step planning is not
redundant with a single window — unlike maze2d, where one window spans the task.
(ii) *The data is sub-optimal:* the DV authors' own analysis (their Fig. 7b) shows
kitchen is the one D4RL family where most trajectories are sub-optimal, and the one
where their MCSS lags guided generation — i.e. the flat best-of-K baseline should sit
below ceiling, leaving the headroom clause testable. (iii) *No locomotion confound:*
a fixed-base Franka arm cannot topple (the antmaze cap, §3), and success decomposes
into 4 discrete subtasks — scores move in interpretable 25-point units.

### 8.2 Setup, scoring, and baseline reproduction

DV components were trained from the released pipeline (`veteran_d4rl_kitchen.py`,
1M steps, H=32/stride=4, mixed split); the DF planner trains on the identical
window distribution after removing the trainer's env-family gate (byte-identical
gather verified against `DV_D4RLKitchenSeqDataset.__getitem__`). Two harness defects
had to be fixed before any number was trustworthy — both are gotcha-class findings
about porting a nav harness to manipulation. First, *scoring:* the harness clipped
non-maze2d cumulative reward to [0,1], capping a 4-subtask episode at 1; kitchen's
metric is cumulative subtask completions in [0,4], normalized to 0–100. Second,
*policy rebase:* the harness hardcoded the plan-to-state rebase (correct for
maze2d/antmaze, whose leading dims are xy); kitchen's leading dims are joint angles,
and rebasing corrupted the inverse-dynamics input — worth 11–15 points (rebased runs:
60–64). With both fixed (per-family in `mcts/specs.py`), **DV-MCSS reproduces the
paper: 75.0 vs published 73.6** (K=150, critic @200k), with SEM ≈ 0 — every rollout
completes *exactly* 3 of 4 subtasks. Width is saturated flat (k150→k300→k600:
75.0/74.5/75.0): more samples never surface a 4-task plan.

### 8.3 Results — the 2×2 (two backbones × flat/tree), all paired

| arm (kitchen-mixed-v0, tree = r150 / b15 / k16 / top-3, critic @200k) | score |
|---|---|
| DV-MCSS k150 (= paper baseline) | 75.0 |
| DV-tree | 74.0 (−0.5, paired t=−0.57, n=50 — **null**) |
| DF-MCSS k150 / k600 | 60.5 / 57.0 (width flat) |
| DF-tree, seeds 0–3 | 68 / 71 / 73 / 68 |
| **DF-tree − DF-MCSS, pooled n=100** | **+10.5, paired t=5.47** (per-seed +8/+10/+17/+7, positive all seeds) |

The DF backbone passed its fidelity gate before any closed-loop run — and passed it
better than maze2d's: generated hop statistics indistinguishable from data (0.259 vs
0.259), critic gen-vs-real gap **0.0004** (maze2d DF: 0.065), prefix `hist_err` = 0.

Because kitchen scores are quantized in 25-point subtask units, the paired
per-rollout differences read directly as subtasks: **the tree completes one extra
subtask in 43 of 100 rollouts** (two extra in 5), loses one in 9, ties in 42 —
sign test 48 wins / 10 losses, p ≈ 4×10⁻⁷.

Two controls come free. *Compute:* DF-MCSS k600 spends **more** planner windows (600)
than the tree (~150 + 15×16 = 390) and scores **13 points below it** (57.0 vs 70.0) —
on kitchen the tree gain is composition/structure, not sampling volume. (The
corresponding maze2d control is the start-matched k256 arm of §7.5, which draws 256
flat samples against the tree's ≈290 and still leaves the DV tree at parity — so the
compute-matched control now exists on both environments.) *Backbone:* the DV-tree
null on the same environment, same tree code, same critic rules out "kitchen just
rewards trees."

### 8.4 Interpretation — one law, now 2 environments × 2 backbones

The maze2d dichotomy replicates exactly: **tree search helps the faithful-conditioning
(DF) backbone (+10.5, t=5.5) and does nothing for the full-sequence (DV) backbone
(−0.5)** — on an environment with a different observation space (60-D manipulation vs
4-D nav), a different metric (discrete subtasks vs camping return), and a different
data regime (sub-optimal mixed vs near-expert). The deciding variable is the
backbone's conditioning faithfulness, not the environment.

What kitchen did *not* deliver: an absolute win. DF-tree (70.0) recovers ~⅔ of the
DF→DV gap but stays under DV-MCSS (75.0); composition did not unlock the 4th subtask.
The diagnosis is sharp, and it is the critic's, not the tree's: the DV critic's
gen-vs-real gap of 0.0004 means it scores *shape-realism*, which every faithful DF
sample saturates — it cannot rank a 4-task plan above a 3-task plan, so search steered
by it cannot find what it cannot see. (The tree is also shallow — depth 2 of max 3 at
budget 15 — but a deeper tree steered by a task-blind value would deepen, not fix,
the problem.)

### 8.5 The open frontier — guidance and noise-aware values (queued, not run)

The §8.4 diagnosis points every remaining lever at the *evaluator*, and converges
with two external observations. The DV paper itself reports kitchen as the one
environment where guided generation (CFG) beats MCSS, attributing it to sub-optimal
data — i.e. when the sample pool lacks near-optimal plans, conditioning generation on
high return beats ranking what an unconditioned sampler happens to produce. And the
Diffusion Forcing paper explicitly advertises "new … guiding schemes that uniquely
profit from Diffusion Forcing's variable-horizon and causal architecture" — guidance
is native to the DF paradigm, not a retrofit. Three queued designs, in rising order
of build cost:

1. **CFG-DF** (retrain): condition the DF net on the window's normalized return with
   condition-dropout, sample with `w_cfg`. Requires retraining (~the standard 400k
   run) — the current DF checkpoint is strictly unconditional (`net(x, k)` only) —
   but the kitchen dataset already carries the return labels DV's own CFG trains on.
   Directly tests whether *generation-side* return-seeking surfaces the 4-task plans
   that *selection-side* ranking provably cannot.
2. **CG on frozen DF with a per-token noise-aware value** (no planner retraining):
   train V(x, k) on windows noised with *per-token* levels k — the diffusion-forcing
   property applied to the critic — and steer sampling with its gradient. Classical
   CG (Diffuser) is noise-aware only at trajectory level (one t per window); the
   per-token version matches DF's asymmetric cleanliness (near tokens nearly clean,
   far tokens noisy) and would let guidance weight each timestep by its reliability.
   Novelty check (2026-07-08): arXiv:2405.20555 is Diffusion Actor-Critic — Q-gradient
   guidance inside *policy training*, no noise-conditioned value model — so the
   token-level noise-aware critic remains open as far as checked. The same model
   doubles as a leaf evaluator that can score partially-denoised expansions in-tree.
3. **Grounded subtask reward on imagined states** (§8.4's direct answer): a
   task-completion checker on hallucinated states as the node value — evaluation that
   improves with depth, independent of the shape-saturated critic.

Supporting queue: the **shortcut-forcing backbone has not been trained on kitchen**
(maze2d only, §7.6); at ~6× cheaper expansion it is the enabler for the deeper-tree
probe (budget 30–45, where depth-2's cap binds) and for sweeping the designs above.
Final numbers stay on the full DF planner per the §7.6 protocol.

### 8.6 Results of design 2 — per-token noise-aware classifier guidance (2026-07-11)

Design 2 was built and run. Terminology note for the write-up: design 2 and the
"noise-aware reward model V(s, noise_lvl)" proposal (supervisor discussion) are ONE
artifact, not alternatives — a CG classifier is by definition a value model that
scores noisy inputs, and ours differs from classical (Diffuser/DV `cg`) guidance
precisely in conditioning at TOKEN-level noise resolution rather than one
trajectory-level t, as DF's asymmetric cleanliness requires. Design 1 (CFG-DF) was
NOT run and is retired by the §8.6 census: it conditions generation on the same
label-capped returns, so it cannot cross the demonstration ceiling either. The
noise critic as an in-tree node value is likewise architecturally supported but
deliberately unrun (the DV critic stays the constant evaluator across all arms;
the label cap applies to it equally). The value model V(x, k) (`mcts/noise_critic.py`) reuses the
DF planner's per-token-adaLN transformer blocks bidirectionally with a pooled scalar
head, trained on windows noised to a *mixture* of i.i.d.-uniform levels and actual
pyramid-schedule rows with clean-history prefixes — the exact query distribution
guidance evaluates at inference (`mcts/df_schedule.py::sample_training_levels`).
Labels are the dataset's normalized return-to-go at the window start (the DV critic's
own target family). Guidance is a per-token eps-shift in the frozen DF sampler:
eps ← eps − w·√(1−ᾱ[k])·∇ₓV(x, k), self-annealing per token as it denoises.

**Open-loop (free gate):** held-out sched-pattern correlation 0.915 (peak at 80k of
200k steps — the V(s) overfit pattern again; the *best* checkpoint is deployed). The
guidance-strength sweep is a clean monotone pass with **no physical cost anywhere in
range**: hop p99 stays at the real-data level (1.03–1.11 vs real 1.09) through w = 8,
while the *independent* DV critic's score of generations rises near-linearly —
0.230 (w0) → 0.237 → 0.249 → 0.262 (w4) → 0.294 (w8) vs real 0.231 — i.e. guided
generations score *above real data* under an evaluator that took no part in guidance.

**Closed-loop (kitchen-mixed, seed 0, n=25 per arm):** the flat baseline
dose-responds — DF-MCSS 60.0 (unguided) → 64.0 (w4) → 66.0 (w8), with guidance
specifically eliminating the 1-task failure rollouts — but the tree lands at the
same point regardless. Seed replication (2026-07-11, flat arms, paired on
matched env instances): **w=8 confirmed at +4.67, paired t=2.22, n=75, 3
seeds; w=4 not separable (+1.50, t=0.46, n=50)** — the guidance claim is
stated at w=8. Seed-0 table:

| guidance | flat MCSS | tree (top-3) | tree − flat |
|---|---|---|---|
| none | 60.0 | 68.0 | +8.0 |
| w = 4 | 64.0 | 70.0 | +6.0 |
| w = 8 | 66.0 | 70.0 | +4.0 |

The tree's landing point is pinned at ~70 across three guidance strengths while its
gain shrinks monotonically (+8 → +6 → +4) as the flat baseline rises underneath it.
This is the **third instance of the headroom law**, now within a single environment:
guidance lifts the flat pool, and the tree equalizes to the same landing point —
composition and guidance partially substitute for each other because both surface
plans from the same manifold. (Provenance note: the w4 tree run's raw per-rollout
vectors were overwritten by a filename collision; its summary statistics — 64.0 ±
2.5 / 70.0 ± 2.0, +6.0 paired — are recorded here and in the runbook.)

**The boundary, made exact.** A census of all 750 kitchen rollouts recorded across
every configuration — DV-MCSS (200/200 rollouts at exactly 75.0), DV width scans,
DV-tree, DF-MCSS to k=600, four DF-tree seeds, and the guided arms — contains **not
a single 4-subtask completion**. Kitchen-mixed's defining property is that its
demonstrations never solve all four tasks, so every *learned* component (planner,
inverse-dynamics policy, DV critic, noise critic) carries labels and targets capped
at the 3-task ceiling: a learned value cannot prefer a 4-task plan, because such a
plan is outside its label range. DV-MCSS's 75.0 is therefore a **hard wall for
learned-value search on this split** — the stack's ceiling equals the dataset's
demonstration ceiling, and search plus guidance close the gap *to* that ceiling
(60 → 70 of the 15-point DF deficit) but cannot pass it. The one design that can even
*express* a preference for the 4th task is the grounded subtask checker (design 3):
computed from state, not learned, hence not label-capped — with the honest caveat
that the policy would still need to execute a never-demonstrated context.

## 9. Threats to validity / open items (state at 2026-07-08)

1. **The DF tree win is CONFIRMED** (2026-07-07): seeds 0–4, pooled n=125, tree−MCSS
   +9.04 at paired t=3.90 — above the 3-seed pooled standard. The shortcut-forcing
   backbone (§7.6) independently reproduces the effect (+37.2, t=6.58) as the third
   point on the headroom curve. [Was flagged here as one seed, t≈1.5.]
2. **Sampling-vs-composition confound — RESOLVED (2026-07-13).** The tree consumes
   ≈290 planner samples/step, so the fair flat comparison is a compute-matched wide
   MCSS. This control has now run start-matched on both environments: on maze2d, the
   DV critic-tree loses −2.16 (t=−2.60) to compute-matched k256 (§7.5 note), and width
   saturates (k50 199.4 → k256 201.2, +1.8 for 5×) — the tree reaches the ceiling by
   efficient sampling, it does not compose beyond it; on kitchen, DF-MCSS k600 spends
   more windows than the tree's ~390 yet scores 13 below it (§8.3). The k≈290 item is
   closed.
3. **Headroom confound is also a prediction:** a stronger DF backbone (K=50, 800k
   steps) should shrink the tree gain — worth one run as an instrument, not a fix.
4. DF-tree 192.2 < DV-MCSS (start-matched k256 201.2): the claim is *when search
   helps*, never a new SOTA on the DV backbone.
5. Critic scores DF samples slightly below real data (−0.065): all DF arms share this
   evaluator bias, so within-backbone comparisons are unaffected.
6. Base-critic near-memorization (train MSE ~1e-6; DV holds nothing out) caveats the
   stitched-vs-dataset MSE ratio's absolute magnitude, not its direction. The ratio is
   quoted throughout as directional evidence that the seam is off-manifold, never as a
   calibrated error figure (§4.4).
7. **Kitchen is DONE as a replication and BOUNDED as a win** (§8.6): the DF-tree
   gain is confirmed (+10.5, t=5.47, n=100, 4 seeds), the DV-tree null replicates
   the dichotomy, and per-token noise-aware CG (design 2) lifts the flat baseline
   60→64→66 with zero physical cost — but the 4-task census (750 rollouts, zero
   completions, every method) shows DV-MCSS 75.0 is the dataset's demonstration
   ceiling: learned-value search cannot pass it. Remaining unexplored: the grounded
   subtask checker (design 3, the only non-label-capped evaluator) as a time-boxed
   moonshot. CG closed-loop gains are single-seed (n=25) — cheap mcss-only seeds
   would firm them. The shortcut backbone is untrained on kitchen.
8. **Kitchen seed count:** 4 seeds (0–3), not the 5 used on maze2d; a 5th seed is
   cheap insurance if a reviewer asks, though the pooled t=5.47 is far past the bar.

## 10. Related work positioning

- **DV** [Lu et al., ICLR 2025]: the base system; also the source of the
  sample-and-rank > guidance finding this project's selection design follows.
- **Diffuser** [Janner et al., ICML 2022]; **Decision Diffuser** [Ajay et al., ICLR
  2023]: full-sequence diffusion planning via guidance — the paradigm whose
  conditioning limits (§6) this project measures directly.
- **Diffusion Forcing** [Chen et al., NeurIPS 2024, arXiv:2407.01392]: provides the
  training paradigm that makes tree expansion a faithful conditional generation; our
  MCG-free, critic-ranked, transformer instantiation is deliberately adapted (§7.2).
- **MCTD — Monte Carlo Tree Diffusion** [Yoon et al., ICML 2025, arXiv:2502.07202]:
  concurrent positive result for tree search over diffusion plans; together with this
  project's negative-regime analysis it brackets the "when does search help" question.
- **AlphaZero** [Silver et al., 2017/2018]: search-in-the-loop value distillation —
  Lever B is its offline reduction. **UCT** [Kocsis & Szepesvári, 2006]; **optimizer's
  curse** [Smith & Winkler, 2006]; **RePaint** [Lugmayr et al., CVPR 2022] (replacement
  conditioning and its limits); **DDPM/DDIM** [Ho et al., 2020; Song et al., 2021];
  **cosine schedule** [Nichol & Dhariwal, 2021]; **D4RL** [Fu et al., 2020];
  **IQL reward tuning** [Kostrikov et al., 2021].

## 11. One-paragraph thesis narrative

We set out to make Monte-Carlo Tree Search beat a state-of-the-art diffusion planner's
flat best-of-K inference, and instead first established — through a causal ladder of
controlled experiments — *why it could not*: the tree's per-state values were ill-posed
(fixed by plan-value distillation, corr 0.74→0.98, +35 closed-loop), its nodes were
scored on incomparable windows (fixed by composed-window scoring), its backups
amplified evaluator noise on stitched plans (winner's curse, measured at −4.29 and
fixed by top-m backup, +4.54 at roll-t 5.00), and its expansion was not a faithful
conditional
generation — unfixable on a frozen full-sequence planner, as shown by the inpainting
experiment (−16, with the diagnosis that the stitching seam had been the critic's
accidental defense against hallucinated depth). Training a Diffusion-Forcing-style
causal planner removed this last obstacle: with exact prefix conditioning the
winner's-curse signature vanishes and tree search flips from harmful to helpful
(+9.04 over its own flat baseline, pooled t=3.90 across five seeds), at the price —
inherent, not incidental — that causal planners sample worse flat plans than
full-sequence ones. The law — **structured search pays exactly where expansion is
faithful and the flat baseline leaves evaluator headroom** — then survived its
designed test: on FrankaKitchen, a second environment with a different observation
space, metric, and data regime, the same dichotomy reappeared (DF-tree +10.5 at
paired t=5.47 over four seeds, one extra subtask in 43% of rollouts, against a null
DV-tree), with the tree beating even a compute-superior flat control. What still
caps the absolute score on both environments is the evaluator, not the search — the
selection critic is saturated on shape-realism and blind to task completion — which
is why the queued frontier (return-conditioned generation, per-token noise-aware
guidance, grounded subtask values) targets the value function rather than the tree.
