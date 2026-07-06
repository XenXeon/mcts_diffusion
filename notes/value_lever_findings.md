# The Value-Lever Arc — COMPLETE (2026-07-04/05)

*Self-contained record of the critic-in-tree + value-improvement investigation on
maze2d-large-v1. Extends notes/PROJECT_HANDOFF.md (stale past §7) and
notes/findings_summary.md. Companion memory: project_critic_tree_decomposition.
All scores = DV camping return (base-pipeline metric); seed 0, n=25 unless noted.*

## 1. The final table (one tree, one env, every value)

| arm | score | what changed |
|---|---|---|
| V(s)-tree, behaviour-return target (b16) | **167.3** (n=150) | ill-posed target (SNR ceiling, corr 0.742) |
| **planv-tree**, distilled plan-value target (b15 k16 MAX) | **202.5** | same tree, same MLP — only the TARGET swapped |
| critic-in-tree, composed windows (k16 MAX) | 203.2 (n=150) | critic scores every node on [s0, s0+H) |
| tree r50 MAX (root ⊇ MCSS pool) | 198.1 | winner's curse: −5.7 despite superset root |
| tree r50 top-3 backup | 202.3 | tempering recovers +4.5 (pooled p<1e-4, 3/3 seeds) |
| tree r50 top-3 + stitch-tuned critic | 201.9 | retraining adds NOTHING over tempering |
| MCSS k50 (baseline) | 204.9 (n=150) | the evaluator's own ceiling −1 |
| MCSS k50 + stitch-tuned critic | 200.8 | fine-tune DAMAGES the baseline (~−4) |
| MCSS k256 | 206.0 (n=150) | width saturation: 5× compute → +1.1 |

**Reading:** the entire 35-point gap between "tree loses badly" and "tree ties" is
value-target posedness. The last ~2–3 points are the whole-trajectory critic's direct
advantage plus selection noise. Nothing is search structure. Depth is informationless
here because the planning window covers the decision-relevant horizon.

## 2. The three levers, verdicts

### Lever A — stitch-aware critic fine-tuning: CLOSED, mechanism-rich negative
- Direct measurement (`finetune_critic_stitched.py`, replica gate = 0.00e+00):
  base critic MSE **0.0766 on stitched windows vs ~1.2e-6 on originals = 63,618×
  off-manifold error ratio** (≈110 dense-steps RMSE vs ≈0.4). This is the winner's
  curse's fuel, quantified. (Caveat: 1e-6 is training-distribution precision — the DV
  pipeline holds nothing out.)
- Fine-tune: stitched 0.0766 → 0.0471 (−38%, plateau by ~10k steps). The floor is
  **aleatoric**: a stitched label imports path B's return BEYOND the visible window
  ((H−j)·stride dense steps of B shown; B's time-to-terminus can be ~800) — no critic
  can remove that variance, only the bias.
- Closed loop: hurts MCSS (204.9→200.8; the sacrificed on-manifold precision was
  load-bearing), adds nothing over tempering (202.3→201.9).
- **Design rule: the curse is VARIANCE-driven → AGGREGATE (top-m backup), don't
  retrain the evaluator on the query distribution.**

### Lever B — distilled plan-value V̂(s): the posedness falsification LANDED
- Same DVStateValue MLP, same optimizer, same dataset start-states; target swapped
  from behaviour-return to plan-value (mean-top-3 of K=16 critic-scored planner
  samples from s; `gen_plan_value_labels.py` + `train_plan_value.py`).
- val corr **0.742 → 0.9805** (MSE 0.00425 ≈ 26 dense-steps RMSE, ~= the labels' own
  sampling-noise floor). Closed loop: planv-tree 202.5 vs behaviour-V(s)-tree 167.3.
- **"The V(s) network/training was suboptimal" is FALSIFIED. Target posedness was the
  entire problem.** Deploy: `--value-mode v_s --value-step planv` (collate: Vplanv).
- Bonus recipe (transfers to kitchen): offline AlphaZero-style distillation — the
  MCSS outcome from s is a well-posed regression target wherever behaviour returns
  are not; usable as a cheap leaf value / prioritized-expansion score.

### Lever C — junction feasibility filter: built, free, undecisive here
`--junction-filter` (reject children whose first continuation hop exceeds the
dataset's p99 stride-step displacement; sentinel −10 dominates UCB bonus). Cheap
safety rail; on maze2d nothing forced it to matter. Keep as a flag for kitchen.

## 3. Why "train them together like DV" was the wrong frame (and what was right)
DV's planner/critic/policy are trained SIDE-BY-SIDE, not jointly
(veteran_d4rl_maze2d.py:219-253 — three losses, three optimizers, zero coupling).
The critic's edge is (a) well-posed target (whole window → determined return) and
(b) evaluating the actual decision object. The correct form of the instinct is
"train each component on the distribution/quantity it is queried with at inference"
— Lever A tested the distribution half (negative, aleatoric floor), Lever B tested
the quantity half (decisively positive). Literal joint training has no sound offline
objective: guidance lost to sample-and-rank in DV's own study; AlphaZero-style
search-in-the-loop reduces offline to Lever B's distillation.

## 4. Tooling added (all torch-free-tested where possible; 31+ local tests green)
- `mcts/window.py` — composed-window scoring (every tree node valued on [s0, s0+H))
- `mcts/stitch.py` + `scripts/finetune_critic_stitched.py` — exact stitched labels
  via the segment identity V_A[sa] − γⁿV_A[sa+n] + γⁿV_B[sb]; maze2d-only gate
- `scripts/gen_plan_value_labels.py` + `scripts/train_plan_value.py` — Lever B
- Sampler/CLI: `--k-root`, `--top-m`, `--junction-filter`, `--critic-step <tag>`,
  `--value-step <tag>`; collate labels all arms distinctly (r50/m3/C…/V…/J)

## 5. What transfers to kitchen (the only open frontier)
Day-one config when DV kitchen checkpoints land: composed windows + **top-3 backup**
+ wide root; V(s) on subtask-return target (`train_state_value.py`) — the pivot's
original hypothesis test (kitchen's value target is well-posed BY CONSTRUCTION);
grounded subtask-reward evaluation on imagined states as the frontier value (the
MuZero ingredient nav never had — evaluation that IMPROVES with depth); plan-value
distillation recipe if a cheap leaf value is needed. Kitchen is where depth finally
has information: window 128/280 dense steps, discrete subtask-order branching.

## 5b. DF-inspired prefix-inpaint expansion (built 2026-07-05, results pending)

Motivated by Diffusion Forcing (Chen et al., NeurIPS 2024, arXiv 2407.01392) via
TA discussion: DF's tree-search capability comes from a CAUSAL planner that
conditions continuations on the search history. Our glue-mode expansion samples
continuations from the leaf state alone and concatenates — the seam is exactly
where the critic's 63,618× off-manifold error lives.

`--expand-mode inpaint` (critic mode only) captures DF's key benefit on the
FROZEN DV planner: the search prefix + node state are clamped into the denoiser
at every diffusion step (`build_inpaint_prior` in mcts/window.py; the same
conditioning-by-replacement the planner already uses for row 0), so the free
rows are generated jointly consistent with the path and the sampled window IS
the composed [s0, s0+H) window — seam-free, scored directly by the critic.
Root expansion is identical to glue (mask = row 0), so baselines are unaffected.
Caveat: the planner was trained with row-0 clamping only; multi-row clamping is
test-time replacement conditioning (approximate, like Diffuser's goal inpainting).

What it can NOT give (needs a real DF planner, i.e. training a new model):
beyond-H lookahead, a causal latent carrying history, per-token uncertainty.

The causal test: r50 MAX + inpaint vs r50 MAX glue (198.1). If the curse is
seam-driven, MAX-backup should recover toward ~204 WITHOUT top-m tempering; if
unchanged, the curse is generic max-over-noise and top-m remains the fix.
Either way kitchen inherits the winning combination.

**RESULT (2026-07-05, seed 0 n=25, config-identical to the glue reference):
inpaint 182.1±11.1 vs glue 198.1±9.9 — paired per-env t=−2.84 (p≈0.009,
identical starts/goals, corr 0.87). Inpaint is genuinely WORSE, and recovery
to ~204 is excluded (~2 SEM below). Provenance verified (expand_mode recorded
in the JSON; root expansion is glue-identical by construction).** Pending
bug-vs-mechanism confirmation via scripts/diag_inpaint.py + code audit, the
reading: replacement conditioning does NOT extend to multi-row prefixes on a
planner trained with row-0 clamping only — the clean-prefix + noisy-future
input is a mixed-noise-level configuration the model never saw (exactly what
Diffusion Forcing trains for, and exactly the approximation error the TA
flagged, now with a measured cost: −16 points). Options if pursued further
(kitchen tooling only): RePaint-style noise-matched clamping (clamp prefix
rows to alpha_t*prefix + sigma_t*eps per step, row 0 kept clean as trained)
keeps the input in-distribution at ~10 lines inside the sample loop; or a
true DF planner (new training, future work). MAX-backup verdict unchanged:
top-m tempering on glue windows remains the configuration of record.

## 5c. planv closed-loop replication (2026-07-05) — transfer CONFIRMED

- maze2d-large planv-tree (b15 k16 MAX): seeds 0/1/2 = **202.5 / 202.4 /
  202.6** — essentially zero seed variance; the 167.3→202.5 jump is not a
  lucky seed.
- maze2d-medium, paired same-run MCSS k50: tree 151.6 vs MCSS 156.6
  (paired t=−1.27, n.s.). maze2d-umaze: tree 129.2 vs MCSS 138.4 (paired
  t=−1.65, n.s.; one env unreached). Same pattern as large: planv-tree sits
  at or a few points below MCSS everywhere — a good V(s) buys tree parity,
  the critic's direct advantage supplies the last few points.
- Offline multi-env corr (same MLP, target swapped behaviour→plan-value):
  medium 0.636→**0.9572**, umaze 0.390→**0.9430**, antmaze-large-diverse
  0.874→**0.9803**, antmaze-medium-diverse 0.513→**0.8999**. The posedness
  claim is general, not a maze2d-large artifact.

## 6. Do NOT re-open
- A "win over MCSS" on maze2d/antmaze — ceiling measured (~206–207 = the critic's
  own selection ceiling); antmaze is locomotion-capped below the sampler.
- Critic retraining for the tree (Lever A closed), behaviour-return V(s) (falsified),
  geodesic/oracle values (dynamics-blind, worst of all — see PROJECT_HANDOFF §1).
