# Research plan v4 — next steps from where this repo stands (2026-06-13)

> **SCOPE DECISION (2026-06-13, supersedes §B/§C below): D4RL only — no OGBench.**
> Consequences accepted: the stitching headline (H1 on stitch datasets) and the
> giant-maze tier are untestable as designed; MCTD-family comparisons become
> published-number context, not same-config head-to-heads. What is kept — and runs
> entirely in THIS repo with no env migration: Stage 0 critic work (V(s,g):
> MC-relabelled first, IQL-u upgrade only if D1 fails), the D1/D4 diagnostics
> (BFS oracle exists in phase6_stage0_oracle.py), the Stage-1 hardened judge, and
> the **H2 mechanism 2×2** on antmaze-large — {MCSS/BoN, tree} × {naive, hardened}
> at matched compute, n=150 paired — which is the dissertation headline experiment.
> Phase-1 evidence makes the 2×2 sharper here than anywhere: the naive/BoN cell is
> already measured to DEGRADE with compute (−7.3pp). The §A asset map and §D
> findings-narrative below remain valid; ignore the new-repo/OGBench items in §B/§C.

Companion to `mcts_diffusion_planner_research_plan_v4.md` (the spec). This document
maps the spec onto the existing assets and findings, lists the decisions that must be
made before week 1 starts, and gives the concrete week-1 build list.

---

## A. How the existing work slots into the plan

The work in this repo (Phases 0–E + the 5-arm grid + the child-index sweep) is the
plan's **Phase-1 / DV-comparability evidence**, already done:

| Plan element | Status from this repo |
|---|---|
| §1 D4RL antmaze-large "ceiling warning" | **Measured, not predicted**: best MCSS 79.3%, MCTS b16 83.3%, +4.0pp n.s. (p=0.44) at n=150 paired — "beats DV on D4RL-large is statistically undetectable" is our pooled result verbatim. Cite as the motivating evidence for moving to giant/stitch. |
| §8 DV reproduction | Done: harness k50 = 79.3±3.3 vs published 76.9±1.3 (n=1000), plus full 5-arm grid with per-scenario vectors. |
| §4 hardened judge motivation | **Measured**: MCSS argmax DEGRADES with candidates (k50→k272 = −7.3pp paired) — critic over-exploitation by maximisation is not a hypothetical; it is the strongest finding of Phase 1 and the empirical case for D2 probes + pessimistic ensembles + the feasibility gate. |
| §3 D1 "compass resolution" Δ* | The child-index sweep is its closed-loop preview: with a goal-agnostic V, useful look-ahead distance ≈ 1 waypoint (L1 86% → L4 74%, fixes constant, breaks double). Value information decays with distance — Δ* is exactly the right diagnostic to build first. |
| §6 burst widening | `mcts/value_forest.py` already does full-wave expansion (k children per selected leaf in one batched call) — the engine is closer to the spec than it looks. |
| §1 Rule 4 evaluation | Our per-scenario paired logging + exact McNemar (collate_mcts) is *stronger* than bootstrap-over-seeds for fixed start/goal configs; port it. |
| Realized-depth instrumentation | Built (depth ≈ 2.5–3.0 at budget 16); feeds the §6 "effective branching"/diagnostics requirements. |

**What is genuinely new** (no existing code): OGBench envs + both configurations,
goal-conditioned IQL critic in u-units (+ quasimetric parallel track), ensembles +
pessimism, relabeling with termination mask, feasibility classifier, executor π_lo,
steps-space tree arithmetic (min-cost backup, point-to-segment goal check, endpoint
dedup/clustering, in-tree normalization, progressive-widening trigger, tree reuse),
MCTD-family baselines, hierarchy (Stage 4).

## B. Decisions needed before week 1 (blockers)

1. **New repo + new env.** This repo pins gym<0.24 / mujoco 2.3.7 / mujoco_py
   (d4rl); OGBench needs gymnasium + mujoco≥3 — irreconcilable in one venv.
   → Create a fresh repo (suggest `mcts_chunks/`) with its own venv. Port (don't
   move): `value_forest.py` engine + tests, `specs.py` pattern, `mcts_loop`
   harness pattern (per-rollout logging), `collate_mcts.py`, `BFSValue` from
   `phase6_stage0_oracle.py` (becomes the dev-only geodesic oracle, Rule 1).
2. **GPU class (Rule 5).** The plan's gates assume A100 ≥40GB. The eval box ran
   50×272-candidate antmaze cells in ~3.1h — fine for the plan's per-decision
   budget (2–4 waves × 16 chunks ≪ our 272/step), but the Stage-3 gate
   (≤3 GPU-h / 100-episode eval) must be restated after one timed run on the
   actual box. → State the GPU model in this doc when known.
3. **Old-stack V(s,g): skip as a standalone experiment.** Its scientific question
   is subsumed: D1/Δ* measures the distance-decay it would have tested; the 2×2
   hardened-vs-naive measures the exploitation it would have mitigated; the plan's
   Stage-0 critic IS the goal-conditioned value, done properly (IQL-u, ensembles,
   relabeling+termination mask vs our MC regression). The pre-registered L4
   interaction test ports to the new stack as a Stage-3 diagnostic (chunk-length
   re-test once the goal-conditioned critic exists). Revisit only if weeks 11–12
   have slack and the dissertation wants the old-stack bridge figure.
4. **MCTD / Fast-MCTD code + released models access** (week-2 start): confirm the
   repos run before committing the customized-OGBench configuration.

## C. Week-1 build list (Stage 0 start + scaffolding), in order

1. `mcts_chunks/` repo skeleton: venv (python ≥3.10, torch, gymnasium, ogbench),
   ported `tree/forest.py` (+8 tests), `specs.py` for OGBench families,
   `collate.py` (per-rollout pairing + McNemar + bootstrap-over-seeds),
   `.gitignore` that does NOT ignore scripts/ or result JSONs (lesson learned).
2. `scripts/measure_dmax.py` — Stage-0 prerequisite: geodesic d_max per maze
   (BFS oracle on the OGBench maze grids) → fixes γ per §2 rule (γ ≥ 1−0.7/d_max).
3. `scripts/train_critic_iql_u.py` — goal-conditioned IQL in u-units: ensemble 5,
   relabeling 70/20/10 with current-state-goal target 0, termination mask +
   truncated n-step, γ from step 2, expectile τ=0.9 (retune list per §3).
4. `scripts/train_critic_quasimetric.py` — parallel track (QRL/MRN-style), same
   d̂ interface so everything downstream is critic-agnostic.
5. `scripts/diag_d1_compass.py` … `diag_d4_calibration.py` — the four diagnostics,
   gates exactly per §3 (D1 proprioception-matched binning; D2 ascent + 10k-chunk
   enrichment; D3 chunk-length window; D4 no-oracle calibration). D1–D4 define
   week-3 exit; build them BEFORE tuning anything.
6. Baseline track (parallel, week 2): MCTD/Fast-MCTD stack on customized OGBench;
   the DV row is already done (this repo).

## D. Findings to carry into the dissertation narrative (Phase 1 → plan)

1. Flat candidate-scaling under argmax backfires (−7.3pp); structured search
   converts the same compute positively (+11.3pp at matched compute, p=0.021,
   n=150 paired) → why the plan hardens the judge and scores with pessimism.
2. Goal-agnostic value caps useful look-ahead at ~1 waypoint (L-sweep) → why the
   plan's critic is goal-conditioned from day one and why D1/Δ* gates chunk length.
3. Failures outside a ~6% hard core are stochastic per run, near-independent across
   configs → why n=150+ paired evaluation and per-scenario McNemar; ports to the
   plan's fixed-start/goal configs directly.
4. D4RL-large is at ceiling for sampler comparisons → why giant/stitch are
   co-primary (§1) — our +4pp n.s. vs the cheap baseline is that warning, realized.

## E. Risks specific to this setup (beyond the plan's register)

| Risk | Mitigation |
|---|---|
| OGBench/mujoco-3 install friction on the eval box | Do the venv + a 10-min env smoke (reset/step/render-free) as literally the first action of week 1 |
| Two stacks drift (d4rl repo vs new repo) | Freeze this repo except for writeup edits; port code by copy, never share files |
| Compute gate on non-A100 | Timed single-episode run in week 1; restate §6 gate numbers |
| 12-week timeline vs dissertation deadline | Weeks 11–12 are writing; if the deadline is earlier than ~mid-September, cut Stage 4 for *large* (keep it for giant only, as the plan already prefers) and reduce headline seeds 8→5 with disclosure |
