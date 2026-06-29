# Failure instrumentation (Tier 0 / 1 / 2) — runbook & design

Dissects the MCSS / DV baseline's ~20–24 % closed-loop failures on
antmaze-large-diverse-v2 into per-mode causes, then bounds how much of that a
better critic (this project's lever) could recover. Built per the agreed
checklist; Tier 1 runs **before** Tier 2 because the oracle-ceiling is only
interpretable once execution / proposal failures are separated out.

## Components

| File | Tier | Deps | Role |
|---|---|---|---|
| `mcts/failure_modes.py` | 1 | **stdlib only** | the science: progress-curve features, candidate-pool verdict, pose-collapse, `classify_failure`. Unit-tested locally. |
| `mcts/instrument.py` | 0, 2 | numpy/torch (lazy) | traced rollout: per-step xy, body-state, BFS-distance, full candidate pool; saves per-**failed**-env npz + index. `value_source="oracle"` = Tier-2 re-rank. |
| `scripts/run_instrumentation.py` | 0, 2 | GPU | CLI driver. |
| `scripts/analyze_failures.py` | 1, 2 | numpy | modes table + oracle-ceiling cross-check. |
| `scripts/plot_failures.py` | 1 | matplotlib | per-episode path-on-map + progress-curve figures. |
| `tests/test_failure_modes.py` | — | stdlib | 24 classifier unit tests (`python -m unittest tests.test_failure_modes`). |

`mcss_propose` was **added** to `Sampler` (additive; `mcss_waypoints`, the
production path, is untouched). It returns the full k=50 candidate pool from one
planner draw so the instrumented loop takes the *same* critic-argmax decision and
logs the pool from one consistent draw.

## Oracle discipline (Rule 1)

`value_source="oracle"` uses the BFS geodesic **as the critic** — privileged
information. It is a **ceiling probe, never reportable**; every oracle dump is
tagged `DIAGNOSTIC_ONLY` and `oracle_used=true`. Per-step BFS distances (used in
both tiers) are a measurement aid only.

## Determinism

Unlike the production harness (diffusion draw unseeded — see `run_episodes`), the
instrumentation seeds torch so a Tier-1 critic run and a Tier-2 oracle run are
reproducible and cross-checkable scenario-by-scenario. The env goal draw is
already a pure function of `--seed`; goals pair across the two runs by
`(seed, env_idx)`. The instrumented reach% is therefore a *fresh, representative*
failure set, not a byte-match of a specific s10 JSON.

## Failure modes

`WRONG_TURN` is the only **critic-fixable** mode (the project's lever); the
value-side of `OSCILLATION` (a goalward candidate went unpicked at the junction) is
**relabelled into** `WRONG_TURN`, so residual `OSCILLATION` is execution/proposal
ping-ponging and stays **critic-immune** alongside `FELL_OVER` (execution),
`NO_GOOD_PLAN` (proposal/coverage), `TIMEOUT_ON_TRACK` / `UNREACHABLE_FAR`
(horizon), `GOAL_RADIUS_ARTIFACT` (measurement), `OFF_GRAPH`. The `WRONG_TURN` vs
`NO_GOOD_PLAN` split is decided by the candidate pool at the closest-approach
("junction") step: a goalward option that went **unpicked** = ranking (critic);
**no** goalward option = proposal. A goalward option that **was** picked yet still
failed routes to execution. `GOAL_RADIUS_ARTIFACT` fires only when the **executed
world distance** to the goal dips within the true 0.5 reward radius yet `success=0`
(a real termination / stride-hop miss) — not a coarse BFS-cell touch (audit F1), so
near-goal *failures* are surfaced rather than buried as immune. All thresholds live
in `ClassifierConfig`.

## Run

```bash
# Tier 0/1 — log the real MCSS baseline's failures (seeds 0,1,2)
python scripts/run_instrumentation.py --env antmaze-large-diverse-v2 \
    --seeds 0 1 2 --n-envs 50 --value-source critic

# Tier 2 — re-run the SAME scenarios with the BFS geodesic as the critic
python scripts/run_instrumentation.py --env antmaze-large-diverse-v2 \
    --seeds 0 1 2 --n-envs 50 --value-source oracle

# Tier 1 modes table + Tier 2 ceiling cross-check
python scripts/analyze_failures.py --in-dir results/instr --out results/instr/summary.json

# optional figures (one per failed episode)
python scripts/plot_failures.py --in-dir results/instr --out-dir results/instr/figs
```

Outputs land in `results/instr/`: `instr_mcss_{critic,oracle}_s{seed}.npz` (heavy,
failed envs only) + `_index.json` (all scenarios + maze geometry). Cost ≈ the
cheap k50 baseline (~0.6 h/seed); the oracle pass adds only BFS lookups.

## First GPU run + recalibration (2026-06-24)

First 3-seed run (seeds 0–2, n=50) surfaced a **Tier-1 ↔ Tier-2 contradiction** that
the cross-check is for: Tier 1 (shape) said 0% critic-fixable, while a **fixes-only**
oracle count said oracle "solves 23/33." **Both were wrong.** Measured PAIRED (the fix
the user's skepticism forced), the oracle's *actual* reach is **78.7% vs the critic's
78.0% — net +0.7 pp (fixes 23 / breaks 22, breaks/fixes 0.96).** The "23 fixes" were
matched by 22 breaks — the same unpaired-noise trap as the n=500 grid. **So selection
is NOT the lever:** a perfect geodesic ranker ≈ the DV critic. Triangulated by
`plot_candidates` (the critic already picks the geodesic-best endpoint; gap≈0) and
`diag_wall_blindness` (`V(s,g)` tracks the geodesic, corr −0.945, *not* wall-blind).
The ~22% failures are planner+policy execution limits; the env is near-saturated at
~78%. Lesson: **attribute by the paired counterfactual, never by fixes alone** — the
analyzer now reports oracle reach + fixes + breaks + net.

A first recalibration attempt (pool-verdict-first, sustained-fall, broadened reverse
check) **overcorrected to 100% `FELL_OVER`** — and the broadened reverse check caught
it (`FELL_OVER oracle-fixed 23/33`). The deeper lesson: **trajectory-shape attribution
structurally conflates symptom with cause.** At the end of a failed episode the ant
genuinely *is* tilted/stalled, but for the 23 oracle-fixable cases that topple is the
*consequence* of an earlier ranking error — no pose/shape threshold can separate
"fell from bad luck" from "fell because ranking steered it into a wall", because only
a counterfactual can. The single-junction candidate-pool proxy is likewise too weak
(the decisive error is usually *earlier* than the critic trajectory's closest
approach, and the DV critic usually picks goalward-but-suboptimal plans).

**Reframe (`analyze_failures.py`):** the headline ATTRIBUTION comes from the oracle
counterfactual, not the shape classifier — and crucially it is measured **PAIRED**:
- **HEADLINE = oracle reach + fixes + breaks + NET.** The first draft reported the
  fixes-only "split" (+15.3 pp, 93.3% ceiling) — *wrong*, it ignored breaks. Paired, the
  oracle reaches **78.7% vs 78.0%, net +0.7 pp (fixes 23 / breaks 22)**: selection is not
  the lever.
- **Tier-1 modes are reported DESCRIPTIVELY** (how failures look), not as the fixable %.
- The script characterises the oracle-immune set with a **per-episode evidence dump**
  (start/min/end geodesic, backslide, off-graph, collapse step, world-min-dist, why).
- The mode×oracle cross-tab + forward/reverse checks remain as validation; when shape
  mislabels (as it does here), the reverse line says so.

The classifier (`failure_modes.py`) keeps the pool-first / sustained-fall logic and 31
passing tests; its labels are now secondary to the oracle attribution.

## Tier 3 + wall-blindness (built 2026-06-24)

Two visualisations that, together with the paired oracle result, show **selection is
not the lever** — the DV critic already ranks the candidates about as well as the true
geodesic, and `V(s,g)` is sound (not wall-blind):

- **`scripts/plot_candidates.py`** (torch-free, reads the critic dumps) — the
  50-candidate dispersion / mis-ranking view. Per decision step it plots the candidate
  endpoints on the wall map coloured by true BFS-geodesic, with the **critic's pick
  (argmax score) vs the oracle's pick (argmin geodesic)** marked, plus a score-vs-
  geodesic scatter. The aggregate reports the selection-relevant **PER-DECISION
  Spearman ρ(score, geodesic)** (within-decision ranking over the 50 — *not* pooled
  across steps, which a between-state level trend would dominate), the **mis-rank rate**
  (% of decisions where a candidate ≥2 cells closer than the critic's pick existed), and
  mean cells left on the table. Rates are **failure-conditioned** (dumps keep failures);
  `run_instrumentation.py --keep-success-frac 0.2` adds the success-episode contrast.
- **`scripts/diag_wall_blindness.py`** (torch/GPU; mirrors `diag_d2`) — tests whether
  `V(s,g)`, which never sees the maze map, judges distance by the wall-respecting
  geodesic or by Euclidean closeness. **Primary, bias-robust verdict: the detour-
  stratified over-optimism gap** (high-detour wall-between minus low-detour open states;
  positive = over-optimistic exactly where walls intervene) — it cancels the global
  steps/cell calibration and isolates the *extrapolation* error (the value is wall-blind
  only where it must cross an un-traversed wall, consistent with D1's stitched-pair
  failure). The corr(value, geodesic-vs-Euclidean) comparison is reported as supporting
  colour only (the two are collinear). Draws a per-cell **error heatmap** (red = "thinks
  it's closer than it is", walls drawn) and an **implied-vs-geodesic scatter** coloured
  by Euclidean detour.

## Not yet built (deferred)

Tier 4 confounder/scope validators (per-scenario cross-seed consistency, start-distance
distribution, k50-vs-k272 failure-set overlap). The npz already stores what they need.
