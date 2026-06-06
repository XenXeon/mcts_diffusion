# Phase 2 — Expansion Primitive Design

## Purpose

Extract the per-step MCSS planning operation from the monolithic eval loop into a
self-contained, testable module (`mcts/expansion.py`).  This is the building block
that future MCTS tree-search code will call at each node expansion.

## What the primitive does

Given a single normalised observation `s_norm` (shape `(obs_dim,)`):

1. Builds `prior = zeros(K, H, planner_dim)` and writes `s_norm` into
   `prior[:, 0, :obs_dim]`.
2. Calls `planner.sample()` — the planner's `fix_mask` clamps position 0 back to
   `s_norm` at every denoising step, so the returned trajectories always start at
   `s_norm` exactly.
3. Scores all K trajectories with `critic(trajs)` → shape `(K,)`.
4. Sorts descending by score and returns an `ExpansionResult`.

## What it does NOT do

- Normalise or unnormalise observations — caller owns that boundary.
- Re-set `fix_mask` — the mask is baked into the planner at construction time.
- Apply the inverse-dynamics policy — action generation is downstream.
- Maintain tree state — this is a stateless function call.

## Key contracts (verified by tests)

| Contract | Test |
|---|---|
| `result.trajs[:, 0, :obs_dim] ≈ s_norm` (fix_mask held) | `test_fix_mask_clamped` |
| `result.trajs.shape == (K, H, planner_dim)` | `test_output_shapes` |
| `result.scores.shape == (K,)`, 1-D | `test_output_shapes` |
| `result.scores[i] >= result.scores[i+1]` for all i | `test_scores_descending` |
| Two calls from same state give different trajs (stochastic) | `test_planner_stochastic` |
| No gradient on outputs | `test_no_grad` |
| `result.best_traj is result.trajs[0]` | `test_best_traj_alias` |
| K=1 is valid | `test_single_candidate` |
| Wrong s_norm shape raises ValueError | `test_bad_input_shape` |

## Dependency map

```
mcts/expansion.py
  ← torch                              (always)
  ← cleandiffuser.diffusion            (ContinuousDiffusionSDE — passed in, not imported)
  ← cleandiffuser.utils                (DVHorizonCritic — passed in, not imported)

tests/test_mcts_expansion.py
  ← mcts.expansion
  ← cleandiffuser.diffusion.ContinuousDiffusionSDE  (to build small test planner)
  ← cleandiffuser.nn_diffusion.DiT1d               (to build small test planner)
  ← cleandiffuser.utils.DVHorizonCritic             (to build small test critic)
  ← pytest

scripts/phase2_smoke_test.py
  ← mcts.expansion
  ← cleandiffuser.* (dataset, diffusion, nn_*)    (real model loading)
  ← d4rl, gym                                     (dataset normalizer)
```

The expansion module itself imports **only `torch`** and `dataclasses`.
No d4rl, no gym, no dataset — fully testable without the full environment stack.

## Configuration (ExpansionConfig)

Mirrors the maze2d-umaze-v1 production values:

| Field | Type | Production value |
|---|---|---|
| `K` | int | 50 |
| `horizon` | int | 32 |
| `obs_dim` | int | 4 |
| `planner_dim` | int | 4 (separate pipeline) |
| `solver` | str | `"ddim"` |
| `sample_steps` | int | 20 |
| `temperature` | float | 1.0 |
| `use_ema` | bool | `True` |
| `device` | str | `"cuda:0"` (tests use `"cpu"`) |

## Phase 1 findings that directly constrain this design

- **fix_mask must clamp position 0** — verified in Phase 1 (max-abs < 1e-4 for all
  100 samples). The test `test_fix_mask_clamped` re-asserts this with the small model.
- **Planner must be stochastic** — Phase 1 confirmed `planner_self_l2 > 0`; the test
  `test_planner_stochastic` re-asserts this. MCTS expansion is meaningless if all K
  samples are identical.
- **Critic scores generated plans >> real plans** — exploitation signal documented in
  `phase1_verification.md` §6. The expansion primitive is intentionally unaware of this;
  the uncertainty-penalty mitigation is a Phase 5 concern.
- **No double-normalisation** — `s_norm` must already be z-scored. The primitive
  asserts `s_norm.shape == (obs_dim,)` but does not validate the value range (that
  check belongs to the integration smoke test).

## Files created in Phase 2

| File | Role |
|---|---|
| `mcts/__init__.py` | package; exports public API |
| `mcts/expansion.py` | `ExpansionConfig`, `ExpansionResult`, `PlannerExpansion` |
| `tests/test_mcts_expansion.py` | unit tests (CPU, small models) + integration tests |
| `scripts/phase2_smoke_test.py` | end-to-end run with real checkpoint |
