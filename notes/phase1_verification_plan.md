# Phase 1 Verification Plan

## Purpose

Verify that DV's frozen planner and critic behave on held-out dataset start
states the way they behave on training start states. This tests
generalisation across the dataset distribution, NOT generalisation to
imagined intermediate states from MCTS expansions (that is a Phase 5
question).

A pass here means the foundation is sound for Phase 2 (building the
expansion primitive). A fail means DV has overfit to the trajectory IDs
it saw during training, and the project pivots before any tree-search
work begins.

## Prerequisites

- `notes/dv_inference_map.md` — Phase 0 codebase audit.
- `notes/phase1_preflight.md` — confirms dataset is pre-normalised,
  trajectory lengths recovered to `results/phase1/path_lengths.npy`,
  held-out pool sized at 74 usable trajectories with 17,676 valid
  offset slots.
- A working `scripts/run_one_episode.py` producing normalised score
  ~114 on maze2d-umaze-v1 (sanity check that DV inference works at all).

## Sampling Procedure

1. Use `np.random.default_rng(0)` for the trajectory-ID permutation
   (matches preflight). Held-out fraction = 0.10 → 156 trajectories,
   74 of which are usable (length >= H*M = 480 dense env steps).
2. Use `np.random.default_rng(42)` for offset sampling within usable
   trajectories.
3. Sample 100 (traj_idx, offset) pairs total, distributed across the
   74 usable trajectories with `samples_per_traj = max(2, ceil(100/74)) = 2`.
   Within each trajectory, draw offsets uniformly from
   `[0, true_length - 480]` without replacement when possible.

## Per-Sample Logging

For each (traj_idx, offset):

| Field | How to compute |
|---|---|
| `traj_idx`, `offset` | from sample list |
| `s_0` | `dataset.seq_obs[traj_idx, offset]` — already normalised |
| `imagined_plan` | `planner.sample(prior with prior[:, 0, :] = s_0, n_samples=1, sample_steps=20, solver="ddim", use_ema=True, temperature=1.0)` |
| `imagined_plan_2nd_draw` | same call, fresh RNG state — used for planner self-consistency baseline |
| `real_plan_jump` | `dataset.seq_obs[traj_idx, offset:offset+480:15]` — shape `(32, 4)` |
| `score_gen` | `critic(imagined_plan).item()` |
| `score_real` | `critic(real_plan_jump[None]).item()` |
| `mean_l2` | mean over jump-steps of L2 distance between `imagined_plan` and `real_plan_jump` |
| `planner_self_l2` | mean over jump-steps of L2 distance between `imagined_plan` and `imagined_plan_2nd_draw` |
| `in_bounds` | `True` iff every position in `imagined_plan` (after unnormalising) lies inside the maze bounds |

## Mandatory Assertions

These run during the loop, not just at the end. If any fails, stop the
script and investigate.

1. `s_0.abs().max() < 10.0` — catches accidentally raw or corrupted
   states. Dataset is pre-normalised, so values should be roughly
   z-scored.
2. `(imagined_plan[0] - s_0).abs().max() < 1e-4` — verifies that the
   `fix_mask` is actually clamping position 0 to `s_0`. If this fails,
   the planner is generating from a different state than intended and
   every score in the run is meaningless. **Critical bug check.**
3. `real_plan_jump.shape == (32, 4)` — ensures the slice
   `[offset:offset+480:15]` produced exactly 32 rows. Off-by-one here
   would silently invalidate the L2 comparison.

## Artefacts to Save

All under `results/phase1/`:

- `heldout_ids.json` — list of 156 held-out trajectory IDs and the
  74 usable subset. Reused verbatim by Phase 5's critic-reliability
  diagnostic, so the splits stay aligned.
- `per_state_results.csv` — one row per sample with all logged fields
  except the full plan tensors.
- `plans.npz` — three arrays each of shape `(100, 32, 4)`:
  `imagined_plans`, `imagined_plans_2nd_draw`, `real_plans`. Stored in
  normalised space.
- `phase1_run_metadata.json` — env name, ckpt path, seed values, git
  commit hash, wall time, normalised baseline score from
  `run_one_episode.py` recorded that day.

## Plots

Produced by a separate script `scripts/phase1_make_plots.py` that loads
the CSV and NPZ. Does NOT re-run the planner. All plots saved to
`results/phase1/`.

### Plot 1: `critic_distributions.png`

Two overlaid histograms on the same axes (alpha=0.5):
- `score_gen` (generated plans)
- `score_real` (real continuations)

Annotate the means and the K–S statistic between the two distributions.

**What it tells you:** marginal calibration. Are generated plans, on
average, scored similarly to real ones?

### Plot 2: `critic_paired_scatter.png`

Scatter plot, one point per sample:
- x-axis: `score_real`
- y-axis: `score_gen`
- y=x diagonal overlaid

**What it tells you:** per-state calibration. Does the critic rank
generated and real plans similarly *for the same start state*? More
informative than the marginal histograms because it controls for
start-state effects (some maze regions are nearer the goal).

### Plot 3: `divergence_vs_horizon.png`

Line plot:
- x-axis: jump-step index, 0 to 31
- y-axis: mean L2 distance across the 100 samples, with shaded
  ±1 standard deviation band
- Two lines: imagined-vs-real (the main signal) and
  imagined-vs-imagined-2nd-draw (the planner self-consistency baseline)

**What it tells you:** how generated plans diverge over the horizon
relative to the planner's own intrinsic stochasticity. If
imagined-vs-real lies near imagined-vs-imagined, the planner is
generating "another valid future" — fine. If imagined-vs-real is much
larger, plans are systematically drifting from the data manifold.
Distance at jump-step 0 must be ~0 for both lines because of the
fix_mask.

### Plot 4: `trajectory_overlays.png`

A 2×4 grid of subplots. Each shows the maze layout (walls drawn as grey
rectangles based on the env's maze spec) with two trajectories overlaid:
- Real continuation in blue.
- Generated plan in red.
- s_0 marked with a black dot.

Pick 8 samples: 4 random from the 100, plus 4 from the lowest-score_gen
tail to stress-test the model's worst cases.

Both trajectories must be unnormalised before plotting (use
`normalizer.unnormalize(...)`) so the spatial layout is interpretable.

**What it tells you:** the visual sanity check. Are generated plans
plausible maze paths, or do they cut through walls / loop / explode?

## Pass Criteria

| Metric | Pass | Borderline | Fail |
|---|---|---|---|
| K–S statistic between `score_gen` and `score_real` distributions | < 0.20 | 0.20–0.40 | > 0.40 |
| `mean(score_gen) - mean(score_real)` | within ±0.10 | ±0.10–0.25 | > ±0.25 |
| In-bounds rate | ≥ 95% | 80–95% | < 80% |
| Trajectory overlays (visual) | plans look like plausible maze paths | some plans wander but mostly OK | clearly broken — straight lines through walls, NaN, jitter |
| `assert (imagined_plan[0] - s_0).abs().max() < 1e-4` | passes for all 100 | n/a | any failure = stop, fix, rerun |

The critic output range is approximately [-1, 1] (per Phase 0 map: MC
returns normalised to that range and trained with MSE), so a 0.10
absolute offset is meaningful but not catastrophic.

**Decision rule:**
- Two metric passes → PASS overall.
- One pass + one borderline → PASS-WITH-LIMITATION (note in the writeup
  which metric is borderline and why it's tolerable).
- Two borderlines, or any fail → STOP, escalate to supervisor before
  Phase 2.

A *systematically higher* `score_gen` than `score_real` (i.e.
mean(gen) − mean(real) > +0.10 with K–S also elevated) is the early
warning sign for model exploitation. This is not a Phase 1 failure; it
is a Phase 5 finding showing up early. Note it in the writeup as
motivating evidence for the uncertainty-penalty work.

## Anomalies to Watch For

- **`score_gen << score_real`:** the critic believes generated plans
  are *worse* than the dataset. Investigate visually — look at the
  worst 5 cases. If they cluster near walls or at high velocity, the
  model has trouble in specific regions, which is a known limitation
  worth flagging but not a project-stopper.
- **`planner_self_l2` ≈ 0:** the planner is essentially deterministic
  given a fixed seed. This would invalidate MCTS expansion (no
  branching diversity from stochastic samples). Should be substantially
  greater than 0.
- **`planner_self_l2` ≈ `imagined_vs_real_l2`:** the planner is
  generating valid alternative futures rather than drifting OOD. Best
  possible outcome.
- **`imagined_vs_real_l2` >> `planner_self_l2`:** plans are
  systematically drifting away from the data manifold, even allowing
  for diffusion stochasticity. This is the model exploitation
  signature. Note for Phase 5; not a Phase 1 fail unless visuals also
  break.
- **`score_gen` has a long tail of very low values:** some specific
  start states produce broken plans. Identify them, check whether they
  cluster (e.g. all near walls). Worth a 30-minute investigation; may
  motivate a "bad starting region" caveat in the dissertation.
- **In-bounds rate 80–95%:** plans occasionally leave the maze. If the
  excursions are small and brief, tolerable. If plans regularly walk
  far outside the maze, the model has a real problem.

## What to Do on Each Failure Mode

| Failure | Likely Cause | Action |
|---|---|---|
| `fix_mask` assertion fails | mask not being applied at every denoising step | Inspect `cleandiffuser/diffusion/diffusionsde.py` line 938; debug before any further work. **Project-blocker until fixed.** |
| In-bounds rate < 80% | wrong checkpoint, wrong noise schedule, or model has overfit to start-of-trajectory states | Recheck `run_one_episode.py` produces ~114 normalised score. If runner works but held-out generalisation fails, escalate to supervisor about retraining DV with random subsequence sampling. |
| K–S > 0.40, `score_gen` >> `score_real` | model exploitation at depth 0 | Not a Phase 1 fail. Proceed to Phase 2 but flag prominently for Phase 5. |
| K–S > 0.40, `score_gen` << `score_real` | planner produces low-quality plans on these specific states | Visual investigation of worst cases. May indicate state-distribution issues worth caveating. |
| `planner_self_l2 ≈ 0` | sampling determinism — possibly a bug in seed handling | Inspect `torch.randn_like` calls in `diffusionsde.py`; verify per-call randomness. **Project-blocker until fixed** because MCTS expansion needs stochastic branching. |
| Anything else surprising | unknown | Save all artefacts, write up what you saw, send plots to supervisor. Pause before Phase 2. |

## Deliverable

`notes/phase1_verification.md` containing:

1. **Setup:** env, n_states, ckpt path, seeds, git commit hash, total
   wall time.
2. **Pre-flight summary:** one paragraph linking to
   `notes/phase1_preflight.md` and confirming nothing changed.
3. **Quantitative results:** K–S statistic, mean score gap, in-bounds
   rate, mean and max L2 at horizon, planner self-consistency baseline.
4. **Plots embedded** (or linked).
5. **Verdict:** PASS / PASS-WITH-LIMITATION / FAIL with one paragraph
   of justification against the pass-criteria table.
6. **Anomalies for Phase 5:** anything worth flagging that doesn't
   block Phase 2 but should inform later work.