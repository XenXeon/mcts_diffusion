# Session handoff prompt

Paste everything below the line as the FIRST message of a fresh session (any model, no
prior context). It is self-contained; the repo docs carry the details.

---

## Context: MSc dissertation — MCTS as the sampler for Diffusion Veteran

**Project brief:** integrate Monte-Carlo Tree Search into a generative-RL codebase
(Diffusion Veteran / cleandiffuser) so it explores multiple imagined rollouts in parallel,
preventing poor early decisions from sinking the final plan. Offline RL on D4RL
(maze2d, antmaze).

**Repo:** `D:\Surrey\Sem 2\Dissertation\backup\mcts_diffusion_back`, branch
`MCTS_Integration`. NOTE: the local Windows shell has **no torch/GPU** — I run all GPU
commands on a separate Linux box (python 3.10) and paste outputs back to you. You write
code/tests/docs; pure-Python tests run locally.

**Read these three files FIRST, in order, before proposing anything:**
1. `notes/writeup_mcts_sampler.md` — the full findings report (architecture, results,
   caveats, next-experiment design). This is the ground truth of where we are.
2. `notes/mcts_sampler_design.md` — design rationale + verified code facts (file:line).
3. `notes/dv_inference_map.md` — how DV's MCSS inference loop actually works.

**Architecture in one breath:** DV = unconditional diffusion **planner** (generates K
state-trajectories from the current state via start-state inpainting, no goal input) +
**trajectory critic** (scores whole plan → scalar; used by stock MCSS argmax) +
**inverse-dynamics policy** (state-pair → action). MCSS = sample K plans, critic-argmax,
execute first waypoint, replan every step (MPC). We added: a retrained per-state value
`V(s)` (`mcts/value_net.py`, trained by `scripts/train_state_value.py`) and a batched
max-backup MCTS sampler (`mcts/value_forest.py` engine — torch-free, 8/8 tests in
`tests/test_value_forest.py`; `mcts/mcts_loop.py` harness; `scripts/run_mcts_compare.py`
CLI; `scripts/collate_mcts.py` collator). Planner and policy are reused untouched.

**Headline result (closed loop, antmaze-large-diverse-v2, paired n=25, seed 0):**
MCSS reach%: k50=76.0, k144=80.0, k272=84.0 (saturating). MCTS (k_mcts=16): budget4=60.0,
budget8=80.0, budget16=**96.0**. At matched candidates/step (272) and equal wall time,
**MCTS beats MCSS by +12 pp**. Budget curve is monotone ⇒ the gain is look-ahead depth,
not sampling volume. Result JSONs in `results/{mcss,mcts}_antmaze_*.json`.

**Hard-won rules — do not relearn these the expensive way:**
- **Closed-loop only.** Single-shot diagnostics ("do any of K plans reach from a standing
  start?") misled us twice; DV replans every step, so only full-episode reach% counts.
  DV pipeline baselines: antmaze-large-diverse 76.9 ± 1.3 (n=1000); maze2d-large saturated
  (all envs reach) — never use maze2d to look for sampler gains, only as a sanity check.
- **Shallow MCTS loses to greedy MCSS** (budget 4 → 60% < 76%): 1-ply state-value
  estimates are noisier than the whole-trajectory critic; the tree needs depth to win.
- **`V(s)` is goal-agnostic** (corr ≈ 0 with goal-closeness on antmaze); direction comes
  from maze geometry only. All checkpoints default to step 1000000; the state-value ckpt
  is `state_value_ckpt_latest.pt` co-located with the planner checkpoints.
- **Verify every hypothesis against code/data before building anything.** This is my
  standing instruction; it has caught several wrong assumptions already.
- n=25 cells: the +12 pp is 24/25 vs 21/25 — treat as strong-but-preliminary.

**Agreed next steps (in priority order):**
1. **Scale the headline cells**: MCSS k=272 vs MCTS budget=16, n=50–100, ≥3 seeds, to make
   the +12 pp tight (ideally record per-env pairing for a McNemar-style test).
2. **Goal-conditioned value `V(s, g)`**: extend `train_state_value.py` with terminus
   relabelling (each trajectory's reached terminus xy is the goal input; same
   negative-time-to-go target), feed the real `env.unwrapped.target_goal` at inference,
   then run the 3-arm ablation MCSS vs MCTS+V(s) vs MCTS+V(s,g). Design and risks are in
   `notes/writeup_mcts_sampler.md` §6.

Start by reading the three docs, then [STATE WHAT YOU WANT THIS SESSION — e.g. "implement
the V(s,g) trainer", "design the scale-up runs", "review the engine for bugs"].
