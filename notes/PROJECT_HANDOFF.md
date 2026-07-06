# MCTS-as-Sampler for Diffusion Veteran — COMPLETE PROJECT HANDOFF

*Self-contained state of the whole project as of 2026-07-02, written so a fresh chat can
continue with zero prior context. Companion docs (all in `notes/`, more detail): `writeup_mcts_sampler.md`
(antmaze depth), `cross_env_results.md` (V(s)+sampler tables), `findings_summary.md` (narrative),
`instrumentation.md` (failure tooling). Memory: `project_mcts_results`, `project_kitchen_pivot`,
`project_dv_pipeline`.*

---

## 0. One-paragraph summary

MSc dissertation: integrate **Monte-Carlo Tree Search** into the **Diffusion Veteran (DV)**
generative planner, replacing DV's stock **MCSS** (best-of-K) sampler with a value-guided tree,
reusing DV's trained diffusion planner + inverse-dynamics policy unchanged. **Result after
exhaustive testing on D4RL nav (antmaze + maze2d): MCTS-as-sampler does NOT beat MCSS, and
usually HURTS. The bottleneck is the VALUE, not the search structure.** MCSS uses DV's
**whole-trajectory critic** — a trained, dynamics-aware, task-aligned value that is *well-posed*
(its input, the whole plan, determines its target). A tree instead needs a **per-state** value;
every per-state value we can supply is worse: a learned **V(s)** is an *ill-posed* regression
(the state doesn't determine the return → SNR ceiling), and the **true geodesic** is
*dynamics-blind* (worst of all — it ignores momentum/topples). So the tree, fed an inferior
value, loses to flat MCSS. This is a clean, mechanistic **negative result** on D4RL. We are
**pivoting to FrankaKitchen**, the one in-stack env with a *clean, well-posed* value target
(discounted subtask-completion return) and no locomotion confound — the test of whether a good
per-state value lets the tree finally help. Gate: train DV on kitchen (~1–2 GPU-days).

---

## 1. THE KEY INSIGHT (read this first — it resolves "why does flat MCSS beat structured search?")

The MCSS-vs-MCTS comparison was **never** "flat vs structured." It is a **value comparison**.
All arms use the *same* DV planner (generate K plans) and the *same* DV inverse-dynamics policy
(execute the first waypoint, replan). They differ only in **which value selects the plan/step**:

| arm | value used | well-posed? | dynamics-aware? | maze2d-large camping |
|---|---|---|---|---|
| **k50 (MCSS)** | DV **trajectory** critic (trained return-to-go over the whole plan) | ✅ input=whole plan → target determined | ✅ trained on real returns | **202.7** (best) |
| b16 (MCTS) | learned **V(s)** (per-state return) | ❌ ill-posed (SNR) | partially | 167.3 |
| k50-fsf (Rule-1) | true geodesic + feasibility filter on first step | ✅ (exact) | ❌ blind | 173.8 |
| **k50-orc (Rule-1)** | true geodesic of the endpoint (a *perfect distance*) | ✅ (exact) | ❌ **blind** | **92.1** (worst; even fails to reach 30–58%) |

**Ordering everywhere (antmaze AND maze2d): DV-critic > noisy-V(s) ≳ feasible-geodesic >
endpoint-geodesic.** The more the value is aligned with the *actual return + dynamics*, the
better. `k50` vs `k50-orc` is the clincher — identical flat structure, only the value differs, and
the trained DV critic beats the *perfect* geodesic by 110 camping points. Why the geodesic is
terrible despite being "perfect distance": on a momentum system (point mass, or the Ant),
"endpoint closest in cells" picks plans that **overshoot** the goal (camping needs *stopping*,
which distance ignores) or that the inverse-dynamics policy can't track → low camping, low reach,
and on the Ant → **topples**. The DV critic is trained on the real return so it picks
dynamically-executable, high-camping plans.

**Consequence for the thesis:** the tree can only help if you can feed it a *per-state* value at
least as good as DV's *whole-trajectory* critic. On D4RL nav, no such per-state value exists
(V(s) is ill-posed; geodesic is dynamics-blind), and the DV critic itself is structurally unusable
in a tree (it scores whole plans, doing all its lookahead at ply-1). Hence MCTS can't win here.
This is the honest, mechanistic negative result.

---

## 2. What Diffusion Veteran (DV) is

DV (ICLR-2025 "What Makes a Good Diffusion Planner", in `pipelines/veteran_d4rl_*.py`,
`cleandiffuser/`) = three trained nets, reused unchanged by us:
- **Planner**: unconditional diffusion over trajectories; given the current state (inpainted at
  t=0) it samples K candidate future trajectories `(H, obs_dim)`.
- **Trajectory critic** (`DVHorizonCritic`): scores a *whole* `(H, D)` plan → one scalar
  return-to-go (read from transformer token 0). This is the MCSS selector.
- **Inverse-dynamics policy** (`DVInvMlp`, diffusion): given (state, next waypoint) → action.
- **MCSS sampler** (stock): sample K plans → **critic-argmax** → execute the **first** waypoint →
  replan every step (MPC). `guidance_type: MCSS`.
- Env families & checkpoints ON DISK: **antmaze** (large-diverse, large-play, medium-diverse,
  medium-play), **maze2d** (umaze, medium, large). Configs also exist for **kitchen** and
  **mujoco** but **no checkpoints** (must train). Ckpt roots in `mcts/specs.py:SPECS`.
- Obs: antmaze 29-dim (xy@0:2, quat@3:7, jointvels…, **linear vel@15:18**); maze2d 4-dim
  `[x,y,vx,vy]`. antmaze goal ≈ fixed far corner; maze2d goal fixed per env, reach saturated.

## 3. What we built (the MCTS integration + tooling)

- **`mcts/value_forest.py`** — torch-free forest of M lockstep trees, UCB descent, **max-backup**,
  one batched planner+value call per round; 8/8 unit tests. `mcts/mcts_loop.py` — shared
  closed-loop harness (`Sampler`, `run_episodes`), per-rollout logging, DV-exact scoring.
- **`mcts/value_net.py` + `scripts/train_state_value.py`** — the retrained per-state **V(s)** MLP
  (needed because the tree needs a per-state value; the trajectory critic can't drive a tree).
  Also a goal-conditioned **V(s,g)** ensemble (relabel + expectile + pessimism).
- **`scripts/run_mcts_compare.py`** — MCSS vs MCTS, `--method both`, `--value-step best`, `--out`
  JSON (has `dv_norm` per-rollout for maze2d camping).
- **`scripts/collate_mcts.py`** — goal-verified paired stats. **Family-aware**: antmaze →
  reach% + exact McNemar; **maze2d → DV camping score + paired-difference test** (reach saturated).
  Has a "DV-score" column.
- **Rule-1 geodesic diagnostics** (DIAGNOSTIC-ONLY, never a reportable number): `mcts/maze_oracle.py`
  (`AntMazeOracle`, `Maze2DOracle`, `make_oracle`), `scripts/diag_oracle_flat.py`
  (flat selectors: `endpoint`/`orc`, `firststep`, `feasible`/`fsf`, `stable`, `gentle`, `smooth`),
  `scripts/diag_oracle_tree.py` (geodesic *in* the tree).
- **Failure instrumentation**: `mcts/instrument.py`, `mcts/failure_modes.py`,
  `scripts/diag_fall_geometry.py`, `scripts/animate_failure.py`, `scripts/make_report_figures.py`
  (5 report figures in `results/figs/`), `scripts/run_compare_trace.py` + `animate_compare.py`.
- **`phase6_stage0_oracle.py`** — maze2d true-sim BFS-value controllers (greedy/MPC). NOTE the MPC
  is mis-tuned/abandoned (a true-sim planner, not DV — a distraction).

## 4. Results — antmaze-large-diverse-v2 (the primary env)

- DV MCSS baseline: **76.9%** (pipeline n=1000); our harness reproduces (~77–79%).
- **n=500 (10 seeds)**: MCSS k50 **78.8**, MCSS k272 **74.8**, MCTS b16 **79.0**. Matched-compute
  k272→b16 **+4.2pp p=0.12 (n.s.)**; k50→b16 +0.2 p=1.0 (tie); k50→k272 −4.0 (flat-scaling
  backfire, critic over-exploitation). *Earlier n=150 gave +11.3pp p=0.021 — small-sample optimism.*
- **Oracle ladder (n=150, Rule-1)**: flat geodesic re-rank **78.7 ≈ critic 78.0**; geodesic-in-tree
  **82.0 ≈ V(s) tree 83.3**; V(s,g) tree 76.7. → **value accuracy is not the lever.**
- **Failure mode = 100% physical TOPPLES** (uprightness ≈ −0.91, motionless). Fall rate rises with
  selection aggressiveness. Every execution-aware selector (stability/gentle/smooth) is null;
  fall-geometry **refutes walls**, shows the sharp pre-topple turn is a **symptom**. → the ~80% cap
  is **locomotion (the DV inverse-dynamics policy tips the Ant)**, below the sampler.

## 5. Results — cross-env broadening (Tier-0), and the maze2d camping headline

**V(s) val_corr by env** (expectile; MSE identical → loss not the bottleneck). Peak = deploy
(`_best`); antmaze critics **overfit** (peak @4–6k then collapse), maze2d overfit slowly:

| env | peak | final | | env | peak | final |
|---|---|---|---|---|---|---|
| antmaze-large-diverse | 0.874 | 0.809 | | maze2d-large | 0.742 | 0.739 |
| antmaze-medium-play | 0.865 | 0.646 | | maze2d-medium | 0.636 | 0.633 |
| antmaze-large-play | 0.665 | 0.495 | | maze2d-umaze | **0.390** | 0.387 |
| antmaze-medium-diverse | 0.513 | 0.360 | | | | |

**maze2d — DV camping score, paired n=150** (reach saturated ~100%, camping is the metric;
`orc`/`fsf` are Rule-1 ceilings):

| env | MCSS k50 | MCTS b16 | fsf (geodesic+feas) | **orc (perfect geodesic)** |
|---|---|---|---|---|
| maze2d-large | **202.7** | 167.3 | 173.8 | **92.1** (reach 52–74%) |
| maze2d-medium | **148.7** | 135.7 | 133.6 | **71.9** (reach 62–78%) |
| maze2d-umaze | **141.2** | 105.6 | 78.2 | **46.8** (reach 38–44%) |

All Δ vs MCSS are **p<1e-3**. MCTS(V(s)) < MCSS (bad V(s) misroutes → slower → camps less). The
*perfect geodesic* (`orc`) is **worst and even fails to reach** — dynamics-blind (§1). This is the
cleanest confirmation of the core finding. **antmaze variants (partial seeds)**: medium-diverse
MCSS 90/84% > MCTS b16 64% > oracle-tree 54% — again, perfect geodesic < mediocre V(s) < DV critic.

## 6. The V(s) SNR diagnosis (why a learned per-state value can't match the critic on nav)

V(s) input = a single state; its target (return-to-go) depends on the **unseen future** the
behaviour policy took AND (maze2d) the **random goal** → the same state maps to many labels → an
**SNR ceiling no network beats** (MLP already memorises `train_mse→0`; a transformer would overfit
more; expectile didn't move corr 0.41→0.39). corr tracks target signal-to-noise, not "difficulty"
(umaze 0.39 is the floor: tiny maze = compressed target range). The MCSS critic avoids this because
its input is the **whole trajectory** → the target is determined by the input (well-posed). The
principled fix is a **goal-conditioned quasimetric distance d(s,g)≈geodesic** (deterministic in
(s,g) → SNR gone; MRN/QRL turns wandering "noise" into coverage) — BUT a better value only pays off
on an env with real planning **headroom** and a **dynamics-aware** objective, which D4RL nav lacks
(antmaze locomotion-capped; maze2d the geodesic is dynamics-blind).

## 7. Unifying conclusion (the dissertation's D4RL result)

**MCTS-as-sampler cannot beat MCSS on DV/D4RL, because the tree requires a per-state value and no
available per-state value matches DV's whole-trajectory critic.** The critic is (a) *well-posed*
(sees the future) and (b) *dynamics/task-aware* (trained on the real return). Every tree-usable
value is worse: learned V(s) is ill-posed (SNR), the oracle geodesic is dynamics-blind (topples /
overshoots). Search **structure** is a red herring; **value quality/alignment** is the lever, and
its ceiling is the DV critic — which the tree structurally cannot use (whole-plan value, all
lookahead at ply-1). Corollary caps: antmaze ≈ locomotion-limited (topples); maze2d ≈ the geodesic
is dynamics-blind (overshoot, no camping). A clean, mechanistic **negative result**, defensible and
publishable.

## 8. THE PIVOT — FrankaKitchen (the current forward direction)

Rationale: the one **in-stack** env that fixes every confound — **sequential** 4-subtask planning
(wrong order caps the rollout = the thesis), **no locomotion** (fixed arm can't fall), and a
**clean, well-posed value target**: `seq_val` = normalised **discounted return of subtask
completions** (discrete events, not wandering-time) → V(s) should *finally* correlate → the test of
whether a good per-state value lets the tree help.
- **WIRED + tested** in `mcts/specs.py` (family `kitchen`, H=32/stride=4/discount=0.997, ckpt path;
  `make_dataset` uses `DV_D4RLKitchenSeqDataset`).
- **GATE = train DV** (no downloadable ckpts): `python pipelines/veteran_d4rl_kitchen.py
  [task=kitchen-partial-v0]` (~1–2 GPU-days planner+critic+invdyn+policy). Then
  `python scripts/train_state_value.py --env kitchen-mixed-v0 --steps 200000` (V(s), now wired),
  then `run_mcts_compare --env kitchen-mixed-v0 --method both --budget 16 --k-mcts 16 --value-step best`.
- **TODO for kitchen** (do when checkpoints exist): extend `collate_mcts` to pair kitchen's
  subtask-score as a *continuous* metric (like the maze2d camping branch), not saturated-reach
  McNemar. kitchen uses `value_mode=v_s` (no xy goal; `get_goal` fails gracefully). NO BFS oracle
  (no maze geometry) → use subtask-order failure analysis instead of a geodesic ceiling.

## 9. Gotchas / things a fresh chat must know (so it doesn't redo them)

- **maze2d oracle transform = `row←x, col←y`** (OPPOSITE of antmaze's `row←y, col←x`). Verified on
  maze2d-large (goal [7,9]→cell (7,9)=the `12` marker; all 8 starts free; row←y gave 12/16 walls).
  umaze/medium are diagonal so don't distinguish — always multi-reset ascii-check. (I flip-flopped
  on this once; it's now settled with 9 data points.) maze2d flatlog **GIFs** would be transposed
  until the animator is swapped too (scores are correct; GIF cosmetic, unfixed).
- **V(s) overfits** on antmaze (peak @4–6k) — deploy `--value-step best` (best-ckpt now saved), not
  `_latest`. Earlier Tier-0 antmaze runs used the overfit `_latest` (numbers not clean).
- **Rule-1 firewall**: the geodesic (`orc`/`fsf`/oracle-tree, `AntMazeOracle`/`Maze2DOracle`) is
  DIAGNOSTIC-ONLY, never a reportable/achievable number.
- **DV-exact scoring**: antmaze = reach% (McNemar); maze2d = camping return >100 (paired-diff).
  The harness/collator already do this per family.
- `phase6_stage0` MPC is broken/abandoned; don't chase it.

## 10. Status of runs

- ✅ antmaze-large-diverse: complete (n=500 + oracle ladder + topple instrumentation + figures).
- ✅ maze2d {umaze,medium,large}: MCSS vs MCTS vs orc/fsf, n=150, camping — **complete** (§5).
- 🟡 antmaze {medium-diverse, medium-play, large-play}: partial seeds (MCSS + b16 + some oracle);
  direction clear (MCTS<MCSS); finish seeds only if the table must be airtight.
- ⏳ **kitchen**: DV training is the gate (start it; everything downstream is wired & fast).

## 11. Open questions / candidate next steps

1. **Kitchen** (primary): does a *clean well-posed* V(s) (subtask-return) let MCTS ≥ MCSS? The
   whole thesis pivot rides on this.
2. **Quasimetric d(s,g)** (`plan_v4` track): the principled value fix (deterministic target) — only
   worth building on an env with headroom + dynamics-aware objective (kitchen, or OGBench-giant).
3. **OGBench** (deferred): ideal planning headroom but a new repo/venv + train DV from scratch
   (gymnasium vs d4rl conflict). Highest cost.
4. Firm the antmaze variant table (more seeds) — optional.

## 12. One-line thesis statement (current best framing)

*"Integrating MCTS into a diffusion planner does not improve closed-loop performance on D4RL
navigation, because the tree needs a per-state value and no per-state value — learned or oracle —
matches the diffusion planner's own whole-trajectory critic, which is well-posed and
dynamics-aware but structurally unusable in a tree; the ceiling is the value, not the search.
FrankaKitchen (clean value target, no locomotion) is the test of whether that changes when a good
per-state value is available."*
