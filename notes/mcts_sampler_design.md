# MCTS-as-Sampler for Diffusion Veteran — Design & Plan

**Project brief:** integrate MCTS into a generative-RL codebase so it "efficiently explores
multiple potential rollouts in parallel," because *poor early decisions in an imagined
rollout limit the final plan's quality*. Codebase = **Diffusion Veteran (DV)**, whose
stock sampler is **MCSS**. Goal: replace MCSS with **MCTS as the sampler**, reusing DV's
trained **planner** (diffusion state-trajectory generator) and **inverse-dynamics policy**
(state-pair → action) unchanged, and **retraining only the critic** to be MCTS-optimal.

---

## 1. The crux: why the MCSS critic cannot drive a tree

Three facts from the code, which together explain the Phase-4 result that *MCTS ≈ MCSS at
matched K*:

1. **`DVHorizonCritic` scores a whole `(H,D)` trajectory → one scalar** = return-to-go of
   the *entire* plan (`cleandiffuser/utils/building_blocks.py:210`, reads token-0). It is
   **plan-sensitive, not a state value** (it must be, or MCSS's argmax over K plans from
   one `s₀` would be random).
2. **Every tree node re-plans a full H-step trajectory**; the child state is just
   `traj[child_state_index]`, one waypoint ahead (`mcts/expansion.py`,
   `mcts/tree.py:_process_expansion_result`). So a depth-3 node carries return-to-go *from
   its state to the goal* — which already contains everything its parent's value contains.
3. **The backup averages those full-return values** (`tree.py:_backprop`,
   `node.py:value()`). Averaging full-trajectory returns from different depths is
   semantically meaningless.

**Conclusion:** because the critic does full lookahead at ply 1, depth re-evaluates states
the root's plan already passed through. A full-trajectory critic *structurally* cannot
benefit from a tree.

---

## 2. The design

For the tree to do what the brief wants — commit to a good prefix, branch the
continuation, prune bad early segments — the value must be **decomposable across depth**:

- **State-value critic `V(s)`**: one normalised state → normalised return-to-go. This is
  what makes a node's value depend on *where you are*, so depth = stitching segments.
- **Short-segment expansion** (advance `L` waypoints per depth, `L < H`): the tree must
  compose multiple segments to reach the goal — where branch-and-prune earns its keep.
- **Bootstrapped backup**: node value derived from `V(leaf_state)` of the best reachable
  leaf (max-backup), not mean-average of full-plan scores.

This is MuZero-*shaped* but uses **DV's planner as the segment generator** (no learned
dynamics model) and **DV's inverse-dynamics policy for low-level actions** — exactly the
stated constraint. Only the **critic** changes (to `V(s)`) plus the **tree backup**.

### Two search configs share ONE critic (the ablation)

| Config | Segment length | Stitch? | Critic call | Notes |
|---|---|---|---|---|
| **Opt 1 — segment stitching** | `L < H` | yes | `V(child_state)` | true parallel-rollout search; the headline integration |
| **Opt 2 — endpoint value** | `L = H−1` | no | `V(plan_endpoint)` | full-H plans, one ply; lighter; mostly == MCSS when planner one-shots |

Both use the **same trained `V(s)`** — Opt 2 is just Opt 1 with `L=H−1` and stitching off.
So one critic retrain yields both, and the dissertation ablation is a clean sweep over
segment length `L` and stitch on/off. (Build Opt 1 first; Opt 2 falls out as a config.)

---

## 3. The retrained critic — verified facts (build on these, not assumptions)

- **Supervision = identical to the MCSS critic, keyed per-state.** The DV dataset already
  computes a per-timestep return-to-go `seq_val[p,t]` and stores normalised states
  `seq_obs[p,t]`. MCSS critic trains on `(whole_traj, seq_val[p,start])`; we train on
  `(seq_obs[p,start], seq_val[p,start])` over the same `dataset.indices` (valid starts).
  Only the input representation changes ⇒ the two critics are directly comparable.
  - `seq_obs` is built from `normed_observations` (GaussianNormalizer) —
    `d4rl_maze2d_dataset.py:134,138` — identical to the tree's `s_norm` at inference.
  - `seq_val[p,t] = seq_rew[p,t] + discount·seq_val[p,t+1]`, then min-max → `[-1,1]`
    (`:165–177`). Restricting to `dataset.indices` avoids the padded terminal-state label.
- **Target config MUST match the pipeline** (`configs/veteran/*/reward_mode/linear.yaml`
  + pipeline line 82): `discount=1.0, continous_reward_at_done=True, reward_tune="iql",
  center_mapping=True` (because `guidance_type=MCSS != cfg`). With these,
  `seq_val` = **normalised negative-time-to-goal** in `[-1,1]` (1 == at goal) — a clean
  monotonic value for a stitching search.
  - ⚠ `phase6_headroom_any.py` used `center_mapping=False` for maze2d — harmless there
    (it only used the normalizer, never `seq_val`) but **wrong for value training**. The
    trainer hard-codes `center_mapping=True` for both families.
- **Per-family geometry** (matches `phase6_headroom_any.py` SPECS): maze2d `H=32,
  stride=15, planner depth=2`; antmaze `H=40, stride=25, planner depth=8`. Checkpoints
  co-located under the MCSS planner dirs.

### Critic architecture / training (locked)
- `DVStateValue` (`mcts/value_net.py`): MLP `obs_dim → [256]×3 (SiLU, LayerNorm) → 1`.
- Loss MSE, Adam lr=3e-4, cosine schedule, batch 128, default 200k steps (matches the
  MCSS critic's 200k checkpoint for a fair comparison). Path-level train/val split.
- Saves `state_value_ckpt_{step}.pt` co-located with the planner/critic; logs `val_corr`
  (V vs target) as the first calibration sanity.

---

## 4. Build plan & status

| Phase | Item | Status |
|---|---|---|
| A — verify design | critic is plan-sensitive; depth-redundancy under MCSS critic; per-timestep target exists; normalization/center_mapping confirmed | ✅ done (this doc) |
| B — retrain `V(s)` | `mcts/value_net.py`, `scripts/train_state_value.py` | ✅ code; ⏳ run on GPU |
| B — validate | `scripts/eval_state_value.py`: calibration + does V-endpoint selection fix the antmaze **critic-miss** seeds | ✅ code; ⏳ run after B |
| C — search engine | `mcts/value_forest.py` (batched value-MCTS, max backup) + `tests/test_value_forest.py` (6/6 pass) + `mcts/mcts_loop.py` (closed-loop MCSS vs MCTS) + `scripts/run_mcts_compare.py` | ✅ code+logic tests; ⏳ run on GPU |
| E — eval | closed-loop reach% vs the real baselines (below) | ⏳ |

## 5. Closed-loop correction (important)

Single-shot diagnostics (`headroom`, `eval_state_value`) measure "can one batch of plans
reach from a standing start" — that is **not** what DV does. DV replans **every step** (MPC)
and takes one step. Verified ground-truth closed-loop baselines:

| Env | MCSS closed-loop | Implication |
|---|---|---|
| maze2d-large-v1 | **201.4**, all 50 reach | saturated — no MCTS headroom |
| antmaze-large-diverse-v2 | **76.9 %** | the ~23 % wrong-early-turn failures are the only headroom |

So: (a) the right metric is **closed-loop reach%**, not single-shot reach; (b) DV is
goal-agnostic (start-only inpaint, `condition_cfg=None`) but still gets 76.9 % via MPC; the
failures are *poor early turns* — exactly what look-ahead targets. The MCTS experiment is
**MCTS vs MCSS's 76.9 % on antmaze-large-diverse**. If look-ahead alone doesn't help, the
fallback lever is a **goal-conditioned value** `V(s, g)` (decided by closed-loop evidence).

### Phase C engine (built)
- `mcts/value_forest.py` — torch-free forest of M lockstep trees; one batched
  `expand_fn` call per round covers all trees × K candidates (the parallelism). Max
  (look-ahead) backup: a node's value = max over children (overrides the prior up *or*
  down); chosen action = first waypoint of the root child with highest value.
- `mcts/mcts_loop.py` — MCSS and MCTS on one harness (shared env loop, normalizer,
  inverse-dynamics policy, replan cadence); reach% directly comparable to the baselines.
- Tunables: `budget` (look-ahead depth), `k_mcts` (branching), `child_index` (segment
  length L; L=1 fine-grained, larger = bigger stitches), `c_ucb`.

### Phase E results — antmaze-large-diverse, paired n=25, seed 0 (PRELIMINARY — SUPERSEDED)
> ⚠ The n=25 single-seed numbers below were superseded by the n=150 confirmatory grid.
> Headline at scale: **MCTS+V(s) 83.3 % vs MCSS k272 72.0 %, +11.3 pp, p=0.021** (the +20 pp
> and 96 % here were a lucky n=25 draw — both arms move down together at scale, gap preserved).
> Flat MCSS scaling *backfires* (k50 79.3 → k272 72.0). The goal-conditioned `V(s, g)`
> experiment (Phase F) was run and is a **tested negative** — it did not beat `V(s)` despite a
> sound value (D2). Full, current analysis: `notes/writeup_mcts_sampler.md` §5–§6.

Closed-loop reach% (harness validated: MCSS reproduces the 76.9 baseline → 76 % here):

| sampler | candidates/step | reach% |
|---|---|---|
| MCSS k=50 | 50 | 76.0 ±8.5 |
| MCTS budget=4 | 80 | 60.0 ±9.8 |
| MCTS budget=8 | 144 | 80.0 ±8.0 |
| MCTS budget=16 | 272 | **96.0 ±3.9** |

**reach% climbs monotonically with look-ahead budget; MCTS-b16 beats MCSS by +20 pp.**
This *refutes* the earlier "goal-agnostic value caps the search" prediction: `V` = steps-to-
terminus carries directional signal via maze geometry (dead-ends → low `V`), so look-ahead
avoids poor early turns — the project brief, validated, stronger with depth. Note MCTS-b4
(80 candidates) < MCSS (50) ⇒ the gain is **not** raw candidate count; the tree structure
matters.

**CONTROL RESOLVED (matched candidates/compute):** MCSS k=144 → 80.0 ±8.0; MCSS k=272 →
84.0 ±7.3 (wall 5896s) vs MCTS b16 → 96.0 ±3.9 (wall 5579s). MCSS saturates in k while
MCTS keeps climbing ⇒ **+12 pp genuine look-ahead win at matched compute and wall time.**
Full analysis, limitations, and the V(s, g) next-experiment design:
`notes/writeup_mcts_sampler.md`.

### Deferred to Phase C (decide with data, not now)
- **Exact backup**: max-backup of `V(leaf)` with an additive per-depth step penalty
  `−λ·depth` (sign-safe in `[-1,1]`), vs a discounted form. Ablate once `V` exists.
- **Segment length `L`**: sweep; `L=1` fine-grained ↔ larger `L` bigger stitches.
- **Where MCTS can beat MCSS**: only the stitching regime (planner-short) and possibly the
  antmaze critic-miss. Measured so far: no clean stitching env; antmaze shows critic-miss.
  The honest deliverable is the *integration + ablation*, with `V(s)` also the lever for
  the measured critic-miss.
