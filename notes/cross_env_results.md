# Cross-environment results — V(s) critic and the MCTS sampler

*Consolidation of the state-value V(s) training and the MCSS-vs-MCTS(b16) sampler results
across every D4RL env tested, plus the geodesic-oracle coverage. As of 2026-07-02 (before the
FrankaKitchen pivot). Companion: [writeup_mcts_sampler.md](writeup_mcts_sampler.md) (antmaze-large
depth), [findings_summary.md](findings_summary.md) (the negative-result narrative),
[project_kitchen_pivot](../memory) (why we pivot).*

---

## 1. V(s) state-value critic — correlation by environment

Held-out `val_corr` from `state_value_train_log.json` (expectile τ=0.9; MSE gave the same, so the
loss is not the bottleneck). **Peak** is the deploy target (`state_value_ckpt_best.pt`); **final**
is `_latest` at 200k/1M (what the pre-fix runs mistakenly loaded — V(s) overfits).

| env | dataset | **peak** val_corr @ step | final (`_latest`) | overfit gap | read |
|---|---|---|---|---|---|
| antmaze-large-diverse-v2 | diverse | **0.874** @ 6k | 0.809 | −0.07 | reference (the one that worked) |
| antmaze-medium-play-v2 | play | **0.865** @ 5k | 0.646 | −0.22 | strong — nearly matches large-diverse |
| antmaze-large-play-v2 | play | **0.665** @ 6k | 0.495 | −0.17 | usable |
| antmaze-medium-diverse-v2 | diverse | **0.513** @ 6k | 0.360 | −0.15 | weak (small maze compresses corr) |
| maze2d-large-v1 | play | **0.742** @ 105k | 0.739 | ~0 | label-noise ceiling (slow overfit) |
| maze2d-medium-v1 | play | **0.636** @ 129k | 0.633 | ~0 | label noise |
| maze2d-umaze-v1 | play | **0.390** @ 125k | 0.387 | ~0 | worst — SNR floor (tiny maze) |

**Three findings:**

1. **V(s) is an ill-posed regression on this data, and that caps the correlation** (not capacity,
   not loss). The input is a single state; the target (return-to-go) depends on the *unseen* future
   the behaviour policy took and — on maze2d — the *random* goal. Same state → many labels → an SNR
   ceiling no network beats. Evidence: the MLP already memorises (`train_mse → 0`), a transformer
   would overfit *more* not correlate better, and switching MSE→expectile moved umaze 0.41→0.39
   (i.e. nothing). The MCSS critic is near-perfect precisely because its input is the *whole
   trajectory*, making its target deterministic given the input.
2. **The correlation tracks target signal-to-noise, not maze "difficulty."** Large mazes / fixed
   goals (antmaze, big spatial gradient) → high corr; small mazes / random goals (umaze 0.39) →
   low, because the return-to-go range is compressed while the behaviour-wandering noise is not.
3. **The antmaze critics overfit hard and early** (peak @ 4–6k, then collapse), so the `_best`
   checkpoint fix matters there (recovers +0.15–0.22 corr); maze2d overfits slowly (`_best ≈ _final`).

## 2. Sampler performance — MCSS vs MCTS (b16)

### 2.1 antmaze-large-diverse-v2 — fully characterised (the headline)

| n | MCSS k50 | MCSS k272 | MCTS b16 | matched-compute (k272→b16) | flat-scale (k50→k272) |
|---|---|---|---|---|---|
| **500** (10 seeds) | **78.8** | **74.8** | **79.0** | +4.2 pp, p=0.12 (n.s.) | −4.0 pp (backfire) |
| 150 (3 seeds) | 79.3 | 72.0 | **83.3** | +11.3 pp, p=0.021 | −7.3 pp |

Oracle ladder (n=150, Rule-1 diagnostic — never citable as achievable): flat geodesic re-rank
**78.7 ≈ critic 78.0**; geodesic *in the tree* **82.0 ≈ V(s) tree 83.3**; V(s,g) tree 76.7. →
**value accuracy is not the lever; the ~80% cap is locomotion (100% topples).**

### 2.2 The other D4RL envs — MCTS(b16) is WORSE than MCSS

The point of maze2d is NOT reach (saturated) but the **DV camping score** (reward accrues every step
at the goal → reaching *faster* scores higher). Measured paired on that metric:

| env | metric | MCSS | MCTS b16 | Δ | p |
|---|---|---|---|---|---|
| maze2d-umaze-v1 | camping | 141.2 | 105.6 | **−35.6** | <1e-3 |
| maze2d-medium-v1 | camping | 148.7 | 135.7 | **−13.1** | <1e-3 |
| maze2d-large-v1 | camping | 202.7 | 167.3 | **−35.4** | <1e-3 |
| antmaze-medium-diverse-v2 | reach% | ~87 (s0–1) | 64 (s0) | **−26 pp** | 0.002 |

**The maze2d result is clean (no topples, no oracle):** MCSS uses the good DV *trajectory* critic;
MCTS uses the bad V(s) (0.39–0.76), so the tree routes worse → reaches slower → camps less (and even
drops *below* 100% reach on umaze/large, a task MCSS solves perfectly). First clean demonstration that
**a bad V(s) in a max-backup tree is worse than flat MCSS with a good critic.** This **revises** §2.1:
`b16 ≈ MCSS` on large-diverse was the *exception* — a hard env (MCSS 78%) where the harm was masked by
headroom; across the family the tree **hurts**.

**Oracle-tree on antmaze-medium-diverse (Rule-1, oracle validated correct):** MCSS 90% > V(s)-tree 64%
> **oracle-tree (perfect geodesic) 54%.** A *perfect distance value does worse than a mediocre learned
one* — the geodesic guides toward aggressive shortest-path steps that topple the Ant (dynamics-blind),
while the DV trajectory critic is dynamically aware. Reinforces: **value accuracy is not the lever; the
ceiling is execution.** (antmaze variants: only partial seeds so far; the maze2d oracle had a row/col
swap, fixed 2026-07-02 — geodesic-flat maze2d numbers pending the re-run.)

## 3. Geodesic-oracle coverage (Rule-1 diagnostics)

| tool | what it is | envs run |
|---|---|---|
| `diag_oracle_flat` (`orc`/`fs`/`fsf`/`stb`/`gnt`/`smt`, `anim_flatlog_*`) | geodesic critic in the DV MCSS pipeline (DV planner + DV policy) | **antmaze-large-diverse only** |
| `diag_oracle_tree` | geodesic *inside* MCTS | antmaze-large-diverse only |
| phase6 `greedy-bfs` | geodesic greedy, true-sim + direct action | **maze2d {umaze; medium partial}** |

**Gap:** the DV-pipeline geodesic oracle (`diag_oracle_flat`/`tree`) was never run on maze2d or the
antmaze medium/play variants. On maze2d it needs the BFS oracle wired into the mcts tooling + the
efficiency metric (reach is saturated), and is largely predicted-null (phase6 already shows
geodesic ≤ DV). On the antmaze variants it would be *informative* (not saturated) and is the higher-
value fill-in if completeness is wanted.

## 4. Bottom line

Value **quality** varies widely by env (SNR + overfit), but value **accuracy is never the lever**:
a perfect geodesic critic ≤ DV-MCSS on every D4RL nav env, because DV's critic already selects the
geodesic-best plan and the ceiling is execution (antmaze topples; maze2d momentum, where DV already
beats a position-value oracle). This motivates the FrankaKitchen pivot: a **clean** value target
(discounted subtask-return → V(s) should correlate) *and* genuine sequential-planning headroom, with
no locomotion confound.
