# Phase 6 — MuZero-style learned model for MCTS (design)

## 0. What we are departing from (DV-MCSS, verified against source)

| component | DV-MCSS | trained how |
|---|---|---|
| planner | full-horizon (H=32, stride 15) **generative** state-sequence diffusion | denoising loss on normalized state sequences |
| critic `DVHorizonCritic` | scores a **whole 32-step trajectory** → scalar | MSE to Monte-Carlo value target |
| policy | diffusion **inverse dynamics**, `(s_t, s_{t+1})→a` (15-step waypoints) | behaviour cloning on dataset actions |
| inference | sample K=50 full plans → critic argmax → execute `traj[:,1]` via inv-dyn | — |

The two measured failure modes (Phases 1–5): **(A) search is redundant** — the full-horizon
planner does the lookahead internally, so re-sampling it in a tree adds nothing; **(B) execution
is the ceiling** — the 15-step inverse-dynamics layer has high tracking variance (open-loop
return 17→222 for plans that all reach the goal).

## 1. Why MuZero could win here — the two-bottleneck reframe

A one-step MuZero is not merely "search done right." It attacks **both** DV failure modes:

1. **Grounded, non-redundant search.** The tree expands a learned **one-step** dynamics model
   `g(z,a)→z'`. Lookahead only exists because search *composes* it — there is no full-horizon
   model short-circuiting it. (vs DV: tree re-samples a full-horizon generator.)
2. **No inverse-dynamics layer.** The policy head outputs the **primitive action** directly
   (force_x, force_y) chosen by search. There is no waypoint→action translation, so DV's
   tracking-variance bottleneck is *removed by construction*.

And the maze2d offline dataset is **ideal for this**: it is wandering data (random waypoints),
which is *bad* for learning a goal-directed generative planner but *excellent* coverage for
learning **dynamics**. MuZero learns dynamics from the wandering data and lets **search** derive
goal-directed behaviour — it never needs expert demonstrations. This is the strongest a-priori
case for MCTS we have had on this task.

**Honest gate (unchanged):** maze2d may still be execution/easy-task-limited. Stage 0 below is the
decisive go/no-go before any model training.

## 2. Architecture — modular so the SOTA endpoint is a *component swap*, not a rewrite

Three core networks (stable across every stage):

- **Representation** `h(o, g) → z₀`. Input = normalized obs **augmented with relative goal**
  `[x, y, vx, vy, gx−x, gy−y]`. Goal-conditioning makes the value signal learnable and the agent
  goal-general.
- **Dynamics** `g(z, a) → (z', r̂, donê)`. One-step (default; horizon is an ablation, §4).
- **Prediction** `f(z) → (policy_prior, value)`.

Three **swappable modules** — this is what makes the plan converge on SOTA:

| module | simple → SOTA | swapped at stage |
|---|---|---|
| `ActionModule` | discrete grid → **Sampled** (sample continuous actions from a flow/Gaussian prior + progressive widening) | Stage 4 |
| `SearchModule` | PUCT-MCTS → **Gumbel** MuZero (Gumbel root + sequential halving; policy-improving with few sims) | Stage 3 |
| `TargetModule` | BC targets → **Reanalyse** (MCTS-on-data as the teacher) | Stage 2 |

The core unroll-trainer (below) never changes; only these three modules are upgraded. **SOTA =
Sampled + Gumbel MuZero + Reanalyse**, reached by swapping the action and search modules — no rewrite.

## 3. The unroll trainer (MuZero, stable across stages)

Sample a length-(K+1) segment `(o_t, a_t, r_t, …)` from the dataset. Encode `z⁰ = h(o_t, g)`.
Unroll the model with the **real actions**: `z^{k+1}, r̂^k = g(z^k, a_{t+k})`. At each step predict
from `z^k`:

- **reward loss**  `ℓ(r̂^k, r_{t+k})`
- **value loss**   `ℓ(v(z^k), V_target_{t+k})`  — n-step bootstrapped return (target net), refreshed by Reanalyse
- **policy loss**  `CE(p(z^k), π_target_{t+k})`  — BC action (Stage 1) → MCTS visits (Stage 2+)
- **consistency loss** (EfficientZero) `‖g(z^k,a)·sg − h(o_{t+k+1})‖` — SimSiam stop-grad; grounds the
  latent so deep unrolls don't drift. **Default-on, not an ablation** — we are fully offline, so the
  model is never corrected by fresh interaction and latent drift is otherwise unchecked.

Standard MuZero details: scale recurrent grads by 1/K, scale losses, use a target network for value
bootstrap. **Not** trained to reconstruct observations by default (value-equivalent model); raw-obs
prediction is a model-target ablation (§4).

**Offline value over-estimation (expected, needs an explicit fix).** Bootstrapping value from a
target net on offline data means out-of-distribution `(s,a)` get an inaccurate `g` rollout and an
optimistic value — the classic offline-RL distribution-shift failure. The consistency loss anchors
latents but does **not** prevent this on unseen *action sequences*. Mitigations to include (not
defer): (i) keep the search near the data — Sampled-MuZero's BC prior naturally constrains actions
to the behavior distribution; (ii) a conservative/pessimistic value penalty (CQL-/ROSMO-style) or
an ensemble-disagreement penalty on `g`. Treat one of these as near-mandatory, not optional.

## 4. Staged plan — every stage is a measurement AND a step toward SOTA

| stage | what it adds | isolates | SOTA-forward? |
|---|---|---|---|
| **0. Oracle decomposition** (see §7.1) | true env dynamics + **oracle state-value** (BFS distance on umaze; momentum-aware VI / true-sim MPC on medium/large — §7.1); compares oracle-greedy / oracle-MCTS / DV-greedy / oracle-value-via-inv-dyn | splits DV's deficit into **value-error vs execution-error vs search**; does NOT use the MCSS critic | gate; reuses SearchModule |
| **1. Offline pretrain (h,g,f)** | learned model, **discrete** actions, **PUCT** MCTS, BC policy target | cost of learned-model error vs Stage 0 | core nets fixed hereafter |
| **2. Reanalyse** | MCTS-on-data as teacher (π, V targets) | value of search-as-teacher vs pure BC | TargetModule → SOTA |
| **3. Gumbel search** | swap PUCT → Gumbel MuZero | value of the SOTA search algorithm at low sim budget | SearchModule → SOTA |
| **4. Sampled continuous** | swap discrete grid → sampled continuous actions + progressive widening | value of removing action discretization | ActionModule → SOTA |

Stages 3–4 give **Sampled + Gumbel MuZero** — the SOTA continuous-control MCTS — as the endpoint.
Each stage is independently evaluable, so gains/limits are attributable to one variable.

**Expected Stage-1 difficulty — weak policy prior.** `f(z)→policy_prior` is BC'd on *wandering*
data, so the prior is biased toward random/aimless actions. In PUCT the prior steers search, so a
near-random prior means MCTS needs *many more* simulations to overcome it. Anticipate this rather
than be surprised by it: down-weight the prior term in PUCT, raise the sim budget at Stage 1, and
treat it as a *positive* argument for moving to **Stage 3 (Gumbel) sooner** — Gumbel's root sampling
+ sequential halving is far more robust to a poor prior and guarantees policy improvement even at low
sim counts. Reanalyse (Stage 2) also progressively replaces the BC prior with the (better) MCTS
visit distribution.

## 5. Ablations (mapped to the modules; each isolates one design choice)

1. **Transition horizon** (`g` granularity): 1 dense step / 5-step chunk / 15-step jump.
   *Critical* — 1-step is faithful but needs deep search; 15-step re-introduces the far-target
   problem we already diagnosed. Hypothesis: 1-step + value-bootstrap at depth ~5 is best.
2. **Action space**: discrete grid vs sampled-continuous (Stage 4). **Default grid ≥ 9×9, not 5×5** —
   a 5×5 grid on `[−1,1]²` gives 0.5-unit force steps, too coarse for the ball's low-speed force
   sensitivity; sweep {9×9, 13×13}. Force discretization is inherently lossy for this env, so the
   continuous Stage-4 endpoint matters *more* here than in discrete-action domains.
3. **Model target**: value-equivalent + EfficientZero consistency (default, §3) vs + raw-obs
   reconstruction. (Consistency itself is no longer an on/off ablation — it is default-on.)
4. **Value target** (maze2d sparse reward — likely *needs* shaping): n-step return (canonical) vs
   **success-probability-within-H (BCE)** vs **negative-time-to-goal**. Default to a shaped value
   (success-prob + time-to-goal) given the wandering, mostly-zero-return data; sparse return is the
   lower-bound comparison.
5. **Search budget**: sims {25, 50, 100, 200}; max depth {5, 10, 20}; exploration const / temperature.
6. **Training regime**: offline BC-only → offline + Reanalyse → (optional) online fine-tune if env
   interaction is allowed.

## 6. Evaluation (seed-matched, same protocol as Phases 0–4)

Baselines on `maze2d-{umaze,medium,large}`: DV-MCSS greedy K=50 · current DV-MCTS · BC policy ·
**true-sim MCTS oracle** · MuZero-MCTS (each stage). Report normalized score, goal_step, and — to
test the execution-bottleneck hypothesis directly — the min-dist/return-variance signature from the
zero-return diagnostic.

## 7. Go/no-go gates (do not skip)

### 7.1 Why Stage 0 uses an oracle value, not the MCSS critic
The `DVHorizonCritic` scores a **whole trajectory → scalar**; primitive-action MCTS needs a
**state-value `V(s)`**, so it is not reusable regardless. On a **deterministic** MDP, greedy w.r.t.
the **true optimal value `V*`** is already optimal (Bellman optimality) — **multi-step search can
never beat it.** Search only helps when the available value is *far* from `V*`.

**Caveat — `V*` is momentum-aware; BFS distance is not.** maze2d is a continuous *physics* sim: the
ball carries velocity and cannot stop or turn instantly, so the true `V*(x,y,vx,vy)` depends on
dynamics (deceleration, overshoot, turning radius at speed). A grid **BFS distance is position-only**
and is *not* `V*`. Consequences for Stage 0:
- BFS is an acceptable near-oracle on **umaze** (low speeds, short paths); on **medium/large** at
  speed, BFS-greedy will visibly overshoot and is *not* optimal.
- The proper oracle is **value iteration on a discretized `(x,y,vx,vy)` grid with the true dynamics**
  (momentum-aware), or, as an empirical stand-in, a large-budget **true-sim MPC**. Stage 0 should
  report the BFS approximation **and** at least one momentum-aware oracle on medium/large.
- This actually *sharpens* the search question: with a **position-only** value, multi-step search
  **does** help (it handles momentum/overshoot that the value ignores); with a **momentum-aware**
  `V*`, greedy is optimal and search is redundant. So **run both** to bracket the answer — and note
  the mapping to MuZero: search helps exactly to the extent the *learned* value `f(z)` fails to
  encode velocity. (`z` can encode it, so a well-trained value should leave little search headroom.)

This momentum-vs-value framing is the *root cause* beneath "the planner does the lookahead",
"re-planning is cheap", and "wp1 is optimal." Stage 0 isolates **value quality vs execution vs
search**, with the critic removed entirely:

| controller / pair | value | execution | isolates |
|---|---|---|---|
| momentum-aware oracle (VI / true-sim MPC) | momentum-aware `V*` | direct | the env's practical **ceiling** |
| **greedy vs MCTS, position-only value** (BFS) | position-only | both direct | search **SHOULD** help — recovers momentum/overshoot the value ignores |
| **greedy vs MCTS, momentum-aware value** | momentum-aware `V*` | both direct | search should **NOT** help — greedy ≈ optimal (Bellman) |
| ceiling vs DV-greedy | — | — | DV's **total** gap |
| oracle-value via DV inv-dyn vs direct action | same value | inv-dyn vs direct | splits the gap into **value-error vs execution-error** |

The two middle rows are the **bracket**: if search helps under the position-only value but vanishes
under the momentum-aware value, then "does search help" reduces to "does the value encode velocity"
— which maps directly to whether MuZero's learned `f(z)` captures the dynamics-relevant state. Log
**both** greedy-vs-MCTS pairs explicitly.

Decision:
- **oracle-greedy ≈ DV-greedy** → DV is at the ceiling; nothing (search/critic/MuZero) has room →
  **escalate the task** (antmaze / shortened-horizon / stochastic) before building the model.
- **oracle-greedy ≫ DV-greedy, gap is execution** → MuZero's **direct primitive action** (no
  inverse-dynamics) is the lever. Clean, publishable on its own. Note: "search is incidental" holds
  **on umaze** (simple enough that greedy-on-a-good-value suffices); on **medium/large** search may
  still help even with direct actions — planning around dead-end corridors that a greedy step on an
  imperfect learned value would enter — so do not generalize the umaze read to bigger mazes.
- **gap is value** → MuZero's **learned state-value** is the lever; search helps only insofar as the
  learned value stays far from `V*`.
- **search beats greedy even with a good value** → the only case that justifies MCTS itself; on
  deterministic maze2d theory predicts this bucket is ~empty, so a positive result here would be the
  most interesting outcome of all.

### 7.2 Model-accuracy gate (before trusting learned-model search)
A learned `g` must roll out ~5–10 steps without drift (check consistency loss / open-loop latent
error) before its search can be trusted.

## 8. First concrete deliverable

**Stage 0 oracle decomposition only.** No model training, no critic. Reuse the gym env as the
simulator (`env.unwrapped.set_state` to copy node states). For the value: start with BFS
shortest-path distance on the maze grid (flag it explicitly as a **position-only approximation** —
valid on umaze, will overshoot at speed on medium/large), and on medium/large add a **momentum-aware
oracle** (value iteration on a discretized `(x,y,vx,vy)` grid, or a large-budget true-sim MPC as the
empirical ceiling). Run the four controllers in §7.1 (oracle-greedy, oracle-MCTS,
oracle-value-via-DV-inverse-dynamics, and re-state the existing DV-greedy number) on the same seeds,
**with both the position-only and momentum-aware value** to bracket the search question. ~1 script.
It decomposes DV's deficit into **value-error vs execution-error vs search**, which determines
(a) whether maze2d has *any* headroom, (b) which MuZero lever (direct action / learned value /
search) matters, and (c) whether we need a harder task before building Stages 1–4.
