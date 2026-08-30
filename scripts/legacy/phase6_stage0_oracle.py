"""scripts/phase6_stage0_oracle.py

Phase 6 — Stage 0: oracle decomposition (the go/no-go gate, no model training).

Question: once hallucination (true dynamics) and the DV inverse-dynamics layer are
removed, where is DV's deficit — value-error, execution-error, or a genuine need for
multi-step search?  See notes/phase6_muzero_design.md §7.1.

Controllers (all use the TRUE env as the transition model via a separate planning env):

  greedy-bfs   horizon-1 greedy w.r.t. a POSITION-only value (BFS distance to goal),
               direct primitive action.  The "no-search, position-value" point.
  mpc-bfs-h{H} receding-horizon MPC (random shooting + optional CEM), terminal value =
               env reward over the rollout + BFS shaping.  Sweeping H is the search
               curve.  Large H = the momentum-aware near-optimal CEILING (the rollout
               in the true sim *is* momentum-aware; BFS only shapes the tail).
  greedy-bfs-dvexec  (optional, needs torch+checkpoint) greedy-bfs picks the target
               next-state, but the DV inverse-dynamics POLICY executes it instead of a
               direct action — isolates the execution gap.

Brackets (design doc §7.1):
  greedy-bfs vs mpc-bfs-h{>1}  → does search help given a position-only value?
                                 (it should: it recovers the momentum the value ignores)
  mpc-bfs-h{large} vs DV-greedy → DV's total gap (baseline read from results/phase0_*).
  greedy-bfs (direct) vs greedy-bfs-dvexec → value-error vs execution-error.

V* CAVEAT: BFS is a POSITION-ONLY approximation; the true V* is momentum-aware.  On
umaze it is an acceptable near-oracle; on medium/large a momentum-aware oracle is the
large-H MPC here.  A proper (x,y,vx,vy) value iteration is a follow-up if the MPC sweep
is ambiguous.  Run with --diagnose first to validate the BFS field visually.

Run:
    python scripts/phase6_stage0_oracle.py --env maze2d-umaze-v1 --diagnose
    python scripts/phase6_stage0_oracle.py --env maze2d-umaze-v1 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict, deque

sys.path.insert(0, ".")

import d4rl  # noqa: F401
import gym
import numpy as np

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="maze2d-umaze-v1",
                    choices=["maze2d-umaze-v1", "maze2d-medium-v1", "maze2d-large-v1"])
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--grid", type=int, default=9,
                    help="discrete action grid per axis for greedy (g×g actions)")
parser.add_argument("--mpc-horizons", type=int, nargs="+", default=[1, 3, 8, 15],
                    help="receding-horizon lengths to sweep (1 = greedy-equivalent)")
parser.add_argument("--mpc-samples", type=int, default=200, help="shooting samples per replan")
parser.add_argument("--mpc-cem-iters", type=int, default=5,
                    help="CEM refinement iterations (>=5 so the momentum-aware ceiling "
                         "is strong; lower for speed at the cost of weaker MPC)")
parser.add_argument("--mpc-vel-pen", type=float, default=0.05,
                    help="terminal-velocity penalty in the MPC objective. The old hardcoded "
                         "0.3 over-penalises: maze2d speeds ~5 give ~1.5 penalty, exceeding "
                         "the ~1-cell/horizon value gain, so the optimiser prefers standing "
                         "still (crawls, never reaches). ~0.05 lets it move; terminal-distance "
                         "still discourages overshoot. Sweep {0, 0.05, 0.15} if needed.")
parser.add_argument("--include-dv-exec", action="store_true",
                    help="also run greedy-bfs target executed via the DV inverse-dynamics policy")
parser.add_argument("--diagnose", action="store_true",
                    help="render the maze + BFS field + goal/start cells and exit")
parser.add_argument("--trace", action="store_true",
                    help="(1) check sim.step == env.step from the same state, then "
                         "(2) trace one greedy episode (pos/vel/value/action) and report "
                         "where it sticks. Decisive for sim≠env vs momentum. Then exit.")
args = parser.parse_args()

ENV_NAME    = args.env
GOAL_RADIUS = 0.5
OUT_DIR     = "results/phase6"
os.makedirs(OUT_DIR, exist_ok=True)
TAG = {"maze2d-umaze-v1": "umaze", "maze2d-medium-v1": "medium",
       "maze2d-large-v1": "large"}[ENV_NAME]

# ── Env (real) + planning env (for lookahead) ──────────────────────────────────
env = gym.make(ENV_NAME)
MAX_T = env._max_episode_steps
sim = gym.make(ENV_NAME)                       # separate env for rollouts
sim._max_episode_steps = 10 ** 9               # never TimeLimit during planning
sim.reset()                                    # satisfy gym OrderEnforcing before set_state/step
                                               # (set_state hits the unwrapped env and won't flip it)


def get_goal(e):
    u = e.unwrapped
    for attr in ("_target", "target", "goal_locations"):
        if hasattr(u, attr):
            g = np.asarray(getattr(u, attr), dtype=np.float32).reshape(-1)
            if g.size >= 2:
                return g[:2]
    if hasattr(u, "get_target"):
        return np.asarray(u.get_target(), dtype=np.float32).reshape(-1)[:2]
    raise RuntimeError("could not locate maze2d target")


def state_from_obs(obs):
    """maze2d obs is [x, y, vx, vy] = [qpos, qvel] — fully determines the point-mass state."""
    return (np.asarray(obs[:2], dtype=np.float64).copy(),
            np.asarray(obs[2:], dtype=np.float64).copy())


def restore_state(e, st):
    e.unwrapped.set_state(st[0].copy(), st[1].copy())


# ── Position-only value: BFS distance to goal on the maze grid ─────────────────
class BFSValue:
    """Grid BFS shortest-path distance to the goal cell (position-only V approximation)."""

    def __init__(self, e, goal_xy):
        u = e.unwrapped
        self.ok = False
        self.goal_xy = goal_xy
        arr = getattr(u, "maze_arr", None)
        if arr is None:
            print("[BFSValue] no maze_arr — falling back to Euclidean distance.")
            return
        arr = np.asarray(arr)
        # wall = 10 (d4rl WALL) or one of the common string wall markers
        if arr.dtype.kind in "iuf":
            self.wall = (arr == 10)
        else:
            self.wall = np.isin(arr, np.array(["#", "x", "1", b"#"], dtype=arr.dtype))
        self.shape = arr.shape
        self._xy_to_rc = getattr(u, "_xy_to_rowcol", None)   # canonical transform if present
        goal_rc = self.pos_to_cell(goal_xy)
        self.dist = self._bfs(goal_rc)
        self.ok = self.dist is not None and np.isfinite(self.dist[goal_rc])
        if not self.ok:
            print(f"[BFSValue] BFS failed (goal cell {goal_rc} unreachable/in wall) — "
                  "falling back to Euclidean.")

    def pos_to_cell(self, xy):
        if self._xy_to_rc is not None:
            rc = self._xy_to_rc(np.asarray(xy))        # canonical d4rl transform if present
            r, c = int(round(rc[0])), int(round(rc[1]))
        else:
            # d4rl maze2d convention: x is horizontal (col), y is vertical (row).
            # MUST be validated with --diagnose (S/G on free cells, distances rising from G).
            r, c = int(round(xy[1])), int(round(xy[0]))
        r = min(max(r, 0), self.shape[0] - 1)
        c = min(max(c, 0), self.shape[1] - 1)
        return (r, c)

    def _bfs(self, goal_rc):
        H, W = self.shape
        dist = np.full((H, W), np.inf)
        if self.wall[goal_rc]:
            return None
        dist[goal_rc] = 0.0
        q = deque([goal_rc])
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not self.wall[nr, nc] \
                        and dist[nr, nc] == np.inf:
                    dist[nr, nc] = dist[r, c] + 1.0
                    q.append((nr, nc))
        return dist

    def __call__(self, xy):
        """Distance-to-goal cost (lower is better). Euclidean fallback if BFS unavailable."""
        if not self.ok:
            return float(np.linalg.norm(np.asarray(xy) - self.goal_xy))
        r, c = self.pos_to_cell(xy)
        d = self.dist[r, c]
        if not np.isfinite(d):                       # in/behind a wall: penalise + Euclid
            return 1e3 + float(np.linalg.norm(np.asarray(xy) - self.goal_xy))
        # cell distance + sub-cell Euclidean refinement for a smooth gradient
        return float(d) + 0.25 * float(np.linalg.norm(np.asarray(xy) - self.goal_xy))


goal = get_goal(env)
sim.unwrapped.set_target(tuple(goal)) if hasattr(sim.unwrapped, "set_target") else None
value = BFSValue(env, goal)
print(f"Env {ENV_NAME}  goal={goal}  MAX_T={MAX_T}  "
      f"value={'BFS' if value.ok else 'Euclidean (BFS unavailable)'}")

# ── Diagnose mode: render maze + BFS field, validate transform, then exit ──────
if args.diagnose:
    env.seed(0); obs = env.reset()
    print(f"\nstart pos={obs[:2]}  start cell={value.pos_to_cell(obs[:2]) if value.ok else 'n/a'}")
    print(f"goal  pos={goal}  goal  cell={value.pos_to_cell(goal) if value.ok else 'n/a'}")
    if value.ok:
        H, W = value.shape
        print("\nmaze (#=wall, .=free, G=goal, S=start) + BFS distance:")
        sc = value.pos_to_cell(obs[:2]); gc = value.pos_to_cell(goal)
        for r in range(H):
            row = ""
            for c in range(W):
                if value.wall[r, c]:
                    row += "  ##"
                elif (r, c) == gc:
                    row += "   G"
                elif (r, c) == sc:
                    row += "   S"
                else:
                    d = value.dist[r, c]
                    row += f"{int(d):4d}" if np.isfinite(d) else "   ?"
            print(row)
        print("\nIf S and G land on free cells and the numbers increase away from G, "
              "the transform is correct. Otherwise adjust pos_to_cell before trusting results.")
    sys.exit(0)

# ── Discrete action grid (for greedy enumeration) ──────────────────────────────
_lin = np.linspace(-1.0, 1.0, args.grid)
ACTION_GRID = np.array([[a, b] for a in _lin for b in _lin], dtype=np.float32)  # (g*g, 2)
ACT_LOW, ACT_HIGH = -1.0, 1.0


# ── Episode runner (shared metrics + reward latch matching DV) ─────────────────
def run_episode(select_action, seed):
    env.seed(seed); env.action_space.seed(seed)
    np.random.seed(seed)                              # MPC shooting reproducibility (global RNG)
    if "torch" in sys.modules:                        # DV-exec diffusion-policy reproducibility
        sys.modules["torch"].manual_seed(seed)
    obs = env.reset()
    ep_reward, finished, t = 0.0, False, 0
    goal_step_val = None                              # step of first goal touch (tracked directly)
    dists = []
    t0 = time.perf_counter()
    while t < MAX_T:
        a = np.asarray(select_action(obs), dtype=np.float32).clip(ACT_LOW, ACT_HIGH)
        obs, rew, done, _ = env.step(a)
        dists.append(float(np.linalg.norm(obs[:2] - goal)))
        if not finished and rew == 1.0:
            finished, goal_step_val = True, t
        ep_reward += float(finished)
        t += 1
        if done:
            break
    return dict(
        reached=int(ep_reward > 0), raw_return=ep_reward,
        normalized_score=round(env.get_normalized_score(ep_reward) * 100, 2),
        goal_step=goal_step_val,
        min_dist=round(float(min(dists)), 3), episode_length=t,
        wall_s=round(time.perf_counter() - t0, 1),
    )


# ── Controllers ────────────────────────────────────────────────────────────────
def greedy_selector(obs):
    """Horizon-1 greedy w.r.t. the BFS value, using the true sim for the 1-step lookahead."""
    st = state_from_obs(obs)                          # current true state
    best_a, best_v = ACTION_GRID[0], np.inf
    for a in ACTION_GRID:
        restore_state(sim, st)
        nobs, _, _, _ = sim.step(a)
        v = value(nobs[:2])
        if v < best_v:
            best_v, best_a = v, a
    return best_a


def mpc_selector(horizon, samples, cem_iters, vel_pen=0.3, discount=0.99):
    """Receding-horizon shooting + CEM.

    Objective rewards ENDING close to the goal and SLOW, not merely *passing* near it:
        score = Σ γ^h·reward  −  value(terminal_pos)  −  vel_pen·‖terminal_vel‖
    Terminal (not min-over-rollout) distance penalises overshoot — a sequence that blows
    through the goal ends far away and scores poorly — which is what forces the controller
    to decelerate into turns and into the goal basin.  (An earlier min-over-rollout
    objective rewarded touch-and-fly-through and overshot the U-turn just like greedy.)"""
    def select(obs):
        st = state_from_obs(obs)
        mu = np.zeros((horizon, 2), dtype=np.float32)
        sigma = np.ones((horizon, 2), dtype=np.float32)
        best_first = None
        for _ in range(max(1, cem_iters)):
            seqs = np.clip(mu[None] + sigma[None] * np.random.randn(samples, horizon, 2),
                           ACT_LOW, ACT_HIGH).astype(np.float32)
            scores = np.empty(samples)
            for i in range(samples):
                restore_state(sim, st)
                R, nobs = 0.0, None
                for h in range(horizon):
                    nobs, rew, _, _ = sim.step(seqs[i, h])
                    R += (discount ** h) * float(rew)
                terminal = value(nobs[:2]) + vel_pen * float(np.linalg.norm(nobs[2:]))
                scores[i] = R * 1000.0 - terminal     # end close AND slow (no overshoot)
            elite = seqs[np.argsort(-scores)[:max(2, samples // 10)]]
            mu, sigma = elite.mean(0), elite.std(0) + 1e-3
            best_first = seqs[int(np.argmax(scores)), 0]
        return best_first
    return select


# Optional: greedy-bfs target executed through the DV inverse-dynamics policy
def make_dv_exec_selector():
    import torch
    from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset
    from cleandiffuser.diffusion import DiscreteDiffusionSDE
    from cleandiffuser.nn_condition import IdentityCondition
    from cleandiffuser.nn_diffusion import DVInvMlp
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    CKPT = ("results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer"
            f"_d2_width256_separate_dpTrue/{ENV_NAME}")
    ds = DV_D4RLMaze2DSeqDataset(env.get_dataset(), horizon=32, stride=15,
                                 learn_policy=False, center_mapping=False,
                                 discount=1.0, continous_reward_at_done=True, reward_tune="iql")
    normalizer = ds.get_normalizer()
    policy = DiscreteDiffusionSDE(
        DVInvMlp(4, 2, emb_dim=64, hidden_dim=256, timestep_emb_type="positional").to(DEVICE),
        IdentityCondition(dropout=0.0).to(DEVICE),
        x_max=+torch.ones((1, 2), device=DEVICE), x_min=-torch.ones((1, 2), device=DEVICE),
        diffusion_steps=10, device=DEVICE)
    policy.load(f"{CKPT}/policy_ckpt_1000000.pt"); policy.eval()

    def select(obs):
        st = state_from_obs(obs)
        best_next, best_v = obs.copy(), np.inf
        for a in ACTION_GRID:                          # greedy-BFS picks the target next-state
            restore_state(sim, st)
            nobs, _, _, _ = sim.step(a)
            v = value(nobs[:2])
            if v < best_v:
                best_v, best_next = v, nobs.copy()
        o = torch.tensor(normalizer.normalize(obs[None]), dtype=torch.float32, device=DEVICE)
        n = torch.tensor(normalizer.normalize(best_next[None]), dtype=torch.float32, device=DEVICE)
        n[:, :2] -= o[:, :2]; o[:, :2] = 0.0           # rebase, as in DV
        with torch.no_grad():
            act, _ = policy.sample(torch.zeros((1, 2), device=DEVICE), solver="ddpm",
                                   n_samples=1, sample_steps=10,
                                   condition_cfg=torch.cat([o, n], dim=-1),
                                   w_cfg=1.0, use_ema=True, temperature=0.5)
        return act.squeeze(0).cpu().numpy()
    return select


# ── Trace mode: sim==env check + one greedy episode trajectory, then exit ──────
if args.trace:
    print("\n[1] sim.step vs env.step from the SAME state (should be ~0):")
    env.seed(0); o = env.reset()
    rng = np.random.RandomState(0)
    max_div = 0.0
    for k in range(6):
        restore_state(sim, state_from_obs(o))          # sim ← env's current state
        a = rng.uniform(-1, 1, 2).astype(np.float32)
        os_, _, _, _ = sim.step(a)
        o, _, _, _ = env.step(a)
        div = float(np.abs(o - os_).max())
        max_div = max(max_div, div)
        print(f"    step {k}: max|env-sim|={div:.2e}  env_pos={o[:2].round(3)} "
              f"sim_pos={os_[:2].round(3)}")
    print(f"  -> max divergence {max_div:.2e}  "
          f"({'OK: sim matches env' if max_div < 1e-3 else 'BUG: sim != env (set_state/obs lossy)'})")

    print("\n[2] greedy-bfs trajectory (true env), looking for where it sticks:")
    env.seed(0); obs = env.reset()
    tr = []
    for t in range(MAX_T):
        a = np.asarray(greedy_selector(obs), dtype=np.float32).clip(-1, 1)
        nobs, rew, done, _ = env.step(a)
        tr.append((t, float(obs[0]), float(obs[1]), float(np.linalg.norm(obs[2:])),
                   float(np.linalg.norm(obs[:2] - goal)), float(value(obs[:2])),
                   float(a[0]), float(a[1]), float(rew)))
        obs = nobs
        if done:
            break
    arr = np.array([(r[4], r[3]) for r in tr])             # (dist_goal, speed) per step
    imin = int(arr[:, 0].argmin())
    print(f"  episode_len={len(tr)}  min_dist={arr[imin,0]:.3f} @ t={imin}  "
          f"pos@min=({tr[imin][1]:.2f},{tr[imin][2]:.2f})  speed@min={arr[imin,1]:.2f}")
    print(f"  mean speed={arr[:,1].mean():.2f}  max speed={arr[:,1].max():.2f}  "
          f"final pos=({tr[-1][1]:.2f},{tr[-1][2]:.2f})  final dist={arr[-1,0]:.2f}")
    print("  dist_goal trajectory (every 15 steps):")
    print("   ", " ".join(f"{arr[i,0]:.2f}" for i in range(0, len(tr), 15)))
    print("  speed trajectory     (every 15 steps):")
    print("   ", " ".join(f"{arr[i,1]:.2f}" for i in range(0, len(tr), 15)))
    tpath = f"{OUT_DIR}/stage0_trace_{TAG}.csv"
    with open(tpath, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "x", "y", "speed", "dist_goal", "value", "ax", "ay", "rew"])
        w.writerows(tr)
    print(f"\n  full per-step trace → {tpath}")
    print("\nRead: if [1] diverges → sim≠env (planning model wrong). If [1] is OK and the "
          "agent reaches low dist then speed stays high while dist plateaus → momentum "
          "overshoot (control problem). If dist plateaus with LOW speed → it's stuck/converged "
          "at a value local-min (value-field problem).")
    sys.exit(0)


# ── Run all controllers over all seeds ─────────────────────────────────────────
controllers = [("greedy-bfs", greedy_selector)]
for H in args.mpc_horizons:
    if H == 1:
        continue                                       # h=1 ≈ greedy-bfs, already included
    controllers.append((f"mpc-bfs-h{H}",
                        mpc_selector(H, args.mpc_samples, args.mpc_cem_iters,
                                     vel_pen=args.mpc_vel_pen)))
if args.include_dv_exec:
    controllers.append(("greedy-bfs-dvexec", make_dv_exec_selector()))

print(f"\nControllers: {[c for c, _ in controllers]}   seeds={args.seeds}\n")
rows = []
for name, sel in controllers:
    for seed in args.seeds:
        r = run_episode(sel, seed)
        r.update(controller=name, seed=seed, env=TAG)
        rows.append(r)
        print(f"{name:<20} seed={seed}  score={r['normalized_score']:>7.1f}  "
              f"reached={r['reached']}  min_dist={r['min_dist']:.2f}  "
              f"goal_step={r['goal_step']}  ({r['wall_s']}s)")

# ── Summary + DV baseline comparison ───────────────────────────────────────────
grp = defaultdict(list)
for r in rows:
    grp[r["controller"]].append(r)

# largest MPC horizon = the momentum-aware near-optimal ceiling
ceiling_name = (f"mpc-bfs-h{max(h for h in args.mpc_horizons if h > 1)}"
                if any(h > 1 for h in args.mpc_horizons) else None)

print("\n" + "=" * 82)
print(f"Stage 0 oracle decomposition — {ENV_NAME}")
print("=" * 82)
print(f"{'controller':<22} {'n':>2} {'norm_score':>11} {'reached':>8} {'min_dist':>9}  note")
print("-" * 82)
# DV references from saved baselines
dv_path = ("results/phase0_baseline.json" if TAG == "umaze"
           else f"results/phase0_baseline_{TAG}.json")
if os.path.exists(dv_path):
    dv = json.load(open(dv_path))
    seen = {}; [seen.__setitem__(r["seed"], r) for r in dv]
    dvs = [r["normalized_score"] for r in seen.values()]
    print(f"{'DV-MCSS greedy (ref)':<22} {len(dvs):>2} {np.mean(dvs):>11.1f} "
          f"{'—':>8} {'—':>9}  baseline to beat")
for name, _ in controllers:
    rs = grp[name]
    sc = [r["normalized_score"] for r in rs]
    reached_str = f"{sum(r['reached'] for r in rs)}/{len(rs)}"
    note = ("← momentum-aware ceiling" if name == ceiling_name
            else "no-search, position value" if name == "greedy-bfs"
            else "execution via DV inv-dyn" if name.endswith("dvexec") else "")
    print(f"{name:<22} {len(rs):>2} {np.mean(sc):>11.1f} "
          f"{reached_str:>8} {np.mean([r['min_dist'] for r in rs]):>9.2f}  {note}")
print("=" * 82)
print("Read: greedy-bfs vs mpc-bfs-h{>1} = does search help given a position value;\n"
      "  largest-H mpc = momentum-aware ceiling; ceiling vs DV-greedy = total gap;\n"
      "  greedy-bfs(direct) vs greedy-bfs-dvexec = value-error vs execution-error.")

csv_path = f"{OUT_DIR}/stage0_oracle_{TAG}.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\n→ {csv_path}")
