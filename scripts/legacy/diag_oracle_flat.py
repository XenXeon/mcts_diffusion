"""scripts/diag_oracle_flat.py — flat, 1-ply selection by FIRST-STEP progress.

The whole loop takes ONE step then replans (MPC), so what matters is the first
waypoint actually executed — yet MCSS (and the oracle tree) rank by the plan's
ENDPOINT (≈40 waypoints out). A plan with a goalward endpoint but a sideways first
step gets picked, and the agent dawdles; with a ~2x-of-optimal time budget that
turns into a timeout. This tests the alternative the user proposed: among plans whose
ENDPOINT lands near the goal (MCSS-like), pick the one whose FIRST waypoint makes the
most geodesic progress toward the goal — directly optimizing the executed step.

  --rank endpoint  : argmin endpoint geodesic        (= the flat oracle re-rank; a control)
  --rank firststep : keep endpoints within --keep-band of the best, then argmin the
                     FIRST-waypoint geodesic among them   (raw idea — found to HURT, because
                     "closest first waypoint" = the most aggressive/infeasible jump)
  --rank feasible  : same pool, but first DROP first waypoints claiming more than
                     --max-progress-cells of geodesic PROGRESS (g_cur - g_fw) from one stride
                     (a hallucinated segment — more progress than one stride delivers), then
                     argmin the rest. NB this caps claimed PROGRESS, not Euclidean displacement.
  --rank stable    : among goal-reaching candidates that still progress (first waypoint closer
                     than now), pick the one the PLANNER predicts is most STABLE — the Ant's
                     ~20% failures are TOPPLES, so this prefers a plan that stays upright over
                     one that lunges and tips. --stability-by {upright,displacement,angvel}.
                     The stability signal is the planner's own prediction (DEPLOYABLE, not the
                     oracle); only the goal-reaching filter uses the geodesic.

A first waypoint is one stride (25 dense steps) ahead; with ~35 dense steps/cell that is
~0.7 cell of feasible progress, so a first waypoint claiming >~1 cell closer than now is a
planner hallucination — `feasible` excludes exactly those before ranking.

Flat: 1 planner call/step (k candidates), ~17x cheaper than b16. Uses the true geodesic
(Rule-1 dev-only). Emits a collate_mcts-compatible JSON (goal-verified + McNemar). With
--log it also dumps a per-failed-episode npz that scripts/animate_failure.py replays.

Run, then collate vs the existing k50 baseline:
    python scripts/diag_oracle_flat.py --env antmaze-large-diverse-v2 \
        --seeds 0 1 2 --n-envs 50 --k 50 --rank feasible --max-progress-cells 1
    python scripts/collate_mcts.py results/scale_mcss_k50_s*.json \
        results/scale_mcss_k50orc_s*.json results/scale_mcss_k50fs*_s*.json

⚠ Rule-1: the geodesic value is privileged — DIAGNOSTIC-ONLY, never reportable.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import torch

from mcts.maze_oracle import AntMazeOracle, make_oracle
from mcts.mcts_loop import Sampler, load_models
from mcts.specs import env_family, get_goal


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _geo(oracle, grid, xy):
    r, c = oracle.cell(xy)
    return float(grid[r][c])                        # cells; inf if in-wall / unreachable


def select_flat(g_end, g_fw, g_cur, rank, keep_band, max_progress):
    """Index of the chosen candidate.

    endpoint : argmin endpoint geodesic.
    firststep: among endpoints within keep_band of the best, argmin first-waypoint geodesic.
    feasible : same pool, but first DROP first waypoints whose claimed progress
               (g_cur - g_fw) exceeds max_progress cells (an infeasible one-stride jump),
               then argmin among the rest — the most progress that is physically reachable.
    """
    safe_end = np.where(np.isfinite(g_end), g_end, np.inf)
    if rank == "endpoint" or not np.isfinite(safe_end.min()):
        return int(safe_end.argmin())
    best = safe_end.min()
    keep = np.isfinite(g_end) & (g_end <= best + keep_band)
    pool = keep & np.isfinite(g_fw)
    if rank == "feasible":
        feasible = pool & ((g_cur - g_fw) <= max_progress)   # drop impossible PROGRESS claims
        pool = feasible if feasible.any() else pool          # fallback if none feasible
    safe_fw = np.where(pool, g_fw, np.inf)
    if not np.isfinite(safe_fw).any():
        return int(safe_end.argmin())                        # fallback: nothing kept
    return int(safe_fw.argmin())


def select_stable(g_end, g_fw, g_cur, keep_band, min_upright, angvel, disp, stability_by):
    """Among GOAL-REACHING candidates that make goalward progress, pick the MOST STABLE one
    — the Ant's ~20% failures are TOPPLES, so prefer a plan predicted to stay upright over a
    plan that lunges and tips. Goal-reaching = endpoint within keep_band of the best;
    goalward = first waypoint closer than now (so it still progresses). stability_by:
       upright      -> max worst-predicted-uprightness over the next few waypoints
       displacement -> min first-step xy move (gentlest, least lunge)
       angvel       -> min predicted angular speed (smoothest)
    The stability features come from the planner's own predicted trajectory (DEPLOYABLE)."""
    safe_end = np.where(np.isfinite(g_end), g_end, np.inf)
    best = safe_end.min()
    if not np.isfinite(best):
        return int(safe_end.argmin())
    goalreach = np.isfinite(g_end) & (g_end <= best + keep_band)
    goalward = goalreach & np.isfinite(g_fw) & (g_fw <= g_cur)  # first step doesn't RETREAT
    pool = goalward if goalward.any() else goalreach
    idxs = np.where(pool)[0]
    if len(idxs) == 0:
        return int(safe_end.argmin())
    key = (-min_upright if stability_by == "upright"
           else angvel if stability_by == "angvel" else disp)   # all minimised
    return int(idxs[int(np.argmin(key[idxs]))])

def select_gentle(g_end, g_fw, g_cur, keep_band, disp, lunge_frac):
    """Progress toward the goal WITHOUT the lunge that tips the Ant. Among goal-reaching,
    goalward candidates, EXCLUDE the biggest-displacement `lunge_frac` (the lunges that the
    data shows raise the fall rate), then pick MAX goal-progress (min g_fw) among the gentle
    rest. Adaptive per-step displacement percentile, so no absolute scale is needed. This is
    the synthesis the logs point to: firststep (max progress) tips; min-displacement creeps;
    'gentle' takes the most-goalward step inside a non-lunging displacement envelope."""
    safe_end = np.where(np.isfinite(g_end), g_end, np.inf)
    best = safe_end.min()
    if not np.isfinite(best):
        return int(safe_end.argmin())
    goalreach = np.isfinite(g_end) & (g_end <= best + keep_band)
    goalward = goalreach & np.isfinite(g_fw) & (g_fw <= g_cur)
    pool = goalward if goalward.any() else goalreach
    idxs = np.where(pool)[0]
    if len(idxs) == 0:
        return int(safe_end.argmin())
    d = disp[idxs]
    thresh = np.quantile(d, max(0.0, 1.0 - lunge_frac)) if lunge_frac < 1.0 else np.inf
    gentle = idxs[d <= thresh]
    if len(gentle) == 0:
        gentle = idxs
    return int(gentle[int(np.argmin(g_fw[gentle]))])   # most progress among the gentle

def select_smooth(g_end, g_fw, g_cur, keep_band, fw_xy, cur_xy, cur_vel, turn_cap):
    """Avoid the sharp commanded turn that PRECEDES topples. diag_fall_geometry shows the
    sharpest steer entering a stall is ~130-170deg (a near-reversal) vs a ~50deg moving
    baseline, while wall clearance at the topple is normal — so the residual falls track a
    momentum-breaking U-turn, not a wall. Among goal-reaching, goalward candidates, compute
    the commanded turn = angle(current MEASURED velocity, candidate_first_waypoint - cur_xy);
    DROP candidates whose turn exceeds turn_cap (deg), then take MAX goal-progress among the
    smooth rest. Uses the REAL observed linear velocity (obs[15:17]), not the planner's
    (empty) predicted-orientation channel — which is why it can bite where stbU could not.
    Below a small speed the heading is undefined, so it falls back to plain goal-progress."""
    safe_end = np.where(np.isfinite(g_end), g_end, np.inf)
    best = safe_end.min()
    if not np.isfinite(best):
        return int(safe_end.argmin())
    goalreach = np.isfinite(g_end) & (g_end <= best + keep_band)
    goalward = goalreach & np.isfinite(g_fw) & (g_fw <= g_cur)
    pool = goalward if goalward.any() else goalreach
    idxs = np.where(pool)[0]
    if len(idxs) == 0:
        return int(safe_end.argmin())
    speed = float(np.linalg.norm(cur_vel[:2]))
    if speed < 0.05:                                  # heading undefined -> plain max progress
        return int(idxs[int(np.argmin(g_fw[idxs]))])
    h = cur_vel[:2] / speed
    d = fw_xy[idxs] - cur_xy[None, :]                 # commanded directions (n,2)
    nd = np.linalg.norm(d, axis=1)
    cosang = np.where(nd > 1e-6, (d @ h) / np.maximum(nd, 1e-9), 1.0)
    turn = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    smooth = idxs[turn <= turn_cap]
    if len(smooth) == 0:                              # nothing under the cap: keep gentlest few
        smooth = idxs[np.argsort(turn)[:max(1, len(idxs) // 4)]]
    return int(smooth[int(np.argmin(g_fw[smooth]))])  # most progress among the smooth

def run_seed(sampler, oracle, seed, n_envs, k, rank, keep_band, max_progress, max_t,
             env_name, stability_by="upright", lunge_frac=0.5, turn_cap=90.0, log=False):
    import gym
    from pipelines.utils import set_seed
    set_seed(seed)
    torch.manual_seed(seed)
    env = gym.vector.make(env_name, n_envs, asynchronous=False)
    try:
        env.seed(seed)
    except Exception:
        pass
    for i, e in enumerate(getattr(env, "envs", None) or []):
        try:
            e.seed(seed + i); e.action_space.seed(seed + i)
        except Exception:
            pass
    normalizer, obs_dim = sampler.m["normalizer"], sampler.m["obs_dim"]
    obs = env.reset()
    goals_raw = np.asarray([get_goal(e) for e in env.envs], dtype=np.float64)
    starts = np.asarray(obs)[:, :2].astype(np.float64).copy()
    grids = [oracle.dist_grid_from(goals_raw[i]) for i in range(n_envs)]
    success = np.zeros(n_envs, dtype=bool)
    active = np.ones(n_envs, dtype=bool)
    arange = np.arange(n_envs)
    fam = env_family(env_name)
    # maze2d: reach is saturated -> the discriminative metric is the DV CAMPING score (reward
    # accrues for every step spent at the goal, so reaching FASTER scores higher). Accumulate
    # it exactly like run_episodes/veteran_d4rl_maze2d (finished |= rew==1; dv_acc += finished).
    dv_finished = np.zeros(n_envs, dtype=bool)
    dv_acc = np.zeros(n_envs, dtype=np.float64)
    # per-env per-step buffers for the animator (filled only with --log, while active)
    buf = ([dict(xy=[], dist=[], cand_xy=[], cand_dist=[], chosen_idx=[], chosen_fw=[])
            for _ in range(n_envs)] if log else None)
    # discriminativeness check: mean per-step std of each stability feature ACROSS the 50
    # candidates. If ~0 the planner predicts every plan alike -> the signal is real-but-empty
    # (a null then means 'policy fix', not 'selection cannot help') — must read it to interpret.
    spread = dict(upright=0.0, disp=0.0, angvel=0.0, n=0)
    t0 = time.perf_counter()
    for t in range(max_t):
        s_norm = normalizer.normalize(obs).astype(np.float32)
        prop = sampler.mcss_propose(s_norm)                       # k flat candidates
        if rank == "stable" and active.any():
            am = active
            spread["upright"] += float(np.std(prop["min_upright"][am], axis=1).sum())
            spread["disp"]    += float(np.std(prop["disp"][am], axis=1).sum())
            spread["angvel"]  += float(np.std(prop["angvel"][am], axis=1).sum())
            spread["n"]       += int(am.sum())
        ep = normalizer.unnormalize(
            prop["endpoints"].reshape(n_envs * k, obs_dim))[:, :2].reshape(n_envs, k, 2)
        fw = normalizer.unnormalize(
            prop["first_wps"].reshape(n_envs * k, obs_dim))[:, :2].reshape(n_envs, k, 2)
        xy_now = np.asarray(obs)[:, :2]
        vel_now = (np.asarray(obs)[:, 15:18] if np.asarray(obs).shape[1] >= 18
                   else np.zeros((n_envs, 3)))       # antmaze vel; maze2d obs=4-dim -> smooth n/a
        chosen = np.empty(n_envs, dtype=np.int64)
        for i in range(n_envs):
            g_end = np.array([_geo(oracle, grids[i], ep[i, j]) for j in range(k)])
            g_fw = np.array([_geo(oracle, grids[i], fw[i, j]) for j in range(k)])
            g_cur = _geo(oracle, grids[i], xy_now[i])
            if rank == "stable":
                chosen[i] = select_stable(g_end, g_fw, g_cur, keep_band,
                                          prop["min_upright"][i], prop["angvel"][i],
                                          prop["disp"][i], stability_by)
            
            elif rank == "gentle":
                chosen[i] = select_gentle(g_end, g_fw, g_cur, keep_band,
                                          prop["disp"][i], lunge_frac)

            elif rank == "smooth":
                chosen[i] = select_smooth(g_end, g_fw, g_cur, keep_band,
                                          fw[i], xy_now[i], vel_now[i], turn_cap)

            else:
                chosen[i] = select_flat(g_end, g_fw, g_cur, rank, keep_band, max_progress)
            if log and active[i]:
                b = buf[i]
                b["xy"].append(xy_now[i].astype(np.float32))
                b["dist"].append(g_cur)
                b["cand_xy"].append(ep[i].astype(np.float32))     # endpoint cloud
                b["cand_dist"].append(g_end.astype(np.float32))
                b["chosen_idx"].append(int(chosen[i]))
                b["chosen_fw"].append(fw[i, chosen[i]].astype(np.float32))  # executed-toward waypoint (the cause)
        wp = prop["first_wps"][arange, chosen]                    # (M, D) normalised
        act = sampler.policy_action(s_norm, wp)
        obs, rew, done, info = env.step(act)
        rew_arr = np.asarray(rew, dtype=np.float64)
        success[active & (rew_arr > 0.0)] = True
        if fam == "maze2d":
            dv_finished |= (rew_arr == 1.0)
            dv_acc += dv_finished
        active &= ~np.asarray(done, dtype=bool)
        if not active.any():
            break
        if (t + 1) % 100 == 0:
            print(f"  [{rank} s{seed}] t={t+1}/{max_t} reached={int(success.sum())}/"
                  f"{n_envs} active={int(active.sum())} {time.perf_counter()-t0:.0f}s")
    env.close()
    # Per-rollout DV-exact score: maze2d camping (dv_acc), antmaze reach indicator.
    env_single = sampler.m["env_single"]
    dv_raw = dv_acc if fam == "maze2d" else np.clip(success.astype(np.float64), 0.0, 1.0)
    dv_norm = np.array([env_single.get_normalized_score(float(x)) for x in dv_raw]) * 100.0
    return success, goals_raw, starts, round(time.perf_counter() - t0, 1), buf, spread, dv_norm


def save_log(out_dir, logtag, seed, env_name, oracle, succ, goals, starts, buf):
    """Per-failed-episode npz (animator-compatible: e{i}_xy/dist/cand_xy/cand_dist/
    chosen_idx) + an index carrying the maze geometry and scenario outcomes."""
    arrays = {}
    failed = [i for i in range(len(succ)) if not succ[i]]
    for i in failed:
        b = buf[i]
        if not b["xy"]:
            continue
        arrays[f"e{i}_xy"] = np.stack(b["xy"])
        arrays[f"e{i}_dist"] = np.asarray(b["dist"], dtype=np.float32)
        arrays[f"e{i}_cand_xy"] = np.stack(b["cand_xy"])
        arrays[f"e{i}_cand_dist"] = np.stack(b["cand_dist"])
        arrays[f"e{i}_chosen_idx"] = np.asarray(b["chosen_idx"], dtype=np.int32)
        arrays[f"e{i}_chosen_fw"] = np.stack(b["chosen_fw"])     # chosen first-waypoint xy (the cause)
    npz = os.path.join(out_dir, f"{logtag}_s{seed}.npz")
    np.savez_compressed(npz, **arrays)
    index = dict(DIAGNOSTIC_ONLY=True, env=env_name, seed=int(seed),
                 npz=os.path.basename(npz),
                 maze=dict(wall=[[1 if w else 0 for w in row] for row in oracle.wall],
                           scaling=oracle.scaling, init_x=oracle.init_x,
                           init_y=oracle.init_y, n_rows=oracle.n_rows, n_cols=oracle.n_cols),
                 scenarios=[dict(env_idx=i, seed=int(seed),
                                 goal=[float(x) for x in goals[i]],
                                 start=[float(x) for x in starts[i]],
                                 success=bool(succ[i])) for i in range(len(succ))])
    with open(os.path.join(out_dir, f"{logtag}_s{seed}_index.json"), "w") as f:
        json.dump(index, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="antmaze-large-diverse-v2")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-envs", type=int, default=50)
    p.add_argument("--k", type=int, default=50, help="flat candidates per step (MCSS-like)")
    p.add_argument("--rank",
                   choices=["endpoint", "firststep", "feasible", "stable", "gentle", "smooth"],
                   default="gentle")
    p.add_argument("--stability-by", choices=["upright", "displacement", "angvel"],
                   default="upright",
                   help="stable: which planner-predicted stability signal to maximise — "
                        "upright (worst predicted uprightness), displacement (gentlest first "
                        "step), or angvel (smoothest). The Ant's failures are topples.")
    p.add_argument("--stability-window", type=int, default=3,
                   help="stable: how many near-term waypoints to assess for stability")
    p.add_argument("--lunge-frac", type=float, default=0.5,
                   help="gentle: exclude the biggest-displacement this-fraction of goalward "
                        "candidates (the lunges) before taking the most-goalward step")
    p.add_argument("--turn-cap", type=float, default=90.0,
                   help="smooth: drop candidates whose commanded turn (angle between the "
                        "measured velocity and the first-waypoint direction) exceeds this many "
                        "degrees, then take the most-goalward of the rest (diag_fall_geometry "
                        "shows ~130-170deg near-reversals precede topples)")
    p.add_argument("--keep-band", type=float, default=2.0,
                   help="keep endpoints within this many cells of the best (the goal-reaching pool)")
    p.add_argument("--max-progress-cells", type=float, default=1.0,
                   help="feasible: cap on the geodesic PROGRESS (g_cur - g_fw) a first waypoint "
                        "may claim from ONE stride (~stride/steps_per_cell ≈ 0.7-1 cell); a larger "
                        "claimed gain is a hallucinated segment, dropped before ranking. NOTE: "
                        "caps claimed progress, not raw Euclidean displacement.")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--log", action="store_true",
                   help="also dump per-failed-episode traces (flatlog_*.npz) for "
                        "scripts/animate_failure.py")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--log-dir", default="results/instr")
    p.add_argument("--critic-step", type=int, default=1000000)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if env_family(args.env) not in ("antmaze", "maze2d"):
        sys.exit("oracle-flat targets antmaze/maze2d (the geodesic-oracle families)")
    models = load_models(args.env, critic_step=args.critic_step, device=args.device)
    sampler = Sampler(models, k_mcss=args.k, value_mode="v_s",   # only uses mcss_propose/policy
                      stability_window=args.stability_window)
    oracle = make_oracle(models["env_single"], env_family(args.env))
    max_t = args.max_steps or models["max_path_length"]
    vmode = {"endpoint": "oracle", "firststep": "oracle_fs", "feasible": "oracle_fsf",
             "stable": "oracle_stb", "gentle": "oracle_gnt", "smooth": "oracle_smt"}[args.rank]
    if args.rank == "endpoint":
        tagk = "orc"
    elif args.rank == "firststep":
        tagk = f"fs{int(args.keep_band)}"
    elif args.rank == "feasible":
        tagk = f"fsf{int(args.keep_band)}m{int(args.max_progress_cells)}"
    elif args.rank == "stable":
        tagk = f"stb{args.stability_by[0].upper()}{int(args.keep_band)}"
    elif args.rank == "smooth":
        tagk = f"smt{int(args.turn_cap)}"
    else:  # gentle
        tagk = f"gnt{int(args.lunge_frac*100)}"
    print(f"oracle-flat: rank={args.rank} k={args.k} keep_band={args.keep_band} "
          f"max_progress_cells={args.max_progress_cells} stability_by={args.stability_by} "
          f"(1 planner call/step)")
    if args.log:
        os.makedirs(args.log_dir, exist_ok=True)

    os.makedirs(args.out_dir, exist_ok=True)
    for seed in args.seeds:
        succ, goals, starts, wall, buf, spread, dv_norm = run_seed(
            sampler, oracle, seed, args.n_envs, args.k, args.rank, args.keep_band,
            args.max_progress_cells, max_t, args.env,
            stability_by=args.stability_by, lunge_frac=args.lunge_frac,
            turn_cap=args.turn_cap, log=args.log)
        if args.rank == "stable" and spread["n"]:
            n = spread["n"]
            print(f"        stability spread across {args.k} candidates (mean per-step std): "
                  f"upright={spread['upright']/n:.3f} disp={spread['disp']/n:.3f} "
                  f"angvel={spread['angvel']/n:.3f}  (~0 => planner can't distinguish plans)")
        pf = float(succ.mean())
        payload = dict(
            env=args.env, seed=int(seed), n_envs=args.n_envs, n_episodes=1,
            max_steps=max_t, k_mcss=args.k, value_mode=vmode, gate="none",
            keep_band=args.keep_band, max_progress_cells=args.max_progress_cells,
            stability_by=args.stability_by, stability_window=args.stability_window,
            lunge_frac=args.lunge_frac, turn_cap=args.turn_cap, DIAGNOSTIC_ONLY=True,
            rule1_note="flat selection by the TRUE geodesic (privileged) — ceiling probe, "
                       "NOT reportable",
            git_commit=_git_commit(), timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            results={"mcss": dict(
                method="mcss", n_rollouts=args.n_envs, reach_pct=100.0 * pf,
                reach_err=math.sqrt(max(pf * (1 - pf), 0.0) / args.n_envs) * 100.0,
                # DV-exact score (maze2d camping / antmaze reach) — collate pairs maze2d on this
                dv_norm_mean=float(np.mean(dv_norm)), dv_norm_err=float(np.std(dv_norm) / math.sqrt(args.n_envs)),
                dv_norm=[float(x) for x in dv_norm],
                success=[int(x) for x in succ],
                goals=[[float(g[0]), float(g[1])] for g in goals],
                starts=[[float(s[0]), float(s[1])] for s in starts], wall_s=wall)})
        fname = os.path.join(args.out_dir, f"scale_mcss_k{args.k}{tagk}_s{seed}.json")
        with open(fname, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  [k{args.k}{tagk} s{seed}] reach={100*pf:.1f}%  {wall:.0f}s "
              f"-> {os.path.basename(fname)}")
        if args.log:
            save_log(args.log_dir, f"flatlog_k{args.k}{tagk}", seed, args.env, oracle,
                     succ, goals, starts, buf)
            print(f"        traces -> {args.log_dir}/flatlog_k{args.k}{tagk}_s{seed}.npz "
                  f"(animate: scripts/animate_failure.py --tag flatlog_k{args.k}{tagk})")

    print("\n" + "=" * 72)
    print("FLAT oracle selection saved. Collate vs the DV-critic k50 baseline (goal-")
    print("verified + McNemar); the k50 -> k50-fsf rung isolates the SELECTION RULE at")
    print("matched compute (endpoint DV-critic vs feasible first-step oracle, both flat 50):")
    print("  python scripts/collate_mcts.py results/scale_mcss_k50_s*.json \\")
    print("      results/scale_mcss_k50orc_s*.json results/scale_mcss_k50fs*_s*.json")
    print("  Read the pooled exact-p. CAVEAT (Rule-1): the geodesic is privileged — a")
    print("  positive result says the MECHANISM helps; the deployable version evaluates")
    print("  V(s,g) at the first waypoint among feasible near-best endpoints.")
    print("=" * 72)


if __name__ == "__main__":
    main()
