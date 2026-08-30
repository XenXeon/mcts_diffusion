"""scripts/diag_oracle_tree.py — does STRUCTURED search with a PERFECT value help?

The flat oracle re-rank (analyze_failures Tier-2) showed flat best-of-N selection is
saturated: a perfect geodesic ranker, picking the closest-endpoint candidate each step,
nets ~0 vs the DV critic. But that tests SELECTION, not SEARCH. This runs the SAME MCTS
forest used by the sampler, but scores tree children by the TRUE BFS geodesic (Rule-1
dev-only) instead of the learned value — so the tree can look ahead over intermediate
states and back up, preferring a first step that LEADS to a reliably-reachable region
rather than greedily chasing the nearest endpoint.

This saves a collate_mcts-compatible JSON (per-rollout success/goals/starts) so the
comparison is done by the battle-tested scripts/collate_mcts.py — which verifies
per-index GOAL identity before any test and reports per-seed + pooled exact McNemar p.
We do NOT re-implement pairing here (that ad-hoc path mixed up baselines and skipped
the significance test). The intended use is the ATTRIBUTION LADDER, each rung isolating
one factor, matched-compute among the trees:
    k50 -> b16      structured tree vs the flat baseline
    b16 -> b16sgP   goal-conditioning the tree value
    b16sgP -> b16orc VALUE ACCURACY (learned V(s,g) -> the perfect geodesic) — the
                     decisive isolation of "does an accurate value inside the tree help"

  net > 0 (pooled, significant) -> structured search + an accurate value IS a lever;
      build V(s,g) toward geodesic accuracy.
  net ~ 0 -> endpoint-geodesic look-ahead cannot beat ~78% (see the F4 caveats: this
      does NOT rule out a segment-feasibility-aware value).

⚠ Rule-1: the geodesic value is privileged — DIAGNOSTIC-ONLY, never reportable.

Run, then collate:
    python scripts/diag_oracle_tree.py --env antmaze-large-diverse-v2 \
        --seeds 0 1 2 --n-envs 50 --budget 16 --k-mcts 16
    python scripts/collate_mcts.py results/scale_mcss_k50_s*.json \
        results/scale_mcts_b16_s*.json results/scale_mcts_b16sgP_s*.json \
        results/scale_mcts_b16orc*_s*.json
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

from mcts.maze_oracle import AntMazeOracle, calibrate_steps_per_cell
from mcts.mcts_loop import Sampler, load_models
from mcts.relabel import build_relabel_inputs
from mcts.specs import env_family, get_goal, make_dataset


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def run_seed(sampler, oracle, scale, spc, seed, n_envs, max_t, env_name):
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
    normalizer = sampler.m["normalizer"]
    obs = env.reset()
    goals_raw = np.asarray([get_goal(e) for e in env.envs], dtype=np.float64)
    starts = np.asarray(obs)[:, :2].astype(np.float64).copy()
    goal_grids = [oracle.dist_grid_from(goals_raw[i]) for i in range(n_envs)]   # env order
    sampler.set_oracle_ctx(dict(normalizer=normalizer, oracle=oracle,
                                goal_grids=goal_grids, scale=scale, spc=spc))
    success = np.zeros(n_envs, dtype=bool)
    active = np.ones(n_envs, dtype=bool)
    t0 = time.perf_counter()
    for t in range(max_t):
        s_norm = normalizer.normalize(obs).astype(np.float32)
        wp = sampler.mcts_waypoints(s_norm)                 # oracle value (goal grids in ctx)
        act = sampler.policy_action(s_norm, wp)
        obs, rew, done, info = env.step(act)
        rew = np.asarray(rew, dtype=np.float64)
        success[active & (rew > 0.0)] = True
        active &= ~np.asarray(done, dtype=bool)
        if not active.any():
            break
        if (t + 1) % 100 == 0:
            print(f"  [oracletree s{seed}] t={t+1}/{max_t} reached={int(success.sum())}/"
                  f"{n_envs} active={int(active.sum())} {time.perf_counter()-t0:.0f}s")
    env.close()
    sampler.set_oracle_ctx(None)
    return success, goals_raw, starts, round(time.perf_counter() - t0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="antmaze-large-diverse-v2")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-envs", type=int, default=50)
    p.add_argument("--budget", type=int, default=16)
    p.add_argument("--k-mcts", type=int, default=16)
    p.add_argument("--child-index", type=int, default=1,
                   help="segment length L (how far each child is from its parent). "
                        "MATCHED-COMPUTE tree knob — sweep this first (L in {1,2,4}); "
                        "with the perfect value, L>1 may finally help (it hurt with V(s)).")
    p.add_argument("--c-ucb", type=float, default=1.4142136,
                   help="UCB exploration constant. With an accurate value, LOWER = exploit "
                        "the value / search deeper; the default is sqrt(2).")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--critic-step", type=int, default=1000000)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if env_family(args.env) != "antmaze":
        sys.exit("oracle-tree targets antmaze (the maze with failure headroom)")
    models = load_models(args.env, critic_step=args.critic_step, device=args.device)
    sampler = Sampler(models, k_mcts=args.k_mcts, budget=args.budget,
                      child_index=args.child_index, c_ucb=args.c_ucb,
                      value_mode="oracle")
    oracle = AntMazeOracle(models["env_single"])
    # value scale + steps/cell (so the geodesic maps onto the learned-value scale)
    _, ds = make_dataset(args.env)
    seq_obs, ends, term_only, scale = build_relabel_inputs(ds)
    path_xys = [models["normalizer"].unnormalize(seq_obs[i, :ends[i] + 1])[:, :2]
                for i in range(len(ends))]
    spc = calibrate_steps_per_cell(oracle, path_xys, ends)
    max_t = args.max_steps or models["max_path_length"]
    print(f"oracle-tree: budget={args.budget} k={args.k_mcts} L={args.child_index} "
          f"steps/cell={spc:.1f} D={scale.D}")

    os.makedirs(args.out_dir, exist_ok=True)
    cidx = args.child_index
    ltag = f"L{cidx}" if cidx != 1 else ""
    # c_ucb in the filename (not in the collate LABEL) so a c-sweep doesn't overwrite;
    # for a c_ucb sweep compare via separate collate runs (label is b{budget}{L}orc).
    ctag = "" if abs(args.c_ucb - 1.4142136) < 1e-6 else "c" + f"{args.c_ucb:.2f}".replace(".", "p")
    for seed in args.seeds:
        succ, goals, starts, wall = run_seed(sampler, oracle, scale, spc, seed,
                                             args.n_envs, max_t, args.env)
        p_frac = float(succ.mean())
        # collate_mcts-compatible schema: per-rollout success/goals/starts so
        # collate does the goal-verified, McNemar-tested PAIRED comparison (F2/F3) —
        # we do NOT re-implement pairing here (that was the F1/F2/F3-flawed path).
        payload = dict(
            env=args.env, seed=int(seed), n_envs=args.n_envs, n_episodes=1,
            max_steps=max_t, budget=args.budget, k_mcts=args.k_mcts, k_mcss=0,
            child_index=cidx, value_mode="oracle", gate="none",
            DIAGNOSTIC_ONLY=True,
            rule1_note="tree children scored by the TRUE geodesic (privileged) — a "
                       "ceiling probe; the 'orc' number is NOT reportable",
            git_commit=_git_commit(), timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            results={"mcts": dict(
                method="mcts", n_rollouts=args.n_envs,
                reach_pct=100.0 * p_frac,
                reach_err=math.sqrt(max(p_frac * (1 - p_frac), 0.0) / args.n_envs) * 100.0,
                success=[int(x) for x in succ],
                goals=[[float(g[0]), float(g[1])] for g in goals],
                starts=[[float(s[0]), float(s[1])] for s in starts],
                wall_s=wall)})
        fname = os.path.join(args.out_dir,
                             f"scale_mcts_b{args.budget}orc{ltag}{ctag}_s{seed}.json")
        with open(fname, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  [b{args.budget}orc s{seed}] reach={100*p_frac:.1f}%  {wall:.0f}s "
              f"-> {os.path.basename(fname)}")

    print("\n" + "=" * 72)
    print("ORACLE-TREE saved. Run the ATTRIBUTION LADDER through collate_mcts (it")
    print("verifies per-index goal identity and reports per-seed + pooled McNemar p):")
    print("  python scripts/collate_mcts.py \\")
    print("      results/scale_mcss_k50_s*.json results/scale_mcts_b16_s*.json \\")
    print(f"      results/scale_mcts_b16sgP_s*.json results/scale_mcts_b{args.budget}orc*_s*.json")
    print("  ladder (each rung isolates ONE factor, matched-compute among the trees):")
    print("    k50 -> b16     : flat baseline -> structured tree (+5.4x compute)")
    print("    b16 -> b16sgP  : goal-conditioning the tree value (matched compute)")
    print("    b16sgP -> b16orc: VALUE ACCURACY (learned V(s,g) -> perfect geodesic)")
    print("  read the POOLED exact-p, NOT the raw net (a +5 net over ~45 discordant is p~0.5).")
    print("\n  CAVEATS when interpreting b16orc (F4):")
    print("   - the geodesic value also GATES off-graph children (->-1), so a positive")
    print("     result motivates value accuracy AND a feasibility gate, not value alone.")
    print("   - it scores ENDPOINT geodesic, not 25-step SEGMENT feasibility, so a NULL")
    print("     bounds endpoint-lookahead only — a segment-feasible value stays untested.")
    print("  Rule-1: 'orc' is privileged/diagnostic — never report it as achievable.")
    print("=" * 72)


if __name__ == "__main__":
    main()
