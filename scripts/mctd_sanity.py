"""scripts/mctd_sanity.py

Phase-1 sanity peek for the faithful MCTD planner (mcts/mctd_planner.py) BEFORE
wiring it into closed-loop eval. Runs MCTDPlanner.plan on a handful of real
starts and prints, per plan, the signals that tell "behaving sensibly" apart
from merely "not crashing":

  * value / info / achieved_t     — the verifier's own verdict;
  * min_d, end_d (WORLD units)    — how close the chosen plan actually GETS to
                                    the goal and where it ENDS, which the binary
                                    Achieved/NotReached hides (a plan can miss
                                    the goal_radius yet still head straight at it);
  * sg_d                          — start->goal distance, for scale;
  * depth / nodes / searches      — did the denoising tree actually deepen.

It also echoes the effective schedule (rows R, block size, terminal_depth) and
the thresholds in play, so a "nothing ever reaches" result can be diagnosed as
low-headroom vs a too-tight goal_radius / warp gate rather than a bug.

Usage (on the training box, needs the DF checkpoint + d4rl):
    python scripts/mctd_sanity.py --env maze2d-large-v1 --n 5
    python scripts/mctd_sanity.py --env antmaze-large-diverse-v2 --n 5 \
        --goal-radius 2.0 --warp -1        # --warp -1 disables the warp gate
"""
import argparse
import sys
import time

sys.path.insert(0, ".")

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="maze2d-large-v1")
    ap.add_argument("--df-ckpt", default="final")
    ap.add_argument("--n", type=int, default=5, help="number of starts to plan")
    ap.add_argument("--seed", type=int, default=0)
    # MCTD knobs
    ap.add_argument("--n-depths", type=int, default=3)
    ap.add_argument("--max-search", type=int, default=32)
    ap.add_argument("--skip", type=int, default=10, help="jumpy-rollout stride")
    ap.add_argument("--guidance", type=float, nargs="+",
                    default=[0.0, 0.1, 0.5, 1.0, 2.0])
    # verifier threshold overrides (default = mcts/mctd_verify.py MCTD_ENV)
    ap.add_argument("--goal-radius", type=float, default=None)
    ap.add_argument("--warp", type=float, default=None,
                    help="warp threshold; negative disables the gate")
    args = ap.parse_args()

    from mcts.mcts_loop import load_models
    from mcts.mctd_planner import MCTDConfig, MCTDPlanner
    from mcts.mctd_verify import MCTD_ENV
    from mcts.specs import env_family, get_goal
    from pipelines.utils import set_seed

    fam = env_family(args.env)
    if fam not in MCTD_ENV:
        sys.exit(f"faithful (geometric) MCTD is not defined for family={fam!r} "
                 f"(no positional goal) — maze2d / antmaze only")

    set_seed(args.seed)
    models = load_models(args.env, df_ckpt=args.df_ckpt)
    if models.get("df_planner") is None:
        sys.exit(f"no DF planner loaded for {args.env} (df_ckpt={args.df_ckpt})")

    env_cfg = dict(MCTD_ENV[fam])
    if args.goal_radius is not None:
        env_cfg["goal_radius"] = args.goal_radius
    if args.warp is not None:
        env_cfg["warp_threshold"] = None if args.warp < 0 else args.warp

    cfg = MCTDConfig(guidance_scales=tuple(args.guidance), n_depths=args.n_depths,
                     max_search_num=args.max_search, skip_level_steps=args.skip)
    planner = MCTDPlanner(df_planner=models["df_planner"],
                          normalizer=models["normalizer"], family=fam,
                          obs_dim=models["obs_dim"], H=models["H"], cfg=cfg,
                          env_cfg=env_cfg, device=models["device"])
    normalizer = models["normalizer"]
    env = models["env_single"]
    pos_dims = list(planner.pos_dims)

    print(f"\n=== MCTD sanity: {args.env} ===")
    print(f"  guidance menu={list(args.guidance)}  n_depths={args.n_depths}  "
          f"max_search={args.max_search}  skip={args.skip}")
    print(f"  goal_radius={env_cfg['goal_radius']}  "
          f"warp_threshold={env_cfg.get('warp_threshold')}  "
          f"H={models['H']} obs_dim={models['obs_dim']}")
    hdr = (f"  {'#':>2} {'solved':>6} {'info':>11} {'value':>6} {'ach_t':>5} "
           f"{'sg_d':>7} {'min_d':>7} {'end_d':>7} {'depth':>5} {'nodes':>5} "
           f"{'srch':>4} {'sec':>5}")
    solved_ct, values, min_ds = 0, [], []
    printed_sched = False
    for i in range(args.n):
        obs = env.reset()
        goal = np.asarray(get_goal(env), dtype=np.float64)
        s_norm = normalizer.normalize(np.asarray(obs)[None])[0].astype(np.float32)
        t0 = time.perf_counter()
        out = planner.plan(s_norm, goal, seed=args.seed + i)
        dt = time.perf_counter() - t0

        if not printed_sched:
            print(f"  schedule: rows={out['n_rows']} block={out['block']} "
                  f"terminal_depth={out['terminal_depth']}\n{hdr}")
            printed_sched = True

        world = normalizer.unnormalize(out["plan_norm"])          # (T, D)
        pos = world[:, pos_dims]                                  # (T, P)
        gpos = goal[:len(pos_dims)]
        start_xy = np.asarray(obs)[pos_dims]
        d = np.linalg.norm(pos - gpos[None], axis=-1)
        sg_d = float(np.linalg.norm(start_xy - gpos))
        min_d, end_d = float(d.min()), float(d[-1])
        solved_ct += int(out["solved"])
        values.append(out["value"])
        min_ds.append(min_d)
        print(f"  {i:>2} {str(out['solved']):>6} {out['info']:>11} "
              f"{out['value']:>6.3f} {str(out['achieved_t']):>5} "
              f"{sg_d:>7.2f} {min_d:>7.2f} {end_d:>7.2f} "
              f"{out['max_depth']:>5} {out['n_nodes']:>5} {out['n_search']:>4} "
              f"{dt:>5.1f}")

    print(f"\n  summary: solved {solved_ct}/{args.n}  "
          f"mean value={np.mean(values):.3f}  "
          f"mean min_d={np.mean(min_ds):.2f} (world units to goal)")
    print("  read: min_d << sg_d means the plan HEADS to the goal even if it "
          "misses the goal_radius; min_d ~ sg_d means it is not goal-directed.\n")


if __name__ == "__main__":
    main()
