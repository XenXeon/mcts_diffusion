"""scripts/run_mctd.py

Closed-loop evaluation of faithful MCTD (mcts/mctd_planner.py + mcts/mctd_loop.py)
on D4RL, Phase 2 of the port. Produces a result JSON with the SAME schema as
scripts/run_mcts_compare.py (results["mctd"] carries reach%, DV-exact score, and
the per-rollout success/dv_norm/goals/starts vectors), seeded so rollouts PAIR
with an MCSS run at the same --env/--seed/--n-envs (scripts/collate_mcts.py).

MCTD is maze2d / antmaze only (geometric verifier needs a positional goal);
kitchen is refused. The DF checkpoints on disk are maze2d-large-v1,
antmaze-large-diverse-v2 (df_planner_ckpt_final.pt).

Cost: MCTD runs a tree search per replan and (for now) loops over envs, so it is
much slower than MCSS. Start small (--n-envs 4 --max-steps 200) to smoke the loop,
then scale. --replan-every trades planning cost against MPC responsiveness.

Examples:
    # smoke: does the closed loop run end to end?
    python scripts/run_mctd.py --env maze2d-large-v1 --n-envs 4 --n-episodes 1 \
        --max-steps 200 --max-search 16

    # a real estimate to compare against MCSS at the same seed
    python scripts/run_mctd.py --env maze2d-large-v1 --n-envs 25 --n-episodes 1 \
        --seed 0 --out results/mctd_maze2d_large_s0.json
    python scripts/run_mcts_compare.py --env maze2d-large-v1 --method mcss \
        --n-envs 25 --n-episodes 1 --seed 0 --out results/mcss_maze2d_large_s0.json
"""
import argparse
import json
import subprocess
import sys
import time

sys.path.insert(0, ".")

from mcts.mcts_loop import load_models
from mcts.mctd_loop import (DFTreeMPCPlanner, GuidedBoNPlanner, MCSSMPCPlanner,
                            run_mctd_episodes)
from mcts.mctd_planner import MCTDConfig, MCTDPlanner
from mcts.mctd_verify import MCTD_ENV
from mcts.specs import SPECS, env_family


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--df-ckpt", type=str, default="final",
                   help="DF planner checkpoint tag (df_planner_ckpt_<tag>.pt)")
    p.add_argument("--n-envs", type=int, default=25)
    p.add_argument("--n-episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    # MCTD search knobs (mcts/mctd_planner.py MCTDConfig)
    p.add_argument("--guidance", type=float, nargs="+",
                   default=[0.0, 0.1, 0.5, 1.0, 2.0], help="guidance-scale menu")
    p.add_argument("--n-depths", type=int, default=3,
                   help="denoising blocks = tree depth (terminal_depth)")
    p.add_argument("--max-search", type=int, default=32, help="expansions per plan")
    p.add_argument("--skip", type=int, default=10, help="jumpy-rollout stride")
    p.add_argument("--c-ucb", type=float, default=1.4142136)
    p.add_argument("--num-tries", type=int, default=3,
                   help="resample attempts for degenerate (non-moving) plans")
    p.add_argument("--early-stopping", choices=["achieved", "none"],
                   default="achieved")
    p.add_argument("--value-mode", choices=["geometric", "critic"],
                   default="geometric",
                   help="MCTD node value: geometric (Way 1, non-learned goal-reach) "
                        "or critic (Way 4c, DV trajectory critic on the clean plan)")
    # geometric verifier overrides (default = mcts/mctd_verify.py MCTD_ENV)
    p.add_argument("--goal-radius", type=float, default=None)
    p.add_argument("--warp", type=float, default=None,
                   help="warp threshold (world units); negative disables the gate")
    # MPC execution
    p.add_argument("--replan-every", type=int, default=30,
                   help="env-steps between replans")
    p.add_argument("--reach-wp", type=float, default=1.0,
                   help="advance to next waypoint when within this (world units)")
    p.add_argument("--rebase-policy", type=int, default=None, choices=[0, 1])
    # controlled baseline: best-of-K MCSS in the IDENTICAL MPC harness (same
    # replan cadence, waypoint-following, DF backbone) — isolates search-vs-flat
    # from the MPC-vs-per-step execution confound (mcts/mctd_loop.py)
    p.add_argument("--flat-mcss", action="store_true",
                   help="run best-of-K MCSS in the MPC harness instead of the "
                        "MCTD tree (the search-isolating control)")
    p.add_argument("--k", type=int, default=50,
                   help="best-of-K for --flat-mcss (MCSS candidate count)")
    p.add_argument("--mcss-backbone", choices=["df", "dv"], default="df",
                   help="--flat-mcss planner: df (search-isolating control, MCTD's "
                        "backbone) or dv (the frozen DV SOTA planner — run at "
                        "--replan-every to get DV-MCSS at a matched cadence, the "
                        "control that separates the DF-vs-DV backbone gap from the "
                        "per-step-vs-MPC cadence gap)")
    p.add_argument("--guided-bon", action="store_true",
                   help="Way 4b: flat best-of-N over guidance weights (critic-"
                        "ranked, no tree) instead of the MCTD tree")
    p.add_argument("--k-per", type=int, default=10,
                   help="--guided-bon: plans per guidance weight (total N = "
                        "len(guidance menu) * k_per)")
    # the DF-tree (this project's trajectory-axis tree) run in the MPC harness —
    # the cadence-matched control that makes it raw-comparable to MCTD-critic
    p.add_argument("--df-tree", action="store_true",
                   help="run this project's DF-tree (trajectory look-ahead + DV "
                        "critic) in the MPC harness at --replan-every, instead of "
                        "the MCTD tree")
    p.add_argument("--tree-budget", type=int, default=15)
    p.add_argument("--tree-k", type=int, default=16, help="DF-tree candidates/expansion")
    p.add_argument("--tree-k-root", type=int, default=16)
    p.add_argument("--tree-top-m", type=int, default=3, help="DF-tree backup: mean of top-m")
    # checkpoints
    p.add_argument("--planner-step", type=int, default=1000000)
    p.add_argument("--policy-step", type=int, default=1000000)
    p.add_argument("--critic-step", type=str, default="1000000")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dv-log", action="store_true")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    fam = env_family(args.env)
    if fam not in MCTD_ENV:
        sys.exit(f"faithful (geometric) MCTD is maze2d / antmaze only — "
                 f"{args.env!r} is family {fam!r} (no positional goal). Kitchen "
                 f"would need a grounded verifier (a separate Way-4c variant).")

    models = load_models(args.env, planner_step=args.planner_step,
                         critic_step=args.critic_step, policy_step=args.policy_step,
                         device=args.device, ckpt_dir=args.ckpt,
                         df_ckpt=args.df_ckpt)
    if models.get("df_planner") is None:
        sys.exit(f"no DF planner loaded (df_ckpt={args.df_ckpt!r})")

    rebase = (SPECS[fam].get("rebase_policy", True)
              if args.rebase_policy is None else bool(args.rebase_policy))

    env_cfg = dict(MCTD_ENV[fam])
    if args.goal_radius is not None:
        env_cfg["goal_radius"] = args.goal_radius
    if args.warp is not None:
        env_cfg["warp_threshold"] = None if args.warp < 0 else args.warp

    if args.df_tree:                             # DF-tree in the MPC harness
        planner = DFTreeMPCPlanner(models, family=fam, budget=args.tree_budget,
                                   k_mcts=args.tree_k, k_root=args.tree_k_root,
                                   top_m=args.tree_top_m)
        method_label = "df_tree_mpc"
        head = (f"DF-tree (trajectory look-ahead, MPC) : {args.env}  "
                f"(n_envs={args.n_envs} x n_episodes={args.n_episodes}"
                f"{'' if args.max_steps is None else f', max_steps={args.max_steps}'}, "
                f"budget={args.tree_budget}, k={args.tree_k}, top_m={args.tree_top_m}, "
                f"replan_every={args.replan_every})")
    elif args.guided_bon:                        # Way 4b
        planner = GuidedBoNPlanner(models, family=fam,
                                   guidance_scales=tuple(args.guidance),
                                   k_per=args.k_per)
        method_label = "guided_bon"
        head = (f"Guided-BoN (Way 4b, {len(args.guidance)}x{args.k_per} critic-"
                f"ranked, no tree) : {args.env}  (n_envs={args.n_envs} x "
                f"n_episodes={args.n_episodes}"
                f"{'' if args.max_steps is None else f', max_steps={args.max_steps}'}, "
                f"replan_every={args.replan_every})")
    elif args.flat_mcss:                         # MCSS-MPC control (df or dv backbone)
        planner = MCSSMPCPlanner(models, family=fam, k=args.k,
                                 backbone=args.mcss_backbone)
        method_label = "mcss_mpc" if args.mcss_backbone == "df" else "dvmcss_mpc"
        head = (f"MCSS-MPC ({args.mcss_backbone.upper()} backbone, best-of-{args.k}) : "
                f"{args.env}  (n_envs={args.n_envs} x n_episodes={args.n_episodes}"
                f"{'' if args.max_steps is None else f', max_steps={args.max_steps}'}, "
                f"replan_every={args.replan_every})")
    else:                                        # MCTD tree (Way 1 or Way 4c)
        cfg = MCTDConfig(guidance_scales=tuple(args.guidance), n_depths=args.n_depths,
                         skip_level_steps=args.skip, max_search_num=args.max_search,
                         c_ucb=args.c_ucb, num_tries_for_bad_plans=args.num_tries,
                         early_stopping=(None if args.early_stopping == "none"
                                         else args.early_stopping))
        planner = MCTDPlanner(df_planner=models["df_planner"],
                              normalizer=models["normalizer"], family=fam,
                              obs_dim=models["obs_dim"], H=models["H"], cfg=cfg,
                              env_cfg=env_cfg, device=models["device"],
                              value_mode=args.value_mode, critic=models["critic"])
        method_label = "mctd" if args.value_mode == "geometric" else "mctd_critic"
        head = (f"MCTD ({args.value_mode} value{'—Way 4c' if args.value_mode == 'critic' else ''}) : "
                f"{args.env}  (n_envs={args.n_envs} x n_episodes={args.n_episodes}"
                f"{'' if args.max_steps is None else f', max_steps={args.max_steps}'}, "
                f"guidance={list(args.guidance)}, depths={args.n_depths}, "
                f"max_search={args.max_search}, replan_every={args.replan_every})")

    print(f"\n=== {head} ===")
    print(f"  goal_radius={env_cfg['goal_radius']} "
          f"warp={env_cfg.get('warp_threshold')} rebase_policy={rebase}")

    result = run_mctd_episodes(planner, models, n_envs=args.n_envs,
                               n_episodes=args.n_episodes, seed=args.seed,
                               max_steps=args.max_steps,
                               replan_every=args.replan_every,
                               reach_wp=args.reach_wp, rebase=rebase,
                               dv_log=args.dv_log, method_label=method_label)

    print("\n" + "=" * 72)
    print(f"{method_label.upper()} — {args.env}  (DV-score = base-pipeline metric)")
    print(f"{'method':>8}  {'reach%':>13}  {'DV-score':>14}  {'rollouts':>8}  {'wall(s)':>8}")
    print(f"{method_label:>8}  {result['reach_pct']:>6.1f}±{result['reach_err']:>4.1f}%  "
          f"{result['dv_norm_mean']:>6.1f} ± {result['dv_norm_err']:>4.1f}  "
          f"{result['n_rollouts']:>8}  {result['wall_s']:>8.0f}")
    print(f"  search: {result['n_plans']} plans, solved_frac="
          f"{result['solved_plan_frac']}, mean_depth={result['tree_depth_mean']}, "
          f"mean_search={result['mean_search']}")
    print("=" * 72)

    if args.out:
        payload = dict(env=args.env, seed=args.seed, n_envs=args.n_envs,
                       n_episodes=args.n_episodes, max_steps=args.max_steps,
                       method=method_label, backbone="df", df_ckpt=args.df_ckpt,
                       flat_mcss=bool(args.flat_mcss), mcss_backbone=args.mcss_backbone,
                       guided_bon=bool(args.guided_bon), k=args.k, k_per=args.k_per,
                       # MCTD config (unused by the flat-mcss / guided-bon arms, kept for record)
                       mctd_value_mode=args.value_mode,
                       guidance_scales=list(args.guidance), n_depths=args.n_depths,
                       max_search=args.max_search, skip_level_steps=args.skip,
                       c_ucb=args.c_ucb, early_stopping=args.early_stopping,
                       goal_radius=env_cfg["goal_radius"],
                       warp_threshold=env_cfg.get("warp_threshold"),
                       replan_every=args.replan_every, reach_wp=args.reach_wp,
                       rebase_policy=rebase,
                       # value_mode label keeps each arm distinct in collate's
                       # config_suffix (method_label already disambiguates the file)
                       value_mode=method_label,
                       gate="none", cg_ckpt=None, cg_w=0.0,
                       planner_step=models["planner_step"],
                       critic_step=models["critic_step"],
                       policy_step=models["policy_step"],
                       ckpt_dir=models["ckpt_dir"],
                       max_path_length=models["max_path_length"],
                       git_commit=git_commit(),
                       timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                       results={method_label: result})
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"saved results -> {args.out}")


if __name__ == "__main__":
    main()
