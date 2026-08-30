"""scripts/run_mcts_compare.py

Closed-loop comparison of MCSS (DV baseline) vs MCTS (state-value look-ahead), on the
identical harness (same env loop, normalizer, inverse-dynamics policy, replan cadence).

The headline metric is `reach%` = fraction of rollouts that reach the goal — directly
comparable to the DV inference baselines you already have:
    antmaze-large-diverse-v2 : 76.9
    maze2d-large-v1          : all reach (saturated)

Cost note: MCTS does ~(budget+1) batched planner calls per env-step vs 1 for MCSS, so start
with modest --n-envs/--n-episodes/--budget and a small --max-steps to smoke-test the loop,
then scale up for a real estimate.

Examples:
    # fast smoke test (does the loop run end to end?)
    python scripts/run_mcts_compare.py --env antmaze-large-diverse-v2 --method both \
        --n-envs 4 --n-episodes 1 --budget 6 --max-steps 120

    # real comparison on the env with headroom
    python scripts/run_mcts_compare.py --env antmaze-large-diverse-v2 --method both \
        --n-envs 25 --n-episodes 2 --budget 15 --k-mcts 16

    # sanity: MCTS should also solve maze2d
    python scripts/run_mcts_compare.py --env maze2d-large-v1 --method both \
        --n-envs 25 --n-episodes 2 --budget 10 --k-mcts 16
"""
import argparse
import json
import subprocess
import sys
import time

sys.path.insert(0, ".")

from mcts.mcts_loop import Sampler, load_models, run_episodes
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
    p.add_argument("--method", type=str, default="both",
                   choices=["mcss", "mcts", "both"])
    p.add_argument("--n-envs", type=int, default=25)
    p.add_argument("--n-episodes", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap episode length (default = env max_path_length)")
    p.add_argument("--seed", type=int, default=0)
    # search params
    p.add_argument("--budget", type=int, default=15, help="MCTS expansion rounds")
    p.add_argument("--k-mcts", type=int, default=16, help="candidates per expansion")
    p.add_argument("--k-root", type=int, default=None,
                   help="ROOT expansion width (default = --k-mcts). The executed "
                        "action is picked among root children, so this is what "
                        "competes with MCSS's --k-mcss pool; set 50 for a "
                        "superset-of-MCSS root")
    p.add_argument("--top-m", type=int, default=1,
                   help="backup = mean of the m best children (1 = MAX backup); "
                        ">1 tempers max-backup optimism on noisy/stitched scores")
    p.add_argument("--k-mcss", type=int, default=50, help="MCSS candidate count")
    p.add_argument("--child-index", type=int, default=1,
                   help="trajectory index used as the child state (segment length)")
    p.add_argument("--c-ucb", type=float, default=1.4142136)
    p.add_argument("--rebase-policy", type=int, default=None, choices=[0, 1],
                   help="policy-input rebasing (subtract xy from obs dims 0-1, as the "
                        "DV separate-pipeline invdyn does). Default: per the DV config — "
                        "maze2d/antmaze=1, kitchen=0 (kitchen dims 0-1 are joint angles, "
                        "NOT xy, so rebasing corrupts the inverse-dynamics policy input)")
    # tree node value: v_s = goal-agnostic baseline; v_sg / v_sg_pess = goal-cond;
    # grounded = ground-truth kitchen subtask count (mcts/grounded.py), exempt from
    # the 3-of-4 label cap every learned value here inherits from kitchen-mixed
    p.add_argument("--value-mode",
                   choices=["v_s", "v_sg", "v_sg_pess", "critic", "grounded"],
                   default="v_s", help="MCTS tree node value (mcts arm only)")

    p.add_argument("--pess-beta", type=float, default=1.0,
                   help="mean−β·std for the (future) mean_std pessimism variant")
    p.add_argument("--expand-mode", choices=["glue", "inpaint"], default="glue",
                   help="critic-mode expansion: glue = continuation sampled from "
                        "the leaf state and concatenated onto the prefix (seam is "
                        "off-manifold for the critic); inpaint = Diffusion-Forcing-"
                        "inspired, prefix clamped into the denoiser so the window "
                        "is generated jointly consistent with the path (seam-free)")
    p.add_argument("--df-ckpt", type=str, default=None,
                   help="tag of a Causal Diffusion Forcing planner checkpoint "
                        "(df_planner_ckpt_<tag>.pt, scripts/train_df_planner.py). "
                        "When set, BOTH arms use the DF backbone: mcss = DF "
                        "sample-and-rank (DV critic), mcts = native prefix-"
                        "conditioned expansion (needs --value-mode critic)")
    p.add_argument("--df-slope", type=int, default=1,
                   help="pyramid schedule slope (levels of extra noise per "
                        "future token)")
    p.add_argument("--df-row-stride", type=int, default=1,
                   help="subsample denoising sweeps (like DDIM step stride)")
    p.add_argument("--sweeps", type=int, default=None,
                   help="shortcut planner only: sampling sweeps (power of 2, e.g. "
                        "8); ignored by the standard DF/DV backbones. None = the "
                        "shortcut model's own cfg default (4)")
    p.add_argument("--cg-ckpt", type=str, default=None,
                   help="tag of a noise-aware value checkpoint "
                        "(noise_critic_ckpt_<tag>.pt, scripts/train_noise_critic.py) "
                        "for classifier guidance on the DF planner. Requires "
                        "--df-ckpt (CG steers the DF sampler only)")
    p.add_argument("--cg-w", type=float, default=0.0,
                   help="classifier-guidance weight (0 = off). Needs --cg-ckpt "
                        "and --df-ckpt; see Sampler's cg_w guards in mcts_loop.py")
    p.add_argument("--grounded-blend", type=float, default=0.25,
                   help="grounded value_mode / --grounded-mcss: weight of the DV "
                        "critic score added as a tiebreaker on top of the grounded "
                        "kitchen subtask count (mcts/grounded.py); 0 = pure grounded")
    p.add_argument("--grounded-mcss", type=int, default=0, choices=[0, 1],
                   help="rerank MCSS candidates by the grounded subtask checker "
                        "(mcts/grounded.py) instead of the DV critic alone — "
                        "independent of --value-mode, kitchen-only")
    p.add_argument("--junction-filter", action="store_true",
                   help="reject tree children whose first continuation step is an "
                        "implausibly large xy hop (> --junction-pct percentile of "
                        "the dataset's stride-spaced steps)")
    p.add_argument("--junction-pct", type=float, default=99.0)
    # checkpoints
    p.add_argument("--value-step", type=str, default="latest")
    p.add_argument("--sg-ckpt", type=str, default="state_value_sg_ckpt_best.pt",
                   help="V(s,g) ensemble checkpoint (best-val, terminus-only by default)")
    p.add_argument("--planner-step", type=int, default=1000000)
    p.add_argument("--critic-step", type=str, default="1000000",
                   help="int step OR a tag: 'stitched'/'stitched_best' loads "
                        "critic_ckpt_<tag>.pt (finetune_critic_stitched.py output)")
    p.add_argument("--policy-step", type=int, default=1000000)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dv-log", action="store_true",
                   help="print the base-DV inference log format ([t=N] rew: [...] "
                        "per step, then the final `mean err` line) with the DV-exact "
                        "per-family score (antmaze reach% <=100; maze2d camping >100)")
    p.add_argument("--out", type=str, default=None,
                   help="path to save the comparison results as JSON")
    args = p.parse_args()

    # value_mode applies to the MCTS arm only — mcss_waypoints uses the DV
    # trajectory critic (or, with --grounded-mcss, the grounded checker instead),
    # so an mcss run with value_mode!=v_s/critic would be falsely labelled (F2).
    # BoN re-ranked by V(s,g) (cell C) is not wired yet.
    if args.method in ("mcss", "both") and args.value_mode not in ("v_s", "critic", "grounded"):
        sys.exit("--value-mode applies to the MCTS arm only (MCSS uses the DV "
                 "critic, or the grounded checker with --grounded-mcss). Run the "
                 "goal-conditioned arm with --method mcts; run MCSS separately "
                 "with the default --value-mode v_s.")
    # Fail before loading any model: the Sampler guard (mcts_loop.py) would also
    # catch a missing --df-ckpt, but that's after a slow load_models() call —
    # catching it here is friendlier.
    if args.cg_w != 0 and not args.df_ckpt:
        sys.exit("--cg-w != 0 needs --df-ckpt (classifier guidance steers the "
                 "DF sampler only; the DV planner has its own guidance stack)")
    # Same friendliness for the grounded checker: it reads kitchen task
    # definitions off the live env (mcts/grounded.py), so fail before the slow
    # load_models() call rather than inside Sampler.__init__.
    if (args.value_mode == "grounded" or args.grounded_mcss) and env_family(args.env) != "kitchen":
        sys.exit(f"--value-mode grounded / --grounded-mcss needs a kitchen env "
                 f"(got {args.env!r}, family={env_family(args.env)!r}) — the "
                 f"grounded checker reads kitchen task definitions off the live env")

    models = load_models(args.env, value_step=args.value_step,
                         planner_step=args.planner_step, critic_step=args.critic_step,
                         policy_step=args.policy_step, device=args.device,
                         ckpt_dir=args.ckpt, sg_ckpt=args.sg_ckpt,
                         df_ckpt=args.df_ckpt, cg_ckpt=args.cg_ckpt)
    # Policy rebasing follows the DV config per family unless overridden (kitchen=0).
    rebase = (SPECS[env_family(args.env)].get("rebase_policy", True)
              if args.rebase_policy is None else bool(args.rebase_policy))
    print(f"  rebase_policy = {rebase} "
          f"({'CLI override' if args.rebase_policy is not None else 'per DV config'})")
    sampler = Sampler(models, k_mcss=args.k_mcss, k_mcts=args.k_mcts, budget=args.budget,
                      child_index=args.child_index, c_ucb=args.c_ucb,
                      value_mode=args.value_mode, pess_beta=args.pess_beta,
                      k_root=args.k_root, top_m=args.top_m,
                      junction_filter=args.junction_filter,
                      junction_pct=args.junction_pct,
                      expand_mode=args.expand_mode,
                      backbone="df" if args.df_ckpt else "dv",
                      df_slope=args.df_slope, df_row_stride=args.df_row_stride,
                      df_sweeps=args.sweeps, cg_w=args.cg_w, rebase=rebase,
                      grounded_blend=args.grounded_blend,
                      grounded_mcss=bool(args.grounded_mcss))

    methods = ["mcss", "mcts"] if args.method == "both" else [args.method]
    results = {}
    for method in methods:
        print(f"\n=== {method.upper()} : {args.env} "
              f"(n_envs={args.n_envs} x n_episodes={args.n_episodes}"
              f"{'' if args.max_steps is None else f', max_steps={args.max_steps}'})"
              f"{f', budget={args.budget}, k={args.k_mcts}, L={args.child_index}' if method=='mcts' else f', k={args.k_mcss}'} ===")
        results[method] = run_episodes(sampler, method, n_envs=args.n_envs,
                                       n_episodes=args.n_episodes, seed=args.seed,
                                       max_steps=args.max_steps, dv_log=args.dv_log)

    print("\n" + "=" * 72)
    print(f"COMPARISON — {args.env}  (DV-score = base-pipeline metric)")
    print(f"{'method':>6}  {'reach%':>13}  {'DV-score':>14}  {'rollouts':>8}  {'wall(s)':>8}")
    for method in methods:
        r = results[method]
        print(f"{method:>6}  {r['reach_pct']:>6.1f}±{r.get('reach_err', float('nan')):>4.1f}%  "
              f"{r['dv_norm_mean']:>6.1f} ± {r['dv_norm_err']:>4.1f}  "
              f"{r['n_rollouts']:>8}  {r['wall_s']:>8.0f}")
    if args.method == "both":
        d = results["mcts"]["reach_pct"] - results["mcss"]["reach_pct"]
        print(f"\n  MCTS − MCSS reach = {d:+.1f} pp "
              f"({'MCTS helps' if d > 0 else 'no gain' if d == 0 else 'MCTS worse'})")
        print("  (paired per-rollout vectors are in the JSON — run "
              "scripts/collate_mcts.py for the McNemar test)")
    print("=" * 72)

    if args.out:
        payload = dict(env=args.env, seed=args.seed, n_envs=args.n_envs,
                       n_episodes=args.n_episodes, max_steps=args.max_steps,
                       budget=args.budget, k_mcts=args.k_mcts, k_mcss=args.k_mcss,
                       k_root=args.k_root or args.k_mcts, top_m=args.top_m,
                       junction_filter=args.junction_filter,
                       junction_pct=args.junction_pct,
                       expand_mode=args.expand_mode,
                       backbone="df" if args.df_ckpt else "dv",
                       df_ckpt=args.df_ckpt, df_slope=args.df_slope,
                       df_row_stride=args.df_row_stride, df_sweeps=args.sweeps,
                       cg_ckpt=args.cg_ckpt, cg_w=args.cg_w,
                       rebase_policy=rebase,
                       child_index=args.child_index, c_ucb=args.c_ucb,
                       # value/gate config — collate_mcts.config_suffix reads these
                       # (B2 write-side): naive cells = v_s + no gate -> no label suffix
                       value_mode=args.value_mode, gate="none", dv_log=args.dv_log,
                       # grounded checker config (mcts/grounded.py) — config_suffix
                       # appends "Gnd" whenever value_mode=grounded or grounded_mcss
                       # is truthy, so a grounded-valued arm never pools with a
                       # critic-valued one at the same base label
                       grounded_blend=args.grounded_blend,
                       grounded_mcss=bool(args.grounded_mcss),
                       sg_ckpt=models["sg_ckpt"],
                       # checkpoint identity (resolved values, not just CLI args)
                       value_step=models["value_step"],
                       planner_step=models["planner_step"],
                       critic_step=models["critic_step"],
                       policy_step=models["policy_step"],
                       ckpt_dir=models["ckpt_dir"],
                       max_path_length=models["max_path_length"],
                       git_commit=git_commit(),
                       timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                       results=results)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"saved results -> {args.out}")


if __name__ == "__main__":
    main()
