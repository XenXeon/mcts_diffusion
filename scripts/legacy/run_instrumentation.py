"""scripts/run_instrumentation.py

Driver for the failure instrumentation (Tier-0 logging and the Tier-2 oracle-V
ceiling) on the MCSS / DV baseline — the sampler whose ~20-24% closed-loop
failures we are dissecting.

    # Tier-0/1: log the real MCSS baseline's failures (seeds 0,1,2)
    python scripts/run_instrumentation.py --env antmaze-large-diverse-v2 \
        --seeds 0 1 2 --n-envs 50 --value-source critic

    # Tier-2: re-run the SAME scenarios with the BFS geodesic as the critic
    #         (Rule-1 dev-only ceiling; never reportable)
    python scripts/run_instrumentation.py --env antmaze-large-diverse-v2 \
        --seeds 0 1 2 --n-envs 50 --value-source oracle

Then analyse with scripts/analyze_failures.py (Tier-1 modes + Tier-2 ceiling),
and optionally figure with scripts/plot_failures.py.

Cost: value_source=critic is one MCSS pass (~k_mcss candidates/step), i.e. the
cheap k50 baseline (~0.6 h/seed). value_source=oracle is the same cost plus the
BFS lookups (negligible). Both far cheaper than the b16 tree.
"""
import argparse
import sys

sys.path.insert(0, ".")

from mcts.mcts_loop import Sampler, load_models
from mcts.instrument import run_traced


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="antmaze-large-diverse-v2")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-envs", type=int, default=50)
    p.add_argument("--k-mcss", type=int, default=50,
                   help="candidates/step (50 = the cheap MCSS baseline being dissected)")
    p.add_argument("--value-source", choices=["critic", "oracle", "both"],
                   default="critic",
                   help="critic = real MCSS (Tier-0/1); oracle = BFS-geodesic ceiling "
                        "(Tier-2, Rule-1 dev-only)")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--keep-success-frac", type=float, default=0.0,
                   help="also dump this fraction of SUCCESSFUL episodes' traces, so "
                        "plot_candidates can contrast the critic's mis-rank rate on "
                        "successes vs failures (default 0 = failures only)")
    p.add_argument("--out-dir", type=str, default="results/instr")
    p.add_argument("--tag", type=str, default=None,
                   help="filename stem (default instr_mcss_<value_source>)")
    p.add_argument("--critic-step", type=int, default=1000000)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    sources = ["critic", "oracle"] if args.value_source == "both" else [args.value_source]
    # An explicit --tag with both sources would write critic then oracle to the same
    # stem (the second overwriting the first); the per-source default keeps them apart.
    if args.tag and len(sources) > 1:
        sys.exit("--tag with --value-source both would overwrite one run with the "
                 "other; drop --tag (defaults to instr_mcss_<source>) or run each "
                 "source separately.")

    models = load_models(args.env, critic_step=args.critic_step, device=args.device)
    sampler = Sampler(models, k_mcss=args.k_mcss, value_mode="v_s")

    for source in sources:
        for seed in args.seeds:
            tag = args.tag or f"instr_mcss_{source}"
            print(f"\n=== TRACED {source.upper()} : {args.env} "
                  f"seed={seed} n_envs={args.n_envs} k_mcss={args.k_mcss} ===")
            run_traced(sampler, seed=seed, n_envs=args.n_envs,
                       max_steps=args.max_steps, value_source=source,
                       out_dir=args.out_dir, tag=tag,
                       keep_success_frac=args.keep_success_frac)

    print("\nNext: python scripts/analyze_failures.py "
          f"--in-dir {args.out_dir}")


if __name__ == "__main__":
    main()
