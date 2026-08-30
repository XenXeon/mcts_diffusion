"""scripts/phase4_mcts_rollout.py

Phase 4 — closed-loop evaluation: MCTS-guided DV-MCSS vs greedy baseline.

For each (method, seed) pair:
    - Greedy (DV-MCSS): one expansion per step, K=50 candidates, pick best.
    - MCTS: tree search with K=10, configurable budget per step.

Results written to:
    results/phase4/episode_results.json   (list of EpisodeResult dicts)
    results/phase4/summary.csv

Baseline comparison loaded from:
    results/phase0_baseline.json

Run:
    python scripts/phase4_mcts_rollout.py
    python scripts/phase4_mcts_rollout.py --seeds 0 1 2 --mcts-budgets 5 10
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, ".")

import d4rl  # noqa: F401
import gym
import torch

from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DiT1d, DVInvMlp
from cleandiffuser.utils import DVHorizonCritic
from mcts.expansion import ExpansionConfig, PlannerExpansion
from mcts.node import TreeConfig
from mcts.rollout import EpisodeResult, RolloutConfig, run_greedy_episode, run_mcts_episode
from pipelines.utils import set_seed

# ── Constants ──────────────────────────────────────────────────────────────────

ENV_NAME     = "maze2d-umaze-v1"   # overridden by --env in main()
OBS_DIM      = 4
ACT_DIM      = 2
H            = 32
M            = 15
CHILD_IDX    = 1
K_GREEDY     = 50     # candidates for greedy DV-MCSS (mirrors Phase 0)
K_MCTS       = 10     # candidates per MCTS expansion (sweet spot from Phase 3)
PLAN_STEPS   = 20
POLICY_STEPS = 10
MAX_T        = 300    # read from env TimeLimit in main() (300/600/800)
DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"

# Env-dependent paths; set in main() once --env is known.
CKPT          = ""
REQUIRED      = []
OUTPUT_DIR    = "results/phase4"
BASELINE_JSON = "results/phase0_baseline.json"

_ENV_TAG = {
    "maze2d-umaze-v1":  "umaze",
    "maze2d-medium-v1": "medium",
    "maze2d-large-v1":  "large",
}


def configure_env_paths(env_name: str) -> None:
    """Set the env-dependent module globals (checkpoint, output dir, baseline)."""
    global ENV_NAME, CKPT, REQUIRED, OUTPUT_DIR, BASELINE_JSON
    ENV_NAME = env_name
    tag = _ENV_TAG[env_name]
    CKPT = (
        "results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer"
        f"_d2_width256_separate_dpTrue/{env_name}"
    )
    REQUIRED = [
        f"{CKPT}/planner_ckpt_1000000.pt",
        f"{CKPT}/critic_ckpt_1000000.pt",
        f"{CKPT}/policy_ckpt_1000000.pt",
    ]
    # Keep umaze artefacts at their original paths; suffix the larger mazes.
    OUTPUT_DIR    = "results/phase4" if tag == "umaze" else f"results/phase4_{tag}"
    BASELINE_JSON = ("results/phase0_baseline.json" if tag == "umaze"
                     else f"results/phase0_baseline_{tag}.json")


DEFAULT_SEEDS        = [0, 1, 2, 3, 4]
# Budgets that actually reach depth ≥ 3 (need budget ≥ K_MCTS + 2 = 12).
# The old [5, 10] sat at depth 2 — no look-ahead. See mcts/tree.py best_path docstring.
DEFAULT_MCTS_BUDGETS = [12, 52]

# ── Helpers ────────────────────────────────────────────────────────────────────

def check_prerequisites() -> None:
    missing = [p for p in REQUIRED if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"[FAIL] Required file not found: {p}")
        sys.exit(1)


def build_models(dataset):
    print(f"Loading planner, critic, policy on {DEVICE} …")
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    fix_mask = torch.zeros((H, obs_dim))
    fix_mask[0, :obs_dim] = 1.0
    planner = ContinuousDiffusionSDE(
        DiT1d(obs_dim, emb_dim=128, d_model=256, n_heads=4, depth=2,
              timestep_emb_type="fourier"),
        fix_mask=fix_mask, noise_schedule="linear", device=DEVICE,
        predict_noise=True, ema_rate=0.9999,
        loss_weight=torch.ones((H, obs_dim)),
    )
    planner.load(f"{CKPT}/planner_ckpt_1000000.pt")
    planner.eval()

    critic_ckpt = torch.load(f"{CKPT}/critic_ckpt_1000000.pt", map_location=DEVICE)
    critic = DVHorizonCritic(
        obs_dim, emb_dim=128, d_model=256, n_heads=4, depth=2, norm_type="pre",
    ).to(DEVICE)
    critic.load_state_dict(critic_ckpt["critic"])
    critic.eval()

    policy = DiscreteDiffusionSDE(
        DVInvMlp(obs_dim, act_dim, emb_dim=64, hidden_dim=256,
                 timestep_emb_type="positional").to(DEVICE),
        IdentityCondition(dropout=0.0).to(DEVICE),
        x_max=+torch.ones((1, act_dim), device=DEVICE),
        x_min=-torch.ones((1, act_dim), device=DEVICE),
        diffusion_steps=POLICY_STEPS, device=DEVICE,
    )
    policy.load(f"{CKPT}/policy_ckpt_1000000.pt")
    policy.eval()

    print(f"  Loaded on {DEVICE}.")
    return planner, critic, policy


def make_expansion(planner, critic, k: int) -> PlannerExpansion:
    cfg = ExpansionConfig(
        K=k, horizon=H, obs_dim=OBS_DIM, planner_dim=OBS_DIM,
        solver="ddim", sample_steps=PLAN_STEPS, temperature=1.0,
        use_ema=True, device=DEVICE,
    )
    return PlannerExpansion(planner, critic, cfg)


def make_tree_cfg(budget: int) -> TreeConfig:
    return TreeConfig(
        obs_dim=OBS_DIM, horizon=H, child_state_index=CHILD_IDX,
        K=K_MCTS, ucb_c=1.414, storage_mode="state_only",
        max_expansions=budget, device=DEVICE,
        leaf_batch_size=10, ucb_tie_breaking="random",
    )


def save_results(results: list[EpisodeResult]) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "episode_results.json")
    rows = [r.to_dict() for r in results]
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
        rows = existing + rows
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    return path


def save_summary(results: list[EpisodeResult]) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "summary.csv")
    if not results:
        return path
    fields = list(results[0].to_dict().keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([r.to_dict() for r in results])
    return path


def load_baseline() -> list[dict]:
    if not os.path.exists(BASELINE_JSON):
        return []
    with open(BASELINE_JSON) as f:
        return json.load(f)


def _mean(values: list) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def print_comparison(results: list[EpisodeResult], baseline: list[dict]) -> None:
    from collections import defaultdict

    grouped: dict[str, list] = defaultdict(list)
    for r in results:
        grouped[r.method].append(r)

    print()
    print("=" * 100)
    print("Phase 4 Evaluation Summary")
    print("=" * 100)
    print(
        f"{'Method':<28}  {'n':>3}  {'norm_score':>12}  "
        f"{'raw_return':>12}  {'goal_step':>10}  "
        f"{'denoise_calls':>14}  {'ms/step':>8}"
    )
    print("-" * 100)

    if baseline:
        bs = baseline
        print(
            f"{'DV-MCSS (Phase 0 ref)':<28}  {len(bs):>3}  "
            f"{_mean([r['normalized_score'] for r in bs]):>12.2f}  "
            f"{_mean([r['raw_return'] for r in bs]):>12.1f}  "
            f"{_mean([r['goal_step'] for r in bs if r['goal_step']]):>10.1f}  "
            f"{_mean([r['denoising_calls'] for r in bs]):>14.0f}  "
            f"{_mean([r['ms_per_step'] for r in bs]):>8.1f}"
        )

    for method in sorted(grouped.keys()):
        rs = grouped[method]
        print(
            f"{method:<28}  {len(rs):>3}  "
            f"{_mean([r.normalized_score for r in rs]):>12.2f}  "
            f"{_mean([r.raw_return for r in rs]):>12.1f}  "
            f"{_mean([r.goal_step for r in rs if r.goal_step is not None]):>10.1f}  "
            f"{_mean([r.denoising_calls for r in rs]):>14.0f}  "
            f"{_mean([r.ms_per_step for r in rs]):>8.1f}"
        )

    mcts_rows = [r for r in results if r.mean_tree_depth is not None]
    if mcts_rows:
        by_method: dict[str, list] = defaultdict(list)
        for r in mcts_rows:
            by_method[r.method].append(r)
        print()
        print("MCTS tree metrics (mean across seeds):")
        for method, rs in sorted(by_method.items()):
            print(
                f"  {method}: depth={_mean([r.mean_tree_depth for r in rs]):.2f}  "
                f"best_score={_mean([r.mean_cumulative_best for r in rs]):.4f}"
            )

    print("=" * 100)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    global K_GREEDY, K_MCTS, MAX_T
    parser = argparse.ArgumentParser(
        description="Phase 4 — MCTS-guided vs greedy closed-loop evaluation"
    )
    parser.add_argument(
        "--env", type=str, default="maze2d-umaze-v1", choices=list(_ENV_TAG),
        help="D4RL maze2d env; selects checkpoint, episode length and output dir",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--mcts-budgets", type=int, nargs="+", default=DEFAULT_MCTS_BUDGETS,
        help=f"MCTS expansions per env step (default: {DEFAULT_MCTS_BUDGETS})",
    )
    parser.add_argument(
        "--k-greedy", type=int, default=K_GREEDY,
        help=f"candidates for greedy DV-MCSS (default {K_GREEDY}); "
             "set to --k-mcts for a matched-K control",
    )
    parser.add_argument(
        "--k-mcts", type=int, default=K_MCTS,
        help=f"candidates per MCTS expansion (default {K_MCTS})",
    )
    parser.add_argument(
        "--skip-greedy", action="store_true",
        help="Skip greedy DV-MCSS rollout (use Phase 0 baseline for comparison only)",
    )
    parser.add_argument(
        "--skip-mcts", action="store_true",
        help="Skip MCTS rollouts (e.g. to run only a greedy-K control)",
    )
    args = parser.parse_args()

    K_GREEDY, K_MCTS = args.k_greedy, args.k_mcts

    configure_env_paths(args.env)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    check_prerequisites()

    env = gym.make(ENV_NAME)
    # Episode length from the env's own TimeLimit (umaze 300 / medium 600 / large 800)
    # so larger mazes are never silently truncated.
    MAX_T = env._max_episode_steps
    dataset = DV_D4RLMaze2DSeqDataset(
        env.get_dataset(), horizon=H, stride=M,
        learn_policy=False, center_mapping=False,
        discount=1.0, continous_reward_at_done=True, reward_tune="iql",
    )
    normalizer = dataset.get_normalizer()
    planner, critic, policy = build_models(dataset)

    print(f"Env: {ENV_NAME}  |  MAX_T={MAX_T}  |  output→{OUTPUT_DIR}  |  baseline={BASELINE_JSON}")

    rollout_cfg = RolloutConfig(
        obs_dim=OBS_DIM, act_dim=ACT_DIM, child_state_index=CHILD_IDX,
        plan_steps=PLAN_STEPS, policy_steps=POLICY_STEPS,
        max_t=MAX_T, device=DEVICE,
    )

    methods = []
    if not args.skip_greedy:
        methods.append(("greedy", None))
    if not args.skip_mcts:
        for budget in args.mcts_budgets:
            methods.append(("mcts", budget))

    total_runs = len(methods) * len(args.seeds)
    print(
        f"\nRunning {total_runs} episodes  "
        f"({len(methods)} methods × {len(args.seeds)} seeds)\n"
    )

    results: list[EpisodeResult] = []
    run_idx = 0

    for method_name, budget in methods:
        if method_name == "greedy":
            expansion = make_expansion(planner, critic, K_GREEDY)
            # Encode K so a matched-K control (e.g. K=10) is not saved as the
            # same "DV-MCSS" label as the K=50 baseline.
            label = "DV-MCSS" if K_GREEDY == 50 else f"DV-MCSS-K{K_GREEDY}"
        else:
            expansion = make_expansion(planner, critic, K_MCTS)
            tree_cfg = make_tree_cfg(budget)
            label = f"MCTS-K{K_MCTS}-exp{budget}"  # matches saved method label

        for seed in args.seeds:
            run_idx += 1
            env.seed(seed)
            env.action_space.seed(seed)
            set_seed(seed)

            print(
                f"[{run_idx:>2}/{total_runs}] {label:<22}  seed={seed}",
                end="  ", flush=True,
            )
            t0 = time.time()

            if method_name == "greedy":
                result = run_greedy_episode(env, expansion, policy, normalizer,
                                            rollout_cfg, seed=seed)
                if label != "DV-MCSS":          # matched-K control: keep label distinct
                    result = dataclasses.replace(result, method=label)
            else:
                result = run_mcts_episode(env, expansion, policy, normalizer,
                                          tree_cfg, rollout_cfg, seed=seed)

            elapsed = time.time() - t0
            print(
                f"return={result.raw_return:.0f}  "
                f"score={result.normalized_score:.1f}  "
                f"goal_step={result.goal_step}  "
                f"denoise={result.denoising_calls}  "
                f"time={elapsed:.1f}s"
            )
            results.append(result)

    json_path = save_results(results)
    csv_path  = save_summary(results)
    baseline  = load_baseline()

    print(f"\nResults → {json_path}")
    print(f"Summary → {csv_path}")
    print_comparison(results, baseline)


if __name__ == "__main__":
    main()
