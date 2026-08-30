"""scripts/phase4_ablation.py

Phase 4 diagnostic ablations — pinpoint why MCTS underperformed greedy.

Ablation B — K-mismatch test
----------------------------------------------------------------------
Hypothesis: Phase 4 MCTS used K=10 while greedy used K=50.  MCTS with
K=50 and budget=1 should behave identically to greedy (same one-shot
expansion, same best trajectory, same waypoint).  Any remaining gap
means best_path()[1] extraction is worse than direct argmax.

    Config          K    Budget   Expected depth
    MCTS-K50-exp1   50   1        1

Decision rule:
    ≈ Greedy  →  K reduction caused the gap.  Try Ablation C.
    < Greedy  →  Waypoint extraction (value-mean vs argmax) is the issue.
                 Fix best_path before running Ablation C.

Ablation C — Depth / budget sweep
----------------------------------------------------------------------
Hypothesis: Sufficient budget lets the tree reach depth ≥ 3, where
meaningful multi-step look-ahead can improve over one-shot greedy.

    Config            K    Budget   Expected depth
    MCTS-K10-exp12    10   12       3
    MCTS-K10-exp22    10   22       3–4
    MCTS-K10-exp52    10   52       5–6
    MCTS-K10-exp102   10   102      ~10

Finding: depth stays at 3 for budgets 12–52 because K=10 creates K²=100
depth-2 nodes; budget ≥ 1+10+100=111 is needed to exhaust depth-2.
exp12 marginally beats greedy (+5.65); exp22 catastrophe at seed=1
is likely random UCB tie-breaking pathology.

Ablation D — Fan-out and tie-breaking diagnosis
----------------------------------------------------------------------
Two hypotheses tested simultaneously:

D1 — Tie-breaking:
    ucb_tie_breaking="greedy" vs "random" at K=10 budgets 12 and 22.
    If exp22 catastrophe disappears under "greedy": random UCB is the
    root cause. If it persists: budget/depth level itself is the issue.

D2 — K=5 fan-out:
    With K=5, depth-2 exhaustion needs only 1+5+25=31 expansions
    (vs 111 for K=10). Budget=31 gives full depth-2 coverage and
    reliable backprop signal. Compare:
        K=5, budgets [6, 12, 31, 52]  (special: 6=depth-2 min, 31=depth-2 full)
    vs K=10-exp12 (current best: 112.85).

    Fan-out table:
        K=5  depth-2 full at budget=31  → first reliable multi-step signal
        K=10 depth-2 full at budget=111 → needs 9× more compute than K=5
        K=5  depth-3 full at budget=156 → ambitious but tractable on GPU

Decision rule:
    Score ↑ monotonically  →  Tree search works; more budget = better.
    Score peaks then ↓     →  Critic noise compounds through backprop.
                               There is an optimal depth (probably shallow).
    Score flat or ↓        →  Critic cannot guide multi-step search.
                               Address critic reliability (Phase 5).

Run:
    python scripts/phase4_ablation.py --ablation B
    python scripts/phase4_ablation.py --ablation C
    python scripts/phase4_ablation.py --ablation D
    python scripts/phase4_ablation.py --ablation B C --seeds 0 1 2
    python scripts/phase4_ablation.py --ablation C --budgets-c 12 52 102
    python scripts/phase4_ablation.py --ablation D --tie-breakings greedy random
    python scripts/phase4_ablation.py --ablation D --k5-budgets 6 31 52
    python scripts/phase4_ablation.py --ablation B --skip-greedy
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import List, Optional

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

# ── Constants ─────────────────────────────────────────────────────────────────

ENV_NAME     = "maze2d-umaze-v1"
OBS_DIM      = 4
ACT_DIM      = 2
H            = 32
M            = 15
CHILD_IDX    = 1
PLAN_STEPS   = 20
POLICY_STEPS = 10
MAX_T        = 300
DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"

CKPT = (
    "results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer"
    "_d2_width256_separate_dpTrue/maze2d-umaze-v1"
)
REQUIRED = [
    f"{CKPT}/planner_ckpt_1000000.pt",
    f"{CKPT}/critic_ckpt_1000000.pt",
    f"{CKPT}/policy_ckpt_1000000.pt",
]
OUTPUT_DIR    = "results/phase4_ablation"
BASELINE_JSON = "results/phase0_baseline.json"

# Ablation B: K=50, budget=1 — should match greedy
ABLATION_B_CONFIGS: list[tuple[int, int]] = [(50, 1)]

# Ablation C: K=10, sweep budgets for depth 3 → ~10
ABLATION_C_K       = 10
ABLATION_C_BUDGETS = [12, 22, 52, 102]

# Ablation D: fan-out and tie-breaking diagnosis
# D1 — tie-breaking: K=10 at budgets 12 and 22, both "greedy" and "random"
ABLATION_D1_K           = 10
ABLATION_D1_BUDGETS     = [12, 22]
ABLATION_D1_TIE_DEFAULT = ["greedy", "random"]
# D2 — K=5 sweep: special budgets match fan-out milestones
#   6  = 1+5     → depth-2 first reached (minimum)
#   12 = 1+5+6   → 6 depth-2 nodes explored (24% of 25)
#   31 = 1+5+25  → depth-2 fully exhausted → first reliable signal
#   52 = 31+21   → 21 depth-3 nodes explored (16.8% of 125)
ABLATION_D2_K           = 5
ABLATION_D2_BUDGETS     = [6, 12, 31, 52]

DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# Ablation E: uncertainty penalty sweep (Phase 5)
# K=10, budget=12, vary uncertainty_beta; seeds 0-14 for matched comparison.
# β=0 reproduces the RNG-fixed K=10-exp12 baseline.
# Sweep covers low (0.5), medium (1.0, 2.0), and aggressive (5.0) penalisation.
ABLATION_E_K         = 10
ABLATION_E_BUDGET    = 12
ABLATION_E_BETAS     = [0.5, 1.0, 2.0, 5.0]
ABLATION_E_SEEDS_ALL = list(range(15))   # 0-14 for full matched comparison

# ── Model loading ──────────────────────────────────────────────────────────────

def check_prerequisites() -> None:
    missing = [p for p in REQUIRED if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"[FAIL] Missing: {p}")
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


def make_expansion(planner, critic, k: int, beta: float = 0.0) -> PlannerExpansion:
    cfg = ExpansionConfig(
        K=k, horizon=H, obs_dim=OBS_DIM, planner_dim=OBS_DIM,
        solver="ddim", sample_steps=PLAN_STEPS, temperature=1.0,
        use_ema=True, device=DEVICE,
        uncertainty_beta=beta,
    )
    return PlannerExpansion(planner, critic, cfg)


def make_tree_cfg(k: int, budget: int, tie_breaking: str = "random") -> TreeConfig:
    # leaf_batch_size=10 keeps GPU utilisation reasonable for large budgets.
    # For budget=1, leaf_batch_size is clamped internally to min(10, budget)=1.
    return TreeConfig(
        obs_dim=OBS_DIM, horizon=H, child_state_index=CHILD_IDX,
        K=k, ucb_c=1.414, storage_mode="state_only",
        max_expansions=budget, device=DEVICE,
        leaf_batch_size=10, ucb_tie_breaking=tie_breaking,
    )


# ── I/O ────────────────────────────────────────────────────────────────────────

def save_results(results: list[EpisodeResult], ablation: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"ablation_{ablation}_results.json")
    rows = [r.to_dict() for r in results]
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
        seen = {(r["method"], r["seed"]) for r in existing}
        rows = existing + [r for r in rows if (r["method"], r["seed"]) not in seen]
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    return path


def save_summary_csv(results: list[EpisodeResult], ablation: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"ablation_{ablation}_summary.csv")
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


# ── Reporting ──────────────────────────────────────────────────────────────────

def _mean(values) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def print_table(results: list[EpisodeResult], baseline: list[dict], ablation: str) -> None:
    grouped: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        grouped[r.method].append(r)

    col_w = 90
    print()
    print("=" * col_w)
    print(f"Ablation {ablation} Summary")
    print("=" * col_w)
    print(
        f"{'Method':<26}  {'n':>3}  {'norm_score':>11}  {'raw_return':>11}  "
        f"{'denoise':>9}  {'depth':>7}  {'cum_best':>9}  {'ms/step':>8}"
    )
    print("-" * col_w)

    if baseline:
        print(
            f"{'DV-MCSS (Phase 0 ref)':<26}  {len(baseline):>3}  "
            f"{_mean([r['normalized_score'] for r in baseline]):>11.2f}  "
            f"{_mean([r['raw_return'] for r in baseline]):>11.1f}  "
            f"{_mean([r['denoising_calls'] for r in baseline]):>9.0f}  "
            f"{'—':>7}  {'—':>9}  "
            f"{_mean([r['ms_per_step'] for r in baseline]):>8.1f}"
        )

    for method in sorted(grouped.keys()):
        rs = grouped[method]
        depth_str = (
            f"{_mean([r.mean_tree_depth for r in rs]):.2f}"
            if any(r.mean_tree_depth is not None for r in rs)
            else "—"
        )
        best_str = (
            f"{_mean([r.mean_cumulative_best for r in rs]):.4f}"
            if any(r.mean_cumulative_best is not None for r in rs)
            else "—"
        )
        print(
            f"{method:<26}  {len(rs):>3}  "
            f"{_mean([r.normalized_score for r in rs]):>11.2f}  "
            f"{_mean([r.raw_return for r in rs]):>11.1f}  "
            f"{_mean([r.denoising_calls for r in rs]):>9.0f}  "
            f"{depth_str:>7}  {best_str:>9}  "
            f"{_mean([r.ms_per_step for r in rs]):>8.1f}"
        )

    print("=" * col_w)
    _print_decision_rule(ablation, grouped, baseline)


def _print_decision_rule(
    ablation: str,
    grouped: dict[str, list[EpisodeResult]],
    baseline: list[dict],
) -> None:
    """Print automated decision-rule verdict based on results."""
    if ablation == "B":
        greedy_score = _mean([r["normalized_score"] for r in baseline]) if baseline else None
        mcts_k50 = grouped.get("MCTS-K50-exp1", [])
        if not mcts_k50 or greedy_score is None:
            return
        mcts_score = _mean([r.normalized_score for r in mcts_k50])
        delta = mcts_score - greedy_score
        n_mcts = len(mcts_k50)
        n_ref  = len(baseline)
        print()
        print("Decision (Ablation B):")
        print(
            f"  NOTE: comparing n={n_mcts} MCTS runs against n={n_ref} Phase 0 reference.\n"
            f"  maze2d-umaze scores range ~69–122; use ≥5 matched seeds for a firm verdict."
        )
        # Threshold is ±12 — roughly half the env's inter-seed std (~20 pts).
        # Comparing different seed sets shifts the mean by ±10 even with identical policies.
        if abs(delta) <= 12.0:
            print(
                f"  MCTS-K50-exp1 ({mcts_score:.1f}) ≈ Greedy ({greedy_score:.1f})  Δ={delta:+.1f}"
            )
            print(
                "  → Within expected variance: MCTS machinery at depth=1 is neutral.\n"
                "    K reduction (50→10) caused the Phase 4 gap.  Proceed to Ablation C."
            )
        elif delta < -12.0:
            print(
                f"  MCTS-K50-exp1 ({mcts_score:.1f}) << Greedy ({greedy_score:.1f})  Δ={delta:+.1f}"
            )
            print(
                "  → Systematic gap: best_path()[1] extraction is worse than direct argmax.\n"
                "    Investigate value() for unvisited nodes before running Ablation C."
            )
        else:
            print(
                f"  MCTS-K50-exp1 ({mcts_score:.1f}) > Greedy ({greedy_score:.1f})  Δ={delta:+.1f}"
            )
            print("  → MCTS tree structure already helps at depth 1.")

    elif ablation == "C":
        greedy_score = _mean([r["normalized_score"] for r in baseline]) if baseline else None
        c_methods = {k: v for k, v in grouped.items() if k.startswith("MCTS-K10-exp")}
        if not c_methods:
            return
        scores = {
            m: _mean([r.normalized_score for r in rs])
            for m, rs in sorted(c_methods.items())
        }
        score_list = list(scores.values())
        print()
        print("Decision (Ablation C):")
        for method, score in scores.items():
            ref = f"  (greedy ref: {greedy_score:.1f})" if greedy_score else ""
            print(f"  {method}: {score:.1f}{ref}")
        if len(score_list) >= 2:
            increasing = all(b >= a - 1.0 for a, b in zip(score_list, score_list[1:]))
            decreasing_after_peak = (
                max(score_list) > score_list[0] + 2.0
                and score_list[-1] < max(score_list) - 2.0
            )
            flat = max(score_list) - min(score_list) < 3.0
            print()
            if flat:
                print(
                    "  → Score flat across all budgets.  Critic cannot guide multi-step\n"
                    "    search.  Deeper trees add compute with no return gain.\n"
                    "    Proceed to Phase 5 (uncertainty penalty / critic recalibration)."
                )
            elif increasing:
                print(
                    "  → Score improves monotonically with budget.  Tree search works.\n"
                    "    More budget = better performance.  Run at higher budgets."
                )
            elif decreasing_after_peak:
                best_method = max(scores, key=scores.get)
                print(
                    f"  → Score peaks at {best_method} then degrades.\n"
                    "    Critic noise compounds through deep backpropagation.\n"
                    f"    Optimal budget ≈ {best_method.split('exp')[-1]} expansions per step."
                )
            else:
                print("  → Mixed result — inspect individual seeds before concluding.")


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_ablation_b(
    env, planner, critic, policy, normalizer,
    rollout_cfg: RolloutConfig,
    seeds: list[int],
    skip_greedy: bool,
) -> list[EpisodeResult]:
    """Ablation B: MCTS-K50-exp1 vs Greedy-K50."""
    results: list[EpisodeResult] = []
    configs: list[tuple[str, int, Optional[int]]] = []

    if not skip_greedy:
        configs.append(("greedy", 50, None))
    for k, budget in ABLATION_B_CONFIGS:
        configs.append(("mcts", k, budget))

    total = len(configs) * len(seeds)
    run_idx = 0
    for method_name, k, budget in configs:
        exp = make_expansion(planner, critic, k)
        if method_name == "greedy":
            label = f"DV-MCSS (K={k})"
        else:
            label = f"MCTS-K{k}-exp{budget}"
            tree_cfg = make_tree_cfg(k, budget)

        for seed in seeds:
            run_idx += 1
            env.seed(seed)
            env.action_space.seed(seed)
            set_seed(seed)
            print(f"[B {run_idx:>2}/{total}] {label:<22}  seed={seed}", end="  ", flush=True)
            t0 = time.time()

            if method_name == "greedy":
                result = run_greedy_episode(env, exp, policy, normalizer, rollout_cfg, seed=seed)
            else:
                result = run_mcts_episode(env, exp, policy, normalizer, tree_cfg, rollout_cfg, seed=seed)

            elapsed = time.time() - t0
            print(
                f"return={result.raw_return:.0f}  score={result.normalized_score:.1f}  "
                f"depth={result.mean_tree_depth}  time={elapsed:.1f}s"
            )
            results.append(result)
    return results


def run_ablation_c(
    env, planner, critic, policy, normalizer,
    rollout_cfg: RolloutConfig,
    seeds: list[int],
    budgets: list[int],
) -> list[EpisodeResult]:
    """Ablation C: K=10, sweep budgets for increasing depth."""
    results: list[EpisodeResult] = []
    exp = make_expansion(planner, critic, ABLATION_C_K)
    total = len(budgets) * len(seeds)
    run_idx = 0
    for budget in budgets:
        tree_cfg = make_tree_cfg(ABLATION_C_K, budget)
        label = f"MCTS-K{ABLATION_C_K}-exp{budget}"
        for seed in seeds:
            run_idx += 1
            env.seed(seed)
            env.action_space.seed(seed)
            set_seed(seed)
            print(f"[C {run_idx:>2}/{total}] {label:<22}  seed={seed}", end="  ", flush=True)
            t0 = time.time()
            result = run_mcts_episode(
                env, exp, policy, normalizer, tree_cfg, rollout_cfg, seed=seed
            )
            elapsed = time.time() - t0
            print(
                f"return={result.raw_return:.0f}  score={result.normalized_score:.1f}  "
                f"depth={result.mean_tree_depth:.2f}  "
                f"denoise={result.denoising_calls}  time={elapsed:.1f}s"
            )
            results.append(result)
    return results


def run_ablation_d(
    env, planner, critic, policy, normalizer,
    rollout_cfg: RolloutConfig,
    seeds: list[int],
    tie_breakings: list[str],
    k5_budgets: list[int],
) -> tuple[list[EpisodeResult], list[EpisodeResult]]:
    """Ablation D: tie-breaking (D1) and K=5 fan-out (D2).

    Returns (results_d1, results_d2) so they can be saved and printed separately.
    """
    # ── D1: tie-breaking comparison ─────────────────────────────────────────────
    results_d1: list[EpisodeResult] = []
    exp10 = make_expansion(planner, critic, ABLATION_D1_K)
    d1_configs = [
        (budget, tie)
        for budget in ABLATION_D1_BUDGETS
        for tie in tie_breakings
    ]
    total_d1 = len(d1_configs) * len(seeds)
    run_idx = 0
    print(f"\n[D1] K={ABLATION_D1_K}, budgets={ABLATION_D1_BUDGETS}, tie-breakings={tie_breakings}")
    for budget, tie in d1_configs:
        tree_cfg = make_tree_cfg(ABLATION_D1_K, budget, tie_breaking=tie)
        # Encode tie-breaking in the method label so results are distinguishable
        label = f"MCTS-K{ABLATION_D1_K}-exp{budget}-{tie[:1].upper()}"
        for seed in seeds:
            run_idx += 1
            env.seed(seed)
            env.action_space.seed(seed)
            set_seed(seed)
            print(
                f"[D1 {run_idx:>2}/{total_d1}] {label:<26}  seed={seed}",
                end="  ", flush=True,
            )
            t0 = time.time()
            result = run_mcts_episode(
                env, exp10, policy, normalizer, tree_cfg, rollout_cfg, seed=seed
            )
            # Override method label to include tie-breaking marker
            result = EpisodeResult(
                method=label, seed=result.seed,
                raw_return=result.raw_return, normalized_score=result.normalized_score,
                goal_step=result.goal_step, episode_length=result.episode_length,
                denoising_calls=result.denoising_calls,
                wall_seconds=result.wall_seconds, ms_per_step=result.ms_per_step,
                mcts_budget=result.mcts_budget,
                mean_tree_depth=result.mean_tree_depth,
                mean_cumulative_best=result.mean_cumulative_best,
            )
            elapsed = time.time() - t0
            print(
                f"return={result.raw_return:.0f}  score={result.normalized_score:.1f}  "
                f"depth={result.mean_tree_depth:.2f}  time={elapsed:.1f}s"
            )
            results_d1.append(result)

    # ── D2: K=5 fan-out sweep ───────────────────────────────────────────────────
    results_d2: list[EpisodeResult] = []
    exp5 = make_expansion(planner, critic, ABLATION_D2_K)
    total_d2 = len(k5_budgets) * len(seeds)
    run_idx = 0
    print(f"\n[D2] K={ABLATION_D2_K}, budgets={k5_budgets}")
    for budget in k5_budgets:
        tree_cfg = make_tree_cfg(ABLATION_D2_K, budget)
        label = f"MCTS-K{ABLATION_D2_K}-exp{budget}"
        for seed in seeds:
            run_idx += 1
            env.seed(seed)
            env.action_space.seed(seed)
            set_seed(seed)
            print(
                f"[D2 {run_idx:>2}/{total_d2}] {label:<22}  seed={seed}",
                end="  ", flush=True,
            )
            t0 = time.time()
            result = run_mcts_episode(
                env, exp5, policy, normalizer, tree_cfg, rollout_cfg, seed=seed
            )
            elapsed = time.time() - t0
            print(
                f"return={result.raw_return:.0f}  score={result.normalized_score:.1f}  "
                f"depth={result.mean_tree_depth:.2f}  "
                f"denoise={result.denoising_calls}  time={elapsed:.1f}s"
            )
            results_d2.append(result)

    return results_d1, results_d2


def _print_decision_d(
    results_d1: list[EpisodeResult],
    results_d2: list[EpisodeResult],
    baseline: list[dict],
) -> None:
    greedy_score = _mean([r["normalized_score"] for r in baseline]) if baseline else None

    # D1 verdict: does greedy tie-breaking eliminate the catastrophe?
    grouped_d1: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results_d1:
        grouped_d1[r.method].append(r)
    if grouped_d1:
        print("\nDecision (D1 — tie-breaking):")
        # Check if any method has a return=0 catastrophe
        for method, rs in sorted(grouped_d1.items()):
            failures = [r for r in rs if r.raw_return == 0]
            mean_s = _mean([r.normalized_score for r in rs])
            tie = "greedy" if method.endswith("-G") else "random"
            fail_str = f"  CATASTROPHIC FAILURES: {len(failures)}/{len(rs)}" if failures else ""
            print(f"  {method}: mean={mean_s:.1f}{fail_str}")
        # Compare greedy vs random at budget=22 specifically
        exp22_g = grouped_d1.get(f"MCTS-K{ABLATION_D1_K}-exp22-G", [])
        exp22_r = grouped_d1.get(f"MCTS-K{ABLATION_D1_K}-exp22-R", [])
        if exp22_g and exp22_r:
            g22_fail = sum(1 for r in exp22_g if r.raw_return == 0)
            r22_fail = sum(1 for r in exp22_r if r.raw_return == 0)
            print()
            if g22_fail < r22_fail:
                print(
                    f"  → exp22 greedy: {g22_fail} failures  random: {r22_fail} failures\n"
                    "    Random UCB tie-breaking is the root cause of the exp22 catastrophe.\n"
                    "    Switch to greedy tie-breaking for production runs."
                )
            elif g22_fail == r22_fail == 0:
                print(
                    "  → No failures under either tie-breaking strategy at exp22.\n"
                    "    The C-run catastrophe was isolated bad luck (re-check C results)."
                )
            else:
                print(
                    f"  → exp22 greedy: {g22_fail} failures  random: {r22_fail} failures\n"
                    "    Failures persist under greedy tie-breaking.\n"
                    "    budget=22 itself is the instability source, not the tie-breaking."
                )

    # D2 verdict: how does K=5 compare with K=10-exp12?
    grouped_d2: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results_d2:
        grouped_d2[r.method].append(r)
    if grouped_d2 and greedy_score is not None:
        print(f"\nDecision (D2 — K=5 fan-out, vs greedy ref={greedy_score:.1f}):")
        k10_exp12_score = 112.85  # from Ablation C
        best_k5_method = None
        best_k5_score  = -float("inf")
        for method, rs in sorted(grouped_d2.items()):
            mean_s = _mean([r.normalized_score for r in rs])
            depth  = _mean([r.mean_tree_depth for r in rs])
            marker = " ← new best" if mean_s > k10_exp12_score else ""
            print(f"  {method}: score={mean_s:.1f}  depth={depth:.2f}{marker}")
            if mean_s > best_k5_score:
                best_k5_score  = mean_s
                best_k5_method = method
        print()
        if best_k5_score > k10_exp12_score + 3.0:
            print(
                f"  → K=5 ({best_k5_method}: {best_k5_score:.1f}) beats K=10-exp12 "
                f"({k10_exp12_score:.1f}).\n"
                "    Smaller fan-out provides better depth-2 coverage at the same compute.\n"
                "    Use K=5 as the production setting."
            )
        elif best_k5_score > greedy_score + 3.0:
            print(
                f"  → K=5 beats greedy ({greedy_score:.1f}) but not K=10-exp12.\n"
                "    K=10 depth-3 signal is more valuable than K=5 depth-3 coverage."
            )
        else:
            print(
                f"  → K=5 does not clearly beat greedy ({greedy_score:.1f}).\n"
                "    More root candidates (larger K) are more important than faster deepening.\n"
                "    Consider K=10-exp12 as the production sweet spot."
            )


def run_ablation_e(
    env, planner, critic, policy, normalizer,
    rollout_cfg: RolloutConfig,
    seeds: list[int],
    betas: list[float],
) -> list[EpisodeResult]:
    """Ablation E: uncertainty penalty sweep.

    Runs MCTS-K10-exp12 with each β in betas.  β=0 is the unpenalised baseline
    (same as RNG-fixed K=10-exp12 from Ablation C).  Results include
    β-encoded method label so all runs can be compared in one table.
    """
    results: list[EpisodeResult] = []
    tree_cfg = make_tree_cfg(ABLATION_E_K, ABLATION_E_BUDGET)
    total = len(betas) * len(seeds)
    run_idx = 0
    for beta in betas:
        exp = make_expansion(planner, critic, ABLATION_E_K, beta=beta)
        label = f"MCTS-K{ABLATION_E_K}-exp{ABLATION_E_BUDGET}-b{beta}"
        for seed in seeds:
            run_idx += 1
            env.seed(seed)
            env.action_space.seed(seed)
            set_seed(seed)
            print(
                f"[E {run_idx:>3}/{total}] {label:<28}  seed={seed}",
                end="  ", flush=True,
            )
            t0 = time.time()
            result = run_mcts_episode(
                env, exp, policy, normalizer, tree_cfg, rollout_cfg, seed=seed
            )
            # Override method label to include beta
            result = EpisodeResult(
                method=label, seed=result.seed,
                raw_return=result.raw_return, normalized_score=result.normalized_score,
                goal_step=result.goal_step, episode_length=result.episode_length,
                denoising_calls=result.denoising_calls,
                wall_seconds=result.wall_seconds, ms_per_step=result.ms_per_step,
                mcts_budget=result.mcts_budget,
                mean_tree_depth=result.mean_tree_depth,
                mean_cumulative_best=result.mean_cumulative_best,
            )
            elapsed = time.time() - t0
            print(
                f"return={result.raw_return:.0f}  score={result.normalized_score:.1f}  "
                f"time={elapsed:.1f}s"
            )
            results.append(result)
    return results


def _print_decision_e(
    results: list[EpisodeResult],
    baseline_scores: dict[int, float],
) -> None:
    """Print per-beta verdict for Ablation E.

    baseline_scores: {seed: greedy_normalized_score} for matched comparison.
    """
    grouped: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        grouped[r.method].append(r)

    print("\nDecision (E — uncertainty penalty, per-seed matched vs greedy):")
    print(
        f"  {'Method':<30}  {'mean':>7}  {'vs greedy':>10}  "
        f"{'fails(score<0)':>15}  {'wins(MCTS>greedy)':>18}"
    )
    print("  " + "-" * 85)

    greedy_mean = _mean(list(baseline_scores.values())) if baseline_scores else None
    best_method, best_score = None, -float("inf")

    for method in sorted(grouped.keys()):
        rs = grouped[method]
        mean_s = _mean([r.normalized_score for r in rs])
        failures = sum(1 for r in rs if r.normalized_score < 0)

        if baseline_scores:
            paired_deltas = [
                r.normalized_score - baseline_scores[r.seed]
                for r in rs if r.seed in baseline_scores
            ]
            wins = sum(1 for d in paired_deltas if d > 0)
            vs_str = f"{_mean(paired_deltas):+.1f}"
            wins_str = f"{wins}/{len(paired_deltas)}"
        else:
            vs_str, wins_str = "—", "—"

        print(
            f"  {method:<30}  {mean_s:>7.1f}  {vs_str:>10}  "
            f"{failures:>15}  {wins_str:>18}"
        )
        if mean_s > best_score:
            best_score, best_method = mean_s, method

    print()
    if greedy_mean is not None and best_method is not None:
        if best_score > greedy_mean + 5.0:
            print(
                f"  → {best_method} beats greedy by {best_score - greedy_mean:.1f} pts.\n"
                "    Uncertainty penalty improves MCTS reliability.  Use this β."
            )
        elif best_score > greedy_mean - 3.0:
            print(
                f"  → Best β ({best_method}: {best_score:.1f}) ≈ greedy ({greedy_mean:.1f}).\n"
                "    Penalty prevents catastrophic failures without clear gain.\n"
                "    Consider as a robustness measure even if mean is similar."
            )
        else:
            print(
                f"  → No β yields reliable improvement over greedy ({greedy_mean:.1f}).\n"
                "    Critic recalibration (train a new critic) is needed before\n"
                "    MCTS can robustly outperform single-shot planning."
            )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 ablations: diagnose MCTS underperformance"
    )
    parser.add_argument(
        "--ablation", nargs="+", choices=["B", "C", "D", "E"], default=["B", "C"],
        help="Which ablation(s) to run (default: B C)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help=f"Seeds to evaluate (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--budgets-c", type=int, nargs="+", default=ABLATION_C_BUDGETS,
        help=f"Budget values for Ablation C (default: {ABLATION_C_BUDGETS})",
    )
    parser.add_argument(
        "--tie-breakings", nargs="+", default=ABLATION_D1_TIE_DEFAULT,
        choices=["greedy", "random"],
        help="UCB tie-breaking strategies for Ablation D1 (default: greedy random)",
    )
    parser.add_argument(
        "--k5-budgets", type=int, nargs="+", default=ABLATION_D2_BUDGETS,
        help=f"Budget values for Ablation D2 K=5 sweep (default: {ABLATION_D2_BUDGETS})",
    )
    parser.add_argument(
        "--betas", type=float, nargs="+", default=ABLATION_E_BETAS,
        help=f"Uncertainty penalty β values for Ablation E (default: {ABLATION_E_BETAS})",
    )
    parser.add_argument(
        "--skip-greedy", action="store_true",
        help="Skip re-running greedy in Ablation B; use Phase 0 reference only",
    )
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    check_prerequisites()

    env = gym.make(ENV_NAME)
    dataset = DV_D4RLMaze2DSeqDataset(
        env.get_dataset(), horizon=H, stride=M,
        learn_policy=False, center_mapping=False,
        discount=1.0, continous_reward_at_done=True, reward_tune="iql",
    )
    normalizer = dataset.get_normalizer()
    planner, critic, policy = build_models(dataset)

    rollout_cfg = RolloutConfig(
        obs_dim=OBS_DIM, act_dim=ACT_DIM, child_state_index=CHILD_IDX,
        plan_steps=PLAN_STEPS, policy_steps=POLICY_STEPS,
        max_t=MAX_T, device=DEVICE,
    )

    baseline = load_baseline()

    if "B" in args.ablation:
        print("\n" + "=" * 60)
        print("Ablation B: K-mismatch test (K=50, budget=1)")
        print("=" * 60)
        results_b = run_ablation_b(
            env, planner, critic, policy, normalizer, rollout_cfg,
            args.seeds, args.skip_greedy,
        )
        json_b = save_results(results_b, "B")
        csv_b  = save_summary_csv(results_b, "B")
        print(f"\nAblation B → {json_b}")
        print(f"Summary   → {csv_b}")
        print_table(results_b, baseline, "B")

    if "C" in args.ablation:
        print("\n" + "=" * 60)
        print(f"Ablation C: depth sweep  K={ABLATION_C_K}, budgets={args.budgets_c}")
        print("=" * 60)
        results_c = run_ablation_c(
            env, planner, critic, policy, normalizer, rollout_cfg,
            args.seeds, args.budgets_c,
        )
        json_c = save_results(results_c, "C")
        csv_c  = save_summary_csv(results_c, "C")
        print(f"\nAblation C → {json_c}")
        print(f"Summary   → {csv_c}")
        print_table(results_c, baseline, "C")

    if "D" in args.ablation:
        print("\n" + "=" * 60)
        print("Ablation D: tie-breaking (D1) + K=5 fan-out (D2)")
        print("=" * 60)
        results_d1, results_d2 = run_ablation_d(
            env, planner, critic, policy, normalizer, rollout_cfg,
            args.seeds, args.tie_breakings, args.k5_budgets,
        )
        json_d1 = save_results(results_d1, "D1")
        csv_d1  = save_summary_csv(results_d1, "D1")
        json_d2 = save_results(results_d2, "D2")
        csv_d2  = save_summary_csv(results_d2, "D2")
        print(f"\nAblation D1 → {json_d1}")
        print(f"Ablation D2 → {json_d2}")
        print_table(results_d1, baseline, "D1")
        print_table(results_d2, baseline, "D2")
        _print_decision_d(results_d1, results_d2, baseline)

    if "E" in args.ablation:
        print("\n" + "=" * 60)
        print(f"Ablation E: uncertainty penalty  K={ABLATION_E_K}, "
              f"budget={ABLATION_E_BUDGET}, β={args.betas}")
        print("=" * 60)

        # Build matched greedy baseline {seed: score} from Ablation B saved JSON
        b_json = os.path.join(OUTPUT_DIR, "ablation_B_results.json")
        baseline_scores: dict[int, float] = {}
        if os.path.exists(b_json):
            with open(b_json) as f:
                b_rows = json.load(f)
            for r in b_rows:
                if r["method"] == "DV-MCSS":
                    baseline_scores[r["seed"]] = r["normalized_score"]

        e_seeds = args.seeds if args.seeds != DEFAULT_SEEDS else ABLATION_E_SEEDS_ALL
        results_e = run_ablation_e(
            env, planner, critic, policy, normalizer, rollout_cfg,
            e_seeds, args.betas,
        )
        json_e = save_results(results_e, "E")
        csv_e  = save_summary_csv(results_e, "E")
        print(f"\nAblation E → {json_e}")
        print(f"Summary   → {csv_e}")
        print_table(results_e, baseline, "E")
        _print_decision_e(results_e, baseline_scores)


if __name__ == "__main__":
    main()
