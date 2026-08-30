"""scripts/phase3_k_ablation.py

Phase 3 K ablation: vary K (candidates per expansion) with fixed total evaluation budget.
Also sweeps leaf_batch_size and UCB tie-breaking strategy so all three dimensions are
measured in one run.

For a fair compute comparison each K uses:
    max_expansions = total_evals // K

so every configuration evaluates the same total number of trajectories from the planner.
Storage mode is fixed to state_only (fastest; all modes give identical search results).

Results written to:
    results/phase3/k_ablation_K{k}_seed{seed}_tie{tie}_batch{batch}.csv
    results/phase3/k_summary.csv

Run (inside Docker):
    python scripts/phase3_k_ablation.py
    python scripts/phase3_k_ablation.py --k-values 5 10 20 50 --total-evals 15000 \\
        --seeds 42 0 123 --leaf-batch-sizes 1 10 --tie-breakings random greedy
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, ".")

import d4rl  # noqa: F401
import gym
import torch

from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import DVHorizonCritic
from mcts.expansion import ExpansionConfig, PlannerExpansion
from mcts.node import TreeConfig
from mcts.tree import MCTSTree, StepRecord

# ── Constants ──────────────────────────────────────────────────────────────────

ENV_NAME = "maze2d-umaze-v1"
OBS_DIM = 4
H = 32
M = 15
CHILD_IDX = 1
STORAGE_MODE = "state_only"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CKPT = (
    "results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer"
    "_d2_width256_separate_dpTrue/maze2d-umaze-v1"
)
REQUIRED = [
    f"{CKPT}/planner_ckpt_1000000.pt",
    f"{CKPT}/critic_ckpt_1000000.pt",
    "results/phase1/per_state_results.csv",
]
OUTPUT_DIR = "results/phase3"

DEFAULT_K_VALUES = [5, 10, 20, 50]
DEFAULT_SEEDS = [42, 0, 123]
DEFAULT_TOTAL_EVALS = 15_000
DEFAULT_LEAF_BATCH_SIZES = [1, 10]
DEFAULT_TIE_BREAKINGS = ["random", "greedy"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def check_prerequisites() -> None:
    missing = [p for p in REQUIRED if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"[FAIL] Required file not found: {p}")
        sys.exit(1)


def load_root_state() -> torch.Tensor:
    import csv as _csv
    print("Loading dataset for normalised start state …")
    env_data = gym.make(ENV_NAME)
    dataset = DV_D4RLMaze2DSeqDataset(
        env_data.get_dataset(), horizon=H, stride=M,
        learn_policy=False, center_mapping=False,
        discount=1.0, continous_reward_at_done=True, reward_tune="iql",
    )
    with open("results/phase1/per_state_results.csv") as f:
        rows = list(_csv.DictReader(f))
    row = rows[0]
    traj_idx, offset = int(row["traj_idx"]), int(row["offset"])
    s_norm = torch.tensor(dataset.seq_obs[traj_idx, offset], dtype=torch.float32)
    print(f"  Root state (traj={traj_idx}, offset={offset}): {s_norm.tolist()}")
    return s_norm


def build_models() -> tuple:
    print(f"Loading planner and critic on {DEVICE} …")
    nn_diff = DiT1d(
        OBS_DIM, emb_dim=128, d_model=256, n_heads=4, depth=2,
        timestep_emb_type="fourier",
    )
    fix_mask = torch.zeros((H, OBS_DIM))
    fix_mask[0, :OBS_DIM] = 1.0
    planner = ContinuousDiffusionSDE(
        nn_diff, nn_condition=None, fix_mask=fix_mask,
        loss_weight=torch.ones((H, OBS_DIM)),
        ema_rate=0.9999, device=DEVICE,
        predict_noise=True, noise_schedule="linear",
    )
    planner.load(f"{CKPT}/planner_ckpt_1000000.pt")
    planner.eval()

    critic_ckpt = torch.load(f"{CKPT}/critic_ckpt_1000000.pt", map_location=DEVICE)
    critic = DVHorizonCritic(
        OBS_DIM, emb_dim=128, d_model=256, n_heads=4, depth=2, norm_type="pre",
    ).to(DEVICE)
    critic.load_state_dict(critic_ckpt["critic"])
    critic.eval()

    print(f"  Loaded on {DEVICE}.")
    return planner, critic


def make_expansion(planner, critic, k: int) -> PlannerExpansion:
    cfg = ExpansionConfig(
        K=k, horizon=H, obs_dim=OBS_DIM, planner_dim=OBS_DIM,
        solver="ddim", sample_steps=20, temperature=1.0,
        use_ema=True, device=DEVICE,
    )
    return PlannerExpansion(planner, critic, cfg)


def gpu_peak_mb() -> float:
    """Peak GPU memory allocated since last reset, in MB. Returns 0 on CPU."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e6


def save_run_csv(records: list[StepRecord], k: int, seed: int,
                 tie: str, batch: int) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"k_ablation_K{k}_seed{seed}_tie{tie}_batch{batch}.csv")
    fields = [f.name for f in dataclasses.fields(StepRecord)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(dataclasses.asdict(rec))
    return path


def save_summary_csv(rows: list[dict]) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "k_summary.csv")
    if not rows:
        return path
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    return mean, std


def print_summary(summary_rows: list[dict], k_values: list[int],
                  tie_breakings: list[str], batch_sizes: list[int],
                  total_evals: int) -> None:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in summary_rows:
        grouped[(row["batch_size"], row["tie_breaking"], row["K"])].append(row)

    print()
    print("=" * 130)
    print(f"Phase 3 K Ablation Summary  (total_evals={total_evals}, mean ± std across seeds)")
    print("=" * 130)
    print(
        f"{'Batch':>6}  {'Tie':>7}  {'K':>5}  {'budget':>8}  {'n':>3}  "
        f"{'depth':>6}  {'best_score':>22}  {'traj/s':>8}  "
        f"{'time_s':>8}  {'peak_GPU_MB':>12}  {'n_nodes':>8}"
    )
    print("-" * 130)
    for batch in batch_sizes:
        for tie in tie_breakings:
            for k in k_values:
                rows = grouped.get((batch, tie, k), [])
                if not rows:
                    continue
                mean_score, std_score = _mean_std([r["cumulative_best"] for r in rows])
                mean_time, _ = _mean_std([r["wall_time"] for r in rows])
                mean_depth, _ = _mean_std([r["tree_depth"] for r in rows])
                mean_nodes, _ = _mean_std([r["n_nodes"] for r in rows])
                mean_tps, _ = _mean_std([r["traj_per_sec"] for r in rows])
                mean_gpu, _ = _mean_std([r["peak_gpu_mb"] for r in rows])
                budget = rows[0]["budget"]
                score_str = f"{mean_score:.4f} ± {std_score:.4f}"
                print(
                    f"{batch:>6}  {tie:>7}  {k:>5}  {budget:>8}  {len(rows):>3}  "
                    f"{mean_depth:>6.1f}  {score_str:>22}  {mean_tps:>8.1f}  "
                    f"{mean_time:>8.1f}  {mean_gpu:>12.1f}  {int(mean_nodes):>8,}"
                )
            print()
    print("=" * 130)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 K ablation — fixed total evaluations, varying K, batch size, tie-breaking"
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES,
    )
    parser.add_argument(
        "--total-evals", type=int, default=DEFAULT_TOTAL_EVALS,
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--ucb-c", type=float, default=math.sqrt(2),
    )
    parser.add_argument(
        "--leaf-batch-sizes", type=int, nargs="+", default=DEFAULT_LEAF_BATCH_SIZES,
        help=f"Leaf batch sizes to sweep (default: {DEFAULT_LEAF_BATCH_SIZES})",
    )
    parser.add_argument(
        "--tie-breakings", type=str, nargs="+", default=DEFAULT_TIE_BREAKINGS,
        choices=["random", "greedy"],
        help=f"UCB tie-breaking strategies to sweep (default: {DEFAULT_TIE_BREAKINGS})",
    )
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    check_prerequisites()
    root_s = load_root_state()
    planner, critic = build_models()

    total_runs = (
        len(args.leaf_batch_sizes) * len(args.tie_breakings)
        * len(args.k_values) * len(args.seeds)
    )
    print(
        f"\nRunning {total_runs} experiments  "
        f"({len(args.leaf_batch_sizes)} batch sizes × {len(args.tie_breakings)} ties "
        f"× {len(args.k_values)} K values × {len(args.seeds)} seeds)\n"
        f"  Storage mode : {STORAGE_MODE}\n"
        f"  Total evals  : {args.total_evals}  (budget = total_evals // K)\n"
    )

    summary_rows: list[dict] = []
    run_idx = 0

    for batch_size in args.leaf_batch_sizes:
        for tie in args.tie_breakings:
            for k in args.k_values:
                budget = args.total_evals // k
                expansion = make_expansion(planner, critic, k)

                for seed in args.seeds:
                    run_idx += 1
                    print(
                        f"[{run_idx:>3}/{total_runs}] "
                        f"batch={batch_size}  tie={tie:>7}  K={k:>3}  "
                        f"budget={budget:>5}  seed={seed:>3}",
                        end="  ", flush=True,
                    )

                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    cfg = TreeConfig(
                        obs_dim=OBS_DIM, horizon=H, child_state_index=CHILD_IDX, K=k,
                        ucb_c=args.ucb_c, storage_mode=STORAGE_MODE,
                        max_expansions=budget, device=DEVICE,
                        leaf_batch_size=batch_size,
                        ucb_tie_breaking=tie,
                    )

                    t0 = time.time()
                    tree = MCTSTree(root_s, expansion, cfg)
                    records = tree.run()
                    elapsed = time.time() - t0
                    peak_gpu = gpu_peak_mb()
                    traj_per_sec = args.total_evals / elapsed if elapsed > 0 else 0.0

                    last = records[-1]
                    save_run_csv(records, k, seed, tie, batch_size)
                    print(
                        f"depth={last.tree_depth}  best={last.cumulative_best:.4f}  "
                        f"traj/s={traj_per_sec:.1f}  peak_GPU={peak_gpu:.0f}MB  "
                        f"n_nodes={last.n_nodes:,}  time={elapsed:.1f}s"
                    )

                    summary_rows.append({
                        "batch_size": batch_size,
                        "tie_breaking": tie,
                        "K": k,
                        "budget": budget,
                        "seed": seed,
                        "wall_time": round(elapsed, 2),
                        "n_nodes": last.n_nodes,
                        "tree_depth": last.tree_depth,
                        "cumulative_best": round(last.cumulative_best, 6),
                        "traj_per_sec": round(traj_per_sec, 2),
                        "peak_gpu_mb": round(peak_gpu, 1),
                    })

    summary_path = save_summary_csv(summary_rows)
    print(f"\nSummary CSV → {summary_path}")
    print_summary(summary_rows, args.k_values, args.tie_breakings,
                  args.leaf_batch_sizes, args.total_evals)


if __name__ == "__main__":
    main()
