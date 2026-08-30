"""scripts/phase3_ablation.py

Phase 3 ablation: compare three MCTS node-storage modes on maze2d-umaze-v1
across multiple random seeds and expansion budgets.

Modes:
    A  state_only              — node stores s_norm only
    B  trajectory_node         — node stores s_norm + full trajectory
    C  state_edge_trajectory   — node stores s_norm; edge stores trajectory + score

Results written to:
    results/phase3/ablation_{mode}_seed{seed}_exp{budget}.csv  (step records per run)
    results/phase3/summary.csv                                   (one row per run, final metrics)

Run (inside Docker):
    python scripts/phase3_ablation.py
    python scripts/phase3_ablation.py --seeds 42 0 123 --budgets 60 120 300
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

MODES = ["state_only", "trajectory_node", "state_edge_trajectory"]
MODE_LABELS = {"state_only": "A", "trajectory_node": "B", "state_edge_trajectory": "C"}

ENV_NAME = "maze2d-umaze-v1"
OBS_DIM = 4
H = 32
M = 15        # dense env stride between trajectory waypoints (dataset constant)
CHILD_IDX = 1 # trajectory index for child state: traj[1] = one planning jump ahead
K = 50
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

DEFAULT_SEEDS = [42, 0, 123]
DEFAULT_BUDGETS = [60, 120, 300]
DEFAULT_LEAF_BATCH = 10

# ── Helpers ────────────────────────────────────────────────────────────────────

def check_prerequisites() -> None:
    missing = [p for p in REQUIRED if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"[FAIL] Required file not found: {p}")
        sys.exit(1)


def load_root_state() -> torch.Tensor:
    """Load the same normalised start state used in the Phase 2 smoke test."""
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
    s_norm = torch.tensor(
        dataset.seq_obs[traj_idx, offset], dtype=torch.float32
    )
    print(f"  Root state (traj={traj_idx}, offset={offset}): {s_norm.tolist()}")
    return s_norm


def build_expansion() -> PlannerExpansion:
    print(f"Loading planner and critic on {DEVICE} …")
    cfg = ExpansionConfig(
        K=K, horizon=H, obs_dim=OBS_DIM, planner_dim=OBS_DIM,
        solver="ddim", sample_steps=20, temperature=1.0,
        use_ema=True, device=DEVICE,
    )
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
    return PlannerExpansion(planner, critic, cfg)


def save_run_csv(records: list[StepRecord], mode: str, seed: int, budget: int) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"ablation_{mode}_seed{seed}_exp{budget}.csv")
    fields = [f.name for f in dataclasses.fields(StepRecord)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(dataclasses.asdict(rec))
    return path


def save_summary_csv(rows: list[dict]) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "summary.csv")
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


def print_summary(summary_rows: list[dict], budgets: list[int]) -> None:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in summary_rows:
        grouped[(row["budget"], row["mode"])].append(row)

    print()
    print("=" * 105)
    print("Phase 3 Ablation Summary  (mean ± std across seeds)")
    print("=" * 105)
    print(
        f"{'Budget':>8}  {'Mode':<30}  {'n':>3}  {'depth':>6}  "
        f"{'best_score':>22}  {'time_s':>8}  {'traj_floats':>12}"
    )
    print("-" * 105)
    for budget in budgets:
        for mode in MODES:
            rows = grouped.get((budget, mode), [])
            if not rows:
                continue
            mean_score, std_score = _mean_std([r["cumulative_best"] for r in rows])
            mean_time, _ = _mean_std([r["wall_time"] for r in rows])
            mean_depth, _ = _mean_std([r["tree_depth"] for r in rows])
            traj_floats = rows[0]["traj_floats"]
            label = f"{mode} ({MODE_LABELS[mode]})"
            score_str = f"{mean_score:.4f} ± {std_score:.4f}"
            print(
                f"{budget:>8}  {label:<30}  {len(rows):>3}  {mean_depth:>6.1f}  "
                f"{score_str:>22}  {mean_time:>8.1f}  {int(traj_floats):>12,}"
            )
        if budget != budgets[-1]:
            print()
    print("=" * 105)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 MCTS ablation — multi-seed, multi-budget"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help=f"Random seeds (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS,
        help=f"Max expansion budgets (default: {DEFAULT_BUDGETS})",
    )
    parser.add_argument(
        "--ucb-c", type=float, default=math.sqrt(2),
        help="UCB exploration constant (default: sqrt(2))",
    )
    parser.add_argument(
        "--leaf-batch-size", type=int, default=DEFAULT_LEAF_BATCH,
        help=f"Leaves expanded per GPU call (default: {DEFAULT_LEAF_BATCH})",
    )
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    check_prerequisites()
    root_s = load_root_state()
    expansion = build_expansion()

    total_runs = len(args.budgets) * len(args.seeds) * len(MODES)
    print(
        f"\nRunning {total_runs} experiments  "
        f"({len(args.budgets)} budgets × {len(args.seeds)} seeds × {len(MODES)} modes)\n"
    )

    summary_rows: list[dict] = []
    run_idx = 0

    for budget in args.budgets:
        for seed in args.seeds:
            for mode in MODES:
                run_idx += 1
                label = MODE_LABELS[mode]
                print(f"[{run_idx:>2}/{total_runs}] budget={budget:>4}  seed={seed:>3}  "
                      f"mode={mode} ({label})", end="  ", flush=True)

                torch.manual_seed(seed)
                cfg = TreeConfig(
                    obs_dim=OBS_DIM, horizon=H, child_state_index=CHILD_IDX, K=K,
                    ucb_c=args.ucb_c, storage_mode=mode,
                    max_expansions=budget, device=DEVICE,
                    leaf_batch_size=args.leaf_batch_size,
                )

                t0 = time.time()
                tree = MCTSTree(root_s, expansion, cfg)
                records = tree.run()
                elapsed = time.time() - t0

                last = records[-1]
                traj_floats = MCTSTree.theoretical_floats(cfg, last.n_nodes)
                path = save_run_csv(records, mode, seed, budget)
                print(f"depth={last.tree_depth}  best={last.cumulative_best:.4f}  "
                      f"time={elapsed:.1f}s")

                summary_rows.append({
                    "budget": budget,
                    "seed": seed,
                    "mode": mode,
                    "wall_time": round(elapsed, 2),
                    "n_nodes": last.n_nodes,
                    "tree_depth": last.tree_depth,
                    "cumulative_best": round(last.cumulative_best, 6),
                    "traj_floats": traj_floats,
                })

    summary_path = save_summary_csv(summary_rows)
    print(f"\nSummary CSV → {summary_path}")
    print_summary(summary_rows, args.budgets)


if __name__ == "__main__":
    main()
