"""scripts/phase5_child_index_ablation.py

Ablation F: child_state_index sweep.

Tests whether branching on a further waypoint improves MCTS performance by
placing children in genuinely diverse regions of the maze rather than in the
tight ~0.1-unit cluster around waypoint 1 (the current design).

Two rollout modes per child_state_index:

  matched  — tree branches at waypoint cidx; policy also targets waypoint cidx.
             Tests the full end-to-end effect, including any policy degradation
             from receiving a target further away than its training distribution
             (policy was trained on 1-step / 15-dense-step targets).

  split    — tree branches at waypoint cidx (diverse children); policy targets
             waypoint 1 of the trajectory that produced path[1], keeping the
             policy command within its training distribution.
             Requires storage_mode="trajectory_node" to retrieve the stored trajectory.

Decision rule:
  cidx>1 matched beats cidx=1  → Further branching helps even with policy degradation.
  cidx>1 split   beats cidx=1  → Diverse branching helps; policy distribution matters.
  cidx>1 split ≈ cidx>1 matched → Policy handles far targets; policy domain is not the issue.
  All cidx>1 ≈ cidx=1           → Branching redundancy is not the bottleneck (critic is).

Outputs:
  results/phase5/ablation_F_results.json
  results/phase5/ablation_F_summary.csv

Usage:
    python scripts/phase5_child_index_ablation.py
    python scripts/phase5_child_index_ablation.py --cidx 1 4 8 --seeds 0 1 2 3 4
    python scripts/phase5_child_index_ablation.py --cidx 4 --mode matched --seeds 0 1 2
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
from mcts.rollout import EpisodeResult, RolloutConfig, run_mcts_episode
from mcts.tree import MCTSTree
from pipelines.utils import set_seed

# ── Constants ─────────────────────────────────────────────────────────────────
OBS_DIM      = 4
ACT_DIM      = 2
H            = 32
M            = 15
PLAN_STEPS   = 20
POLICY_STEPS = 10
K            = 10
BUDGET       = 12
DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT_DIR      = "results/phase5"

# child_state_index values to sweep (H-1=31 is the maximum useful index)
DEFAULT_CIDX  = [1, 4, 8]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cidx",  type=int, nargs="+", default=DEFAULT_CIDX,
                    help="child_state_index values to sweep (default: 1 4 8)")
parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
parser.add_argument("--mode",  nargs="+", choices=["matched", "split"],
                    default=["matched", "split"],
                    help="'matched': policy targets cidx; 'split': policy targets 1")
parser.add_argument("--k",      type=int, default=K)
parser.add_argument("--budget", type=int, default=BUDGET)
parser.add_argument("--env", default="maze2d-umaze-v1",
                    choices=["maze2d-umaze-v1", "maze2d-medium-v1", "maze2d-large-v1"])
args = parser.parse_args()

ENV_NAME      = args.env
ENV_TAG       = ENV_NAME.replace("maze2d-", "").replace("-v1", "")
CKPT          = (
    "results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer"
    f"_d2_width256_separate_dpTrue/{ENV_NAME}"
)
BASELINE_JSON = f"results/phase0_baseline_{ENV_TAG}.json"

os.makedirs(OUT_DIR, exist_ok=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Model loading ──────────────────────────────────────────────────────────────
print("Loading dataset …")
env = gym.make(ENV_NAME)
MAX_T = env._max_episode_steps
dataset = DV_D4RLMaze2DSeqDataset(
    env.get_dataset(), horizon=H, stride=M,
    learn_policy=False, center_mapping=False,
    discount=1.0, continous_reward_at_done=True, reward_tune="iql",
)
normalizer = dataset.get_normalizer()

print(f"Loading models on {DEVICE} …")
fix_mask = torch.zeros((H, OBS_DIM))
fix_mask[0, :OBS_DIM] = 1.0
planner = ContinuousDiffusionSDE(
    DiT1d(OBS_DIM, emb_dim=128, d_model=256, n_heads=4, depth=2,
          timestep_emb_type="fourier"),
    fix_mask=fix_mask, noise_schedule="linear", device=DEVICE,
    predict_noise=True, ema_rate=0.9999,
    loss_weight=torch.ones((H, OBS_DIM)),
)
planner.load(f"{CKPT}/planner_ckpt_1000000.pt")
planner.eval()

critic_ckpt = torch.load(f"{CKPT}/critic_ckpt_1000000.pt", map_location=DEVICE)
critic = DVHorizonCritic(
    OBS_DIM, emb_dim=128, d_model=256, n_heads=4, depth=2, norm_type="pre",
).to(DEVICE)
critic.load_state_dict(critic_ckpt["critic"])
critic.eval()

policy = DiscreteDiffusionSDE(
    DVInvMlp(OBS_DIM, ACT_DIM, emb_dim=64, hidden_dim=256,
              timestep_emb_type="positional").to(DEVICE),
    IdentityCondition(dropout=0.0).to(DEVICE),
    x_max=+torch.ones((1, ACT_DIM), device=DEVICE),
    x_min=-torch.ones((1, ACT_DIM), device=DEVICE),
    diffusion_steps=POLICY_STEPS, device=DEVICE,
)
policy.load(f"{CKPT}/policy_ckpt_1000000.pt")
policy.eval()
print("Models loaded.")

# ── Config helpers ─────────────────────────────────────────────────────────────

def make_expansion(k: int) -> PlannerExpansion:
    cfg = ExpansionConfig(
        K=k, horizon=H, obs_dim=OBS_DIM, planner_dim=OBS_DIM,
        solver="ddim", sample_steps=PLAN_STEPS, temperature=1.0,
        use_ema=True, device=DEVICE,
    )
    return PlannerExpansion(planner, critic, cfg)


def make_tree_cfg(k: int, budget: int, cidx: int,
                  storage: str = "state_only") -> TreeConfig:
    return TreeConfig(
        obs_dim=OBS_DIM, horizon=H, child_state_index=cidx,
        K=k, ucb_c=1.414, storage_mode=storage,
        max_expansions=budget, device=DEVICE,
        leaf_batch_size=10, ucb_tie_breaking="random",
    )

# ── Split-mode rollout ─────────────────────────────────────────────────────────

def _policy_action(obs_norm: torch.Tensor, next_norm: torch.Tensor) -> torch.Tensor:
    """Inverse-dynamics policy: returns action (act_dim,)."""
    obs_r  = obs_norm.unsqueeze(0).to(DEVICE).clone()
    next_r = next_norm.unsqueeze(0).to(DEVICE).clone()
    next_r[:, :2] -= obs_r[:, :2]
    obs_r[:, :2] = 0.0
    prior = torch.zeros((1, ACT_DIM), device=DEVICE)
    with torch.no_grad():
        act, _ = policy.sample(
            prior, solver="ddpm", n_samples=1,
            sample_steps=POLICY_STEPS,
            condition_cfg=torch.cat([obs_r, next_r], dim=-1),
            w_cfg=1.0, use_ema=True, temperature=0.5)
    return act.squeeze(0).cpu()


def run_mcts_split(
    env_ep, expansion, tree_cfg: TreeConfig, rollout_cfg: RolloutConfig, seed: int,
) -> EpisodeResult:
    """MCTS with far branching (child_state_index=cidx) but policy targets waypoint 1.

    The tree branches at tree_cfg.child_state_index, placing children in diverse
    regions of state space.  However, the policy receives waypoint 1 of the
    trajectory that produced path[1], not path[1].s_norm itself.

    Requires storage_mode='trajectory_node' so path[1].traj is available.
    """
    obs = env_ep.reset()
    ep_reward, finished, t, denoise_calls = 0.0, False, 0, 0
    depths: List[float] = []
    cum_bests: List[float] = []
    t0 = time.perf_counter()

    while t < rollout_cfg.max_t:
        s_norm = torch.tensor(
            normalizer.normalize(obs[None]), dtype=torch.float32).squeeze(0)

        tree    = MCTSTree(s_norm, expansion, tree_cfg)
        records = tree.run()
        denoise_calls += tree_cfg.max_expansions * rollout_cfg.plan_steps

        path = tree.best_path()
        if len(path) >= 2:
            child = path[1]
            if child.traj is not None:
                # Trajectory stored on node — use waypoint 1 as policy target
                next_s_norm = child.traj[1, :OBS_DIM].cpu()
            else:
                # Fallback: child.s_norm is at cidx (policy may degrade for cidx > 1)
                next_s_norm = child.s_norm.cpu()
        else:
            next_s_norm = path[0].s_norm.cpu()

        last = records[-1]
        depths.append(float(last.tree_depth))
        cum_bests.append(float(last.cumulative_best))

        act = _policy_action(s_norm, next_s_norm)
        denoise_calls += rollout_cfg.policy_steps

        obs, rew, done, _ = env_ep.step(act.numpy())
        finished = finished or (rew == 1.0)
        ep_reward += float(finished)
        t += 1
        if done:
            break

    wall       = time.perf_counter() - t0
    norm_score = env_ep.get_normalized_score(ep_reward) * 100
    goal_step  = int(rollout_cfg.max_t - ep_reward) if ep_reward > 0 else None

    return EpisodeResult(
        method=f"MCTS-K{tree_cfg.K}-exp{tree_cfg.max_expansions}"
               f"-cidx{tree_cfg.child_state_index}-split",
        seed=seed,
        raw_return=ep_reward,
        normalized_score=round(norm_score, 2),
        goal_step=goal_step,
        episode_length=t,
        denoising_calls=denoise_calls,
        wall_seconds=round(wall, 2),
        ms_per_step=round(wall / t * 1000, 1) if t > 0 else 0.0,
        mcts_budget=tree_cfg.max_expansions,
        mean_tree_depth=sum(depths) / len(depths) if depths else 0.0,
        mean_cumulative_best=sum(cum_bests) / len(cum_bests) if cum_bests else 0.0,
    )

# ── Runner ─────────────────────────────────────────────────────────────────────

def run_all() -> List[EpisodeResult]:
    expansion = make_expansion(args.k)
    rollout_cfg = RolloutConfig(
        obs_dim=OBS_DIM, act_dim=ACT_DIM, child_state_index=1,
        plan_steps=PLAN_STEPS, policy_steps=POLICY_STEPS,
        max_t=MAX_T, device=DEVICE,
    )

    results: List[EpisodeResult] = []
    run_list = []

    for cidx in args.cidx:
        if "matched" in args.mode:
            run_list.append(("matched", cidx))
        # split is only distinct from matched when cidx > 1
        if "split" in args.mode and cidx > 1:
            run_list.append(("split", cidx))

    total = len(run_list) * len(args.seeds)
    run_idx = 0

    for mode, cidx in run_list:
        if mode == "matched":
            tree_cfg = make_tree_cfg(args.k, args.budget, cidx,
                                     storage="state_only")
            rc = RolloutConfig(
                obs_dim=OBS_DIM, act_dim=ACT_DIM,
                # NOTE: child_state_index here is dead code for run_mcts_episode.
                # That function derives next_s_norm from path[1].s_norm (the child
                # node's state, which tree_cfg already placed at waypoint cidx).
                # It never reads rollout_cfg.child_state_index.  The field only
                # matters for run_greedy_episode; setting it to cidx just keeps
                # the RolloutConfig self-consistent for logging purposes.
                child_state_index=cidx,
                plan_steps=PLAN_STEPS, policy_steps=POLICY_STEPS,
                max_t=MAX_T, device=DEVICE)
            label = f"MCTS-K{args.k}-exp{args.budget}-cidx{cidx}-matched"
        else:
            # tree branches at cidx, policy always targets waypoint 1
            tree_cfg = make_tree_cfg(args.k, args.budget, cidx,
                                     storage="trajectory_node")
            rc = rollout_cfg   # child_state_index=1 for policy
            label = f"MCTS-K{args.k}-exp{args.budget}-cidx{cidx}-split"

        for seed in args.seeds:
            run_idx += 1
            env.seed(seed)
            env.action_space.seed(seed)
            set_seed(seed)

            print(f"[F {run_idx:>3}/{total}] {label:<38}  seed={seed}",
                  end="  ", flush=True)
            t0 = time.time()

            if mode == "matched":
                result = run_mcts_episode(
                    env, expansion, policy, normalizer, tree_cfg, rc, seed=seed)
                # Override method label to include cidx and mode
                result = EpisodeResult(
                    method=label, seed=result.seed,
                    raw_return=result.raw_return,
                    normalized_score=result.normalized_score,
                    goal_step=result.goal_step,
                    episode_length=result.episode_length,
                    denoising_calls=result.denoising_calls,
                    wall_seconds=result.wall_seconds,
                    ms_per_step=result.ms_per_step,
                    mcts_budget=result.mcts_budget,
                    mean_tree_depth=result.mean_tree_depth,
                    mean_cumulative_best=result.mean_cumulative_best,
                )
            else:
                result = run_mcts_split(env, expansion, tree_cfg, rc, seed=seed)

            elapsed = time.time() - t0
            print(
                f"return={result.raw_return:.0f}  score={result.normalized_score:.1f}  "
                f"depth={result.mean_tree_depth:.2f}  time={elapsed:.1f}s")
            results.append(result)

    return results

# ── I/O ────────────────────────────────────────────────────────────────────────

def _mean(vals) -> float:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def load_baseline() -> list:
    if not os.path.exists(BASELINE_JSON):
        return []
    with open(BASELINE_JSON) as f:
        return json.load(f)


def save_and_print(results: List[EpisodeResult]) -> None:
    # JSON
    json_path = os.path.join(OUT_DIR, f"ablation_F_results_{ENV_TAG}.json")
    rows = [r.to_dict() for r in results]
    if os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
        seen = {(r["method"], r["seed"]) for r in existing}
        rows = existing + [r for r in rows if (r["method"], r["seed"]) not in seen]
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    # CSV
    csv_path = os.path.join(OUT_DIR, f"ablation_F_summary_{ENV_TAG}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Table
    baseline = load_baseline()
    grouped: dict[str, List[EpisodeResult]] = defaultdict(list)
    for r in results:
        grouped[r.method].append(r)

    col_w = 100
    print()
    print("=" * col_w)
    print("Ablation F — child_state_index sweep")
    print("=" * col_w)
    print(f"{'Method':<42}  {'n':>3}  {'norm_score':>11}  "
          f"{'raw_return':>11}  {'depth':>7}  {'ms/step':>8}")
    print("-" * col_w)

    if baseline:
        print(f"{'DV-MCSS (Phase 0 ref)':<42}  {len(baseline):>3}  "
              f"{_mean([r['normalized_score'] for r in baseline]):>11.2f}  "
              f"{_mean([r['raw_return'] for r in baseline]):>11.1f}  "
              f"{'—':>7}  "
              f"{_mean([r['ms_per_step'] for r in baseline]):>8.1f}")

    for method in sorted(grouped.keys()):
        rs = grouped[method]
        depth_str = (f"{_mean([r.mean_tree_depth for r in rs]):.2f}"
                     if any(r.mean_tree_depth is not None for r in rs) else "—")
        print(f"{method:<42}  {len(rs):>3}  "
              f"{_mean([r.normalized_score for r in rs]):>11.2f}  "
              f"{_mean([r.raw_return for r in rs]):>11.1f}  "
              f"{depth_str:>7}  "
              f"{_mean([r.ms_per_step for r in rs]):>8.1f}")

    print("=" * col_w)
    print()
    _print_decision(grouped, baseline)
    print(f"\n→ {json_path}")
    print(f"→ {csv_path}")


def _print_decision(
    grouped: dict[str, List[EpisodeResult]],
    baseline: list,
) -> None:
    greedy_score = (_mean([r["normalized_score"] for r in baseline])
                    if baseline else None)
    cidx1_matched = grouped.get(
        f"MCTS-K{args.k}-exp{args.budget}-cidx1-matched", [])
    cidx1_score   = _mean([r.normalized_score for r in cidx1_matched]) if cidx1_matched else None

    print("Decision (Ablation F — child_state_index):")
    ref_str = (f"greedy={greedy_score:.1f}" if greedy_score is not None
               else "no greedy ref")
    print(f"  Reference: {ref_str}")
    if cidx1_score is not None:
        print(f"  cidx=1 matched (current design): {cidx1_score:.1f}")

    for cidx in [c for c in args.cidx if c > 1]:
        m_key = f"MCTS-K{args.k}-exp{args.budget}-cidx{cidx}-matched"
        s_key = f"MCTS-K{args.k}-exp{args.budget}-cidx{cidx}-split"
        m_score = _mean([r.normalized_score for r in grouped.get(m_key, [])]) if m_key in grouped else None
        s_score = _mean([r.normalized_score for r in grouped.get(s_key, [])]) if s_key in grouped else None

        matched_str = f"{m_score:.1f}" if m_score is not None else "—"
        split_str   = f"{s_score:.1f}" if s_score is not None else "—"
        ref          = cidx1_score if cidx1_score is not None else (greedy_score or 0)
        m_delta      = (m_score - ref) if m_score is not None else None
        s_delta      = (s_score - ref) if s_score is not None else None
        print(f"\n  cidx={cidx}:")
        print(f"    matched: {matched_str}"
              + (f"  (Δ={m_delta:+.1f} vs cidx=1)" if m_delta is not None else ""))
        print(f"    split  : {split_str}"
              + (f"  (Δ={s_delta:+.1f} vs cidx=1)" if s_delta is not None else ""))

        if m_score is not None and s_score is not None:
            if s_score > ref + 3.0 and s_score > m_score - 2.0:
                print("    → Split ≈ matched or better: diverse branching helps; "
                      "policy handles 1-step target fine.")
            elif m_score < ref - 3.0 and s_score > m_score + 3.0:
                print("    → Split >> matched: policy degrades with far targets. "
                      "Decouple branching from policy waypoint.")
            elif s_score > ref + 3.0:
                print("    → Split beats cidx=1: diverse branching with 1-step "
                      "policy target is beneficial.")
            elif m_score > ref + 3.0:
                print("    → Matched beats cidx=1: far targets help the policy too.")
            else:
                print("    → Neither variant beats cidx=1 clearly.")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nAblation F: child_state_index sweep")
    print(f"  env={ENV_NAME}, cidx={args.cidx}, modes={args.mode}, seeds={args.seeds}")
    print(f"  K={args.k}, budget={args.budget}\n")

    results = run_all()
    save_and_print(results)
