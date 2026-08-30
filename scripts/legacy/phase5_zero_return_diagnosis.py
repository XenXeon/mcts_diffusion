"""scripts/phase5_zero_return_diagnosis.py

Diagnose, per step, why some MCTS episodes return 0 (never reach the goal).

This script does NOT assume a cause.  It records the evidence needed to
*separate* the candidate explanations, and reports the numbers — it does not
hard-code an interpretation.  (An earlier version claimed "split reaches the
goal like cidx=1"; the data contradicts that — split also returns 0 sometimes —
so all conclusions here are computed from the current run, not asserted.)

Candidate causes a return=0 could have, and the signal that isolates each:

  1. Policy out-of-distribution (far target) — the policy is commanded toward a
     waypoint far outside its 1-step training range.
       signal: tgt_disp_norm high  AND  act_sat high (actions pinned at ±1)
  2. Tree / critic selected a bad target — the tree steered toward a waypoint
     that is itself farther from the goal.
       signal: policy_tgt_progress < 0  (commanded target is *farther* from goal)
       and/or tree_child_progress < 0   (the tree's chosen child is farther too)
  3. Execution failure — a good target was commanded, but the agent did not move
     toward it.
       signal: policy_tgt_progress > 0  but  step_progress <= 0
  4. Unlucky navigation — per-step signals look fine, but the agent never enters
     the 0.5 goal radius.
       signal: progress mostly >= 0 yet min_dist_goal stays > 0.5

Per step it logs (full trace written to CSV):
  pos (x,y), dist_to_goal before/after, step_progress,
  tree-selected child waypoint (raw x,y) and its progress toward goal,
  the actual policy target (raw x,y) and its progress,
  tgt_disp_norm (policy conditioning magnitude), action, act_sat,
  selected child value(), cumulative_best critic score, tree path length.

Outputs:
  results/phase5/zero_return_trace_<env>.csv      (one row per env step)
  results/phase5/zero_return_summary_<env>.csv    (one row per episode)

Run:
    python scripts/phase5_zero_return_diagnosis.py
    python scripts/phase5_zero_return_diagnosis.py --env maze2d-large-v1 \
        --cidx 1 8 16 --modes matched split --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, ".")

import d4rl  # noqa: F401
import gym
import numpy as np
import torch

from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DiT1d, DVInvMlp
from cleandiffuser.utils import DVHorizonCritic
from mcts.expansion import ExpansionConfig, PlannerExpansion
from mcts.node import TreeConfig
from mcts.tree import MCTSTree
from pipelines.utils import set_seed

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="maze2d-umaze-v1",
                    choices=["maze2d-umaze-v1", "maze2d-medium-v1", "maze2d-large-v1"])
parser.add_argument("--cidx", type=int, nargs="+", default=[1, 8])
parser.add_argument("--modes", nargs="+", choices=["matched", "split"],
                    default=["matched", "split"])
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--budget", type=int, default=12)
parser.add_argument("--trace-all", action="store_true",
                    help="write per-step trace for every episode (default: only "
                         "return=0 episodes + one reached reference per config)")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"
ENV_NAME     = args.env
OBS_DIM      = 4
ACT_DIM      = 2
H            = 32
M            = 15
PLAN_STEPS   = 20
POLICY_STEPS = 10
GOAL_RADIUS  = 0.5
CKPT = (
    "results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer"
    f"_d2_width256_separate_dpTrue/{ENV_NAME}"
)
OUT_DIR = "results/phase5"
os.makedirs(OUT_DIR, exist_ok=True)
TAG = {"maze2d-umaze-v1": "umaze", "maze2d-medium-v1": "medium",
       "maze2d-large-v1": "large"}[ENV_NAME]

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Env + normalizer ──────────────────────────────────────────────────────────
print(f"Loading dataset ({ENV_NAME}) …")
env = gym.make(ENV_NAME)
MAX_T = env._max_episode_steps
dataset = DV_D4RLMaze2DSeqDataset(
    env.get_dataset(), horizon=H, stride=M,
    learn_policy=False, center_mapping=False,
    discount=1.0, continous_reward_at_done=True, reward_tune="iql")
normalizer = dataset.get_normalizer()


def get_goal(e):
    """Robust maze2d target lookup (checks _target, target, goal_locations, get_target)."""
    u = e.unwrapped
    for attr in ("_target", "target", "goal_locations"):
        if hasattr(u, attr):
            g = np.asarray(getattr(u, attr), dtype=np.float32).reshape(-1)
            if g.size >= 2:
                return g[:2]
    if hasattr(u, "get_target"):
        return np.asarray(u.get_target(), dtype=np.float32).reshape(-1)[:2]
    raise RuntimeError("could not locate maze2d target")


goal = get_goal(env)
print(f"Goal (raw x,y): {goal}   MAX_T={MAX_T}   goal_radius={GOAL_RADIUS}")

# ── Models ────────────────────────────────────────────────────────────────────
print(f"Loading models on {DEVICE} …")
fix_mask = torch.zeros((H, OBS_DIM))
fix_mask[0, :OBS_DIM] = 1.0
planner = ContinuousDiffusionSDE(
    DiT1d(OBS_DIM, emb_dim=128, d_model=256, n_heads=4, depth=2,
          timestep_emb_type="fourier"),
    fix_mask=fix_mask, noise_schedule="linear", device=DEVICE,
    predict_noise=True, ema_rate=0.9999, loss_weight=torch.ones((H, OBS_DIM)))
planner.load(f"{CKPT}/planner_ckpt_1000000.pt")
planner.eval()

critic_ckpt = torch.load(f"{CKPT}/critic_ckpt_1000000.pt", map_location=DEVICE)
critic = DVHorizonCritic(OBS_DIM, emb_dim=128, d_model=256, n_heads=4, depth=2,
                         norm_type="pre").to(DEVICE)
critic.load_state_dict(critic_ckpt["critic"])
critic.eval()

policy = DiscreteDiffusionSDE(
    DVInvMlp(OBS_DIM, ACT_DIM, emb_dim=64, hidden_dim=256,
             timestep_emb_type="positional").to(DEVICE),
    IdentityCondition(dropout=0.0).to(DEVICE),
    x_max=+torch.ones((1, ACT_DIM), device=DEVICE),
    x_min=-torch.ones((1, ACT_DIM), device=DEVICE),
    diffusion_steps=POLICY_STEPS, device=DEVICE)
policy.load(f"{CKPT}/policy_ckpt_1000000.pt")
policy.eval()
print("Models loaded.\n")


def make_expansion(k):
    return PlannerExpansion(planner, critic, ExpansionConfig(
        K=k, horizon=H, obs_dim=OBS_DIM, planner_dim=OBS_DIM,
        solver="ddim", sample_steps=PLAN_STEPS, temperature=1.0,
        use_ema=True, device=DEVICE))


def make_tree_cfg(cidx, storage):
    return TreeConfig(
        obs_dim=OBS_DIM, horizon=H, child_state_index=cidx,
        K=args.k, ucb_c=1.414, storage_mode=storage,
        max_expansions=args.budget, device=DEVICE,
        leaf_batch_size=10, ucb_tie_breaking="random")


def policy_action(s_norm, next_norm):
    obs_r  = s_norm.unsqueeze(0).to(DEVICE).clone()
    next_r = next_norm.unsqueeze(0).to(DEVICE).clone()
    next_r[:, :2] -= obs_r[:, :2]
    obs_r[:, :2] = 0.0
    prior = torch.zeros((1, ACT_DIM), device=DEVICE)
    with torch.no_grad():
        act, _ = policy.sample(prior, solver="ddpm", n_samples=1,
                               sample_steps=POLICY_STEPS,
                               condition_cfg=torch.cat([obs_r, next_r], dim=-1),
                               w_cfg=1.0, use_ema=True, temperature=0.5)
    return act.squeeze(0).cpu().numpy()


def unnorm_xy(s_norm_tensor):
    """Unnormalise a (obs_dim,) normalised state → raw (x, y)."""
    arr = s_norm_tensor.detach().cpu().numpy()[None]   # (1, obs_dim)
    return normalizer.unnormalize(arr)[0, :2]


# ── Instrumented episode ──────────────────────────────────────────────────────

def diagnose_episode(expansion, cidx, mode, seed):
    storage = "trajectory_node" if mode == "split" else "state_only"
    tree_cfg = make_tree_cfg(cidx, storage)
    env.seed(seed); env.action_space.seed(seed); set_seed(seed)
    obs = env.reset()
    # Re-read the goal AFTER reset, per episode — never trust the module-level
    # goal, in case maze2d refreshes the target during reset().
    episode_goal = get_goal(env)
    ep_reward, finished, t = 0.0, False, 0
    trace = []

    while t < MAX_T:
        pos = obs[:2].copy()
        dist_before = float(np.linalg.norm(pos - episode_goal))

        s_norm = torch.tensor(normalizer.normalize(obs[None]),
                              dtype=torch.float32).squeeze(0)
        tree = MCTSTree(s_norm, expansion, tree_cfg)
        records = tree.run()
        path = tree.best_path()

        # Tree-selected child (always at waypoint=cidx) — the tree's "intent"
        if len(path) >= 2:
            child = path[1]
            tree_child_xy = unnorm_xy(child.s_norm)
            child_value = float(child.value())
            # Policy target depends on mode
            if mode == "split" and child.traj is not None:
                next_s_norm = child.traj[1, :OBS_DIM].cpu()   # 1-step (in-dist)
            else:
                next_s_norm = child.s_norm.cpu()              # wp cidx (far if cidx>1)
        else:
            child = path[0]
            tree_child_xy = unnorm_xy(child.s_norm)
            child_value = float(child.value()) if child.visit_count > 0 else float("nan")
            next_s_norm = child.s_norm.cpu()

        policy_tgt_xy = unnorm_xy(next_s_norm)
        tree_child_dg = float(np.linalg.norm(tree_child_xy - episode_goal))
        policy_tgt_dg = float(np.linalg.norm(policy_tgt_xy - episode_goal))
        tree_child_progress = dist_before - tree_child_dg     # >0: tree intends toward goal
        policy_tgt_progress = dist_before - policy_tgt_dg     # >0: commanded toward goal
        tgt_disp_norm = float(torch.norm(next_s_norm[:2] - s_norm[:2]))

        a = policy_action(s_norm, next_s_norm)
        act_sat = float((np.abs(a) > 0.95).mean())

        obs, rew, done, _ = env.step(a)
        dist_after = float(np.linalg.norm(obs[:2] - episode_goal))
        step_progress = dist_before - dist_after              # >0: actually moved toward goal
        finished = finished or (rew == 1.0)
        ep_reward += float(finished)

        trace.append(dict(
            env=TAG, cidx=cidx, mode=mode, seed=seed, t=t,
            x=round(float(pos[0]), 4), y=round(float(pos[1]), 4),
            dist_before=round(dist_before, 4),
            tree_child_x=round(float(tree_child_xy[0]), 4),
            tree_child_y=round(float(tree_child_xy[1]), 4),
            tree_child_dist_goal=round(tree_child_dg, 4),
            tree_child_progress=round(tree_child_progress, 4),
            policy_tgt_x=round(float(policy_tgt_xy[0]), 4),
            policy_tgt_y=round(float(policy_tgt_xy[1]), 4),
            policy_tgt_dist_goal=round(policy_tgt_dg, 4),
            policy_tgt_progress=round(policy_tgt_progress, 4),
            tgt_disp_norm=round(tgt_disp_norm, 4),
            act0=round(float(a[0]), 4), act1=round(float(a[1]), 4),
            act_sat=round(act_sat, 3),
            dist_after=round(dist_after, 4),
            step_progress=round(step_progress, 4),
            child_value=round(child_value, 4) if child_value == child_value else "nan",
            cum_best=round(float(records[-1].cumulative_best), 4),
            path_len=len(path),
            # raw env reward + latched return: lets you verify the return matches
            # the env exactly, and cross-check the 0.5-radius proxy against rew==1.
            rew=round(float(rew), 4),
            latched_return=ep_reward,
            in_goal_zone=int(dist_after < GOAL_RADIUS),
        ))
        t += 1
        if done:
            break

    # ── Episode aggregates (computed from the trace; no assumptions) ──
    A = lambda key: np.array([r[key] for r in trace], dtype=float)
    ptp = A("policy_tgt_progress")
    sp  = A("step_progress")
    summary = dict(
        env=TAG, cidx=cidx, mode=mode, seed=seed,
        reached=int(ep_reward > 0), raw_return=ep_reward,
        episode_length=t,
        min_dist_goal=round(float(A("dist_after").min()), 4),
        final_dist_goal=round(float(trace[-1]["dist_after"]), 4),
        frac_in_goal_zone=round(float(A("in_goal_zone").mean()), 4),
        mean_tgt_disp_norm=round(float(A("tgt_disp_norm").mean()), 4),
        mean_act_sat=round(float(A("act_sat").mean()), 4),
        mean_tree_child_progress=round(float(A("tree_child_progress").mean()), 4),
        mean_policy_tgt_progress=round(float(ptp.mean()), 4),
        frac_policy_tgt_toward_goal=round(float((ptp > 0).mean()), 4),
        mean_step_progress=round(float(sp.mean()), 4),
        frac_step_toward_goal=round(float((sp > 0).mean()), 4),
        # causal-separation rates
        bad_target_rate=round(float((ptp < 0).mean()), 4),       # tree/critic steered away
        exec_fail_rate=round(float(((ptp > 0) & (sp <= 0)).mean()), 4),  # good cmd, no move
        mean_child_value=round(float(np.nanmean(A("child_value"))), 4),
        mean_cum_best=round(float(A("cum_best").mean()), 4),
    )
    return summary, trace


# ── Run ───────────────────────────────────────────────────────────────────────
configs = []
for cidx in args.cidx:
    for mode in args.modes:
        if mode == "split" and cidx == 1:
            continue
        configs.append((cidx, mode))

print(f"Diagnosing {len(configs)} configs × {len(args.seeds)} seeds "
      f"(cidx={args.cidx}, modes={args.modes}, seeds={args.seeds})\n")

summaries, all_traces = [], []
exp = make_expansion(args.k)
reached_ref_written = set()   # one reached-reference trace per config

for cidx, mode in configs:
    for seed in args.seeds:
        t0 = time.time()
        summ, trace = diagnose_episode(exp, cidx, mode, seed)
        summaries.append(summ)
        # keep trace for: all (if --trace-all), every zero-return, and one reached ref/config
        keep = args.trace_all or summ["reached"] == 0
        if not keep and (cidx, mode) not in reached_ref_written:
            keep = True
            reached_ref_written.add((cidx, mode))
        if keep:
            all_traces.extend(trace)
        flag = "" if summ["reached"] else "  <-- RETURN 0"
        print(f"cidx={cidx:<2} {mode:<8} seed={seed}  ret={summ['raw_return']:>5.0f}  "
              f"min_dist={summ['min_dist_goal']:.2f}  "
              f"tgt_disp={summ['mean_tgt_disp_norm']:.2f}  "
              f"act_sat={summ['mean_act_sat']:.2f}  "
              f"tgt→goal={summ['frac_policy_tgt_toward_goal']:.2f}  "
              f"step→goal={summ['frac_step_toward_goal']:.2f}  "
              f"({time.time()-t0:.0f}s){flag}")

# ── Write CSVs ────────────────────────────────────────────────────────────────
summ_csv  = f"{OUT_DIR}/zero_return_summary_{TAG}.csv"
with open(summ_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
    w.writeheader(); w.writerows(summaries)

trace_csv = f"{OUT_DIR}/zero_return_trace_{TAG}.csv"
if all_traces:
    with open(trace_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_traces[0].keys()))
        w.writeheader(); w.writerows(all_traces)

# ── Per-episode evidence-based attribution (heuristic, derived from THIS run) ──
print("\n" + "=" * 104)
print("Evidence per episode  (attribution is a heuristic read of the logged "
      "signals, not an assumption)")
print("=" * 104)
print(f"{'cidx':>4} {'mode':<7} {'seed':>4} {'ret':>5} {'reach':>5} "
      f"{'min_d':>6} {'tgt_disp':>8} {'act_sat':>7} {'tgt→gl':>7} "
      f"{'step→gl':>7} {'badT':>5} {'execF':>6}  attribution(heuristic)")
print("-" * 104)
for s in summaries:
    if s["reached"]:
        attrib = "reached goal"
    elif s["mean_act_sat"] > 0.5 and s["mean_tgt_disp_norm"] > 0.5:
        attrib = "policy OOD (far target → saturated)"
    elif s["bad_target_rate"] > 0.5:
        attrib = "tree/critic steered away from goal"
    elif s["exec_fail_rate"] > 0.4:
        attrib = "execution failure (good target, no move)"
    else:
        attrib = "navigation/other (per-step ok, never entered zone)"
    print(f"{s['cidx']:>4} {s['mode']:<7} {s['seed']:>4} {s['raw_return']:>5.0f} "
          f"{s['reached']:>5} {s['min_dist_goal']:>6.2f} "
          f"{s['mean_tgt_disp_norm']:>8.2f} {s['mean_act_sat']:>7.2f} "
          f"{s['frac_policy_tgt_toward_goal']:>7.2f} {s['frac_step_toward_goal']:>7.2f} "
          f"{s['bad_target_rate']:>5.2f} {s['exec_fail_rate']:>6.2f}  {attrib}")
print("=" * 104)
print("Columns: tgt_disp=policy conditioning magnitude (norm space; large=OOD); "
      "act_sat=frac actions pinned at ±1;\n  tgt→gl=frac steps the commanded target "
      "is closer to goal; step→gl=frac steps agent moved closer;\n  badT=frac steps "
      "target was FARTHER from goal (tree/critic error); execF=frac steps good target "
      "but no progress.")
print(f"\n→ {summ_csv}")
if all_traces:
    print(f"→ {trace_csv}  ({len(all_traces)} step rows)")
