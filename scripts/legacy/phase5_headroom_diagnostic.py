"""scripts/phase5_headroom_diagnostic.py

Headroom diagnostic — does MCTS have anything to do on this env?

For each seed it resets the env, generates K planner candidates from the real
start state, picks the critic-best plan, and asks two questions:

  1. SUFFICIENCY — does the single best plan reach the goal within its own
     horizon (H jump-steps)?  If yes, one diffusion sample already solves the
     task and tree search has no headroom (the umaze/medium regime).  If the
     plan runs out of waypoints short of the goal, the task needs multi-plan
     stitching — the regime MCTS is built for.

  2. MULTIMODALITY — how much do the K plans disagree?  Spread at waypoint[1]
     (the immediate move) measures junction ambiguity: greedy picks one branch
     by critic score; only if the branches genuinely diverge can look-ahead
     beat greedy.  Low spread => even in the stitching regime, MPC re-planning
     is unambiguous and MCTS still won't help.

Goal-reaching threshold is the maze2d reward radius (0.5 in maze coords).

Run:
    python scripts/phase5_headroom_diagnostic.py --env maze2d-large-v1
    python scripts/phase5_headroom_diagnostic.py --env maze2d-large-v1 --seeds 0 1 2 3 4 --k 50
"""
import argparse
import os
import sys

sys.path.insert(0, ".")

import d4rl  # noqa: F401 — registers D4RL envs
import gym
import numpy as np
import torch

from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import DVHorizonCritic
from pipelines.utils import set_seed

# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="MCTS headroom diagnostic")
parser.add_argument("--env", type=str, default="maze2d-large-v1",
                    choices=["maze2d-umaze-v1", "maze2d-medium-v1", "maze2d-large-v1"])
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--k", type=int, default=50, help="planner candidates per state")
args = parser.parse_args()

# ── Config (mirrors run_one_episode.py exactly) ─────────────────────────────
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
ENV        = args.env
H          = 32     # planner_horizon (jump-steps)
M          = 15     # stride (dense steps per jump)
K          = args.k
D_MODEL    = 256
DEPTH      = 2
EMB_DIM    = 128
PLAN_STEPS = 20
HORIZON_DENSE = H * M            # 480 dense steps depicted by one plan
GOAL_RADIUS   = 0.5              # maze2d reward threshold (maze coords)
CKPT = (
    "results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer"
    f"_d2_width256_separate_dpTrue/{ENV}"
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Env + normalizer ────────────────────────────────────────────────────────
env = gym.make(ENV)
dataset = DV_D4RLMaze2DSeqDataset(
    env.get_dataset(), horizon=H, stride=M,
    learn_policy=False, center_mapping=False,
    discount=1.0, continous_reward_at_done=True, reward_tune="iql")
normalizer = dataset.get_normalizer()
obs_dim, act_dim = dataset.o_dim, dataset.a_dim


def get_goal(e):
    """Fetch the maze2d eval target (x, y) robustly across d4rl versions."""
    u = e.unwrapped
    for attr in ("_target", "target", "goal_locations"):
        if hasattr(u, attr):
            g = np.asarray(getattr(u, attr), dtype=np.float32).reshape(-1)
            if g.size >= 2:
                return g[:2]
    if hasattr(u, "get_target"):
        return np.asarray(u.get_target(), dtype=np.float32).reshape(-1)[:2]
    raise RuntimeError("could not locate maze2d target")


# ── Models (identical to run_one_episode.py) ────────────────────────────────
fix_mask = torch.zeros((H, obs_dim))
fix_mask[0, :obs_dim] = 1.0
planner = ContinuousDiffusionSDE(
    DiT1d(obs_dim, emb_dim=EMB_DIM, d_model=D_MODEL,
          n_heads=D_MODEL // 64, depth=DEPTH, timestep_emb_type="fourier"),
    fix_mask=fix_mask, noise_schedule="linear", device=DEVICE)
planner.load(f"{CKPT}/planner_ckpt_1000000.pt")
planner.eval()

critic = DVHorizonCritic(
    obs_dim, emb_dim=EMB_DIM, d_model=D_MODEL,
    n_heads=D_MODEL // 64, depth=2, norm_type="pre").to(DEVICE)
critic.load_state_dict(
    torch.load(f"{CKPT}/critic_ckpt_1000000.pt", map_location=DEVICE)["critic"])
critic.eval()

print(f"\nHeadroom diagnostic — {ENV}  (K={K}, horizon={H} jumps = {HORIZON_DENSE} dense steps)")
print("=" * 100)
print(f"{'seed':>4}  {'start→goal':>11}  {'best reaches?':>13}  {'reach_wp':>9}  "
      f"{'min_dist':>9}  {'K reach':>8}  {'wp1 spread':>11}  {'end spread':>11}")
print("-" * 100)

rows = []
for seed in args.seeds:
    set_seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    obs = env.reset()
    goal = get_goal(env)
    start = obs[:2].copy()
    start_dist = float(np.linalg.norm(start - goal))

    obs_t = torch.tensor(normalizer.normalize(obs[None]),
                         device=DEVICE, dtype=torch.float32)
    prior = torch.zeros((K, H, obs_dim), device=DEVICE)
    prior[:, 0, :] = obs_t.expand(K, -1)
    with torch.no_grad():
        traj, _ = planner.sample(prior, solver="ddim", n_samples=K,
                                 sample_steps=PLAN_STEPS, use_ema=True, temperature=1.0)
        value = critic(traj).squeeze(-1)             # (K,)
        best_i = int(torch.argmax(value).item())

    traj_np = traj.cpu().numpy()                     # (K, H, obs_dim) normalized
    traj_raw = normalizer.unnormalize(traj_np.reshape(-1, obs_dim)).reshape(K, H, obs_dim)
    pos = traj_raw[:, :, :2]                          # (K, H, 2) maze coords

    # best plan: reach analysis
    dist_best = np.linalg.norm(pos[best_i] - goal[None], axis=1)   # (H,)
    hit = np.where(dist_best < GOAL_RADIUS)[0]
    reach_wp = int(hit[0]) if hit.size else -1
    reaches = reach_wp >= 0
    min_dist = float(dist_best.min())

    # how many of the K plans reach the goal at all
    dist_all = np.linalg.norm(pos - goal[None, None], axis=2)      # (K, H)
    k_reach = int((dist_all.min(axis=1) < GOAL_RADIUS).sum())

    # multimodality: spread of immediate move (wp1) and endpoint (wp-1)
    wp1_spread = float(np.linalg.norm(pos[:, 1, :].std(axis=0)))
    end_spread = float(np.linalg.norm(pos[:, -1, :].std(axis=0)))

    rows.append(dict(seed=seed, start_dist=start_dist, reaches=reaches,
                     reach_wp=reach_wp, min_dist=min_dist, k_reach=k_reach,
                     wp1_spread=wp1_spread, end_spread=end_spread))
    print(f"{seed:>4}  {start_dist:>11.2f}  {('YES' if reaches else 'NO'):>13}  "
          f"{(str(reach_wp) if reaches else '—'):>9}  {min_dist:>9.2f}  "
          f"{f'{k_reach}/{K}':>8}  {wp1_spread:>11.3f}  {end_spread:>11.3f}")

print("=" * 100)

# ── Verdict ─────────────────────────────────────────────────────────────────
n = len(rows)
n_fail = sum(1 for r in rows if not r["reaches"])
mean_wp1 = float(np.mean([r["wp1_spread"] for r in rows]))
print("\nVerdict:")
print(f"  single best plan FAILS to reach goal in {n_fail}/{n} seeds "
      f"(these need multi-plan stitching = MCTS-relevant regime)")
print(f"  mean wp1 spread (junction ambiguity) = {mean_wp1:.3f} maze units")
if n_fail == 0:
    print("  → DEGENERATE: one diffusion sample solves every start. MCTS has no headroom here.")
elif mean_wp1 < 0.15:
    print("  → Stitching needed, but K plans agree on the immediate move (low wp1 spread).")
    print("    MPC re-planning is unambiguous; MCTS unlikely to beat greedy. Inspect per-seed.")
else:
    print("  → HEADROOM: some starts exceed single-plan horizon AND plans diverge at the")
    print("    first move. Greedy can mis-commit at junctions; look-ahead can help.")
    print("    Run the MCTS-vs-greedy comparison on the failing seeds.")
