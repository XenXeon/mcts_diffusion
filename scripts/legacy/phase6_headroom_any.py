"""scripts/phase6_headroom_any.py

Generalized single-shot headroom diagnostic — maze2d AND antmaze.

Question: from the eval start states, does the DV planner one-shot the goal, or is
the goal beyond a single plan (the stitching regime where MCTS could help)?

For each seed: reset env, generate K planner candidates, and report how many of the K
plans reach within the success radius of the goal, plus the critic-best plan's closest
approach.  If `0/K` reach for some seeds (as maze2d-large seeds 0/2 already showed),
that env is a genuine stitching testbed — no retraining needed.

Verified per-env specifics (do NOT change without re-checking the configs/pipeline):
  maze2d : obs from dataset, H=32, stride=15, planner depth=2, ckpt suffix
           _d2_width256_separate_dpTrue, goal = env.unwrapped._target
  antmaze: H=40, stride=25, planner depth=8, ckpt suffix _d8_width256_separate_dp1,
           goal = env.unwrapped.target_goal, obs[:2] = ant torso xy (pipeline line 446)
  success radius 0.5 (d4rl reward fires within 0.5 of goal) for both.

Run:
    python scripts/phase6_headroom_any.py --env antmaze-large-diverse-v2 --diagnose
    python scripts/phase6_headroom_any.py --env antmaze-large-diverse-v2 --seeds 0 1 2 3 4
    python scripts/phase6_headroom_any.py --env maze2d-large-v1 --seeds 0 1 2 3 4   # re-confirm
"""
import argparse
import os
import sys

sys.path.insert(0, ".")

import d4rl  # noqa: F401
import gym
import numpy as np
import torch

from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import DVHorizonCritic
from pipelines.utils import set_seed

# ── Per-env-family spec (verified against configs + pipelines) ─────────────────
def env_family(env_name):
    return "maze2d" if env_name.startswith("maze2d") else "antmaze"


SPECS = {
    "maze2d": dict(
        H=32, stride=15, planner_depth=2,
        planner_step=1000000, critic_step=1000000,   # maze2d configs select 1M
        ckpt=("results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer"
              "_d2_width256_separate_dpTrue"),
    ),
    "antmaze": dict(
        H=40, stride=25, planner_depth=8,
        planner_step=1000000, critic_step=1000000,   # override with --critic-step if needed
        ckpt=("results/veteran_d4rl_antmaze_H40_Jump25_next1_MCSS_transformer"
              "_d8_width256_separate_dp1"),
    ),
}

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--k", type=int, default=50, help="planner candidates per start")
parser.add_argument("--ckpt", type=str, default=None,
                    help="override checkpoint dir (else derived from env family)")
parser.add_argument("--planner-step", type=int, default=None,
                    help="planner checkpoint step (else per-family default)")
parser.add_argument("--critic-step", type=int, default=None,
                    help="critic checkpoint step (else per-family default)")
parser.add_argument("--diagnose", action="store_true",
                    help="print goal/start/obs-dim sanity for one seed and exit")
args = parser.parse_args()

FAM = env_family(args.env)
SPEC = SPECS[FAM]
H, M, DEPTH = SPEC["H"], SPEC["stride"], SPEC["planner_depth"]
PLAN_STEPS = 20
GOAL_RADIUS = 0.5
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CKPT = (args.ckpt or SPEC["ckpt"]) + f"/{args.env}"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Env + dataset (normalizer) ─────────────────────────────────────────────────
print(f"Loading {args.env}  (family={FAM}, H={H}, stride={M}, depth={DEPTH}) …")
env = gym.make(args.env)
raw = env.get_dataset()

if FAM == "maze2d":
    from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset
    dataset = DV_D4RLMaze2DSeqDataset(
        raw, horizon=H, stride=M, learn_policy=False, center_mapping=False,
        discount=1.0, continous_reward_at_done=True, reward_tune="iql")
else:
    from cleandiffuser.dataset.d4rl_antmaze_dataset import DV_D4RLAntmazeSeqDataset
    dataset = DV_D4RLAntmazeSeqDataset(
        raw, horizon=H, stride=M, learn_policy=False, center_mapping=True,
        discount=1.0, continous_reward_at_done=True, reward_tune="iql")
normalizer = dataset.get_normalizer()
OBS_DIM, ACT_DIM = dataset.o_dim, dataset.a_dim


def get_goal(e):
    u = e.unwrapped
    for attr in ("target_goal", "_target", "target"):
        if hasattr(u, attr):
            g = np.asarray(getattr(u, attr), dtype=np.float32).reshape(-1)
            if g.size >= 2:
                return g[:2]
    if hasattr(u, "get_target"):
        return np.asarray(u.get_target(), dtype=np.float32).reshape(-1)[:2]
    raise RuntimeError("could not locate goal for this env")


# ── Planner + critic (architecture from the verified config) ───────────────────
fix_mask = torch.zeros((H, OBS_DIM))
fix_mask[0, :OBS_DIM] = 1.0
planner = ContinuousDiffusionSDE(
    DiT1d(OBS_DIM, emb_dim=128, d_model=256, n_heads=4, depth=DEPTH,
          timestep_emb_type="fourier"),
    fix_mask=fix_mask, noise_schedule="linear", device=DEVICE,
    predict_noise=True, ema_rate=0.9999, loss_weight=torch.ones((H, OBS_DIM)))
PLANNER_STEP = args.planner_step or SPEC["planner_step"]
CRITIC_STEP = args.critic_step or SPEC["critic_step"]
planner.load(f"{CKPT}/planner_ckpt_{PLANNER_STEP}.pt")
planner.eval()

critic_ckpt = torch.load(f"{CKPT}/critic_ckpt_{CRITIC_STEP}.pt", map_location=DEVICE)
critic = DVHorizonCritic(OBS_DIM, emb_dim=128, d_model=256, n_heads=4, depth=2,
                         norm_type="pre").to(DEVICE)
critic.load_state_dict(critic_ckpt["critic"])
critic.eval()
print(f"  loaded planner+critic on {DEVICE}  (obs_dim={OBS_DIM}, act_dim={ACT_DIM})")

# ── Diagnose: sanity for one seed, then exit ──────────────────────────────────
if args.diagnose:
    set_seed(0); env.seed(0); obs = env.reset()
    goal = get_goal(env)
    print(f"\nobs shape={np.asarray(obs).shape}  obs[:2] (xy)={np.asarray(obs)[:2]}")
    print(f"goal (xy)={goal}   start→goal dist={np.linalg.norm(obs[:2]-goal):.2f}")
    print(f"success radius={GOAL_RADIUS}")
    s = torch.tensor(normalizer.normalize(obs[None]), dtype=torch.float32).squeeze(0)
    prior = torch.zeros((4, H, OBS_DIM), device=DEVICE)
    prior[:, 0, :] = s.to(DEVICE)
    with torch.no_grad():
        traj, _ = planner.sample(prior, solver="ddim", n_samples=4,
                                 sample_steps=PLAN_STEPS, use_ema=True, temperature=1.0)
    raw_traj = normalizer.unnormalize(traj.cpu().numpy().reshape(-1, OBS_DIM)).reshape(4, H, OBS_DIM)
    d0 = np.linalg.norm(raw_traj[0, :, :2] - goal[None], axis=1)
    print(f"\nsample plan 0 dist-to-goal over waypoints (min={d0.min():.2f}):")
    print("  ", " ".join(f"{x:.1f}" for x in d0))
    print("\nIf obs[:2] looks like a sane xy, goal is plausible, and the plan's dist "
          "decreases toward the goal, the setup is correct. Then run without --diagnose.")
    sys.exit(0)

# ── Headroom over seeds ────────────────────────────────────────────────────────
COVERAGE_OK = 0.85   # best plan must cover ≥85% of start→goal to count as "reached vicinity"


def classify(reaches, k_reach, coverage):
    """Separate the four distinct causes a seed can 'fail to reach':
       one-shot      : critic-best reaches (planner + critic both fine)
       critic-miss   : reaching plans EXIST but the critic didn't pick one  → CRITIC problem
       near-miss     : no plan reaches, but planner got to the vicinity     → precision/execution
       planner-short : planner falls genuinely short (low coverage)         → horizon/STITCHING"""
    if reaches:
        return "one-shot"
    if k_reach > 0:
        return "critic-miss"
    if coverage >= COVERAGE_OK:
        return "near-miss"
    return "planner-short"


print(f"\n{'seed':>4}  {'start→goal':>11}  {'best_min':>9}  {'cover%':>7}  "
      f"{'K reach':>8}  category")
print("-" * 72)
rows = []
for seed in args.seeds:
    set_seed(seed); env.seed(seed); env.action_space.seed(seed)
    obs = env.reset()
    goal = get_goal(env)
    start_dist = float(np.linalg.norm(obs[:2] - goal))

    s = torch.tensor(normalizer.normalize(obs[None]), dtype=torch.float32).squeeze(0)
    prior = torch.zeros((args.k, H, OBS_DIM), device=DEVICE)
    prior[:, 0, :] = s.to(DEVICE)
    with torch.no_grad():
        traj, _ = planner.sample(prior, solver="ddim", n_samples=args.k,
                                 sample_steps=PLAN_STEPS, use_ema=True, temperature=1.0)
        scores = critic(traj).squeeze(-1)
        best_i = int(torch.argmax(scores).item())
    raw_traj = normalizer.unnormalize(
        traj.cpu().numpy().reshape(-1, OBS_DIM)).reshape(args.k, H, OBS_DIM)
    pos = raw_traj[:, :, :2]
    dist_all = np.linalg.norm(pos - goal[None, None], axis=2)   # (K, H)
    k_reach = int((dist_all.min(axis=1) < GOAL_RADIUS).sum())
    best_min = float(dist_all[best_i].min())
    reaches = best_min < GOAL_RADIUS
    coverage = (start_dist - best_min) / max(start_dist, 1e-6)   # fraction of distance covered
    cat = classify(reaches, k_reach, coverage)

    rows.append(dict(seed=seed, start_dist=start_dist, reaches=reaches,
                     best_min=best_min, k_reach=k_reach, coverage=coverage, cat=cat))
    print(f"{seed:>4}  {start_dist:>11.2f}  {best_min:>9.2f}  {100*coverage:>6.1f}%  "
          f"{f'{k_reach}/{args.k}':>8}  {cat}")

print("=" * 72)
from collections import Counter
cnt = Counter(r["cat"] for r in rows)
n = len(rows)
print(f"\nVerdict for {args.env}  (n={n} seeds):")
print(f"  one-shot      {cnt['one-shot']}/{n}   planner + critic both fine")
print(f"  critic-miss   {cnt['critic-miss']}/{n}   reaching plans EXIST but critic didn't pick them"
      "  → CRITIC")
print(f"  near-miss     {cnt['near-miss']}/{n}   planner reached vicinity, just outside radius"
      "  → precision/execution")
print(f"  planner-short {cnt['planner-short']}/{n}   planner falls genuinely short"
      "  → horizon/STITCHING")
print()
# dominant-cause read
if cnt["planner-short"] >= max(1, n // 2):
    print("  → STITCHING/horizon regime: a real MCTS-stitching testbed.")
elif cnt["critic-miss"] >= 1 and cnt["planner-short"] == 0:
    print("  → NOT stitching. The planner one-shots; the CRITIC is the bottleneck "
          "(mis-selects reaching plans).\n    Lever = retrain the critic (helps MCSS directly); "
          "search does not fix mis-ranking.")
elif cnt["near-miss"] >= 1 and cnt["critic-miss"] == 0 and cnt["planner-short"] == 0:
    print("  → NOT stitching. Planner reaches the vicinity everywhere; the gap is precision/"
          "execution (radius or inverse-dynamics), not search.")
else:
    print("  → Mixed; read the per-seed categories above before concluding.")
