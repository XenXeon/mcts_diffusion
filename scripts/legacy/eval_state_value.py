"""scripts/eval_state_value.py

Validate the retrained state-value critic V(s) BEFORE wiring it into the MCTS search.

Two questions, both answered offline on planner-generated plans (no env rollout):

  1. Calibration — does V rank plans by how close they actually get to the goal?
     For each start, generate K plans; correlate V(plan_endpoint) with
     −(min dist-to-goal over the plan).  High positive corr ⇒ V is a usable selector.

  2. Does V fix the measured CRITIC-MISS?  For each seed, compare the plan the MCSS
     critic (DVHorizonCritic) selects vs the plan a V-based selector picks, and whether
     each selected plan actually reaches the goal.  This is exactly Option 2
     (endpoint-value, full-H, one ply) evaluated offline.  If V-selection reaches on the
     antmaze seeds where MCSS-selection did not (large seeds 2,3; medium seed 2 in the
     headroom run), the retrained critic fixes the critic-miss.

Selectors compared per start (argmax over K plans):
    mcss      : DVHorizonCritic(full_traj)            (the stock critic)
    v_end     : V(traj[:, -1, :obs_dim])              (Option-2 endpoint value)
    v_max     : max_h V(traj[:, h, :obs_dim])         (best state the plan passes through)

Run (after training V):
    python scripts/eval_state_value.py --env antmaze-large-diverse-v2 --seeds 0 1 2 3 4
    python scripts/eval_state_value.py --env maze2d-large-v1 --value-step 200000
"""
import argparse
import os
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import DVHorizonCritic
from mcts.specs import SPECS, env_family, get_goal, make_dataset
from mcts.value_net import load_state_value
from pipelines.utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--k", type=int, default=50)
parser.add_argument("--value-step", type=str, default="latest",
                    help="state_value checkpoint step, or 'latest'")
parser.add_argument("--planner-step", type=int, default=1000000)
parser.add_argument("--critic-step", type=int, default=1000000)
parser.add_argument("--ckpt", type=str, default=None)
args = parser.parse_args()

FAM = env_family(args.env)
SPEC = SPECS[FAM]
H, M, DEPTH = SPEC["H"], SPEC["stride"], SPEC["planner_depth"]
PLAN_STEPS, GOAL_RADIUS = 20, 0.5
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CKPT = (args.ckpt or SPEC["ckpt"]) + f"/{args.env}"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Env + dataset (normalizer) ─────────────────────────────────────────────────
print(f"Loading {args.env}  (family={FAM}, H={H}, stride={M}) …")
env, dataset = make_dataset(args.env, H=H, stride=M)
normalizer = dataset.get_normalizer()
OBS_DIM = dataset.o_dim

# ── Planner + MCSS critic + state-value V ──────────────────────────────────────
fix_mask = torch.zeros((H, OBS_DIM))
fix_mask[0, :OBS_DIM] = 1.0
planner = ContinuousDiffusionSDE(
    DiT1d(OBS_DIM, emb_dim=128, d_model=256, n_heads=4, depth=DEPTH,
          timestep_emb_type="fourier"),
    fix_mask=fix_mask, noise_schedule="linear", device=DEVICE,
    predict_noise=True, ema_rate=0.9999, loss_weight=torch.ones((H, OBS_DIM)))
planner.load(f"{CKPT}/planner_ckpt_{args.planner_step}.pt")
planner.eval()

critic = DVHorizonCritic(OBS_DIM, emb_dim=128, d_model=256, n_heads=4, depth=2,
                         norm_type="pre").to(DEVICE)
critic.load_state_dict(torch.load(f"{CKPT}/critic_ckpt_{args.critic_step}.pt",
                                  map_location=DEVICE, weights_only=False)["critic"])
critic.eval()

v_path = f"{CKPT}/state_value_ckpt_{args.value_step}.pt"
if not os.path.exists(v_path):
    sys.exit(f"state-value checkpoint not found: {v_path}\n"
             f"Train it first:  python scripts/train_state_value.py --env {args.env}")
value = load_state_value(v_path, device=DEVICE)
print(f"  loaded planner + MCSS critic + V(s)  (obs_dim={OBS_DIM})\n")

# ── Per-seed comparison ────────────────────────────────────────────────────────
hdr = (f"{'seed':>4}  {'start→goal':>10}  {'K_reach':>7}  "
       f"{'mcss':>12}  {'v_end':>12}  {'v_max':>12}  {'V~closeness':>11}")
print(hdr); print("-" * len(hdr))


def sel_reaches(scores, reach_mask):
    i = int(torch.argmax(scores).item())
    return bool(reach_mask[i]), i


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
        mcss_scores = critic(traj).squeeze(-1).cpu()
        v_all = value(traj[..., :OBS_DIM]).squeeze(-1).cpu()      # (K, H)
        v_end = v_all[:, -1]                                       # endpoint value
        v_max = v_all.max(dim=1).values                           # best state on the plan

    raw_traj = normalizer.unnormalize(
        traj.cpu().numpy().reshape(-1, OBS_DIM)).reshape(args.k, H, OBS_DIM)
    dist_all = np.linalg.norm(raw_traj[:, :, :2] - goal[None, None], axis=2)  # (K, H)
    min_dist = dist_all.min(axis=1)                                # (K,)
    reach_mask = min_dist < GOAL_RADIUS
    k_reach = int(reach_mask.sum())

    r_mcss, i_mcss = sel_reaches(mcss_scores, reach_mask)
    r_vend, i_vend = sel_reaches(v_end, reach_mask)
    r_vmax, i_vmax = sel_reaches(v_max, reach_mask)
    # calibration: does endpoint-V rank plans by closeness? corr(V_end, -min_dist)
    corr = (float(np.corrcoef(v_end.numpy(), -min_dist)[0, 1])
            if v_end.numpy().std() > 1e-8 else 0.0)

    def mark(reaches):
        return "REACH" if reaches else "miss "

    rows.append(dict(seed=seed, k_reach=k_reach, mcss=r_mcss, vend=r_vend, vmax=r_vmax,
                     idx_mcss=i_mcss, idx_vend=i_vend, idx_vmax=i_vmax))
    print(f"{seed:>4}  {start_dist:>10.2f}  {f'{k_reach}/{args.k}':>7}  "
          f"{mark(r_mcss):>12}  {mark(r_vend):>12}  {mark(r_vmax):>12}  {corr:>11.3f}")

print("=" * len(hdr))
n = len(rows)
have_reaching = [r for r in rows if r["k_reach"] > 0]   # only seeds where a fix is possible
fixed = [r for r in have_reaching if r["vend"] and not r["mcss"]]
regress = [r for r in have_reaching if r["mcss"] and not r["vend"]]
print(f"\nSelection summary for {args.env} (n={n} seeds; "
      f"{len(have_reaching)} have ≥1 reaching plan):")
print(f"  mcss selects reaching : {sum(r['mcss'] for r in rows)}/{n}")
print(f"  v_end selects reaching: {sum(r['vend'] for r in rows)}/{n}")
print(f"  v_max selects reaching: {sum(r['vmax'] for r in rows)}/{n}")
print(f"  critic-miss FIXED by v_end (v reaches where mcss missed): {len(fixed)}  "
      f"{[r['seed'] for r in fixed]}")
print(f"  regressions (mcss reached, v_end missed):                {len(regress)}  "
      f"{[r['seed'] for r in regress]}")
print("\nRead: v_end ≥ mcss on 'selects reaching' with 0 regressions ⇒ the retrained "
      "critic is a strict-or-better selector (Option 2 alone already helps). Then proceed "
      "to wire V into the segment-stitching backup (Option 1).")
