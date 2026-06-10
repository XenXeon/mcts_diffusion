"""mcts/mcts_loop.py

Closed-loop sampler comparison on the SAME harness:
    method="mcss"  — the DV baseline (batched planner -> critic argmax -> first waypoint)
    method="mcts"  — state-value look-ahead search (ValueForest) -> first waypoint

Both share the identical env loop, normalizer, inverse-dynamics policy, and per-step
cadence (replan every step, take one step), so any difference is the sampler alone.

Parallelism
-----------
The MCTS search grows M trees (one per parallel env) in lockstep: every expansion round
batches all M trees' candidate states into ONE planner.sample + value pass (see expand_fn).
So a search of `budget` rounds costs ~budget+1 batched planner calls per env-step, each of
shape (M * k_mcts, H, obs_dim) — not M*budget separate calls.

Reuses DV's trained planner + inverse-dynamics policy untouched; only the value head is new.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from cleandiffuser.diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DiT1d, DVInvMlp
from cleandiffuser.utils import DVHorizonCritic
from mcts.value_forest import ForestConfig, ValueForest
from mcts.value_net import load_state_value


def env_family(env_name: str) -> str:
    return "maze2d" if env_name.startswith("maze2d") else "antmaze"


SPECS = {
    "maze2d": dict(
        H=32, stride=15, planner_depth=2, max_path_length=800,
        ckpt=("results/veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer"
              "_d2_width256_separate_dpTrue")),
    "antmaze": dict(
        H=40, stride=25, planner_depth=8, max_path_length=1000,
        ckpt=("results/veteran_d4rl_antmaze_H40_Jump25_next1_MCSS_transformer"
              "_d8_width256_separate_dp1")),
}
# Dataset target cfg (only the normalizer + dims are used here; matches the pipeline).
TARGET_CFG = dict(discount=1.0, continous_reward_at_done=True,
                  reward_tune="iql", center_mapping=True)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_models(env_name: str, value_step: str = "latest",
                planner_step: int = 1000000, critic_step: int = 1000000,
                policy_step: int = 1000000, device: Optional[str] = None,
                ckpt_dir: Optional[str] = None) -> Dict[str, Any]:
    import d4rl  # noqa: F401
    import gym

    fam = env_family(env_name)
    spec = SPECS[fam]
    H, stride, depth = spec["H"], spec["stride"], spec["planner_depth"]
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env_single = gym.make(env_name)
    raw = env_single.get_dataset()
    if fam == "maze2d":
        from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset
        ds = DV_D4RLMaze2DSeqDataset(raw, horizon=H, stride=stride, learn_policy=False,
                                     **TARGET_CFG)
    else:
        from cleandiffuser.dataset.d4rl_antmaze_dataset import DV_D4RLAntmazeSeqDataset
        ds = DV_D4RLAntmazeSeqDataset(raw, horizon=H, stride=stride, learn_policy=False,
                                      **TARGET_CFG)
    normalizer = ds.get_normalizer()
    obs_dim, act_dim = ds.o_dim, ds.a_dim
    ckpt = (ckpt_dir or spec["ckpt"]) + f"/{env_name}"

    fix_mask = torch.zeros((H, obs_dim))
    fix_mask[0, :obs_dim] = 1.0
    planner = ContinuousDiffusionSDE(
        DiT1d(obs_dim, emb_dim=128, d_model=256, n_heads=4, depth=depth,
              timestep_emb_type="fourier"),
        fix_mask=fix_mask, noise_schedule="linear", device=device,
        predict_noise=True, ema_rate=0.9999, loss_weight=torch.ones((H, obs_dim)))
    planner.load(f"{ckpt}/planner_ckpt_{planner_step}.pt")
    planner.eval()

    critic = DVHorizonCritic(obs_dim, emb_dim=128, d_model=256, n_heads=4, depth=2,
                             norm_type="pre").to(device)
    critic.load_state_dict(torch.load(f"{ckpt}/critic_ckpt_{critic_step}.pt",
                                      map_location=device, weights_only=False)["critic"])
    critic.eval()

    value = load_state_value(f"{ckpt}/state_value_ckpt_{value_step}.pt", device=device)

    policy = DiscreteDiffusionSDE(
        DVInvMlp(obs_dim, act_dim, emb_dim=64, hidden_dim=256,
                 timestep_emb_type="positional").to(device),
        IdentityCondition(dropout=0.0).to(device),
        predict_noise=True, optim_params={"lr": 3e-4},
        x_max=+1. * torch.ones((1, act_dim), device=device),
        x_min=-1. * torch.ones((1, act_dim), device=device),
        diffusion_steps=10, ema_rate=0.995, device=device)
    policy.load(f"{ckpt}/policy_ckpt_{policy_step}.pt")
    policy.eval()

    print(f"[{env_name}] loaded planner+critic+V+policy on {device} "
          f"(obs_dim={obs_dim}, act_dim={act_dim}, H={H}, stride={stride})")
    return dict(planner=planner, critic=critic, value=value, policy=policy,
                normalizer=normalizer, obs_dim=obs_dim, act_dim=act_dim, H=H,
                stride=stride, max_path_length=spec["max_path_length"],
                env_single=env_single, env_name=env_name, device=device, family=fam)


# ── Sampler (MCSS + MCTS share the policy & env loop) ──────────────────────────

class Sampler:
    def __init__(self, models: Dict[str, Any], k_mcss: int = 50, k_mcts: int = 16,
                 budget: int = 15, child_index: int = 1, c_ucb: float = 1.4142136,
                 plan_steps: int = 20, policy_steps: int = 10, planner_temp: float = 1.0,
                 policy_temp: float = 0.5, solver: str = "ddim",
                 policy_solver: str = "ddpm", rebase: bool = True) -> None:
        self.m = models
        self.dev = models["device"]
        self.obs_dim = models["obs_dim"]
        self.act_dim = models["act_dim"]
        self.H = models["H"]
        self.k_mcss, self.k_mcts, self.budget = k_mcss, k_mcts, budget
        self.child_index, self.c_ucb = child_index, c_ucb
        self.plan_steps, self.policy_steps = plan_steps, policy_steps
        self.planner_temp, self.policy_temp = planner_temp, policy_temp
        self.solver, self.policy_solver, self.rebase = solver, policy_solver, rebase
        if not (1 <= child_index < self.H):
            raise ValueError(f"child_index must be in [1, H-1]={self.H-1}, got {child_index}")

    # one batched planner+value pass; the search's only GPU touch-point
    def expand_fn(self, states: List[np.ndarray]):
        K, H, D = self.k_mcts, self.H, self.obs_dim
        B = len(states)
        s = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.dev)  # (B,D)
        prior = torch.zeros((B * K, H, D), device=self.dev)
        prior[:, 0, :] = s.repeat_interleave(K, dim=0)
        with torch.no_grad():
            trajs, _ = self.m["planner"].sample(
                prior, solver=self.solver, n_samples=B * K, sample_steps=self.plan_steps,
                use_ema=True, condition_cfg=None, w_cfg=1.0, temperature=self.planner_temp)
            child_states = trajs[:, self.child_index, :D]      # (B*K, D) tree child = L ahead
            # Action target is ALWAYS the immediate next waypoint traj[1], regardless of
            # child_index: the tree stitches in L-step segments, but we still execute exactly
            # one step per env-step (then replan). first_wp matters only for root children.
            first_wps = trajs[:, 1, :D]                         # (B*K, D)
            vvals = self.m["value"](child_states).squeeze(-1)  # (B*K,)
        cs = child_states.cpu().numpy().reshape(B, K, D)
        fw = first_wps.cpu().numpy().reshape(B, K, D)
        vv = vvals.cpu().numpy().reshape(B, K)
        return [([cs[i, j] for j in range(K)],
                 [fw[i, j] for j in range(K)],
                 [float(vv[i, j]) for j in range(K)]) for i in range(B)]

    def mcts_waypoints(self, s_norm: np.ndarray) -> np.ndarray:
        M = s_norm.shape[0]
        forest = ValueForest([s_norm[i] for i in range(M)], self.expand_fn,
                             ForestConfig(k=self.k_mcts, budget=self.budget, c_ucb=self.c_ucb))
        forest.run()
        wps = forest.best_first_waypoints()
        return np.stack([wps[i] if wps[i] is not None else s_norm[i] for i in range(M)])

    def mcss_waypoints(self, s_norm: np.ndarray) -> np.ndarray:
        M, K, H, D = s_norm.shape[0], self.k_mcss, self.H, self.obs_dim
        s = torch.as_tensor(s_norm, dtype=torch.float32, device=self.dev)
        prior = torch.zeros((M * K, H, D), device=self.dev)
        prior[:, 0, :] = s.repeat_interleave(K, dim=0)
        with torch.no_grad():
            trajs, _ = self.m["planner"].sample(
                prior, solver=self.solver, n_samples=M * K, sample_steps=self.plan_steps,
                use_ema=True, condition_cfg=None, w_cfg=1.0, temperature=self.planner_temp)
            scores = self.m["critic"](trajs).squeeze(-1).view(M, K)
            idx = scores.argmax(dim=1)
            trajs = trajs.view(M, K, H, D)
            best = trajs[torch.arange(M, device=self.dev), idx]   # (M, H, D)
            wp = best[:, 1, :D].cpu().numpy()
        return wp

    def policy_action(self, s_norm: np.ndarray, next_wp: np.ndarray) -> np.ndarray:
        M = s_norm.shape[0]
        obs_r = torch.as_tensor(s_norm, dtype=torch.float32, device=self.dev).clone()
        next_r = torch.as_tensor(next_wp, dtype=torch.float32, device=self.dev).clone()
        if self.rebase:
            next_r[:, :2] -= obs_r[:, :2]
            obs_r[:, :2] = 0.0
        prior = torch.zeros((M, self.act_dim), device=self.dev)
        with torch.no_grad():
            act, _ = self.m["policy"].sample(
                prior, solver=self.policy_solver, n_samples=M,
                sample_steps=self.policy_steps,
                condition_cfg=torch.cat([obs_r, next_r], dim=-1), w_cfg=1.0,
                use_ema=True, temperature=self.policy_temp)
        return act.cpu().numpy()


# ── Closed-loop evaluation ──────────────────────────────────────────────────────

def run_episodes(sampler: Sampler, method: str, n_envs: int, n_episodes: int,
                 seed: int = 0, max_steps: Optional[int] = None,
                 verbose: bool = True) -> Dict[str, Any]:
    import gym
    from pipelines.utils import set_seed

    assert method in ("mcss", "mcts")
    m = sampler.m
    env_name = m["env_name"]
    normalizer = m["normalizer"]
    env_single = m["env_single"]
    max_t = max_steps or m["max_path_length"]

    set_seed(seed)
    env = gym.vector.make(env_name, n_envs)
    try:
        env.seed(seed)
    except Exception:
        pass

    all_success: List[np.ndarray] = []
    t0 = time.perf_counter()
    for ep in range(n_episodes):
        obs = env.reset()
        ep_rew = np.zeros(n_envs, dtype=np.float64)
        active = np.ones(n_envs, dtype=bool)   # still in the FIRST episode (count rewards)
        for t in range(max_t):
            s_norm = normalizer.normalize(obs).astype(np.float32)   # (n_envs, obs_dim)
            if method == "mcss":
                next_wp = sampler.mcss_waypoints(s_norm)
            else:
                next_wp = sampler.mcts_waypoints(s_norm)
            act = sampler.policy_action(s_norm, next_wp)
            obs, rew, done, info = env.step(act)
            # gym.vector auto-resets an env on done; count rewards only within each env's
            # first episode (freeze once done) so a reset second episode can't be mixed in.
            ep_rew += np.asarray(rew, dtype=np.float64) * active
            active &= ~np.asarray(done, dtype=bool)
            if not active.any():
                break
        succ = np.clip(ep_rew, 0.0, 1.0)
        all_success.append(succ)
        if verbose:
            print(f"  [{method}] ep {ep+1}/{n_episodes}  reach={succ.mean()*100:5.1f}%  "
                  f"elapsed={time.perf_counter()-t0:6.0f}s")
    env.close()

    flat = np.concatenate(all_success)
    norm = np.array([env_single.get_normalized_score(x) for x in flat]) * 100.0
    out = dict(method=method, n_rollouts=int(flat.size),
               reach_pct=float(flat.mean() * 100.0),
               norm_mean=float(norm.mean()),
               norm_err=float(norm.std() / np.sqrt(flat.size)),
               wall_s=round(time.perf_counter() - t0, 1))
    if verbose:
        print(f"  [{method}] DONE  reach={out['reach_pct']:.1f}%  "
              f"norm={out['norm_mean']:.1f}±{out['norm_err']:.1f}  "
              f"(n={out['n_rollouts']}, {out['wall_s']:.0f}s)")
    return out
