"""mcts/mctd_loop.py

Closed-loop evaluation for the faithful MCTD planner (mcts/mctd_planner.py),
Phase 2 of the MCTD port. MCTD is an MPC TRAJECTORY planner (the reference runs
it as: plan a whole trajectory, follow its waypoints open-loop for a horizon,
replan), NOT a per-step waypoint proposer like MCSS/MCTS — and a full tree search
every env-step would be intractable. So this is a SEPARATE loop rather than a
method= branch of mcts_loop.run_episodes (which replans every step); run_episodes
is left untouched so every existing arm stays bit-identical.

Comparability is preserved on purpose:
  * identical env construction + seeding to run_episodes, so goals are the same
    pure function of (seed, index) and MCTD rollouts PAIR with MCSS/MCTS rollouts
    on (seed, index) for McNemar (scripts/collate_mcts.py);
  * the SAME DV inverse-dynamics policy and rebasing produce the executed action
    from (state, target waypoint) — so planner+search is the only thing that
    differs from the DV arms;
  * the SAME DV-exact per-family accounting (maze2d camping latch, antmaze reach
    clip) and the SAME result-dict schema as run_episodes.

Execution cadence: replan every `replan_every` env-steps; between replans follow
the planned trajectory by advancing to the next stride-spaced waypoint once the
agent is within `reach_wp` (world units) of the current target — the reference's
sub-goal-advancement idea, adapted to this repo's waypoint plans.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from mcts.df_schedule import pyramid_matrix
from mcts.mctd_denoise import denoise_rows, fresh_plan
from mcts.mctd_guidance import GoalGuide
from mcts.mctd_verify import MCTD_ENV
from mcts.specs import env_family, get_goal, normalize_goal_xy


class MCSSMPCPlanner:
    """Best-of-K MCSS as a drop-in plan provider for run_mctd_episodes — the
    controlled baseline that isolates SEARCH from EXECUTION. It exposes the same
    .plan(s_norm, goal_raw, seed) -> {plan_norm, ...} interface and .pos_dims as
    MCTDPlanner, so the MPC harness (replan cadence, waypoint-following, DV
    inverse-dynamics policy) is byte-identical; the ONLY difference from an MCTD
    run is how the plan is produced:

        MCTD       — tree search over the denoising axis, geometric value;
        MCSS (here)— sample K plans from the SAME DF planner, rank by the SAME
                     DV critic, return the best FULL trajectory.

    So MCTD-MPC vs MCSS-MPC differs only in search-vs-flat (same backbone, same
    execution) — the clean control for "is MCTD's loss the weaker value or the
    MPC execution model?". Plan generation is byte-identical to
    mcts_loop.Sampler.mcss_waypoints (DF backbone), just returning the whole
    ranked-best window instead of only its first waypoint. Goal-agnostic exactly
    as MCSS is (the critic ranks; goal_raw is ignored).

    backbone="df" uses the DF planner (the search-isolating control, same backbone
    as MCTD); backbone="dv" uses the frozen DV full-sequence planner (the SOTA
    baseline). Running backbone="dv" in this harness gives DV-MCSS at an arbitrary
    replan cadence, the control that disentangles the DF-vs-DV backbone gap from the
    per-step-vs-MPC cadence gap (an absolute-score confound flagged in review:
    DF-MCSS is 183.4 per-step but 238.8 at rp50, same backbone, so raw scores must
    never be juxtaposed across cadences)."""

    def __init__(self, models: Dict[str, Any], family: str, k: int = 50,
                 temperature: float = 1.0, slope: int = 1, row_stride: int = 1,
                 backbone: str = "df", plan_steps: int = 20, solver: str = "ddim"):
        if backbone == "df":
            if models.get("df_planner") is None:
                raise ValueError("MCSSMPCPlanner backbone='df' needs a DF planner (df_ckpt=)")
            self.p = models["df_planner"]
        elif backbone == "dv":
            self.p = models["planner"]                 # frozen DV full-sequence planner
        else:
            raise ValueError(f"backbone must be 'df' or 'dv', got {backbone!r}")
        self.backbone = backbone
        self.critic = models["critic"]
        self.dev = models["device"]
        self.H = models["H"]
        self.obs_dim = models["obs_dim"]
        self.k = int(k)
        self.temperature, self.slope, self.row_stride = temperature, slope, row_stride
        self.plan_steps, self.solver = int(plan_steps), solver
        self.pos_dims = tuple(MCTD_ENV[family]["pos_dims"])

    @torch.no_grad()
    def plan(self, s_norm: np.ndarray, goal_raw=None, seed: int = 0) -> Dict[str, Any]:
        K, H, D = self.k, self.H, self.obs_dim
        s = torch.as_tensor(np.asarray(s_norm, dtype=np.float32),
                            device=self.dev).view(1, D)
        prior = torch.zeros((K, H, D), device=self.dev)
        prior[:, 0, :] = s
        if self.backbone == "df":
            trajs = self.p.sample(prior, torch.ones(K, dtype=torch.long, device=self.dev),
                                  H, slope=self.slope, row_stride=self.row_stride,
                                  temperature=self.temperature)             # (K,H,D)
        else:                                          # DV: matches mcss_waypoints exactly
            trajs, _ = self.p.sample(prior, solver=self.solver, n_samples=K,
                                     sample_steps=self.plan_steps, use_ema=True,
                                     condition_cfg=None, w_cfg=1.0,
                                     temperature=self.temperature)          # (K,H,D)
        scores = self.critic(trajs).squeeze(-1)                            # (K,)
        idx = int(scores.argmax())
        best = trajs[idx]                                                  # (H,D)
        return dict(plan_norm=best.detach().cpu().numpy(), solved=True,
                    info=f"mcss_mpc_{self.backbone}", value=float(scores[idx]),
                    achieved_t=None, n_search=self.k, n_nodes=self.k, max_depth=0)


class GuidedBoNPlanner:
    """Way 4b: flat best-of-N over GUIDANCE WEIGHTS, critic-ranked, NO tree — the
    ablation that keeps MCTD's 'guidance scale is a knob' idea but drops the
    denoising search. For each guidance weight in the menu it draws k_per fully-
    denoised guided plans, pools them, and returns the one the DV critic scores
    best. Same .plan/.pos_dims contract as MCTDPlanner, so it runs through the
    identical MPC harness; it differs from MCSSMPCPlanner only in where the
    candidate diversity comes from — varying the guidance weight (0..2) rather
    than sampling noise alone — isolating 'does guidance-diversity help the flat
    pool?' from the tree. Total candidates N = len(menu) * k_per."""

    def __init__(self, models: Dict[str, Any], family: str,
                 guidance_scales=(0.0, 0.1, 0.5, 1.0, 2.0), k_per: int = 10,
                 reach_scale: float = 2.0, slope: int = 1, row_stride: int = 1,
                 temperature: float = 1.0):
        if models.get("df_planner") is None:
            raise ValueError("GuidedBoNPlanner needs a DF planner (df_ckpt=)")
        self.p = models["df_planner"]
        self.critic = models["critic"]
        self.normalizer = models["normalizer"]
        self.dev = models["device"]
        self.H = models["H"]
        self.obs_dim = models["obs_dim"]
        self.guidance_scales = list(guidance_scales)
        self.k_per = int(k_per)
        self.reach_scale, self.slope, self.row_stride = reach_scale, slope, row_stride
        self.temperature = temperature
        self.pos_dims = tuple(MCTD_ENV[family]["pos_dims"])

    @torch.no_grad()
    def plan(self, s_norm: np.ndarray, goal_raw, seed: int = 0) -> Dict[str, Any]:
        H, D = self.H, self.obs_dim
        mat = pyramid_matrix(self.p.K, H, slope=self.slope, row_stride=self.row_stride)
        s = torch.as_tensor(np.asarray(s_norm, dtype=np.float32),
                            device=self.dev).view(1, D)
        x_hist = torch.zeros((self.k_per, H, D), device=self.dev)
        x_hist[:, 0, :] = s
        goal_world = np.asarray(goal_raw, dtype=np.float64).reshape(-1)[:len(self.pos_dims)]
        goal_norm = normalize_goal_xy(self.normalizer, goal_world.astype(np.float32))
        guide = GoalGuide(goal_norm, pos_dims=self.pos_dims,
                          reach_scale=self.reach_scale, device=self.dev)
        cands = []
        for gsc in self.guidance_scales:
            gd, w = (guide, float(gsc)) if gsc else (None, 0.0)
            x = fresh_plan(self.p, x_hist, 1, self.temperature)
            clean = denoise_rows(self.p, x, mat, hist_len=1, x_hist=x_hist,
                                 guide=gd, w=w)                     # (k_per,H,D)
            cands.append(clean)
        allc = torch.cat(cands, 0)                                 # (N,H,D)
        scores = self.critic(allc).reshape(-1)                     # (N,)
        idx = int(scores.argmax())
        N = allc.shape[0]
        return dict(plan_norm=allc[idx].detach().cpu().numpy(), solved=True,
                    info="guided_bon", value=float(scores[idx]), achieved_t=None,
                    n_search=N, n_nodes=N, max_depth=0)


class DFTreeMPCPlanner:
    """This project's DF-tree (trajectory-axis look-ahead + DV critic) as a
    drop-in plan provider for run_mctd_episodes, so it can be driven at any
    replan cadence. The DF-tree's native harness (mcts_loop.run_episodes)
    re-plans every step and takes only the first waypoint; here we take its full
    committed plan and follow it under the SAME MPC cadence as MCTD-critic and
    MCSS-MPC. This is the cadence-matched control that makes the DF-tree
    raw-comparable to MCTD-critic (both DF backbone, same DV critic, same rp50):
    it answers whether the DF-tree's per-step win survives at the MPC cadence.

    Wraps mcts_loop.Sampler (backbone='df', value_mode='critic') and calls its
    additive mcts_best_plans (the full stitched best-branch plan)."""

    def __init__(self, models: Dict[str, Any], family: str, budget: int = 15,
                 k_mcts: int = 16, k_root: Optional[int] = 16, top_m: int = 3,
                 c_ucb: float = 1.4142136, child_index: int = 1):
        from mcts.mcts_loop import Sampler          # lazy: avoids import cycle
        if models.get("df_planner") is None:
            raise ValueError("DFTreeMPCPlanner needs a DF planner (df_ckpt=)")
        self.sampler = Sampler(models, k_mcss=50, k_mcts=k_mcts, budget=budget,
                               child_index=child_index, c_ucb=c_ucb,
                               value_mode="critic", k_root=k_root, top_m=top_m,
                               backbone="df")
        self.pos_dims = tuple(MCTD_ENV[family]["pos_dims"])
        self.budget = budget

    def plan(self, s_norm: np.ndarray, goal_raw=None, seed: int = 0) -> Dict[str, Any]:
        plans = self.sampler.mcts_best_plans(
            np.asarray(s_norm, dtype=np.float32)[None])        # (1, H, D)
        stt = (self.sampler.last_tree_stats or [{}])[0]
        return dict(plan_norm=plans[0], solved=True, info="df_tree",
                    value=float(stt.get("root_best_value", 0.0)), achieved_t=None,
                    n_search=self.budget, n_nodes=int(stt.get("n_nodes", 0)),
                    max_depth=int(stt.get("max_depth", 0)))


def _policy_action(models: Dict[str, Any], s_norm: np.ndarray, next_wp: np.ndarray,
                   rebase: bool, policy_solver: str = "ddpm",
                   policy_steps: int = 10, policy_temp: float = 0.5) -> np.ndarray:
    """Byte-for-byte the same action production as Sampler.policy_action
    (mcts_loop.py): DV diffusion inverse-dynamics policy on (state, next
    waypoint), with the DV separate-pipeline xy rebasing."""
    policy = models["policy"]
    dev = models["device"]
    act_dim = models["act_dim"]
    M = s_norm.shape[0]
    obs_r = torch.as_tensor(s_norm, dtype=torch.float32, device=dev).clone()
    next_r = torch.as_tensor(next_wp, dtype=torch.float32, device=dev).clone()
    if rebase:
        next_r[:, :2] -= obs_r[:, :2]
        obs_r[:, :2] = 0.0
    prior = torch.zeros((M, act_dim), device=dev)
    with torch.no_grad():
        act, _ = policy.sample(prior, solver=policy_solver, n_samples=M,
                               sample_steps=policy_steps,
                               condition_cfg=torch.cat([obs_r, next_r], dim=-1),
                               w_cfg=1.0, use_ema=True, temperature=policy_temp)
    return act.cpu().numpy()


def run_mctd_episodes(planner, models: Dict[str, Any], n_envs: int,
                      n_episodes: int, seed: int = 0, max_steps: Optional[int] = None,
                      replan_every: int = 30, reach_wp: float = 1.0,
                      rebase: bool = True, verbose: bool = True,
                      dv_log: bool = False, method_label: str = "mctd") -> Dict[str, Any]:
    """Closed-loop MPC eval; returns the SAME aggregates + per-rollout vectors as
    mcts_loop.run_episodes, plus search diagnostics.

    planner: any plan provider with .plan(s_norm, goal_raw, seed) -> {plan_norm,
             solved, n_search, n_nodes, max_depth} and .pos_dims — MCTDPlanner
             (the tree) or MCSSMPCPlanner (the best-of-K control). method_label
             tags the output ("mctd" / "mcss_mpc") so both run through the
             identical harness and pair in scripts/collate_mctd.py.
    models:  the mcts_loop.load_models(...) dict (needs df_planner, policy,
             normalizer, env_single, env_name, ...).
    """
    import gym
    from pipelines.utils import set_seed

    env_name = models["env_name"]
    normalizer = models["normalizer"]
    env_single = models["env_single"]
    obs_dim = models["obs_dim"]
    H = models["H"]
    pos_dims = list(planner.pos_dims)
    max_t = max_steps or models["max_path_length"]
    fam = env_family(env_name)

    set_seed(seed)
    env = gym.vector.make(env_name, n_envs, asynchronous=False)
    # identical seeding to run_episodes -> identical (seed, index) goals for pairing
    seed_failures = []
    try:
        env.seed(seed)
    except Exception as exc:
        seed_failures.append(f"VectorEnv.seed: {exc!r}")
    for i, e in enumerate(getattr(env, "envs", None) or []):
        try:
            e.seed(seed + i)
            e.action_space.seed(seed + i)
        except Exception as exc:
            seed_failures.append(f"env[{i}].seed: {exc!r}")
    if seed_failures:
        print(f"  [{method_label}] WARNING: seeding incomplete "
              f"({'; '.join(seed_failures[:3])})")

    all_success, all_reach_step, all_starts = [], [], []
    all_goals, all_dv = [], []
    # MCTD search diagnostics (averaged over every plan() call)
    diag = dict(n_plans=0, solved_plans=0, sum_search=0, sum_nodes=0, sum_depth=0)
    t0 = time.perf_counter()

    for ep in range(n_episodes):
        obs = env.reset()
        all_starts.append(np.asarray(obs)[:, :2].astype(np.float64).copy())
        try:
            goals_raw = np.asarray([get_goal(e) for e in env.envs], dtype=np.float64)
            all_goals.append(goals_raw.tolist())
        except Exception as e:
            print(f"  [{method_label}] WARNING: could not read per-env goals ({e!r})")
            all_goals.append([None] * n_envs)
            raise RuntimeError("MPC loop needs per-env goals but none could be read") from e

        # per-env MPC state
        plans_norm: List[Optional[np.ndarray]] = [None] * n_envs
        plans_world: List[Optional[np.ndarray]] = [None] * n_envs
        ptr = np.ones(n_envs, dtype=np.int64)
        since_replan = np.full(n_envs, replan_every, dtype=np.int64)  # force t=0 replan

        ep_rew = np.zeros(n_envs, dtype=np.float64)
        reach_step = np.full(n_envs, -1, dtype=np.int64)
        active = np.ones(n_envs, dtype=bool)
        dv_acc = np.zeros(n_envs, dtype=np.float64)
        dv_finished = np.zeros(n_envs, dtype=bool)

        for t in range(max_t):
            s_norm = normalizer.normalize(obs).astype(np.float32)
            next_wp = s_norm.copy()          # normalized target; inactive envs -> self
            for i in range(n_envs):
                if not active[i]:
                    continue
                if (plans_norm[i] is None or since_replan[i] >= replan_every
                        or ptr[i] >= H):
                    out = planner.plan(s_norm[i], goals_raw[i], seed=seed + i)
                    plans_norm[i] = out["plan_norm"]
                    plans_world[i] = normalizer.unnormalize(out["plan_norm"])
                    ptr[i] = 1
                    since_replan[i] = 0
                    diag["n_plans"] += 1
                    diag["solved_plans"] += int(out["solved"])
                    diag["sum_search"] += out["n_search"]
                    diag["sum_nodes"] += out["n_nodes"]
                    diag["sum_depth"] += out["max_depth"]
                # advance to the next waypoint once we're near the current target
                cur_xy = np.asarray(obs[i])[pos_dims]
                while (ptr[i] < H - 1 and
                       np.linalg.norm(plans_world[i][ptr[i]][pos_dims] - cur_xy) < reach_wp):
                    ptr[i] += 1
                next_wp[i] = plans_norm[i][ptr[i]]
                since_replan[i] += 1

            act = _policy_action(models, s_norm, next_wp, rebase)
            obs, rew, done, info = env.step(act)

            rew = np.asarray(rew, dtype=np.float64)
            hit = active & (rew > 0.0) & (reach_step < 0)
            reach_step[hit] = t
            ep_rew += rew * active
            if fam == "maze2d":
                dv_finished |= (rew == 1.0)
                dv_acc += dv_finished
            else:
                dv_acc += rew
            active &= ~np.asarray(done, dtype=bool)
            if dv_log:
                print(f"[t={t+1}] rew: {dv_acc}")
            if not active.any() and not dv_log:
                break

        succ = np.clip(ep_rew, 0.0, 1.0)
        all_success.append(succ)
        all_reach_step.append(reach_step)
        if fam == "maze2d":
            all_dv.append(dv_acc)
        elif fam == "kitchen":
            all_dv.append(np.clip(dv_acc, 0.0, 4.0))
        else:
            all_dv.append(np.clip(dv_acc, 0.0, 1.0))
        if verbose:
            print(f"  [{method_label}] ep {ep+1}/{n_episodes}  "
                  f"reach={succ.mean()*100:5.1f}%  "
                  f"elapsed={time.perf_counter()-t0:6.0f}s")
    env.close()

    flat = np.concatenate(all_success)
    flat_reach_step = np.concatenate(all_reach_step)
    norm = np.array([env_single.get_normalized_score(x) for x in flat]) * 100.0
    p = float(flat.mean())
    flat_dv = np.concatenate(all_dv)
    dv_norm = np.array([env_single.get_normalized_score(x) for x in flat_dv]) * 100.0
    np_ = max(1, diag["n_plans"])
    out = dict(method=method_label, n_rollouts=int(flat.size),
               reach_pct=float(p * 100.0),
               reach_err=float(np.sqrt(p * (1.0 - p) / flat.size) * 100.0),
               norm_mean=float(norm.mean()),
               norm_err=float(norm.std() / np.sqrt(flat.size)),
               dv_norm_mean=float(dv_norm.mean()),
               dv_norm_err=float(dv_norm.std() / np.sqrt(flat_dv.size)),
               wall_s=round(time.perf_counter() - t0, 1),
               success=[int(x > 0) for x in flat],
               dv_norm=[float(x) for x in dv_norm],
               reach_step=[int(s) if s >= 0 else None for s in flat_reach_step],
               starts=[xy.tolist() for ep_s in all_starts for xy in ep_s],
               goals=[g for ep_g in all_goals for g in ep_g],
               # MCTD search diagnostics
               n_plans=diag["n_plans"],
               solved_plan_frac=round(diag["solved_plans"] / np_, 3),
               mean_search=round(diag["sum_search"] / np_, 2),
               mean_nodes=round(diag["sum_nodes"] / np_, 2),
               tree_depth_mean=round(diag["sum_depth"] / np_, 2))
    if verbose:
        print(f"  [{method_label}] DONE  reach={out['reach_pct']:.1f}%±{out['reach_err']:.1f}  "
              f"norm={out['norm_mean']:.1f}±{out['norm_err']:.1f}  "
              f"(n={out['n_rollouts']}, {out['wall_s']:.0f}s)  "
              f"plans={out['n_plans']} solved_frac={out['solved_plan_frac']} "
              f"mean_depth={out['tree_depth_mean']}")
    return out
