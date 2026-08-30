"""mcts/mctd_planner.py

Faithful Monte Carlo Tree Diffusion (MCTD, Yoon et al., ICML 2025) planner on
this repo's D4RL Diffusion-Forcing stack — the "Way 1" full port. It reproduces
the reference algorithm (mctd-main/.../df_planning.py::p_mctd_plan, sequential /
parallel_search_num=1 config) end to end:

  Selection  (UCT over guidance-scale children; mcts/mctd_tree.py)
  Expansion  (advance the plan one denoising BLOCK under the chosen guidance
              scale; mcts/mctd_denoise.py block_rows + denoise_rows)
  Simulation (JUMPY-denoise the child to clean; score with the geometric goal-
              reach verifier; mcts/mctd_verify.py) — with bad-plan resampling
  Backprop   (MAX-backup up to the root)

Only the ENVIRONMENT (D4RL, not OGBench) and the BACKBONE (this repo's DFPlanner,
not the reference's DF) differ; the algorithm is the same. The value function is
the reference's non-learned geometric heuristic — no learned critic — so this is
MCTD as published, not a hybrid.

Scope: geometric MCTD needs a positional goal, i.e. maze2d and antmaze. Kitchen
has no positional goal and is refused here (it would need a grounded verifier — a
separate Way-4c variant; see mcts/mctd_verify.py). The design is otherwise
env-agnostic: horizon, stride, checkpoint and the per-family verifier config all
come from mcts/specs.py + mcts/mctd_verify.py::MCTD_ENV, so adding maze2d-medium,
antmaze-large-play, etc. is config, not code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mcts.df_schedule import pyramid_matrix
from mcts.mctd_denoise import block_rows, denoise_rows, fresh_plan, jumpy_rows
from mcts.mctd_guidance import GoalGuide
from mcts.mctd_tree import (ExpandResult, MCTDSearchConfig, MCTDTreeNode,
                            run_mctd_search)
from mcts.mctd_verify import MCTD_ENV, geometric_values, is_degenerate
from mcts.specs import normalize_goal_xy


@dataclass
class MCTDConfig:
    """MCTD search knobs (defaults = the reference's, adapted to our schedule).

    n_depths: tree depth = number of denoising BLOCKS the schedule is cut into
              (the reference sets this via mctd_num_denoising_steps; we set the
              depth directly and derive the block size, so it adapts across
              horizons). terminal_depth == n_depths.
    """
    guidance_scales: Sequence[float] = (0.0, 0.1, 0.5, 1.0, 2.0)
    n_depths: int = 3
    skip_level_steps: int = 10          # jumpy-rollout subsampling (simulation)
    max_search_num: int = 64            # expansions per plan() call
    c_ucb: float = math.sqrt(2.0)
    num_tries_for_bad_plans: int = 3    # resample degenerate (non-moving) plans
    early_stopping: Optional[str] = "achieved"
    temperature: float = 1.0
    reach_scale: float = 2.0            # GoalGuide tanh softening (normalized)
    slope: int = 1                      # pyramid schedule slope
    row_stride: int = 1                 # pyramid schedule row subsampling


class MCTDPlanner:
    """MCTD denoising-axis tree. value_mode selects the node value:

      "geometric" — Way 1, the reference's non-learned goal-reach heuristic
                    (mcts/mctd_verify.py). Needs a positional goal.
      "critic"    — Way 4c: the SAME tree/guidance/rollout, but the jumpy-
                    denoised clean plan is scored by the DV trajectory critic
                    (the strong learned selector MCSS uses) instead of geometry.
                    Isolates MCTD's search STRUCTURE from its weak value — the
                    controlled test of "does the denoising tree help once the
                    value is good?". No reach/achieved concept, so early
                    stopping is disabled and the output is the max-critic plan.
    """

    def __init__(self, df_planner, normalizer, family: str, obs_dim: int,
                 H: int, cfg: Optional[MCTDConfig] = None,
                 env_cfg: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None, value_mode: str = "geometric",
                 critic=None):
        if value_mode not in ("geometric", "critic"):
            raise ValueError(f"value_mode must be 'geometric' or 'critic', "
                             f"got {value_mode!r}")
        # geometric needs a positional goal (kitchen has none); critic does not,
        # but this port still only wires maze2d/antmaze (pos_dims for the MPC
        # waypoint-following live in MCTD_ENV).
        if family not in MCTD_ENV and env_cfg is None:
            raise ValueError(
                f"MCTD is only wired for maze2d / antmaze here (got family="
                f"{family!r}); kitchen needs the grounded verifier variant")
        if df_planner is None:
            raise ValueError("MCTDPlanner needs a DF planner (load_models df_ckpt=)")
        if value_mode == "critic" and critic is None:
            raise ValueError("value_mode='critic' (Way 4c) needs the DV critic "
                             "(pass critic=models['critic'])")
        self.p = df_planner
        self.normalizer = normalizer
        self.family = family
        self.obs_dim = obs_dim
        self.H = H
        self.cfg = cfg or MCTDConfig()
        self.env = dict(env_cfg or MCTD_ENV[family])
        self.pos_dims = tuple(int(d) for d in self.env["pos_dims"])
        self.goal_radius = float(self.env["goal_radius"])
        self.warp_threshold = self.env.get("warp_threshold")
        self.dev = device or self.p.dev
        self.value_mode = value_mode
        self.critic = critic
        if len(self.cfg.guidance_scales) < 1:
            raise ValueError("guidance_scales must be non-empty")

    @classmethod
    def from_models(cls, models: Dict[str, Any],
                    cfg: Optional[MCTDConfig] = None,
                    value_mode: str = "geometric") -> "MCTDPlanner":
        """Build from an mcts.mcts_loop.load_models(...) dict (must have been
        loaded with df_ckpt=... so models['df_planner'] is present)."""
        return cls(df_planner=models.get("df_planner"),
                   normalizer=models["normalizer"], family=models["family"],
                   obs_dim=models["obs_dim"], H=models["H"], cfg=cfg,
                   device=models["device"], value_mode=value_mode,
                   critic=models.get("critic"))

    # ── helpers ───────────────────────────────────────────────────────────
    def _unnorm_pos(self, x_norm: torch.Tensor) -> np.ndarray:
        """(n, T, D) normalized -> (n, T, P) raw-world positions."""
        n, T, D = x_norm.shape
        flat = x_norm.detach().cpu().numpy().reshape(-1, D)
        world = self.normalizer.unnormalize(flat).reshape(n, T, D)
        return world[..., list(self.pos_dims)]

    # ── the plan call ───────────────────────────────────────────────────────
    def plan(self, start_norm: np.ndarray, goal_raw: np.ndarray,
             seed: int = 0) -> Dict[str, Any]:
        """Run MCTD from a single normalized start toward a raw-world goal.

        start_norm: (D,) normalized current observation.
        goal_raw:   (2,) or (P,) raw-world goal position.
        Returns a dict with the chosen normalized plan and search diagnostics.
        """
        T, D = self.H, self.obs_dim
        cfg = self.cfg
        mat = pyramid_matrix(self.p.K, T, slope=cfg.slope, row_stride=cfg.row_stride)
        R = mat.shape[0]
        block = max(1, math.ceil((R - 1) / cfg.n_depths))
        terminal_depth = cfg.n_depths

        start_norm = np.asarray(start_norm, dtype=np.float32).reshape(D)
        goal_world = np.asarray(goal_raw, dtype=np.float64).reshape(-1)[:len(self.pos_dims)]
        start_world = self.normalizer.unnormalize(
            start_norm[None])[0][list(self.pos_dims)].astype(np.float64)

        # history template: row 0 = start (clamped clean), hist_len = 1
        x_hist = torch.zeros((1, T, D), device=self.dev)
        x_hist[0, 0] = torch.as_tensor(start_norm, device=self.dev)

        goal_norm = normalize_goal_xy(self.normalizer, goal_world.astype(np.float32))
        guide = GoalGuide(goal_norm, pos_dims=self.pos_dims,
                          reach_scale=cfg.reach_scale, device=self.dev)

        def expand_eval(node: MCTDTreeNode, cand: Dict[str, Any]) -> ExpandResult:
            g = float(cand["guidance_scale"])
            gd, w = (guide, g) if g else (None, 0.0)
            depth = cand["depth"]
            b_rows = block_rows(mat, depth, block)
            j_rows = jumpy_rows(mat, depth, block, cfg.skip_level_steps)
            child_partial = clean = None
            world_pos = None
            for _ in range(max(1, cfg.num_tries_for_bad_plans)):
                x = (fresh_plan(self.p, x_hist, 1, cfg.temperature)
                     if node.payload is None else node.payload.clone())
                child_partial = denoise_rows(self.p, x, b_rows, hist_len=1,
                                             x_hist=x_hist, guide=gd, w=w)
                clean = denoise_rows(self.p, child_partial.clone(), j_rows,
                                     hist_len=1, x_hist=x_hist, guide=gd, w=w)
                world_pos = self._unnorm_pos(clean)              # (1, T, P)
                if not bool(is_degenerate(world_pos)[0]):
                    break
            if self.value_mode == "critic":
                # Way 4c: score the clean plan with the DV critic (same evaluator
                # MCSS ranks by), NOT the geometric heuristic. No reach concept ->
                # info="NotReached" so every plan pools into the max-value output.
                v = float(self.critic(clean).reshape(-1)[0])
                return ExpandResult(value=v, info="NotReached",
                                    child_partial=child_partial, clean_plan=clean,
                                    achieved_t=None)
            values, infos, ach_t = geometric_values(
                world_pos, start_world, goal_world,
                self.goal_radius, self.warp_threshold)
            at = int(ach_t[0]) if ach_t[0] >= 0 else None
            return ExpandResult(value=float(values[0]), info=infos[0],
                                child_partial=child_partial, clean_plan=clean,
                                achieved_t=at)

        root = MCTDTreeNode("0", 0, terminal_depth, list(cfg.guidance_scales))
        root.value = 0.0
        # critic mode has no "achieved" event -> run the full budget, then take the
        # max-critic plan (like MCSS's argmax, but over the tree-explored pool).
        early = None if self.value_mode == "critic" else cfg.early_stopping
        search_cfg = MCTDSearchConfig(
            guidance_scales=list(cfg.guidance_scales), terminal_depth=terminal_depth,
            max_search_num=cfg.max_search_num, c_ucb=cfg.c_ucb,
            early_stopping=early)
        rng = np.random.default_rng(seed)
        res = run_mctd_search(root, expand_eval, search_cfg, rng)

        # ── choose the plan to return (reference's output selection) ──────
        info, value, achieved_t = "Failed", 0.0, None
        if res.achieved:
            clean, value, achieved_t = max(res.achieved, key=lambda z: z[1])
            info = "Achieved"
        elif res.not_reached:
            clean, value = max(res.not_reached, key=lambda z: z[1])
            info = "NotReached"
        else:                                    # nothing sampled: static plan
            clean = x_hist[:, :1].repeat(1, T, 1)
        plan_norm = clean.detach().cpu().numpy()[0]              # (T, D)

        return dict(plan_norm=plan_norm, solved=res.solved, info=info,
                    value=float(value), achieved_t=achieved_t,
                    n_search=res.n_search, n_nodes=res.n_nodes,
                    max_depth=res.max_depth, terminal_depth=terminal_depth,
                    n_rows=int(R), block=int(block),
                    guidance_scales=list(cfg.guidance_scales))
