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

Per-rollout logging
-------------------
run_episodes() records, for every rollout (episode-major order — for n_episodes=1 the
list index IS the vector-env index): binary success, the step of first goal touch,
the start xy, and the episode's goal xy.  Two runs with the same --seed see the same
scenario at the same index (VectorEnv.seed(seed) seeds sub-env i with seed+i), so the
(seed, index) key pairs MCSS and MCTS rollouts for McNemar-style paired tests
(scripts/collate_mcts.py consumes this).
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
import os

from mcts.specs import (SPECS, TARGET_CFG, env_family, get_goal,  # noqa: F401
                        make_dataset, max_episode_steps, normalize_goal_xy)
from mcts.value_forest import ForestConfig, ValueForest
from mcts.value_net import load_state_value, load_value_ensemble
from mcts.window import build_inpaint_prior, compose_window, extend_prefix


# ── Model loading ──────────────────────────────────────────────────────────────

def load_models(env_name: str, value_step: str = "latest",
                planner_step: int = 1000000, critic_step: int = 1000000,
                policy_step: int = 1000000, device: Optional[str] = None,
                ckpt_dir: Optional[str] = None,
                sg_ckpt: str = "state_value_sg_ckpt_best.pt",
                df_ckpt: Optional[str] = None,
                cg_ckpt: Optional[str] = None) -> Dict[str, Any]:
    """Build + load planner, MCSS critic, V(s), and inverse-dynamics policy.

    NOTE on critic_step: the official DV inference config defaults to
    critic_ckpt=200000 (configs/veteran/*/: critic_ckpt), while this harness
    defaults to 1000000.  Harness validation showed the two are empirically
    equivalent (MCSS k=50 reach 76.0% vs the pipeline's 76.9% on
    antmaze-large-diverse-v2); pass critic_step=200000 to match the official
    config exactly.  Whichever is used is recorded in the run JSON.
    """
    fam = env_family(env_name)
    spec = SPECS[fam]
    H, stride, depth = spec["H"], spec["stride"], spec["planner_depth"]
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env_single, ds = make_dataset(env_name, H=H, stride=stride)
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

    # Plain V(s) — optional like the ensemble (F3): a V(s,g)-only run shouldn't
    # require it. The Sampler enforces presence for value_mode=v_s.
    value = None
    v_path = f"{ckpt}/state_value_ckpt_{value_step}.pt"
    if os.path.exists(v_path):
        value = load_state_value(v_path, device=device)

    # Goal-conditioned V(s, g) ensemble — loaded lazily (only present once trained);
    # v_s runs don't need it, so a missing checkpoint is not fatal here.
    value_sg = None
    sg_path = f"{ckpt}/{sg_ckpt}"
    if os.path.exists(sg_path):
        value_sg = load_value_ensemble(sg_path, device=device)
        print(f"  loaded V(s,g) ensemble: {sg_ckpt} "
              f"(full_data={value_sg.meta.get('full_data')}, D={value_sg.meta.get('D')})")

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

    # Episode length from the env's own TimeLimit (umaze 300 / medium 600 / large
    # 800 / antmaze 1000) — the family-level spec value is a fallback only.
    max_path_length = max_episode_steps(env_single, env_name)

    # Consecutive-waypoint xy displacement sample (normalized units, stride-spaced,
    # REAL steps only) — the reference distribution for the optional junction
    # feasibility filter (Sampler(junction_filter=True)). None when the dataset
    # class lacks the needed attributes (the filter then refuses to arm).
    wp_disp_sample = None
    try:
        paths = getattr(ds, "paths", None)
        seq_obs_np = np.asarray(ds.seq_obs) if hasattr(ds, "seq_obs") else None
        if paths is not None and seq_obs_np is not None:
            rng = np.random.default_rng(0)
            chunks = []
            for p in rng.permutation(len(paths))[:512]:
                pl = paths[p][1] - paths[p][0] + 1
                hi = min(pl - 1, seq_obs_np.shape[1] - 1) - stride
                if hi <= 0:
                    continue
                t = rng.integers(0, hi + 1, size=min(256, hi + 1))
                d = np.linalg.norm(seq_obs_np[p, t + stride, :2]
                                   - seq_obs_np[p, t, :2], axis=-1)
                chunks.append(d)
            if chunks:
                wp_disp_sample = np.concatenate(chunks).astype(np.float32)
    except Exception as exc:              # diagnostic feature — never block loading
        print(f"  (junction-filter reference sample unavailable: {exc!r})")

    # Diffusion Forcing planner (mcts/df_model.py) — optional alternate backbone:
    # trained by scripts/train_df_planner.py, loaded only when a run asks for it.
    df_planner = None
    if df_ckpt:
        from mcts.df_model import load_df_planner
        df_planner = load_df_planner(f"{ckpt}/df_planner_ckpt_{df_ckpt}.pt",
                                     device=device)
        print(f"  loaded DF planner: df_planner_ckpt_{df_ckpt}.pt "
              f"(cfg={df_planner.cfg})")

    # Per-token noise-aware value V(x, k) (mcts/noise_critic.py, trained by
    # scripts/train_noise_critic.py) — optional classifier-guidance signal for
    # the frozen DF planner. Loaded only when a run asks for it (--cg-ckpt);
    # guide=None keeps every existing DF/DV arm bit-identical (see Sampler).
    cg_critic = None
    if cg_ckpt:
        from mcts.noise_critic import NoiseAwareCritic
        cg_critic = NoiseAwareCritic.load(f"{ckpt}/noise_critic_ckpt_{cg_ckpt}.pt",
                                          device=device)
        print(f"  loaded noise-critic: noise_critic_ckpt_{cg_ckpt}.pt "
              f"(cfg={cg_critic.cfg})")

    print(f"[{env_name}] loaded planner+critic+V+policy on {device} "
          f"(obs_dim={obs_dim}, act_dim={act_dim}, H={H}, stride={stride}, "
          f"max_path_length={max_path_length})")
    return dict(planner=planner, critic=critic, value=value, value_sg=value_sg,
                df_planner=df_planner, df_ckpt=df_ckpt,
                cg_critic=cg_critic, cg_ckpt=cg_ckpt,
                wp_disp_sample=wp_disp_sample,
                policy=policy, normalizer=normalizer, obs_dim=obs_dim,
                act_dim=act_dim, H=H, stride=stride, max_path_length=max_path_length,
                env_single=env_single, env_name=env_name, device=device, family=fam,
                ckpt_dir=ckpt, value_step=value_step, planner_step=planner_step,
                critic_step=critic_step, policy_step=policy_step,
                sg_ckpt=sg_ckpt if value_sg is not None else None)


# ── Sampler (MCSS + MCTS share the policy & env loop) ──────────────────────────

class Sampler:
    def __init__(self, models: Dict[str, Any], k_mcss: int = 50, k_mcts: int = 16,
                 budget: int = 15, child_index: int = 1, c_ucb: float = 1.4142136,
                 plan_steps: int = 20, policy_steps: int = 10, planner_temp: float = 1.0,
                 policy_temp: float = 0.5, solver: str = "ddim",
                 policy_solver: str = "ddpm", rebase: bool = True,
                 value_mode: str = "v_s", pess_beta: float = 1.0,
                 stability_window: int = 3, k_root: Optional[int] = None,
                 top_m: int = 1, junction_filter: bool = False,
                 junction_pct: float = 99.0, expand_mode: str = "glue",
                 backbone: str = "dv", df_slope: int = 1,
                 df_row_stride: int = 1, df_sweeps: Optional[int] = None,
                 cg_w: float = 0.0, grounded_blend: float = 0.25,
                 grounded_mcss: bool = False) -> None:
        self.m = models
        self.dev = models["device"]
        self.obs_dim = models["obs_dim"]
        self.act_dim = models["act_dim"]
        self.H = models["H"]
        self.k_mcss, self.k_mcts, self.budget = k_mcss, k_mcts, budget
        # Root expansion width (default = k_mcts). The EXECUTED action is chosen among
        # root children, so root width is what competes with MCSS's per-step pool of
        # k_mcss candidates; deep rounds only refine the ranking of those children.
        self.k_root = int(k_root) if k_root else k_mcts
        self.top_m = int(top_m)   # backup = mean of the m best children (1 = MAX)
        if self.k_root < 1 or self.top_m < 1:
            raise ValueError(f"k_root and top_m must be >= 1, "
                             f"got {self.k_root}, {self.top_m}")
        # Junction feasibility filter: children whose FIRST continuation step is an
        # implausibly large hop (xy displacement above the junction_pct percentile
        # of the dataset's real stride-spaced steps) get a sentinel value far below
        # the [-1, 1] value range (and below any UCB exploration bonus) — they lose
        # every selection/backup but the search stays well-defined if a node's
        # children are all filtered.
        self._junction_thr: Optional[float] = None
        self.last_junction_reject: Optional[float] = None
        if junction_filter:
            sample = models.get("wp_disp_sample")
            if sample is None:
                raise ValueError("junction_filter=True but the dataset offered no "
                                 "displacement sample (see load_models)")
            self._junction_thr = float(np.percentile(sample, junction_pct))
            print(f"  junction filter armed: reject first-step xy hop > "
                  f"{self._junction_thr:.4f} (p{junction_pct:g} of dataset steps)")
        self.child_index, self.c_ucb = child_index, c_ucb
        self.plan_steps, self.policy_steps = plan_steps, policy_steps
        self.planner_temp, self.policy_temp = planner_temp, policy_temp
        self.solver, self.policy_solver, self.rebase = solver, policy_solver, rebase
        # Tree node value:
        #   v_s        – goal-agnostic V(s)         (the measured baseline arm)
        #   v_sg       – goal-conditioned mean over the ensemble
        #   v_sg_pess  – goal-conditioned ensemble-min (pessimistic, the hardened value)
        #   critic     – DV trajectory critic on the COMPOSED window (search prefix +
        #                continuation, mcts/window.py): same evaluator as MCSS, every
        #                depth scored on the same [s0, s0+H) window -> comparable backups
        #   grounded   – ground-truth subtask count from the env's own task definitions
        #                (mcts/grounded.py KitchenGroundedChecker), reused on the SAME
        #                composed window critic mode builds — the one value in this repo
        #                NOT capped at 3-of-4 subtasks by kitchen-mixed's demonstrations
        #                (see _window_value). kitchen-only.
        self.value_mode, self.pess_beta = value_mode, pess_beta
        self.grounded_blend, self.grounded_mcss = float(grounded_blend), bool(grounded_mcss)
        self.stability_window = stability_window   # near-term waypoints scored for stability
        # cache the GaussianNormalizer mean/std as tensors so mcss_propose can UNNORMALIZE
        # the planner's (normalized) trajectory before reading physical pose/velocity for the
        # stability features — computing uprightness on standardized quat dims is meaningless.
        nz = models["normalizer"]
        self._norm_mean = (torch.as_tensor(np.asarray(nz.mean), dtype=torch.float32,
                                           device=self.dev) if hasattr(nz, "mean") else None)
        self._norm_std = (torch.as_tensor(np.asarray(nz.std), dtype=torch.float32,
                                          device=self.dev) if hasattr(nz, "std") else None)
        if value_mode not in ("v_s", "v_sg", "v_sg_pess", "oracle", "critic", "grounded"):
            raise ValueError(f"unknown value_mode {value_mode!r}")
        # Grounded subtask checker (mcts/grounded.py): a NON-learned evaluator read
        # straight off the env's own task definitions, exempt from the 3-of-4-subtask
        # label cap every LEARNED value in this stack inherits from kitchen-mixed's
        # demonstrations (no demo ever solves all 4). Built ONCE here (not per-call)
        # since it reads the live env; needed whenever it will be READ, i.e. either
        # as the tree's node value (value_mode="grounded") or as an MCSS reranker
        # (grounded_mcss=True, independent of value_mode — see mcss_waypoints).
        self.grounded = None
        if value_mode == "grounded" or grounded_mcss:
            if models.get("family") != "kitchen":
                raise ValueError(
                    f"value_mode='grounded' / grounded_mcss=True needs "
                    f"models['family']=='kitchen' (got {models.get('family')!r}) — "
                    f"the checker reads kitchen task definitions off the live env")
            from mcts.grounded import KitchenGroundedChecker
            self.grounded = KitchenGroundedChecker.from_env(
                models["env_single"], models["normalizer"])
        # Expansion mode (critic/grounded mode only):
        #   glue    – continuations sampled from the leaf state alone, then
        #             concatenated onto the search prefix (mcts/window.py). The
        #             seam is off-manifold for the critic (measured 63,618x MSE
        #             ratio on maze2d-large) — the winner's-curse fuel.
        #   inpaint – Diffusion-Forcing-inspired: the prefix + node state are
        #             CLAMPED into the denoiser at every diffusion step, so the
        #             continuation is generated jointly consistent with the path
        #             and the sampled window IS the composed window, seam-free.
        self.expand_mode = expand_mode
        if expand_mode not in ("glue", "inpaint"):
            raise ValueError(f"unknown expand_mode {expand_mode!r}")
        if expand_mode == "inpaint" and value_mode not in ("critic", "grounded"):
            raise ValueError("expand_mode='inpaint' needs value_mode in "
                             "{'critic', 'grounded'} (it conditions on the search "
                             "prefix, which only critic/grounded mode tracks)")
        # inpaint clamps d+1 rows and must leave >=1 free row at the deepest
        # expandable node — one row tighter than the glue/compose guard below
        if expand_mode == "inpaint" and budget * child_index > self.H - 2:
            raise ValueError(
                f"expand_mode=inpaint: budget*child_index = {budget * child_index} "
                f"> H-2 = {self.H - 2} — the deepest node's clamped prefix would "
                f"leave no free rows to generate (build_inpaint_prior would raise "
                f"mid-run)")
        # Planner backbone:
        #   dv – the frozen DV full-sequence planner (all established arms)
        #   df – the Causal Diffusion Forcing planner (mcts/df_model.py): tree
        #        expansion conditions on the search prefix NATIVELY (clean
        #        history tokens are in-distribution by training), replacing
        #        both the glue seam and the inpaint replacement hack. The DV
        #        critic stays the evaluator, so DF arms compare like-for-like.
        self.backbone = backbone
        self.df_slope, self.df_row_stride = int(df_slope), int(df_row_stride)
        # shortcut planner only: sampling sweeps (power of 2); None -> the model's
        # own cfg default. DFPlanner.sample / the DV path ignore it (absorbed by
        # **_), so threading it through unconditionally stays backbone-agnostic
        # and does NOT change any existing DF/DV arm.
        self.df_sweeps = int(df_sweeps) if df_sweeps else None
        if backbone not in ("dv", "df"):
            raise ValueError(f"unknown backbone {backbone!r}")
        if backbone == "df":
            if models.get("df_planner") is None:
                raise ValueError("backbone='df' needs a DF planner checkpoint "
                                 "(load_models df_ckpt= / --df-ckpt)")
            if budget * child_index > self.H - 2:
                raise ValueError(
                    f"backbone=df: budget*child_index = {budget * child_index} "
                    f"> H-2 = {self.H - 2} — deepest history would leave no "
                    f"free tokens to generate")
        # Classifier guidance (mcts/noise_critic.py, scripts/train_noise_critic.py):
        # steers the FROZEN DF planner's sampling with a per-token noise-aware
        # value's gradient (DFPlanner.sample guide=/w_cg=). Every guard below
        # exists because a silently-ignored --cg-w would report an unguided
        # number under a guided label — refuse instead of no-op.
        self.cg_w = float(cg_w)
        if self.cg_w != 0:
            if backbone != "df":
                raise ValueError("cg_w != 0 needs backbone='df' — CG steers the "
                                 "DF sampler; the DV planner has its own guidance "
                                 "stack (this repo's own finding: rank beats "
                                 "guidance there)")
            if models.get("cg_critic") is None:
                raise ValueError("cg_w needs a noise-critic checkpoint "
                                 "(--cg-ckpt / load_models cg_ckpt=)")
            if models["df_planner"].cfg.get("kind") == "shortcut":
                raise ValueError(
                    "cg_w != 0: the shortcut-forcing planner (mcts/shortcut_df.py, "
                    "few-step bootstrap sampling) has no guidance hook — silently "
                    "ignoring --cg-w would be a silent no-op, refusing instead")
            if models["cg_critic"].cfg["K"] != models["df_planner"].cfg["K"]:
                raise ValueError(
                    f"cg_w != 0: noise-critic K={models['cg_critic'].cfg['K']} != "
                    f"DF planner K={models['df_planner'].cfg['K']} — the critic's "
                    f"alpha-bar table must match the planner's noise schedule")
        # NOTE: the v_s / v_sg checkpoint-presence checks are enforced LAZILY in
        # mcts_waypoints (the tree entry point), NOT here — an MCSS-only run never
        # uses a tree value function, so it must not require a V(s)/V(s,g) checkpoint
        # (e.g. kitchen, where V(s) is not trained yet).
        if value_mode in ("critic", "grounded") and budget * child_index > self.H - 1:
            raise ValueError(
                f"value_mode={value_mode}: budget*child_index = {budget * child_index} "
                f"> H-1 = {self.H - 1} — the deepest node's prefix would fill the "
                f"critic window with no continuation left (see mcts/window.py)")
        # Per-step normalised goals (M, 2), set by mcts_waypoints; expand_fn reads
        # them positionally (forest leaves stay in tree/env order every round).
        self._cur_goals: Optional[torch.Tensor] = None
        # value_mode="oracle" — DEV-ONLY (Rule-1): children scored by the TRUE BFS
        # geodesic, set via set_oracle_ctx() by scripts/diag_oracle_tree.py. It tests
        # whether STRUCTURED search with a perfect value beats the learned critic (the
        # flat oracle re-rank already showed flat selection does not). It needs no
        # learned value and must NEVER appear in a reportable run.
        self._oracle_ctx: Optional[Dict[str, Any]] = None
        # Per-step forest stats from the latest mcts_waypoints call (one dict per tree:
        # n_nodes / max_depth / root_best_value); run_episodes aggregates realized depth.
        self.last_tree_stats: Optional[List[dict]] = None
        if not (1 <= child_index < self.H):
            raise ValueError(f"child_index must be in [1, H-1]={self.H-1}, got {child_index}")

    def _child_values(self, child_states: torch.Tensor, B: int, K: int) -> torch.Tensor:
        """(B*K, D) child states -> (B*K,) node values, per value_mode.

        For goal-conditioned modes each env's goal (in tree/env order) is broadcast
        over its K children and concatenated with the state, exactly as training fed
        [state, goal_xy]; the goal was normalised once via normalize_goal_xy."""
        if self.value_mode == "oracle":
            return self._oracle_child_values(child_states, B, K)
        if self.value_mode == "v_s":
            return self.m["value"](child_states).squeeze(-1)
        g = self._cur_goals.repeat_interleave(K, dim=0)          # (B*K, 2)
        x = torch.cat([child_states, g], dim=-1)                 # (B*K, D+2)
        ens = self.m["value_sg"]
        if self.value_mode == "v_sg_pess":
            return ens.pessimistic(x, mode="min").squeeze(-1)
        return ens(x).mean(dim=-1)                               # v_sg: ensemble mean

    def set_oracle_ctx(self, ctx: Optional[Dict[str, Any]]) -> None:
        """DEV-ONLY (Rule-1): give the oracle value its per-episode context, in tree/env
        order. ctx keys: normalizer, oracle (AntMazeOracle), goal_grids (list of M BFS
        grids), scale (StepScale), spc (steps/cell). Set by scripts/diag_oracle_tree.py."""
        self._oracle_ctx = ctx

    def _oracle_child_values(self, child_states: torch.Tensor, B: int, K: int) -> torch.Tensor:
        """DEV-ONLY (Rule-1): score children by the TRUE BFS geodesic to each env's goal,
        mapped onto the SAME value scale as the learned critic so c_ucb is comparable.
        child_states is (B*K, D) with the b-th block of K belonging to tree/env b, so
        goal_grids[b] is the right grid; unreachable cells map to -1 via the scale clip."""
        ctx = self._oracle_ctx
        if ctx is None:
            raise RuntimeError("value_mode='oracle' needs set_oracle_ctx() (dev-only)")
        xy = ctx["normalizer"].unnormalize(
            child_states.detach().cpu().numpy())[:, :2]          # (B*K, 2) world
        grids, oracle = ctx["goal_grids"], ctx["oracle"]
        geo = np.empty(B * K, dtype=np.float64)
        for idx in range(B * K):
            r, c = oracle.cell(xy[idx])
            geo[idx] = grids[idx // K][r][c]                     # cells; inf if unreachable
        vals = ctx["scale"].val_array(geo * ctx["spc"])         # -> [-1, 1] (inf -> -1)
        return torch.as_tensor(np.asarray(vals, dtype=np.float32), device=self.dev)

    def _window_value(self, win: torch.Tensor) -> torch.Tensor:
        """(N, H, D) composed/native windows -> (N,) node value, for value_mode in
        {'critic', 'grounded'} — the two modes share every plumbing site (composed-
        window construction, prefix tracking, budget guards; see expand_fn/
        mcts_waypoints, every widened `critic_mode` check) and differ ONLY in this
        scoring step:

          critic   – the DV trajectory critic's raw score on the window: the
                     established MCSS-comparable evaluator.
          grounded – self.grounded.score(win) (mcts/grounded.py) is the PRIMARY
                     term: it is the only signal in this stack that can see a 4th-
                     subtask completion, since every learned value INCLUDING this
                     same DV critic is trained on labels capped at 3-of-4 (kitchen-
                     mixed never demonstrates all 4). self.grounded_blend * the
                     critic score is added purely as a TIEBREAKER among windows the
                     grounded checker scores equally (a realism/plausibility ranking
                     on top of the ground-truth subtask count); grounded_blend=0 is
                     pure grounded.
        """
        critic_v = self.m["critic"](win).squeeze(-1)
        if self.value_mode == "grounded":
            return self.grounded.score(win) + self.grounded_blend * critic_v
        return critic_v

    # one batched planner+value pass; the search's only GPU touch-point
    def expand_fn(self, states: List[Any], k: Optional[int] = None):
        K = int(k) if k else self.k_mcts     # k: per-call width (root may be wider)
        H, D = self.H, self.obs_dim
        B = len(states)
        # critic/grounded share plumbing (state=(vec, prefix) pairs, composed-window
        # construction, native/inpaint expansion): "grounded" is critic-mode with a
        # different final scorer (self._window_value below decides which).
        critic_mode = self.value_mode in ("critic", "grounded")
        # critic mode: node states are (state_vec, prefix) pairs — the prefix is the
        # search-chosen waypoint path from the CURRENT REAL state s0 to this node, so
        # children can be scored on the shared [s0, s0+H) window (mcts/window.py).
        if critic_mode:
            prefixes = [p for _, p in states]
            s_np = np.stack([v for v, _ in states])
        else:
            prefixes = None
            s_np = np.stack(states)
        s = torch.as_tensor(s_np, dtype=torch.float32, device=self.dev)   # (B, D)
        inpaint = critic_mode and self.expand_mode == "inpaint"
        df = critic_mode and self.backbone == "df"
        native = inpaint or df    # prefix lives INSIDE the sampled window
        if native:
            # Both native modes generate the whole [s0, s0+H) window in one shot,
            # prefix included:
            #   inpaint – DV planner, prefix CLAMPED via replacement conditioning
            #             (approximate: the planner never trained on multi-row
            #             clamping — measured -16 closed-loop, kept as ablation);
            #   df      – Causal Diffusion Forcing planner: clean history tokens
            #             are in-distribution by training -> EXACT conditioning.
            prior_np, mask_np, d_lens = build_inpaint_prior(prefixes, s_np, H, K)
            prior = torch.as_tensor(prior_np, device=self.dev)
            # per-sample row offset of the node state inside its window
            dt = torch.as_tensor(np.repeat(d_lens, K), device=self.dev)   # (B*K,)
            ar = torch.arange(B * K, device=self.dev)
            if inpaint:
                inp_mask = torch.as_tensor(mask_np, device=self.dev)
        else:
            prior = torch.zeros((B * K, H, D), device=self.dev)
            prior[:, 0, :] = s.repeat_interleave(K, dim=0)
        with torch.no_grad():
            planner = self.m["planner"]
            if df:
                hist_len = torch.as_tensor(np.repeat(d_lens + 1, K),
                                           device=self.dev)
                trajs = self.m["df_planner"].sample(
                    prior, hist_len, H, slope=self.df_slope,
                    row_stride=self.df_row_stride, sweeps=self.df_sweeps,
                    temperature=self.planner_temp,
                    guide=(self.m.get("cg_critic") if self.cg_w else None),
                    w_cg=self.cg_w)
            else:
                if inpaint:
                    base_mask = planner.fix_mask   # (1, H, D) row-0 mask; restore after
                    planner.fix_mask = inp_mask
                try:
                    trajs, _ = planner.sample(
                        prior, solver=self.solver, n_samples=B * K,
                        sample_steps=self.plan_steps, use_ema=True, condition_cfg=None,
                        w_cfg=1.0, temperature=self.planner_temp)
                finally:
                    if inpaint:
                        planner.fix_mask = base_mask
            if native:
                # rows [0:d] replicate the prefix exactly (clamped), row d is the node
                # state, so continuation indexing shifts by the per-node prefix length.
                child_states = trajs[ar, dt + self.child_index, :D]       # (B*K, D)
                first_wps = trajs[ar, dt + 1, :D]                         # (B*K, D)
            else:
                child_states = trajs[:, self.child_index, :D]  # (B*K, D) tree child = L ahead
                # Action target is ALWAYS the immediate next waypoint traj[1], regardless of
                # child_index: the tree stitches in L-step segments, but we still execute exactly
                # one step per env-step (then replan). first_wp matters only for root children.
                first_wps = trajs[:, 1, :D]                         # (B*K, D)
            if native:
                # the sampled window is already the composed window — score directly
                # (value_mode critic OR grounded: see _window_value)
                vvals = self._window_value(trajs)
                trj = trajs.view(B, K, H, D).cpu().numpy()
            elif critic_mode:
                # Score prefix + continuation on the shared window. A naive critic(trajs)
                # would score each depth on its OWN later window, which systematically
                # inflates expanded children on progress tasks (window-shift bias) —
                # max-backup then rewards visits, not merit.
                trj = trajs.view(B, K, H, D).cpu().numpy()
                composed = np.concatenate(
                    [compose_window(prefixes[i], trj[i]) for i in range(B)], axis=0)
                # value_mode critic OR grounded: see _window_value
                vvals = self._window_value(torch.as_tensor(
                    composed, dtype=torch.float32, device=self.dev))
            else:
                vvals = self._child_values(child_states, B, K)  # (B*K,) per value_mode
            if self._junction_thr is not None:
                if native:
                    disp = (trajs[ar, dt + 1, :2] - trajs[ar, dt, :2]).norm(dim=-1)
                else:
                    disp = (trajs[:, 1, :2] - trajs[:, 0, :2]).norm(dim=-1)   # (B*K,)
                bad = disp > self._junction_thr
                # -10 dominates value range [-1,1] PLUS the UCB exploration bonus
                # (~<=2.4 at zero visits), so filtered children are never selected
                # for expansion and never win top-m backups unless ALL siblings
                # are filtered (search stays well-defined).
                vvals = torch.where(bad, torch.full_like(vvals, -10.0), vvals)
                self.last_junction_reject = float(bad.float().mean().item())
        cs = child_states.cpu().numpy().reshape(B, K, D)
        fw = first_wps.cpu().numpy().reshape(B, K, D)
        vv = vvals.cpu().numpy().reshape(B, K)
        if native:
            # child prefix = window rows [0 : d + child_index] — identical semantics
            # to extend_prefix (rows [0:d] == parent prefix by clamping, row d ==
            # node state), just sliced from the jointly-generated window.
            return [([(cs[i, j],
                       trj[i, j, : int(d_lens[i]) + self.child_index].copy())
                      for j in range(K)],
                     [fw[i, j] for j in range(K)],
                     [float(vv[i, j]) for j in range(K)]) for i in range(B)]
        if critic_mode:
            return [([(cs[i, j], extend_prefix(prefixes[i], trj[i, j], self.child_index))
                      for j in range(K)],
                     [fw[i, j] for j in range(K)],
                     [float(vv[i, j]) for j in range(K)]) for i in range(B)]
        return [([cs[i, j] for j in range(K)],
                 [fw[i, j] for j in range(K)],
                 [float(vv[i, j]) for j in range(K)]) for i in range(B)]

    def mcts_waypoints(self, s_norm: np.ndarray,
                       goals_norm: Optional[np.ndarray] = None) -> np.ndarray:
        M = s_norm.shape[0]
        if self.backbone == "df" and self.value_mode not in ("critic", "grounded"):
            raise ValueError("backbone='df' tree search needs value_mode in "
                             "{'critic', 'grounded'} (prefix bookkeeping lives there)")
        # Lazy value-checkpoint guards (moved from __init__): the tree needs its
        # value function here, but an MCSS-only run must not require one.
        if self.value_mode == "v_s" and self.m.get("value") is None:
            raise ValueError("value_mode=v_s needs the V(s) checkpoint (none loaded)")
        if self.value_mode in ("v_sg", "v_sg_pess") and self.m.get("value_sg") is None:
            raise ValueError(f"value_mode={self.value_mode} needs a V(s,g) ensemble "
                             f"checkpoint (none loaded — pass --sg-ckpt)")
        if self.value_mode in ("v_sg", "v_sg_pess"):
            if goals_norm is None:
                raise ValueError(f"value_mode={self.value_mode} needs per-env goals")
            self._cur_goals = torch.as_tensor(goals_norm, dtype=torch.float32,
                                              device=self.dev)   # (M, 2)
        # critic/grounded mode packs (state, prefix) so expand_fn can compose windows;
        # the forest treats states as opaque either way. Root prefix = None: the
        # root's window starts at the real state itself.
        roots = ([(s_norm[i], None) for i in range(M)]
                 if self.value_mode in ("critic", "grounded")
                 else [s_norm[i] for i in range(M)])
        # The FIRST expand call is the root round (ValueForest.__init__ batches all M
        # roots into it); widen it to k_root, deep rounds use k_mcts.
        call = {"root": True}

        def expand(states):
            k = self.k_root if call["root"] else None
            call["root"] = False
            return self.expand_fn(states, k=k)

        forest = ValueForest(roots, expand,
                             ForestConfig(k=self.k_mcts, budget=self.budget,
                                          c_ucb=self.c_ucb, top_m=self.top_m))
        forest.run()
        # Realized look-ahead: with child_index=L, a depth-d tree has seen d*L waypoints
        # (= d*L*stride dense steps) before committing. Logged so the depth the UCB
        # search ACTUALLY reaches is measured, not assumed.
        self.last_tree_stats = forest.stats()
        wps = forest.best_first_waypoints()
        return np.stack([wps[i] if wps[i] is not None else s_norm[i] for i in range(M)])

    def mcts_best_plans(self, s_norm: np.ndarray,
                        goals_norm: Optional[np.ndarray] = None) -> np.ndarray:
        """Like mcts_waypoints, but returns each tree's FULL committed plan (the
        stitched best-branch path from s0, padded to H with its last waypoint)
        instead of only the first waypoint. Additive: lets the DF-tree be driven
        at any replan cadence in the MPC harness (mcts/mctd_loop.py
        DFTreeMPCPlanner) so it is raw-comparable to MCTD-critic. Returns
        (M, H, D). Duplicates mcts_waypoints' forest setup deliberately, so that
        production path stays untouched."""
        M, H, D = s_norm.shape[0], self.H, self.obs_dim
        if self.backbone == "df" and self.value_mode not in ("critic", "grounded"):
            raise ValueError("backbone='df' tree search needs value_mode in "
                             "{'critic', 'grounded'}")
        if self.value_mode == "v_s" and self.m.get("value") is None:
            raise ValueError("value_mode=v_s needs the V(s) checkpoint")
        if self.value_mode in ("v_sg", "v_sg_pess") and self.m.get("value_sg") is None:
            raise ValueError(f"value_mode={self.value_mode} needs a V(s,g) ensemble")
        if self.value_mode in ("v_sg", "v_sg_pess"):
            if goals_norm is None:
                raise ValueError(f"value_mode={self.value_mode} needs per-env goals")
            self._cur_goals = torch.as_tensor(goals_norm, dtype=torch.float32,
                                              device=self.dev)
        roots = ([(s_norm[i], None) for i in range(M)]
                 if self.value_mode in ("critic", "grounded")
                 else [s_norm[i] for i in range(M)])
        call = {"root": True}

        def expand(states):
            k = self.k_root if call["root"] else None
            call["root"] = False
            return self.expand_fn(states, k=k)

        forest = ValueForest(roots, expand,
                             ForestConfig(k=self.k_mcts, budget=self.budget,
                                          c_ucb=self.c_ucb, top_m=self.top_m))
        forest.run()
        self.last_tree_stats = forest.stats()
        leaves = forest.best_leaf_states()
        plans = np.zeros((M, H, D), dtype=np.float32)
        for i in range(M):
            st = leaves[i]
            if isinstance(st, tuple):                 # critic/grounded: (vec, prefix)
                vec, prefix = st
                if prefix is None or len(prefix) == 0:
                    path = np.asarray(vec, dtype=np.float32)[None]
                else:
                    path = np.concatenate(
                        [np.asarray(prefix, dtype=np.float32),
                         np.asarray(vec, dtype=np.float32)[None]], axis=0)
            else:                                     # non-critic: state IS the vec
                path = np.asarray(st, dtype=np.float32)[None]
            L = min(path.shape[0], H)
            plans[i, :L] = path[:L]
            if L < H:                                 # pad with the last waypoint
                plans[i, L:] = path[L - 1]
        return plans

    def mcss_waypoints(self, s_norm: np.ndarray) -> np.ndarray:
        M, K, H, D = s_norm.shape[0], self.k_mcss, self.H, self.obs_dim
        s = torch.as_tensor(s_norm, dtype=torch.float32, device=self.dev)
        prior = torch.zeros((M * K, H, D), device=self.dev)
        prior[:, 0, :] = s.repeat_interleave(K, dim=0)
        with torch.no_grad():
            if self.backbone == "df":
                # DF-MCSS: same sample-and-rank loop, DF backbone, DV critic —
                # calibrates DF plan quality against every DV arm.
                trajs = self.m["df_planner"].sample(
                    prior, torch.ones(M * K, dtype=torch.long, device=self.dev),
                    H, slope=self.df_slope, row_stride=self.df_row_stride,
                    sweeps=self.df_sweeps, temperature=self.planner_temp,
                    guide=(self.m.get("cg_critic") if self.cg_w else None),
                    w_cg=self.cg_w)
            else:
                trajs, _ = self.m["planner"].sample(
                    prior, solver=self.solver, n_samples=M * K, sample_steps=self.plan_steps,
                    use_ema=True, condition_cfg=None, w_cfg=1.0, temperature=self.planner_temp)
            scores = self.m["critic"](trajs).squeeze(-1).view(M, K)
            if self.grounded_mcss:
                # Rerank by the grounded subtask count (mcts/grounded.py) instead of
                # the learned critic alone — same primary-signal-plus-tiebreaker
                # combination as _window_value's tree-node scoring, but MCSS windows
                # are already single windows FROM s0 (trajs, M*K, H, D), so no window
                # composition is needed here.
                g = self.grounded.score(trajs).view(M, K)
                scores = g + self.grounded_blend * scores
            idx = scores.argmax(dim=1)
            trajs = trajs.view(M, K, H, D)
            best = trajs[torch.arange(M, device=self.dev), idx]   # (M, H, D)
            wp = best[:, 1, :D].cpu().numpy()
        return wp

    def mcss_propose(self, s_norm: np.ndarray) -> Dict[str, np.ndarray]:
        """INSTRUMENTATION-ONLY: one planner.sample, return the FULL candidate pool.

        Additive hook for the failure-instrumentation harness (mcts/instrument.py).
        It does NOT alter mcss_waypoints (the production path) — it reproduces MCSS's
        single planner call so the instrumented loop can, from ONE consistent draw,
        (a) take the same DV-critic argmax decision MCSS would, (b) log every
        candidate's value, and (c) let a Tier-2 oracle re-rank pick over the identical
        candidates. Returned arrays (numpy, M = n_envs, K = k_mcss, D = obs_dim):
            first_wps  (M, K, D)  candidate immediate next waypoints (executed if chosen)
            endpoints  (M, K, D)  candidate plan endpoints (where each plan heads)
            scores     (M, K)     DV trajectory-critic score per candidate

        Note: argmax over `scores` here is byte-for-byte the same selection rule as
        mcss_waypoints; the two differ only in that this returns the internals.
        """
        M, K, H, D = s_norm.shape[0], self.k_mcss, self.H, self.obs_dim
        s = torch.as_tensor(s_norm, dtype=torch.float32, device=self.dev)
        prior = torch.zeros((M * K, H, D), device=self.dev)
        prior[:, 0, :] = s.repeat_interleave(K, dim=0)
        with torch.no_grad():
            trajs, _ = self.m["planner"].sample(
                prior, solver=self.solver, n_samples=M * K, sample_steps=self.plan_steps,
                use_ema=True, condition_cfg=None, w_cfg=1.0, temperature=self.planner_temp)
            scores = self.m["critic"](trajs).squeeze(-1).view(M, K)
            trajs = trajs.view(M, K, H, D)
            first_wps = trajs[:, :, 1, :D]                       # (M, K, D)
            endpoints = trajs[:, :, H - 1, :D]                   # (M, K, D)
            # predicted near-term STABILITY features (DEPLOYABLE — straight from the
            # planner's own trajectory, no privileged info). The Ant's ~20% failures are
            # topples, so a stability-aware selector can prefer plans predicted to stay
            # upright. MUST be read in RAW physical units: trajs is GaussianNormalizer-space,
            # so uprightness on standardized quat dims is garbage — unnormalize first.
            sd = self._norm_std
            # first-step xy MOVE in raw units (mean cancels in the difference, scale by std)
            disp = (((trajs[:, :, 1, :2] - trajs[:, :, 0, :2]) * sd[:2]).norm(dim=-1)
                    if sd is not None else
                    (trajs[:, :, 1, :2] - trajs[:, :, 0, :2]).norm(dim=-1))   # (M,K)
            if D >= 21 and sd is not None:                       # antmaze (quat@3:7, angvel@18:21)
                Ls = max(1, min(self.stability_window, H - 1))
                seg = trajs[:, :, 1:1 + Ls, :] * sd + self._norm_mean    # (M,K,Ls,D) RAW state
                up = 1.0 - 2.0 * (seg[..., 4] ** 2 + seg[..., 5] ** 2)   # true uprightness/waypoint
                min_upright = up.min(dim=2).values              # (M,K) worst predicted uprightness
                angvel = seg[..., 18:21].norm(dim=-1).mean(dim=2)        # (M,K) mean angular speed
            else:                                               # maze2d point mass: no pose
                min_upright = torch.ones((M, K), device=self.dev)
                angvel = torch.zeros((M, K), device=self.dev)
        return dict(first_wps=first_wps.cpu().numpy(),
                    endpoints=endpoints.cpu().numpy(),
                    scores=scores.cpu().numpy(),
                    min_upright=min_upright.cpu().numpy(),       # (M,K) higher = predicted-stabler
                    angvel=angvel.cpu().numpy(),                 # (M,K) lower = smoother
                    disp=disp.cpu().numpy())                     # (M,K) lower = gentler first step

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
                 verbose: bool = True, dv_log: bool = False,
                 trace: bool = False) -> Dict[str, Any]:
    """Closed-loop evaluation; returns aggregates PLUS per-rollout vectors.

    DV-compatible score (`dv_norm_mean ± dv_norm_err`, always computed; the per-step
    `[t=N] rew:` lines printed when dv_log=True) replicates the base DV pipeline's
    EXACT per-family accounting so the numbers are directly comparable to its logs:
      * antmaze (veteran_d4rl_antmaze.py:445,449): ep_reward += rew, then
        np.clip(ep_reward, 0, 1) -> a reach indicator -> get_normalized_score
        gives 0 or 100, mean = reach% in [0, 100].
      * maze2d (veteran_d4rl_maze2d.py:445,449): finished |= (rew==1); ep_reward +=
        finished -> a latched camping count, NOT clipped -> get_normalized_score can
        exceed 100/200 (the "201.4 norm" regime). The metric is camping return, not
        a reach indicator.
    The binary `success` vector below (for McNemar) is the active-masked reach
    indicator either way; on antmaze it equals the DV reach% exactly.

    Per-rollout fields (episode-major order: [ep0_env0..ep0_envN, ep1_env0..]):
        success    1/0 — did the rollout ever touch the goal
        reach_step step index of the first goal touch (None if never)
        starts     raw start xy per rollout
        goals      eval goal xy per rollout (None if it could not be read)

    Pairing contract (verified on the eval box): with the same seed and n_envs,
    index i sees the SAME GOAL in every run — the goal is a pure function of
    (seed, index) — so success[i] pairs with success[i] of another method's run.
    Start positions share the same cell up to the env's unseedable ±0.1 reset
    jitter (see the seeding comment below). collate_mcts checks the goal vectors
    before computing McNemar. Prefer n_episodes=1 and vary seeds across
    replicates — that keeps the pairing key explicitly (seed, index).
    """
    import gym
    from pipelines.utils import set_seed

    assert method in ("mcss", "mcts")
    m = sampler.m
    env_name = m["env_name"]
    normalizer = m["normalizer"]
    env_single = m["env_single"]
    max_t = max_steps or m["max_path_length"]

    set_seed(seed)
    # Synchronous vector env: stepping cost is negligible next to the diffusion
    # sampling, and the sub-envs stay reachable (env.envs) for per-episode goal
    # capture — the async default hides them in worker processes.
    env = gym.vector.make(env_name, n_envs, asynchronous=False)
    # Scenario pairing — verified on the eval box (smoke runs, 2026-06-10):
    #   * GOALS are a pure function of (seed, index): d4rl antmaze samples the goal
    #     via the GLOBAL np.random (d4rl/locomotion/maze_env.py set_target_goal),
    #     which set_seed() above controls. Identical across method arms AND across
    #     separate invocations — this is the pairing key, and collate_mcts verifies
    #     it from the stored goal vectors before computing McNemar.
    #   * START positions keep the env's ±0.1 qpos reset jitter UNPAIRED: the ant's
    #     reset noise uses the instance RNG, and AntMazeEnv.seed() routes into gym's
    #     deprecated no-op seeding on the pinned gym/d4rl combo, so no env.seed call
    #     reaches it. Same start cell every time; the sub-cell jitter is unseedable
    #     entropy of the same character as the diffusion sampling noise.
    # The explicit per-sub-env seeding below is kept belt-and-braces for env
    # families whose seed() path does work; failures are aggregated and loud.
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
        print(f"  [{method}] WARNING: seeding incomplete ({'; '.join(seed_failures[:3])}) "
              f"— check the goal vectors via scripts/collate_mcts.py before trusting "
              f"McNemar pairing.")

    fam = env_family(env_name)
    all_success: List[np.ndarray] = []
    all_reach_step: List[np.ndarray] = []
    all_starts: List[np.ndarray] = []
    all_goals: List[List[Optional[List[float]]]] = []
    all_dv: List[np.ndarray] = []                # DV-exact per-env score input
    traj_xy: List[np.ndarray] = []               # trace=True: per-step executed xy (T,n_envs,2)
    depth_sum, depth_n, depth_max = 0.0, 0, 0   # realized tree depth (mcts only)
    t0 = time.perf_counter()
    goal_conditioned = (method == "mcts" and sampler.value_mode in ("v_sg", "v_sg_pess"))
    for ep in range(n_episodes):
        obs = env.reset()
        all_starts.append(np.asarray(obs)[:, :2].astype(np.float64).copy())
        goals_raw = None
        try:
            goals_raw = np.asarray([get_goal(e) for e in env.envs], dtype=np.float64)
            all_goals.append(goals_raw.tolist())
        except Exception as e:
            print(f"  [{method}] WARNING: could not read per-env goals ({e!r})")
            all_goals.append([None] * n_envs)
            if goal_conditioned:
                raise RuntimeError("value_mode needs per-env goals but none could "
                                   "be read — abort rather than score blind") from e
        # Goals are fixed per episode; normalise ONCE with the shared helper (C1)
        # so the tree's V(s,g) call matches training exactly.
        goals_norm = (normalize_goal_xy(normalizer, goals_raw)
                      if goal_conditioned else None)
        ep_rew = np.zeros(n_envs, dtype=np.float64)
        reach_step = np.full(n_envs, -1, dtype=np.int64)   # first goal touch, -1 = never
        active = np.ones(n_envs, dtype=bool)   # still in the FIRST episode (count rewards)
        dv_acc = np.zeros(n_envs, dtype=np.float64)        # DV-exact accumulator
        dv_finished = np.zeros(n_envs, dtype=bool)         # maze2d latch
        for t in range(max_t):
            if trace:                              # executed path: pre-step xy, NaN once done
                xy_rec = np.asarray(obs)[:, :2].astype(np.float32).copy()
                xy_rec[~active] = np.nan
                traj_xy.append(xy_rec)
            s_norm = normalizer.normalize(obs).astype(np.float32)   # (n_envs, obs_dim)
            if method == "mcss":
                next_wp = sampler.mcss_waypoints(s_norm)
            else:
                next_wp = sampler.mcts_waypoints(s_norm, goals_norm=goals_norm)
                for st_t in (sampler.last_tree_stats or []):
                    depth_sum += st_t["max_depth"]
                    depth_n += 1
                    if st_t["max_depth"] > depth_max:
                        depth_max = st_t["max_depth"]
            act = sampler.policy_action(s_norm, next_wp)
            obs, rew, done, info = env.step(act)
            # gym.vector auto-resets an env on done; count rewards only within each env's
            # first episode (freeze once done) so a reset second episode can't be mixed in.
            rew = np.asarray(rew, dtype=np.float64)
            hit = active & (rew > 0.0) & (reach_step < 0)
            reach_step[hit] = t
            ep_rew += rew * active
            # DV-exact accumulator, per family (matches the base pipelines verbatim):
            if fam == "maze2d":
                dv_finished |= (rew == 1.0)               # latch at first goal touch
                dv_acc += dv_finished                     # camping count (no clip)
            else:
                dv_acc += rew                             # antmaze raw (clipped at end)
            active &= ~np.asarray(done, dtype=bool)
            if dv_log:
                print(f"[t={t+1}] rew: {dv_acc}")
            if not active.any() and not dv_log and not trace:
                break       # dv_log/trace run the full horizon so the per-step log is complete
        succ = np.clip(ep_rew, 0.0, 1.0)
        all_success.append(succ)
        all_reach_step.append(reach_step)
        # DV-exact per-family final accounting (matches the base pipelines verbatim):
        #   maze2d  = un-clipped latched camping count (can exceed 100/200)
        #   kitchen = cumulative subtask reward clipped [0,4] (veteran_d4rl_kitchen.py:447)
        #   antmaze = reach indicator clipped [0,1]
        if fam == "maze2d":
            all_dv.append(dv_acc)
        elif fam == "kitchen":
            all_dv.append(np.clip(dv_acc, 0.0, 4.0))
        else:
            all_dv.append(np.clip(dv_acc, 0.0, 1.0))
        if verbose:
            print(f"  [{method}] ep {ep+1}/{n_episodes}  reach={succ.mean()*100:5.1f}%  "
                  f"elapsed={time.perf_counter()-t0:6.0f}s")
    env.close()

    flat = np.concatenate(all_success)
    flat_reach_step = np.concatenate(all_reach_step)
    norm = np.array([env_single.get_normalized_score(x) for x in flat]) * 100.0
    p = float(flat.mean())
    # DV-exact score: get_normalized_score on the per-family accumulator (reach
    # indicator for antmaze -> [0,100]; un-clipped camping return for maze2d ->
    # can exceed 100/200). This is the number the base DV pipelines print.
    flat_dv = np.concatenate(all_dv)
    dv_norm = np.array([env_single.get_normalized_score(x) for x in flat_dv]) * 100.0
    dv_norm_mean = float(dv_norm.mean())
    dv_norm_err = float(dv_norm.std() / np.sqrt(flat_dv.size))
    if dv_log:
        print(f"{dv_norm_mean} {dv_norm_err}")     # the base DV pipeline's final line
    out = dict(method=method, n_rollouts=int(flat.size),
               reach_pct=float(p * 100.0),
               # binomial SEM of reach% — the honest error bar for a success rate
               reach_err=float(np.sqrt(p * (1.0 - p) / flat.size) * 100.0),
               norm_mean=float(norm.mean()),
               norm_err=float(norm.std() / np.sqrt(flat.size)),
               # the base-DV-comparable score (== norm_mean on antmaze; camping on maze2d)
               dv_norm_mean=dv_norm_mean, dv_norm_err=dv_norm_err,
               wall_s=round(time.perf_counter() - t0, 1),
               # per-rollout vectors, episode-major; (seed, index) is the pairing key
               success=[int(x > 0) for x in flat],
               # per-rollout DV-exact normalized score — the metric to PAIR on for maze2d
               # (camping return, >100), where the binary `success` above is saturated.
               dv_norm=[float(x) for x in dv_norm],
               reach_step=[int(s) if s >= 0 else None for s in flat_reach_step],
               starts=[xy.tolist() for ep_s in all_starts for xy in ep_s],
               goals=[g for ep_g in all_goals for g in ep_g],
               # realized search depth (mcts only): mean/max of per-tree max depth over
               # all env-steps; look-ahead distance = depth × child_index × stride
               tree_depth_mean=round(depth_sum / depth_n, 2) if depth_n else None,
               tree_depth_max=int(depth_max) if depth_n else None)
    if trace and traj_xy:
        out["trace_xy"] = np.stack(traj_xy)        # (T, n_envs, 2); use with n_episodes=1
    if verbose:
        depth_str = (f"  tree_depth={out['tree_depth_mean']:.1f} (max {out['tree_depth_max']})"
                     if out["tree_depth_mean"] is not None else "")
        print(f"  [{method}] DONE  reach={out['reach_pct']:.1f}%±{out['reach_err']:.1f}  "
              f"norm={out['norm_mean']:.1f}±{out['norm_err']:.1f}  "
              f"(n={out['n_rollouts']}, {out['wall_s']:.0f}s){depth_str}")
    return out
