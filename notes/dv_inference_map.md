# DV Inference Loop — End-to-End Map

**Scope:** maze2d-umaze-v1, MCSS guidance, `separate` pipeline, `use_diffusion_invdyn=True`.  
All line numbers are stable as of the `MCTS_Integration` branch snapshot.

---

## 1. Where Is the MCSS Implementation?

MCSS (Monte Carlo Self-Sampling) is **not a standalone function**. It is the inference pattern
implemented inline in the environment rollout loop.

**File:** [pipelines/veteran_d4rl_maze2d.py](../pipelines/veteran_d4rl_maze2d.py)  
**Lines:** 357–377 (planning step) and 414–433 (action step)

```python
# lines 357–377 — MCSS planning step
if args.guidance_type == "MCSS":
    planner_prior = torch.zeros(
        (args.num_envs * args.planner_num_candidates, args.task.planner_horizon, planner_dim),
        device=args.device)

    obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
    obs_repeat = obs.unsqueeze(1).repeat(1, args.planner_num_candidates, 1).view(-1, obs_dim)

    planner_prior[:, 0, :obs_dim] = obs_repeat          # write start state into prior
    with torch.no_grad():
        traj, log = planner.sample(
            planner_prior, solver=args.planner_solver,
            n_samples=args.num_envs * args.planner_num_candidates,
            sample_steps=args.planner_sampling_steps,
            use_ema=args.planner_use_ema,
            condition_cfg=None, w_cfg=1.0,
            temperature=args.task.planner_temperature)

    with torch.no_grad():                               # critic reranking
        value = critic(traj)                            # (num_envs*K, 1)
        value = value.view(args.num_envs, args.planner_num_candidates)
        idx = torch.argmax(value, -1)
        traj = traj.reshape(args.num_envs, args.planner_num_candidates,
                            args.task.planner_horizon, planner_dim)
        traj = traj[torch.arange(args.num_envs), idx]  # (num_envs, H, D)
```

### planner.sample() — Concrete Signature

**Class:** `ContinuousDiffusionSDE`  
**File:** [cleandiffuser/diffusion/diffusionsde.py](../cleandiffuser/diffusion/diffusionsde.py)  
**Line:** 743

```python
def sample(
    self,
    prior: torch.Tensor,                          # (B, H, planner_dim) — fixed portion / start state
    solver: str = "ddpm",                          # "ddim" in production (maze2d config)
    n_samples: int = 1,
    sample_steps: int = 5,                         # 20 for planner, 10 for policy
    sample_step_schedule: Union[str, Callable] = "uniform_continuous",
    use_ema: bool = True,
    temperature: float = 1.0,
    condition_cfg=None,
    mask_cfg=None,
    w_cfg: float = 0.0,
    condition_cg=None,
    w_cg: float = 0.0,
    diffusion_x_sampling_steps: int = 0,
    warm_start_reference: Optional[torch.Tensor] = None,
    warm_start_forward_level: float = 0.3,
    requires_grad: bool = False,
    preserve_history: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, dict]:
    ...
    return xt, log    # xt: (n_samples, H, planner_dim), log: {"sample_history": ...}
```

**Input contract for the planner (MCSS, maze2d-umaze):**

| Argument | Shape / Value | Notes |
|---|---|---|
| `prior` | `(num_envs*K, H, obs_dim)` = `(50, 32, 4)` per env | zeros except `[:, 0, :]` = normalised start obs |
| `n_samples` | `num_envs * K` = 50 | |
| `solver` | `"ddim"` | from `planner_solver` config |
| `sample_steps` | 20 | `planner_sampling_steps` |
| `condition_cfg` | `None` | MCSS uses no CFG |
| `w_cfg` | 1.0 | identity (unconditional) |
| `temperature` | 1.0 | `planner_temperature` |

**Return:**

| Value | Shape | Contents |
|---|---|---|
| `xt` (trajectory) | `(n_samples, H, obs_dim)` = `(50, 32, 4)` | normalised observation states only (no actions) |
| `log["sample_history"]` | `None` | not requested during inference |

**Scoring:** `DVHorizonCritic.forward(traj)` returns `(n_samples, 1)` — scalar per trajectory.
Best candidate selected by `argmax` over the K dimension.

---

## 2. Where in the Env-Rollout Loop Is planner.sample() Called?

**File:** [pipelines/veteran_d4rl_maze2d.py](../pipelines/veteran_d4rl_maze2d.py)

```
lines 351–453 — outer episode loop
  line 354  — while not done and t < max_path_length + 1:
    lines 357–377 — STEP 1: generate plan (MCSS)   ← planner.sample() called here
    lines 414–433 — STEP 2: generate action        ← policy.sample() called here
    line  444     — env.step(act)
    line  447     — t += 1
```

**State form received:** The raw gym observation `obs` (shape `(num_envs, obs_dim)`) is first
**normalised** at line 360 before being passed to the planner:

```python
obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
```

`normalizer` is a `GaussianNormalizer` fit to the training dataset observations
(zero-mean, unit-variance per dimension). It is obtained via
`planner_dataset.get_normalizer()` at line 348.

The planner therefore **never sees raw env observations** — only normalised ones.

**Replanning cadence:** The planner is called at **every single dense env timestep** `t`.
There is no receding-horizon skip; a fresh 32-step plan is generated at each step.

---

## 3. Start-State Conditioning — Inpainting

**Mask creation:** [pipelines/veteran_d4rl_maze2d.py](../pipelines/veteran_d4rl_maze2d.py) lines 145–146

```python
fix_mask = torch.zeros((args.task.planner_horizon, planner_dim))  # (32, 4) for separate pipeline
fix_mask[0, :obs_dim] = 1.                                         # clamp position 0, all obs dims
```

**How it's used at every denoising step:**  
[cleandiffuser/diffusion/diffusionsde.py](../cleandiffuser/diffusion/diffusionsde.py) line 938 (and init at line 494):

```python
xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
```

This is applied **after every denoising iteration** and once at initialisation (line 494), keeping
the clamped dimensions fixed to whatever was written into `prior` before calling `sample()`.

**Scope of the mask:**
- `fix_mask[0, :obs_dim] = 1` → position 0 only, all observation channels (no action channels
  because `planner_dim = obs_dim` in the `separate` pipeline)
- Positions 1 … H-1 are fully unconstrained (mask = 0 everywhere else)
- The mechanism supports **arbitrary mask patterns** (any subset of `(H, D)` positions can be
  fixed by setting those entries to 1), but in this configuration only the start state is fixed

---

## 4. What Does the Trajectory Tensor Contain?

**Pipeline type:** `separate` (from `pipeline_type: separate` in `maze2d.yaml` line 20)

```python
# pipelines/veteran_d4rl_maze2d.py line 97
planner_dim = obs_dim if args.pipeline_type == "separate" else obs_dim + act_dim
```

For `separate`: `planner_dim = obs_dim = 4`

| Tensor | Shape | Contents |
|---|---|---|
| `traj` (raw from planner) | `(K, H, obs_dim)` = `(50, 32, 4)` | normalised `[x, y, vx, vy]` states — **no actions** |
| After reranking | `(num_envs, H, obs_dim)` = `(num_envs, 32, 4)` | best plan per env |

**Actions are inferred separately** by the inverse-dynamics policy from the transition
`(traj[:, 0, :], traj[:, 1, :])` — i.e., the current state and the planned next state.

For the `joint` pipeline (`planner_dim = obs_dim + act_dim`), the trajectory would contain
`[obs | action]` concatenated — but that path is not used in the current checkpoint.

**maze2d-umaze-v1 dimensionality:**
- `obs_dim = 4` — `(x, y, vx, vy)` position + velocity in the 2-D maze
- `act_dim = 2` — `(dx, dy)` velocity commands

---

## 5. Jump Step M and Planning Horizon H

**Source:** [configs/veteran/maze2d/task/maze2d-umaze-v1.yaml](../configs/veteran/maze2d/task/maze2d-umaze-v1.yaml)

```yaml
planner_horizon: 32    # H — number of waypoints in the trajectory
stride: 15             # M — dense env timesteps between adjacent waypoints
```

| Quantity | Value | Meaning |
|---|---|---|
| H (jump-steps) | 32 | number of distinct observation slots in the trajectory |
| M (stride) | 15 | dense env timesteps skipped between adjacent planner waypoints |
| Dense steps spanned | (H−1)×M = 31×15 = **465** | total future horizon in env timesteps |
| Jump-step distance | 1 step in the trajectory = 15 env steps | |

The dataset builds sequences by subsampling at stride M:
[cleandiffuser/dataset/d4rl_maze2d_dataset.py](../cleandiffuser/dataset/d4rl_maze2d_dataset.py) line 190:
```python
horizon_state = self.seq_obs[path_idx, start:end:self.stride]
```

At **inference**, only the first waypoint transition is consumed per env step:
`traj[:, 1, :]` is used as the target for the inverse-dynamics policy (lines 418–419),
and the planner replans from scratch at the next env step.

---

## 6. The Critic

**Class:** `DVHorizonCritic`  
**File:** [cleandiffuser/utils/building_blocks.py](../cleandiffuser/utils/building_blocks.py) line 176

```python
class DVHorizonCritic(nn.Module):
    def __init__(
        self,
        in_dim: int,       # = planner_dim = obs_dim = 4 (maze2d-umaze)
        emb_dim: int,      # = 128
        d_model: int = 384,   # = 256 (from config)
        n_heads: int = 6,     # = 4  (256//64)
        depth: int = 12,      # = 2  (from pipeline line 116)
        dropout: float = 0.0,
        norm_type: str = "post"   # = "pre" (from pipeline line 116)
    ):
```

**Forward pass:** (line 210)
```python
def forward(self, x: torch.Tensor):
    # x:  (B, horizon, in_dim)  — full trajectory in planner space
    # returns: (B, 1)           — scalar value per trajectory
    x = self.x_proj(x) + positional_embedding     # project + add sinusoidal pos emb
    for block in self.blocks: x = block(x)         # depth=2 transformer blocks
    x = self.final_layer(x)                        # linear → (B, H, 1)
    return x[:, 0, :]                              # take position-0 token → (B, 1)
```

**Architecture summary:**
- Single network (no ensemble)
- Input: full trajectory `(B, 32, 4)` in planner space (normalised obs only)
- Output: scalar value `(B, 1)` extracted from the first transformer token
- Sinusoidal position embeddings (cached)
- 2 × `DVTransformerBlock` (pre-norm, multi-head self-attention + MLP)

**Training loss:** plain MSE regression  
[pipelines/veteran_d4rl_maze2d.py](../pipelines/veteran_d4rl_maze2d.py) lines 230–232:

```python
val_pred = critic(planner_horizon_data)            # (B, 1)
critic_loss = F.mse_loss(val_pred, planner_td_val) # planner_td_val is MC return in [-1, 1]
```

`planner_td_val` is the **Monte Carlo discounted return** normalised to `[-1, 1]`
(see dataset line 174–176 for the normalisation). **Not** IQL-style expectile regression,
**not** CQL-style conservative loss.

---

## 7. Seed Management

**Seeding function:** [pipelines/utils.py](../pipelines/utils.py) lines 70–73

```python
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
```

Called once at startup: [pipelines/veteran_d4rl_maze2d.py](../pipelines/veteran_d4rl_maze2d.py) line 44:
```python
set_seed(args.seed)   # args.seed = 0 (default from maze2d.yaml line 8)
```

**What is and is not seeded:**
- `torch`, `numpy`, `random` RNGs are seeded once globally
- **No per-episode re-seeding** — episode variance comes from diffusion stochasticity
- **No gym env seed** — `env_eval.reset()` at line 352 is not seeded per episode
- All diffusion randomness (`torch.randn_like(prior)` in `diffusionsde.py` line 835)
  flows from the global PyTorch seed set at startup
- CUDA non-determinism (cuDNN) is **not** suppressed — results may vary across runs
  even with the same seed unless `torch.backends.cudnn.deterministic = True`

---

## Data Flow Summary

```
env.reset()
  │  raw obs  (num_envs, 4)
  ▼
normalizer.normalize()
  │  normed obs  (num_envs, 4)
  ▼
tile K=50 times → (num_envs*50, 4)
  │  write into prior[:, 0, :]
  ▼
ContinuousDiffusionSDE.sample()  [solver=ddim, steps=20, no CFG]
  │  fix_mask clamps prior[:, 0, :obs_dim] at each of 20 denoising iters
  ▼
traj  (num_envs*50, 32, 4)  — normalised obs trajectories
  │
  ▼
DVHorizonCritic.forward(traj) → (num_envs*50, 1) scalar values
  │  argmax over K dim
  ▼
best_traj  (num_envs, 32, 4)
  │  traj[:, 1, :] = planned next obs
  │  rebase: next_obs[:, :2] -= obs[:, :2];  obs[:, :2] = 0
  ▼
DiscreteDiffusionSDE.sample()  [solver=ddpm, steps=10, CFG w=1.0]
  │  condition_cfg = cat([obs_rebased, next_obs_rebased], dim=-1)  shape (num_envs, 8)
  ▼
act  (num_envs, 2)
  │
  ▼
env.step(act)  →  next raw obs, reward, done
  │
  └─── repeat at next dense timestep (re-plan every step)
```

---

## Key Configuration Reference (maze2d-umaze-v1)

| Parameter | Value | Source |
|---|---|---|
| `obs_dim` | 4 | dataset |
| `act_dim` | 2 | dataset |
| `planner_horizon` H | 32 | `maze2d-umaze-v1.yaml` |
| `stride` M | 15 | `maze2d-umaze-v1.yaml` |
| `planner_num_candidates` K | 50 | `maze2d.yaml` |
| `planner_solver` | `"ddim"` | `maze2d.yaml` |
| `planner_sampling_steps` | 20 | `maze2d.yaml` |
| `policy_solver` | `"ddpm"` | `maze2d.yaml` |
| `policy_sampling_steps` | 10 | `maze2d.yaml` |
| `planner_d_model` | 256 | `maze2d.yaml` |
| `planner_depth` | 2 | `maze2d.yaml` |
| `noise_schedule` | `"linear"` | pipeline line 154 |
| `rebase_policy` | `True` | `maze2d.yaml` line 22 |
| `max_path_length` | 300 | `maze2d-umaze-v1.yaml` |
| Critic checkpoint | 200000 steps | `maze2d.yaml` line 62 |
| Planner/policy ckpt | 1000000 steps | `maze2d.yaml` lines 61,63 |
| NN calls per env step | 30 (20 planner + 10 policy) | — |
