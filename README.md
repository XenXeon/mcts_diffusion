<a name="readme-top"></a>

# Integrating Monte Carlo Tree Search into Offline Diffusion Planners

An experimental study of **when a structured tree search helps a state-of-the-art
diffusion planner** for offline reinforcement learning, built on top of
[CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) and its
Diffusion Veteran (DV) planner. All experiments run on the
[D4RL](https://github.com/Farama-Foundation/D4RL) benchmarks (maze2d, antmaze,
FrankaKitchen).

---

## Overview

The base system (DV) draws a fixed number of candidate trajectories, ranks the
whole trajectory with a learned critic, executes the first step, and repeats —
a flat best-of-K rule called **Monte Carlo Sampling with Selection (MCSS)**.
This project asks whether replacing that flat selection with a **Monte Carlo
Tree Search** over trajectories (or over the denoising process) does better,
and under what conditions.

Everything is added on top of a **frozen** copy of DV — the same inverse-dynamics
policy is shared across every arm, and the planner and critic are the only
experimental factors, so each comparison isolates one variable. The code
contributed by this work lives in **`mcts/`** (tree search, node value
functions, the causal Diffusion-Forcing and shortcut planners, the per-token
noise-aware guidance model, and a faithful port of Monte Carlo Tree Diffusion)
and **`scripts/`** (experiment drivers, training, analysis, figures).

Two entry points run every experiment:

| Script | Execution model | Used for |
|---|---|---|
| `scripts/run_mcts_compare.py` | per-step MPC (Setup 1) | flat MCSS vs the **trajectory-axis** tree; all node-value, backbone, and guidance experiments (dissertation Chapters 4–5) |
| `scripts/run_mctd.py` | periodic-replan MPC (Setup 2) | the **denoising-axis** search (MCTD) and its controls; cadence and target-selection controls (Chapter 6) |

Each run writes a self-describing **result JSON** (full config, git commit,
per-rollout scores, and the start/goal arrays) so any comparison can be
re-paired and re-checked after the fact.

---

## Getting Started

The environment is a Docker image (CUDA + PyTorch nightly + D4RL/mujoco-py,
pre-compiled). You need an **NVIDIA GPU**, **Docker**, and the
**[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**
(so `--gpus all` works).

### 1. Build the images

```bash
make build
```

This builds `cleandiffuser:base` (from `Dockerfile`) and `cleandiffuser:dev`
(from `Dockerfile.dev`, which adds D4RL from source and pre-compiles
`mujoco_py`).

### 2a. Open in VS Code (devcontainer — recommended)

The repo ships a `.devcontainer/` config. In VS Code with the *Dev Containers*
extension, run **"Dev Containers: Reopen in Container"**. It mounts the repo at
`/workspace`, requests all GPUs, and installs the Python/Jupyter extensions.
`WANDB_API_KEY`, `WANDB_ENTITY`, and `HF_TOKEN` are forwarded from your host
environment if set (only needed for training/logging, not for evaluation).

### 2b. Or run the container directly

```bash
make run          # interactive shell in cleandiffuser:dev, repo mounted at /workspace
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 3. Datasets and checkpoints

- **D4RL datasets** download automatically on first use and cache to `~/.d4rl`
  (mounted into the container by the Makefile targets).
- **Checkpoints are not tracked in git.** The closed-loop scripts expect the
  frozen DV planner / critic / policy under
  `results/veteran_d4rl_<family>_.../<env-name>/` (see `mcts/specs.py` for the
  exact directory per family). Train them with the Makefile's Hydra pipeline
  targets, e.g.:

  ```bash
  make train PIPELINE=veteran_d4rl_maze2d  TASK=maze2d-large-v1
  make train PIPELINE=veteran_d4rl_kitchen TASK=kitchen-mixed-v0
  # (make eval PIPELINE=... TASK=... STEP=latest  runs the DV inference baseline)
  ```

  The additional models this project introduces are produced by the training
  scripts and then selected at evaluation time by the flags in brackets:

  | Model | Train with | Selected at eval by |
  |---|---|---|
  | behaviour-return value `V(s)` | `scripts/train_state_value.py` | `--value-mode v_s` |
  | distilled plan-value `V̂(s)` | `scripts/gen_plan_value_labels.py` → `scripts/train_plan_value.py` | `--value-mode v_s --value-step <tag>` |
  | goal-conditioned `V(s,g)` | `scripts/train_state_value.py` (goal-cond) | `--value-mode v_sg` / `v_sg_pess` |
  | stitched-window critic | `scripts/finetune_critic_stitched.py` | `--critic-step stitched` |
  | causal Diffusion-Forcing planner | `scripts/train_df_planner.py` | `--df-ckpt <tag>` |
  | per-token noise-aware guidance | `scripts/train_noise_critic.py` | `--cg-ckpt <tag> --cg-w <w>` |

### 4. Smoke test

Confirm the closed loop runs end-to-end before launching anything long:

```bash
python scripts/run_mcts_compare.py --env maze2d-large-v1 --method both \
    --n-envs 4 --n-episodes 1 --budget 6 --max-steps 200
```

Unit tests (the tree, verifier, scheduling, and window composition are
torch-free and run anywhere):

```bash
pytest tests/ -m "not integration"
```

---

## Running the experiments

Every command below saves a result JSON via `--out`. Runs at the same
`--env` / `--seed` / `--n-envs` are **paired by construction** (shared RNG
stream → identical start/goal draws), so differences are computed per-rollout;
pair and compute the seed-level statistics with `scripts/collate_mcts.py` and
`scripts/seed_level_stats.py`.

Supported envs: `maze2d-large-v1`, `maze2d-medium-v1`, `maze2d-umaze-v1`,
`antmaze-large-diverse-v2`, `kitchen-mixed-v0`.

### A. Reproduce the flat DV baseline (MCSS)

```bash
# DV MCSS at K=50 and K=256, maze2d-large, 10 seeds
for s in 0 1 2 3 4 5 6 7 8 9; do
  python scripts/run_mcts_compare.py --env maze2d-large-v1 --method mcss --k-mcss 50  --seed $s --out results/dv_mcss_k50_s$s.json
  python scripts/run_mcts_compare.py --env maze2d-large-v1 --method mcss --k-mcss 256 --seed $s --out results/dv_mcss_k256_s$s.json
done
```

### B. Trajectory tree on the DV backbone (the tree that loses, and why)

The full-sequence DV planner cannot be conditioned on a prefix exactly, so the
tree loses here. These arms decompose the failure (node value, backup, and
expansion fidelity).

```bash
# tree with the DV trajectory critic, seam-glue expansion, tempered top-3 backup,
# root width 50 (a strict superset of the MCSS pool)
for s in 0 1 2; do
  python scripts/run_mcts_compare.py --env maze2d-large-v1 --method both \
    --value-mode critic --expand-mode glue --k-root 50 --k-mcts 16 --budget 15 --top-m 3 \
    --seed $s --out results/dv_tree_top3_s$s.json
done
```

Vary **one** knob at a time to reproduce the defect table:

```bash
# defect 2 — backup:      --top-m 1   (MAX backup)   vs  --top-m 3  (tempered)
# defect 3 — expansion:   --expand-mode glue          vs  --expand-mode inpaint
# node value alternatives: --value-mode v_sg  (goal-conditioned, mean)
#                          --value-mode v_sg_pess  (pessimistic ensemble-min)
# defect 1 — the value target: behaviour-return vs distilled plan-value are the
#   same --value-mode v_s, chosen by different --value-step checkpoints.
```

### C. Trajectory tree on the causal Diffusion-Forcing backbone (the tree that wins)

With `--df-ckpt`, **both** arms use the DF backbone: `mcss` becomes DF
sample-and-rank, `mcts` becomes exact prefix-conditioned expansion.

```bash
for s in 0 1 2 3 4; do
  python scripts/run_mcts_compare.py --env maze2d-large-v1 --method both \
    --df-ckpt final --value-mode critic --k-root 50 --k-mcts 16 --budget 15 --top-m 3 \
    --seed $s --out results/df_both_s$s.json
done
# repeat with --env maze2d-medium-v1 / maze2d-umaze-v1 / kitchen-mixed-v0 (use --k-mcss 150 for kitchen)
```

### D. Shortcut-forcing backbone (the weaker third point for the headroom curve)

```bash
for s in 0 1 2 3 4; do
  python scripts/run_mcts_compare.py --env maze2d-large-v1 --method both \
    --df-ckpt <shortcut-tag> --sweeps 8 --value-mode critic --k-root 50 --top-m 3 \
    --seed $s --out results/short_both_s$s.json
done
```

### E. FrankaKitchen: guidance and the grounded evaluator (Chapter 5)

```bash
# DF flat + tree
python scripts/run_mcts_compare.py --env kitchen-mixed-v0 --method both --df-ckpt final --k-mcss 150 --seed 0 --out results/kitchen_df_s0.json

# per-token noise-aware classifier guidance on the DF sampler (w = guidance strength)
python scripts/run_mcts_compare.py --env kitchen-mixed-v0 --method both --df-ckpt final \
    --cg-ckpt <cg-tag> --cg-w 8 --k-mcss 150 --seed 0 --out results/kitchen_cg8_s0.json

# grounded subtask evaluator (kitchen only) as the tree node value, critic-blended tiebreak
python scripts/run_mcts_compare.py --env kitchen-mixed-v0 --method mcts --df-ckpt final \
    --value-mode grounded --grounded-blend 0.25 --k-mcss 150 --seed 0 --out results/kitchen_grounded_s0.json
```

### F. Denoising-axis search: Monte Carlo Tree Diffusion (Chapter 4.4)

`run_mctd.py` uses the periodic-replan MPC harness (Setup 2). MCTD is maze2d /
antmaze only (its geometric verifier needs a positional goal).

```bash
for s in 0 1 2 3 4; do
  # MCTD as published (geometric verifier)
  python scripts/run_mctd.py --env maze2d-large-v1 --replan-every 50 --seed $s --out results/mctd_s$s.json
  # the execution-matched flat control: best-of-K MCSS in the identical harness
  python scripts/run_mctd.py --env maze2d-large-v1 --flat-mcss --mcss-backbone df --replan-every 50 --seed $s --out results/mctd_ctrl_s$s.json
done

# ablations (change one component):
#   --value-mode critic   MCTD with the DV critic instead of the geometric verifier
#   --guided-bon          flat best-of-N over guidance weights, no tree
```

### G. Execution-model controls (Chapter 6)

Absolute scores move sharply with the execution setup, so these isolate the
**target-selection rule** and the **backbone-vs-cadence** gap inside one harness.

```bash
# target-selection rule isolated on the DF-tree: aim-next (reach-wp 0) vs advance-past (reach-wp 1)
for s in 0 1 2; do
  python scripts/run_mctd.py --env maze2d-large-v1 --df-tree --tree-k-root 50 --tree-k 16 --tree-budget 15 --tree-top-m 3 --replan-every 1 --reach-wp 0.0 --seed $s --out results/treerule_aimnext_s$s.json
  python scripts/run_mctd.py --env maze2d-large-v1 --df-tree --tree-k-root 50 --tree-k 16 --tree-budget 15 --tree-top-m 3 --replan-every 1 --reach-wp 1.0 --seed $s --out results/treerule_advance_s$s.json
done

# backbone-vs-cadence: DV and DF flat MCSS in the same MPC harness at matched cadence
python scripts/run_mctd.py --env maze2d-large-v1 --flat-mcss --mcss-backbone dv --replan-every 50 --seed 0 --out results/cad_dv_rp50_s0.json
python scripts/run_mctd.py --env maze2d-large-v1 --flat-mcss --mcss-backbone df --replan-every 50 --seed 0 --out results/cad_df_rp50_s0.json
```

---

## Command reference (what each flag does)

### Shared / scale

| Flag | What it is | Changing it |
|---|---|---|
| `--env` | D4RL environment | `maze2d-{large,medium,umaze}-v1`, `antmaze-large-diverse-v2`, `kitchen-mixed-v0`. Picks the checkpoint family and geometry automatically. |
| `--seed` | RNG seed = unit of replication | Fixes the whole set of start/goal draws; different seeds give paired repeats. Confirmed claims use ≥3 seeds. |
| `--n-envs` | parallel environments per seed | More envs = a tighter per-seed mean but more GPU memory. Default 25. |
| `--n-episodes` | episodes per env | Usually 1 (maze2d/kitchen). |
| `--max-steps` | cap episode length | Lower it (e.g. 200) to smoke-test; default = the env's own time limit. |
| `--out` | result JSON path | Where the per-rollout scores + config are saved. |
| `--dv-log` | print DV-style inference log | Emits the base system's per-step log and DV-exact score. |

### `run_mcts_compare.py` — trajectory-axis tree

| Flag | What it is | Changing it |
|---|---|---|
| `--method` | which arm(s) to run | `mcss` (flat), `mcts` (tree), or `both` (paired). |
| `--k-mcss` | flat MCSS candidate count | The flat pool size (50 default, 256 for the width control, 150 on kitchen). Wider ≈ flat "best-of-more". |
| `--k-mcts` | candidates per expansion | Fan-out at each expanded node. Higher = wider per-node search, more planner calls (cost); returns saturate by ~16. |
| `--k-root` | root expansion width | The pool the executed action is chosen from — set `50` to make the root a strict superset of MCSS. Default = `--k-mcts`. |
| `--budget` | expansion rounds | Tree depth/size budget. More rounds = a bigger tree at linear cost; the search budget has the smallest effect on the score. |
| `--top-m` | backup = mean of top-m children | `1` = MAX backup (optimistic, overestimates on noisy scores); `>1` tempers it. `3` is the default tempered value. |
| `--child-index` | segment length per edge | Which continuation index becomes the child state (edge length L). |
| `--c-ucb` | UCB exploration constant | Higher explores rarely-visited children more. Default √2. |
| `--value-mode` | tree node value (mcts arm) | `critic` (DV trajectory critic), `v_s` (per-state value), `v_sg` / `v_sg_pess` (goal-conditioned mean / pessimistic-min), `grounded` (kitchen subtask count, non-learned). |
| `--value-step` | value checkpoint tag | Selects which `V(s)` checkpoint — this is how behaviour-return vs distilled plan-value are switched. |
| `--pess-beta` | pessimism strength | For the `mean − β·std` variant. |
| `--expand-mode` | prefix-conditioning method | `glue` (continuation from the leaf state, concatenated — leaves a seam) vs `inpaint` (prefix clamped into the denoiser — no seam but off-distribution for a full-sequence net). |
| `--df-ckpt` | DF planner checkpoint tag | Switches **both** arms to the causal Diffusion-Forcing backbone (`df_planner_ckpt_<tag>.pt`); enables exact prefix conditioning. |
| `--sweeps` | shortcut sampling sweeps | Shortcut backbone only — a power of 2 (e.g. `8`); fewer sweeps = faster, lower-quality sampling. |
| `--df-slope` | pyramid schedule slope | Extra noise levels per future token in the DF schedule. |
| `--cg-ckpt` / `--cg-w` | noise-aware guidance | `--cg-ckpt` picks the per-token guidance model; `--cg-w` is its strength (`0` = off). Needs `--df-ckpt` (guidance steers the DF sampler). |
| `--grounded-blend` | grounded tiebreak weight | Weight of the DV critic added on top of the grounded subtask count (`0` = pure grounded). |
| `--grounded-mcss` | grounded rerank of MCSS | `1` reranks flat candidates by the grounded checker instead of the critic (kitchen only). |
| `--junction-filter` / `--junction-pct` | implausible-hop guard | Reject tree children whose first step exceeds the `--junction-pct` percentile of dataset stride. |
| `--critic-step` | critic checkpoint | An int step, or `stitched` to load the stitched-window fine-tuned critic. |
| `--rebase-policy` | policy-input rebasing | `0/1`; default follows the DV config per family (kitchen = 0, its first dims are joint angles not xy). |

### `run_mctd.py` — denoising-axis search + MPC controls

Selects the arm by flag: **default** = faithful MCTD; `--flat-mcss` = flat
control; `--guided-bon` = guided best-of-N; `--df-tree` = this project's
trajectory tree run inside the same MPC harness.

| Flag | What it is | Changing it |
|---|---|---|
| `--replan-every` | env-steps between replans | The MPC cadence (Setup 2). `1` = replan every step; `50` = the reference open-loop horizon. Strongly affects absolute score. |
| `--reach-wp` | waypoint-advance threshold | The **target-selection rule**: `0.0` aims at the next waypoint (aim-next); `1.0` advances past waypoints already reached (advance-past). |
| `--guidance` | MCTD meta-action menu | The set of guidance scales the denoising tree chooses among (default `0 0.1 0.5 1 2`). |
| `--n-depths` | denoising-tree depth | Number of denoising blocks = tree depth. |
| `--max-search` | expansions per plan | MCTD search budget per replan. |
| `--skip` | jumpy-rollout stride | Stride of the fast rollout that scores a node. |
| `--value-mode` | MCTD node value | `geometric` (non-learned goal-reach verifier, as published) or `critic` (DV trajectory critic on the clean plan). |
| `--flat-mcss` / `--k` | flat control | Run best-of-`--k` MCSS in the identical MPC harness (isolates search from the execution model). |
| `--mcss-backbone` | control planner | `df` (MCTD's backbone) or `dv` (the frozen SOTA planner — separates the backbone gap from the cadence gap). |
| `--guided-bon` / `--k-per` | guided best-of-N | Flat best-of-N over guidance weights, no tree; `--k-per` plans per weight. |
| `--df-tree` + `--tree-{budget,k,k-root,top-m}` | the trajectory tree in the MPC harness | Runs this project's DF-tree at `--replan-every`, making it raw-comparable to MCTD. The `--tree-*` knobs mirror `--budget` / `--k-mcts` / `--k-root` / `--top-m` above. |
| `--df-ckpt` | DF checkpoint tag | `df_planner_ckpt_<tag>.pt` (default `final`). |

---

## Repository layout

```
mcts/          tree search, node value functions, expansion mechanisms,
               causal DF + shortcut planners, per-token guidance, MCTD port
scripts/       experiment drivers (run_mcts_compare.py, run_mctd.py),
               training, analysis/collation, figure generation
pipelines/     Hydra training/inference pipelines for the DV base system
cleandiffuser/ the upstream CleanDiffuser library (diffusion models, datasets)
tests/         unit tests (tree/verifier/scheduling are torch-free) + GPU smoke tests
results/       result JSONs (checkpoints live here too, but are git-ignored)
```

---

## Acknowledgement and attribution

This work is built on **[CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser)**
and its **Diffusion Veteran** planner
([*What Makes a Good Diffusion Planner for Decision Making?*](https://openreview.net/forum?id=7BQkXXM8Fy)),
and uses the **[D4RL](https://github.com/Farama-Foundation/D4RL)** benchmarks.
The CleanDiffuser library retains its original Apache License 2.0 (see
`LICENSE.txt`). The `mcts/` package, the experiment drivers and analysis tooling
in `scripts/`, and the evaluation harness are contributions of this dissertation.

```
@article{cleandiffuser,
  author  = {Zibin Dong and Yifu Yuan and Jianye Hao and Fei Ni and Yi Ma and Pengyi Li and Yan Zheng},
  title   = {CleanDiffuser: An Easy-to-use Modularized Library for Diffusion Models in Decision Making},
  journal = {arXiv preprint arXiv:2406.09509},
  year    = {2024},
  url     = {https://arxiv.org/abs/2406.09509},
}
```
