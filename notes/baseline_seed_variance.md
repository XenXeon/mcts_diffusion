# DV-MCSS Baseline — Seed Variance

**Task:** maze2d-umaze-v1  
**Method:** DV-MCSS (planner: DiT-256 d=2, DDIM 20 steps, K=50 candidates; policy: diffusion inv-dyn DDPM 10 steps)  
**Checkpoint:** 1 000 000 gradient steps  
**Git commit:** 5ae067313c2ffe1b7211208edb7bdc2a3db0e529  
**Accumulation:** latch (`ep_reward += float(finished)`) — mirrors reference pipeline  

Run with:
```bash
bash scripts/run_baseline_seeds.sh
```

## Results

| Seed | Norm. score | Raw return | Goal step | Length | Wall (s) | ms/step |
|------|-------------|------------|-----------|--------|----------|---------|
| 0    | 114.6       | 182.0      | 118       | 300    | 28.62    | 95.4    |
| 1    | TBD         |            |           |        |          |         |
| 2    | TBD         |            |           |        |          |         |
| 3    | TBD         |            |           |        |          |         |
| 4    | TBD         |            |           |        |          |         |
| **mean ± std** | | | | | | |

## Interpretation guide

| Outcome | Meaning |
|---|---|
| All scores 110–130 | Tight baseline; 3 seeds per condition is enough for Phase 4 |
| Scores swing 80–150 | High variance; use 5+ seeds and report confidence intervals |
| Any seed ≤ 50 | Something is wrong — check env reset, normaliser, or checkpoint |
| Any seed ≥ 160 | Suspiciously high — check latch accumulation isn't double-counting |

## DV paper reference

The DV paper reports ~140 normalised score on maze2d-umaze-v1 (averaged over multiple seeds and
episodes). A single-seed single-episode score of 114.6 is below that mean but well within the
expected single-episode distribution. The reference pipeline evaluates over 20 episodes × 50 envs;
single-episode variance on this task is known to be high.

`REF_MIN_SCORE = 23.85`, `REF_MAX_SCORE = 161.86` (D4RL infos.py)  
A normalised score of 114.6 corresponds to raw return 182, meaning the agent first reached the
goal at step 118 and the latch accumulated 182 subsequent steps.
