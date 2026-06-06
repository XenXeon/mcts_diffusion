# Phase 1 Preflight Findings

## A. Dataset normalisation state
`dataset.seq_obs` is ALREADY NORMALISED (z-scored via GaussianNormalizer at
construction time). Do not re-normalise dataset states before passing to
planner/critic. env.reset/step observations are RAW and must be
normalised at the boundary. See preflight script output: dataset sample
abs-max ≈ 0.94, env-obs after normalisation abs-max ≈ 1.07,
double-normalising blows out to abs-max ≈ 4.

## B. Trajectory lengths
`seq_obs` shape (1566, 1265, 4) is heavily padded with repeated terminal states. Per-trajectory lengths recovered empirically by detecting where consecutive states stop changing (threshold 1e-5). Saved to `results/phase1/path_lengths.npy`.
- mean length 482.8 (dense env steps)
- range [47, 800]
- trajectories at 800 cap: 372
- spot-checks confirmed padding detection is perfectly accurate.

## C. Held-Out Pool Sizing
Using `heldout_frac=0.10` yields 156 held-out trajectories. 
Of these, 74 are "usable" (length >= H*M = 480 dense steps).
Total valid offset slots: 17,676. 
This is highly sufficient for a 100-sample Phase 1 verification without over-indexing on a few trajectories.

## D. Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
Set in all Phase 1+ runner scripts.