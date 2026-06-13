#!/usr/bin/env bash
set -euo pipefail
# Budget curve at scale: b4/b8 on the SAME 50 scenarios per seed as the existing
# headline cells (same seed + --n-envs 50 + --n-episodes 1 => same goal draws;
# collate_mcts verifies pairing per cell before computing McNemar).
for B in 4 8; do
    for S in 0 1 2; do
        python scripts/run_mcts_compare.py --env antmaze-large-diverse-v2 --method mcts \
            --n-envs 50 --n-episodes 1 --seed "$S" --budget "$B" --k-mcts 16 \
            --out "results/scale_mcts_b${B}_s${S}.json"
    done
done
# MCSS k=50 control: anchors the harness to DV's published 76.9 baseline AND gives
# the MCSS saturation point (k50 vs k272) at scale. Cheap (~35 min/seed).
for S in 0 1 2; do
    python scripts/run_mcts_compare.py --env antmaze-large-diverse-v2 --method mcss \
        --n-envs 50 --n-episodes 1 --seed "$S" --k-mcss 50 \
        --out "results/scale_mcss_k50_s${S}.json"
done
python scripts/collate_mcts.py results/scale_*.json
