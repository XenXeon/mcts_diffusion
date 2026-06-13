#!/usr/bin/env bash
set -euo pipefail
for L in 2 4; do
    python scripts/run_mcts_compare.py --env antmaze-large-diverse-v2 --method mcts \
        --n-envs 50 --n-episodes 1 --seed 0 --budget 16 --k-mcts 16 --child-index "$L" \
        --out "results/scale_mcts_b16L${L}_s0.json"
done
python scripts/collate_mcts.py results/scale_*.json
