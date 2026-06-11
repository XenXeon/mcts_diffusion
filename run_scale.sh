#!/usr/bin/env bash
set -euo pipefail
for s in 0 1 2; do
    python scripts/run_mcts_compare.py --env antmaze-large-diverse-v2 --method mcss \
        --n-envs 50 --n-episodes 1 --seed "$s" --k-mcss 272 \
        --out "results/scale_mcss_k272_s${s}.json"
    python scripts/run_mcts_compare.py --env antmaze-large-diverse-v2 --method mcts \
        --n-envs 50 --n-episodes 1 --seed "$s" --budget 16 --k-mcts 16 \
        --out "results/scale_mcts_b16_s${s}.json"
done
python scripts/collate_mcts.py results/scale_*.json
