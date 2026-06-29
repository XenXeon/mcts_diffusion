#!/usr/bin/env bash
set -euo pipefail
for S in 0 1 2 3 4 5 6 7 8 9; do
  # MCSS k=50 (cheap baseline; ~0.6 h/seed)
  python scripts/run_mcts_compare.py --env antmaze-large-diverse-v2 --method mcss \
      --n-envs 50 --n-episodes 1 --seed "$S" --k-mcss 50 --dv-log \
      --out "results/s10_mcss_k50_s${S}.json"
  # MCSS k=272 (matched-compute flat baseline; ~3.1 h/seed)
  python scripts/run_mcts_compare.py --env antmaze-large-diverse-v2 --method mcss \
      --n-envs 50 --n-episodes 1 --seed "$S" --k-mcss 272 --dv-log \
      --out "results/s10_mcss_k272_s${S}.json"
  # MCTS b16 + V(s) (the winner; ~3.1 h/seed)
  python scripts/run_mcts_compare.py --env antmaze-large-diverse-v2 --method mcts \
      --n-envs 50 --n-episodes 1 --seed "$S" --budget 16 --k-mcts 16 --dv-log \
      --out "results/s10_mcts_b16_s${S}.json"
done
python scripts/collate_mcts.py results/s10_*.json
