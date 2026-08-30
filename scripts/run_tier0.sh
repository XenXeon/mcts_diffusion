#!/usr/bin/env bash
# scripts/run_tier0.sh — Tier-0 multi-environment broadening (no new training beyond a cheap
# V(s) MLP; all checkpoints already on disk). Runs MCSS (DV baseline) vs MCTS (b16 look-ahead)
# across the D4RL navigation family and collates with the DV-EXACT per-family metric:
#   antmaze  -> reach%      (paired McNemar)
#   maze2d   -> camping DV-score >100  (paired difference test; reach is saturated there)
#
# Envs (all have planner/critic/policy checkpoints already):
#   maze2d  {umaze, medium, large}          -- point mass, NO locomotion (the clean control)
#   antmaze {medium-diverse, medium-play, large-play}  -- does the locomotion ceiling scale?
# (antmaze-large-diverse-v2 is the already-measured reference; not re-run here.)
#
# Usage (from the repo root, on the GPU box):
#   bash scripts/run_tier0.sh                 # seeds 0 1 2, MCSS k50 + MCTS b16
#   SEEDS="0" bash scripts/run_tier0.sh       # fast single-seed first look
#   RUN_K272=1 bash scripts/run_tier0.sh      # also the flat-scaling backfire control (antmaze)
#
# Time: maze2d cells are quick; antmaze b16 ~3.1 h/seed (the bulk). Start with SEEDS="0".

set -u
SEEDS="${SEEDS:-0 1 2}"
RUN_K272="${RUN_K272:-0}"
NENVS="${NENVS:-50}"

MAZE2D="maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1"
ANTMAZE="antmaze-medium-diverse-v2 antmaze-medium-play-v2 antmaze-large-play-v2"
ALL="$MAZE2D $ANTMAZE"

# 0) Cheap V(s) MLP for the MCTS arm — only the variants that lack one.
#    (maze2d-large-v1 and antmaze-large-diverse-v2 already have state_value_ckpt_latest.pt.)
echo "=== [0] training V(s) for variants without it ==="
for ENV in maze2d-umaze-v1 maze2d-medium-v1 \
           antmaze-medium-diverse-v2 antmaze-medium-play-v2 antmaze-large-play-v2; do
  CK="results/veteran_d4rl_$( [[ $ENV == maze2d* ]] && echo maze2d_H32_Jump15_next1_MCSS_transformer_d2_width256_separate_dpTrue || echo antmaze_H40_Jump25_next1_MCSS_transformer_d8_width256_separate_dp1 )/$ENV/state_value_ckpt_best.pt"
  if [[ -f "$CK" ]]; then echo "  $ENV: V(s) best-ckpt already present, skip"; continue; fi
  echo "  training V(s): $ENV  (expectile tau=0.9; saves state_value_ckpt_best.pt at peak val_corr)"
  python scripts/train_state_value.py --env "$ENV" --steps 200000 \
      --loss expectile --tau 0.9 --log-interval 1000 \
      || echo "  !! V(s) train failed: $ENV"
done

# 1) MCSS (k50) vs MCTS (b16) per env, per seed.
echo "=== [1] MCSS vs MCTS runs ==="
for ENV in $ALL; do
  OUT="results/tier0/$ENV"; mkdir -p "$OUT"
  for S in $SEEDS; do
    echo "--- $ENV seed $S ---"
    python scripts/run_mcts_compare.py --env "$ENV" --method mcss --n-envs "$NENVS" \
        --n-episodes 1 --seed "$S" --k-mcss 50  --out "$OUT/mcss_k50_s$S.json" \
        || echo "  !! mcss k50 failed: $ENV s$S"
    python scripts/run_mcts_compare.py --env "$ENV" --method mcts --n-envs "$NENVS" \
        --n-episodes 1 --seed "$S" --budget 16 --k-mcts 16 --value-step best \
        --out "$OUT/mcts_b16_s$S.json" || echo "  !! mcts b16 failed: $ENV s$S"
    if [[ "$RUN_K272" == "1" ]]; then      # optional flat-scaling backfire control
      python scripts/run_mcts_compare.py --env "$ENV" --method mcss --n-envs "$NENVS" \
          --n-episodes 1 --seed "$S" --k-mcss 272 --out "$OUT/mcss_k272_s$S.json" \
          || echo "  !! mcss k272 failed: $ENV s$S"
    fi
  done
done

# 2) Collate per env — DV-correct metric chosen automatically by family.
echo "=== [2] collation (antmaze=reach%/McNemar, maze2d=camping/paired-diff) ==="
for ENV in $ALL; do
  echo; echo "################## $ENV ##################"
  python scripts/collate_mcts.py results/tier0/"$ENV"/*.json
done
echo "=== done ==="
