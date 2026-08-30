#!/usr/bin/env bash
# Run run_one_episode.py for seeds 0–4 and accumulate DV-MCSS Phase 0 baselines.
# Writes a separate JSON per env so umaze/medium/large stay distinct.
#
#   bash scripts/run_baseline_seeds.sh                  # umaze (default)
#   bash scripts/run_baseline_seeds.sh maze2d-large-v1  # large
#
set -euo pipefail

ENV="${1:-maze2d-umaze-v1}"
case "$ENV" in
    maze2d-umaze-v1)  TAG="umaze"  ;;
    maze2d-medium-v1) TAG="medium" ;;
    maze2d-large-v1)  TAG="large"  ;;
    *) echo "Unknown env: $ENV"; exit 1 ;;
esac
# umaze keeps the original un-suffixed filename: every consumer
# (phase4_mcts_rollout.py, phase4_ablation.py, phase6_stage0_oracle.py) reads
# results/phase0_baseline.json for umaze and the _${TAG} variants otherwise.
if [ "$TAG" = "umaze" ]; then
    OUTFILE="results/phase0_baseline.json"
else
    OUTFILE="results/phase0_baseline_${TAG}.json"
fi

# Fresh file each run so re-runs don't append duplicates.
rm -f "$OUTFILE"

for seed in 0 1 2 3 4; do
    echo "=== $ENV  seed $seed ==="
    python scripts/run_one_episode.py --env "$ENV" --seed "$seed" --save-json "$OUTFILE"
done

echo ""
echo "All seeds done. Results in $OUTFILE"
python - "$OUTFILE" <<'EOF'
import json, statistics, sys
with open(sys.argv[1]) as f:
    rows = json.load(f)
scores = [r["normalized_score"] for r in rows]
print(f"scores:  {scores}")
print(f"mean:    {statistics.mean(scores):.1f}")
print(f"std:     {statistics.stdev(scores):.1f}")
print(f"min/max: {min(scores):.1f} / {max(scores):.1f}")
EOF
