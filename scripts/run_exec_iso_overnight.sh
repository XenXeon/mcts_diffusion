#!/usr/bin/env bash
# run_exec_iso_overnight.sh  (Git Bash / MSYS)
# Execution-model isolation for dissertation issue #2 (target-selection confound).
# Flat DF best-of-K on maze2d-large at MATCHED cadence (--replan-every 1),
# varying ONLY the target rule:
#     --reach-wp 0.0  -> aim at the immediate next waypoint  ("aimnext")
#     --reach-wp 1.0  -> advance past reached waypoints        ("advance")
# 3 seeds each; splits the ~55-point per-step-vs-rp50 gap into cadence vs rule.
#
# Safety: quick preflight, a 7.25h global deadline, and a 90-min per-run kill.
# Total wall-clock bounded under 8 hours. Safe to leave overnight.

cd "D:/Surrey/Sem 2/Dissertation/backup/mcts_diffusion_back" || { echo "cd failed"; exit 1; }
export PYTHONIOENCODING=utf-8

LOGDIR="results/exec_iso_logs"
mkdir -p "$LOGDIR"
PER_RUN_CAP=5400          # 90 min hard cap per run
BUDGET_SEC=26100          # 7.25h real-run budget (preflight adds <=15 min on top)
ENV="maze2d-large-v1"

run_py () {   # $1=timeout_sec  $2=logbase  $3.. = python args
  local to="$1"; local logbase="$2"; shift 2
  timeout -k 30 "$to" python "$@" >"$logbase.log" 2>"$logbase.err"
  return $?
}

# ---- preflight: validate the whole pipeline quickly (<=15 min) ---------------
echo "[PREFLIGHT] 2-env / 50-step validation..."
run_py 900 "$LOGDIR/preflight" scripts/run_mctd.py --env "$ENV" --flat-mcss \
  --mcss-backbone df --k 50 --replan-every 1 --reach-wp 1.0 --seed 0 \
  --n-envs 2 --max-steps 50 --out results/exec_iso_preflight.json
if [ ! -f results/exec_iso_preflight.json ]; then
  echo "[PREFLIGHT-FAIL] no valid output (see $LOGDIR/preflight.err). Aborting so the night is not wasted."
  exit 1
fi
echo "[PREFLIGHT-OK] pipeline works. Starting the real runs."

# ---- real runs, ordered so each seed's PAIR finishes before the next seed ----
DEADLINE=$(( $(date +%s) + BUDGET_SEC ))
rws=(1.0 0.0 1.0 0.0 1.0 0.0)
seeds=(0 0 1 1 2 2)
names=(advance_s0 aimnext_s0 advance_s1 aimnext_s1 advance_s2 aimnext_s2)

for i in 0 1 2 3 4 5; do
  now=$(date +%s)
  remaining=$(( DEADLINE - now ))
  if [ "$remaining" -le 180 ]; then
    echo "[BUDGET] Out of time - skipping ${names[$i]} and the rest."
    break
  fi
  to=$(( PER_RUN_CAP < remaining ? PER_RUN_CAP : remaining ))
  out="results/exec_iso_${names[$i]}.json"
  echo "[$(date +%H:%M:%S)] START ${names[$i]}  (timeout ${to}s, budget ends $(date -d @${DEADLINE} +%H:%M 2>/dev/null || echo soon))"
  run_py "$to" "$LOGDIR/${names[$i]}" scripts/run_mctd.py --env "$ENV" --flat-mcss \
    --mcss-backbone df --k 50 --replan-every 1 --reach-wp "${rws[$i]}" \
    --seed "${seeds[$i]}" --out "$out"
  rc=$?
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "[TIMEOUT] ${names[$i]} exceeded ${to}s - killed, moving on."
  else
    echo "[DONE] ${names[$i]}  exit=$rc  outputWritten=$([ -f "$out" ] && echo true || echo false)"
  fi
done

echo "[FINISHED] $(date +%H:%M:%S). Results:"
ls -1 results/exec_iso_*.json 2>/dev/null | sed 's/^/  /'
