#!/usr/bin/env bash
# Usage: bash scripts/run_gate.sh <doctor|prepare|gate-a|gate-b|gate-c|report> [extra cli args]
# Runs one BCCD command under pipefail, streams output to logs/<cmd>.log and writes
# logs/<cmd>.exit.json with exit code, UTC times and the CLI status file (MLflow run IDs).
set -uo pipefail
CMD="${1:?command required}"; shift || true
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT"
mkdir -p logs
export PATH="$HOME/.local/bin:$PATH"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5001}"
export PYTHONUNBUFFERED=1
if ! curl -fsS "$MLFLOW_TRACKING_URI/health" >/dev/null 2>&1; then
  echo "MLflow not reachable at $MLFLOW_TRACKING_URI (start with scripts/start_mlflow.sh)" | tee -a "logs/${CMD}.log"
  exit 2
fi
START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CHILD=""
on_signal() {
  echo "[run_gate] caught signal; forwarding to $CHILD (state is checkpointed per unit)" | tee -a "logs/${CMD}.log"
  [ -n "$CHILD" ] && kill -INT "$CHILD" 2>/dev/null
  wait "$CHILD" 2>/dev/null
}
trap on_signal INT TERM
( uv run --frozen python -m bayescircuits.cli "$CMD" --config configs/preregistration.yaml "$@" 2>&1 | tee -a "logs/${CMD}.log" ) &
CHILD=$!
wait "$CHILD"
RC=$?
END="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STATUS_JSON="logs/${CMD}.status.json"
python3 - "$CMD" "$RC" "$START" "$END" "$STATUS_JSON" <<'PY'
import json, sys, os
cmd, rc, start, end, sp = sys.argv[1:6]
st = json.load(open(sp)) if os.path.exists(sp) else None
json.dump({"command": cmd, "exit_code": int(rc), "started_utc": start, "finished_utc": end,
           "success": int(rc) == 0, "cli_status": st}, open(f"logs/{cmd}.exit.json", "w"), indent=2)
PY
echo "[run_gate] $CMD exit=$RC"
exit $RC
