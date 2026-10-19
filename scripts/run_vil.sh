#!/usr/bin/env bash
# Usage: bash scripts/run_vil.sh <log-name> <vil-cli-command> [cli args...]
#   e.g. bash scripts/run_vil.sh vil-confirmatory-0 vil-run --set confirmatory --shard 0 --n-shards 4
# Runs one BCCD-VIL-1.0 command under pipefail, streams output to logs/<log-name>.log and writes
# logs/<log-name>.exit.json (exit code, UTC times). Seeds are checkpointed in runs/verification_v1/<set>/seed-*/status.json.
set -uo pipefail
NAME="${1:?log name required}"; shift
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT"
mkdir -p logs
export PATH="$HOME/.local/bin:$PATH"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5001}"
export PYTHONUNBUFFERED=1
if ! curl -fsS "$MLFLOW_TRACKING_URI/health" >/dev/null 2>&1; then
  echo "MLflow not reachable at $MLFLOW_TRACKING_URI (start with scripts/start_mlflow.sh)" | tee -a "logs/${NAME}.log"
  exit 2
fi
START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
uv run --frozen python -m bayescircuits.cli "$@" 2>&1 | tee -a "logs/${NAME}.log"
RC=${PIPESTATUS[0]}
END="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 - "$NAME" "$RC" "$START" "$END" "$*" <<'PY'
import json, sys
name, rc, start, end, args = sys.argv[1:6]
json.dump({"name": name, "args": args, "exit_code": int(rc), "started_utc": start, "finished_utc": end, "success": int(rc) == 0},
          open(f"logs/{name}.exit.json", "w"), indent=2)
PY
echo "[run_vil] $NAME exit=$RC"
exit "$RC"
