#!/usr/bin/env bash
# Start the BCCD MLflow tracking server on the remote host, loopback only.
# Port 5000 on the shared host is already used by another project's MLflow server, so the
# BCCD server listens on 127.0.0.1:5001 (deviation D-INF-1). Never bind 0.0.0.0.
set -euo pipefail
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
PORT="${BCCD_MLFLOW_PORT:-5001}"
SESSION=bccd-mlflow
cd "$PROJECT"
mkdir -p mlruns/artifacts logs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists; not starting another server"; exit 0
fi
if ss -ltn | grep -q "127.0.0.1:${PORT} "; then
  echo "port ${PORT} already in use; refusing to start"; exit 1
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
tmux new-session -d -s "$SESSION" -c "$PROJECT" \
  "$UV run --frozen mlflow server \
    --backend-store-uri sqlite:///mlruns/mlflow.db \
    --artifacts-destination $PROJECT/mlruns/artifacts \
    --host 127.0.0.1 --port ${PORT} --workers 2 2>&1 | tee -a logs/mlflow.log"
for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then echo "MLflow healthy on 127.0.0.1:${PORT}"; exit 0; fi
  sleep 2
done
echo "MLflow did not become healthy"; exit 1
