# 03 — Infrastructure and operations: uv + SSH + tmux + local MLflow

## 1. Host assumptions and safety boundaries

Target one remote Linux host with an **NVIDIA RTX A4000 (16 GB VRAM)**, SSH access, Git, `uv`, Python 3.11, and `tmux`. No Docker, Kubernetes, cloud experiment platform, or privileged admin operation is required. The project may be developed locally then executed remotely over SSH.

Required nonsecret connection variables (fill out in your own shell/agent environment):

```bash
export REMOTE_SSH='gpu-host-alias'                   # alias in ~/.ssh/config or user@host
export REMOTE_PROJECT_DIR="$HOME/bayesian-circuit-poc"  # actual path on remote host
```

Do not put credentials, private SSH keys, access tokens or literal host secrets in committed config files. If SSH host/access is not provided, perform local CPU development and provide deployment commands; mark unexecuted remote runs BLOCKED, not PASS.

**Security:** the user-provided example `--host 0.0.0.0 --allowed-hosts '*' --cors-allowed-origins '*'` with UFW port 5000 open would expose a tracking server without adequate access control. For a single-user research host, **bind MLflow to 127.0.0.1 and access through an SSH tunnel**. Do not run `sudo ufw enable`, change UFW rules or expose port 5000 without explicit approval. SSH tunneling requires only existing SSH access (usually TCP/22).

## 2. Remote preflight

```bash
ssh "$REMOTE_SSH" 'uname -a; command -v uv; command -v tmux; nvidia-smi; df -h "$HOME"; free -h'
```

On the remote host:

```bash
nvidia-smi
uv --version
python3 --version
git --version
tmux -V
```

Record exact GPU name, total/free VRAM, driver version, CUDA compatibility as reported by `nvidia-smi`, torch build and `torch.cuda.is_available()`. Do not assume a CUDA toolkit needs to be installed: PyTorch CUDA wheels contain user-space components but still require an adequately compatible NVIDIA driver.

If remote Python is too new for the selected dependency stack, install a suitable interpreter via uv (`uv python install 3.11`) without modifying the system Python. Keep the active Python version in `pyproject.toml` and `uv.lock`.

## 3. Repository and uv dependencies

From the remote project root:

```bash
mkdir -p logs mlruns/artifacts reports/figures reports/tables
uv sync                    # initial development only, after the agent has finalized pyproject.toml
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
uv run python -m bayescircuits.cli doctor
```

Once locked, use `uv sync --frozen` for replication and commit `uv.lock`. Do not mix the project's environment with global pip installs. The coding agent should pin compatible versions of PyTorch, TransformerLens, MLflow, shapiq/LightGBM, NumPy/SciPy, scikit-learn, test tools and any benchmark wrappers.

For PyTorch CUDA, **determine driver compatibility first**. Follow the official uv PyTorch guide for an explicitly selected CUDA wheel index in `pyproject.toml`, preferably an `explicit=true` named index with `torch` pinned to it. Do not blindly install the most recent CUDA wheel; record the selected wheel and why it works on the A4000. On macOS/CPU development use a separate documented CPU environment, not a silent change to the remote CUDA lock.

Useful documentation: https://docs.astral.sh/uv/guides/integration/pytorch/ ; https://docs.astral.sh/uv/concepts/indexes/

## 4. Launch MLflow on the remote host

Run these **on the remote host** from the absolute project root:

```bash
export PROJECT="$HOME/bayesian-circuit-poc"   # adjust if needed
cd "$PROJECT"
mkdir -p mlruns/artifacts logs

tmux new-session -d -s bccd-mlflow -c "$PROJECT" \
  "uv run mlflow server \
    --backend-store-uri sqlite:///mlruns/mlflow.db \
    --artifacts-destination $PROJECT/mlruns/artifacts \
    --host 127.0.0.1 \
    --port 5000 2>&1 | tee -a logs/mlflow.log"

# Verify from the remote host:
curl -fsS http://127.0.0.1:5000/health || curl -I http://127.0.0.1:5000/
tmux ls
```

The SQLite path above is relative to the tmux session's working directory; `--artifacts-destination` is passed the absolute project-local artifacts directory. MLflow versions can differ in artifact serving behavior; the agent MUST run an end-to-end `mlflow.log_artifact()` / `mlflow.artifacts.download_artifacts()` round trip and verify that artifact locations are durable and readable. If the installed MLflow version requires a different explicit artifact-serving flag, document the exact compatible command and run ID, not a theoretical command.

For experiment processes executing **on the same remote machine**:

```bash
export MLFLOW_TRACKING_URI='http://127.0.0.1:5000'
```

**MLflow UI on your local computer:** run a local tunnel in a terminal:

```bash
ssh -N -L 127.0.0.1:5000:127.0.0.1:5000 "$REMOTE_SSH"
```

Open http://127.0.0.1:5000 in the local browser. No inbound TCP/5000 firewall rule is necessary. If your local port 5000 is occupied, forward local port 5001 to remote port 5000 and open http://127.0.0.1:5001.

Diagnostics:

```bash
tmux attach -t bccd-mlflow
# Detach: Ctrl-b, then d
tail -f logs/mlflow.log
```

Do not kill another user's tmux sessions or overwrite an existing MLflow instance. Check `tmux has-session -t bccd-mlflow` and the listener before starting another server.

## 5. Long-running gate sessions

Each experimental gate must run in a separate, named tmux session (e.g., `bccd-a`, `bccd-b`, `bccd-c`). The coding agent should implement `scripts/run_gate.sh <gate>` which:

1. `cd`s to the canonical project directory.
2. Checks MLflow is reachable and sets `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`.
3. Runs `uv run python -m bayescircuits.cli gate-a|gate-b|gate-c --config configs/preregistration.yaml` under `set -o pipefail`.
4. Streams stdout/stderr to `logs/gate-{a,b,c}.log` and writes a final machine-readable status file with exit code, UTC completion time and MLflow run IDs.
5. Handles SIGINT gracefully: checkpoint the current method/world/mask state, then exit without fabricating completion.

After the agent creates that script, start a session using:

```bash
cd "$PROJECT"
tmux new-session -d -s bccd-a -c "$PROJECT" \
  'bash scripts/run_gate.sh gate-a'

tmux attach -t bccd-a
# Detach without stopping: Ctrl-b, then d
```

Repeat with `bccd-b` and `gate-b`, then `bccd-c` and `gate-c`. **Run GPU gates serially** on a single A4000; don't launch B and C concurrently. Gate A may use CPU only. Use `tmux capture-pane -p -t bccd-c -S -100` and `tail -f logs/gate-c.log` to inspect progress.

After recovery from SSH disconnects, `tmux ls` reveals sessions. To distinguish a finished process from a failed one, inspect status files and MLflow run statuses; absence of a tmux session is not evidence of successful completion.

## 6. MLflow experiment scheme

Experiment names:

- `bccd/dev-smoke` — tests and development. Never used for GO.
- `bccd/gate-a` — 4 regimes, 20 worlds each, confirmatory.
- `bccd/gate-b` — 10 validated planted transformer instances.
- `bccd/gate-c` — GPT-2 Small IOI, 5 discovery seeds.
- `bccd/aggregate` — formal results, tables and verdict.

Tags on *every run*:

```text
protocol_version, protocol_sha256, config_sha256, git_sha, uv_lock_sha256,
model_id, model_revision, task, gate, world_id, seed, method,
budget_masks, prompt_batch_id, intervention_semantics, candidate_set_sha256,
split_sha256, confirmatory (true/false), device, torch_version
```

Metrics: `train_mae`, `test_mae`, `test_rmse`, `coverage_90`, `interval_width_90`, `circuit_R_k`, `S`, `top_pair_accuracy`, `alt_circuit_recall`, `discovery_masks`, `discovery_forward_evals`, `screening_forward_evals`, `gpu_seconds`, `wall_seconds`, `peak_vram_mb`, `invalid_high_conf_claims`, etc. Only log metrics that were actually measured; don't log null as zero.

Artifacts: `config_snapshot.yaml`, `split_hashes.json`, `candidate_heads.json` (Gate C), `intervention_queries.parquet`, `posterior_samples.npz` or lightweight summary, `candidate_circuits.json`, `blind_predictions.parquet`, figures, `status.json`, and a run-specific reproducibility manifest.

Do not store full model checkpoints or enormous activation tensors in MLflow artifacts unless unavoidable. Store source/model revision IDs and checksums instead. Keep transient caches outside Git.

## 7. A4000 resource guidelines

- GPT-2 Small should fit comfortably; keep Gate C batch/sequence conservative and dynamically **microbatch within fixed prompt batches** (e.g., 4–8 prompts per forward). This does not change the statistical design.
- Use no-grad/inference mode for oracle interventions. Attention-attribution screening may need gradients; isolate and free its graph afterward.
- Test `hook_z` shape and mask wiring before the first confirmatory run. Do not enable unrestricted `run_with_cache` over all layers and prompts; use selective caching or stateless hooks.
- Monitor `nvidia-smi` and record actual peak memory. Avoid concurrent GPU jobs. Drop all unused tensors and call `torch.cuda.empty_cache()` only when it demonstrably helps fragmentation; do not use it as a substitute for correct memory lifecycle.
- Use fp32 initially for small models/accurate interventions. Lower precision only after a numerical equivalence check and explicit documentation; freeze numerical settings for comparisons.
- If OOM, lower **execution microbatch** only. Never reduce the number of masks/prompts/seeds and still mark a gate PASS.

## 8. Fallbacks and non-success states

- **No SSH host or credentials:** local CPU implementation/tests only, `BLOCKED_REMOTE` for remote gates; provide exact commands to finish.
- **No NVIDIA GPU or incompatible driver:** Gate A runnable CPU; mark B/C BLOCKED, do not claim they ran.
- **MIB dependencies incompatible:** use only documented, license-compliant portions; a scientifically different substitute makes Gate C INCONCLUSIVE unless preregistered equivalence is established.
- **MLflow unavailable:** fix tracking before confirmatory runs. Offline run logs can be retained temporarily, but do not declare completion without importing and verifying the corresponding MLflow artifacts.
- **External code unavailable:** implement transparent baseline equivalent and report it; never claim the official external method was run if it was not.
- **No internet/model weights:** keep model revision and blocked status; don't fabricate results.

## 9. Recommended project exclusions

`mlruns/`, `logs/`, `.venv/`, model caches, Hugging Face credentials, giant result matrices, third-party Git clones, and generated artifacts should be excluded from Git by `.gitignore`. Logs/reports remain present on remote storage and are cross-referenced in MLflow. Preserve checksums and source revisions in tracked manifests.
