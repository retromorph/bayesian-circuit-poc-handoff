# Bayesian Causal Circuit Discovery — Go/No-Go proof of concept

**Status:** research design and agent handoff only. **No experiments have been run** by this starter package. This repository is intended for a coding agent to implement and execute the pilot on a single NVIDIA RTX A4000 (16 GB).

## Scientific objective

Test whether a **sparse Bayesian surrogate of head-ablation effects**, with **decision-focused active intervention selection**, discovers faithful, interacting and redundant computational circuits with fewer expensive model evaluations than strong non-Bayesian baselines.

It is not enough for the Bayesian model to recover planted interactions. The pass decision requires gains over pairwise sparse regression and interaction-aware models on a frozen pretrained GPT-2 Small experiment, plus calibrated predictions for unseen interventions.

## Reading order

1. [`AGENTS.md`](AGENTS.md) — inviolable rules for the coding agent.
2. [`docs/01_PROTOCOL.md`](docs/01_PROTOCOL.md) — experimental gates, definitions, thresholds, and evaluation.
3. [`docs/02_METHODS_AND_BASELINES.md`](docs/02_METHODS_AND_BASELINES.md) — models, acquisition functions, and baseline fairness.
4. [`docs/03_INFRA_AND_OPERATIONS.md`](docs/03_INFRA_AND_OPERATIONS.md) — A4000, uv, SSH, tmux, local MLflow, logs.
5. [`docs/04_REPRODUCIBILITY_AND_QA.md`](docs/04_REPRODUCIBILITY_AND_QA.md) — tests, leakage prevention, inference checks, and provenance.
6. [`docs/05_REPORT_TEMPLATE.md`](docs/05_REPORT_TEMPLATE.md) — report schema and verdict expectations.
7. [`docs/06_REFERENCES.md`](docs/06_REFERENCES.md) — relevant published work, repos, and compatibility cautions.
8. [`CODING_AGENT_PROMPT.md`](CODING_AGENT_PROMPT.md) — paste into the coding agent to launch the end-to-end task.

`configs/preregistration.yaml` records the machine-readable experimental defaults. `docs/07_DECISIONS_AND_DEVIATIONS.md` documents initial clarifications and provides an append-only deviation log.

## Three gates

| Gate | Setting | Primary pass requirement |
|---|---|---|
| A | 32-component synthetic intervention oracle | Recover redundant interactions and alternative circuits; predictive calibration and misspecification detection. |
| B | 3-layer, 8-head/layer planted XOR transformer | Recover redundant head pathways in at least 8/10 valid instances, with significant mask-efficiency gains. |
| C | GPT-2 Small, IOI, 48 fixed candidate attention heads | At 128 intervention masks: ≥15% better faithfulness–size score than the strongest non-Bayesian baseline, and match its quality at 256 masks; calibrated uncertainty and independent evidence. |

**Overall GO iff Gates A, B and C all PASS.** `BLOCKED` and `INCONCLUSIVE` are not GO. The exact pass conditions are in the protocol and YAML.

## Suggested implementation layout (to be built by the agent)

```text
src/bayescircuits/
  cli.py
  config.py
  oracle/{base,synthetic,transformer}.py
  data/{splits,ioi,planted}.py
  models/{bayesian,lasso,gbdt,proxyspex}.py
  acquisition/{random,uncertainty,decision}.py
  circuits/{search,validation}.py
  evaluate/{metrics,calibration,bootstrap,decisions}.py
  tracking/{mlflow,manifest}.py
  reporting/{tables,figures,verdict}.py
tests/
configs/preregistration.yaml
reports/
logs/
mlruns/
third_party/
```

The agent may alter internal modules if it preserves the public CLI, data/protocol definitions, and full audit trail.

## Getting started on the remote server

1. Copy the repository to the server using Git or `rsync`; ensure `uv`, `tmux`, Git, Python 3.11, and a compatible NVIDIA driver are installed.
2. Build the project's `pyproject.toml` and lockfile as part of the agent task; `uv sync --frozen` should reproduce the locked environment afterward.
3. Run `uv run python -m bayescircuits.cli doctor` to check device, dependencies, free disk, GPU memory, and tracking connectivity.
4. Start MLflow in a **dedicated tmux session, bound to loopback**; see [operations runbook](docs/03_INFRA_AND_OPERATIONS.md).
5. Run Gates A, B, C in distinct tmux sessions and generate the report.

Do not copy/paste commands that expose unauthenticated MLflow to the public internet. SSH tunneling avoids opening port 5000.

## Agent handoff

Open the repo in the coding agent and paste the complete contents of `CODING_AGENT_PROMPT.md`. Fill in `REMOTE_SSH` and `REMOTE_PROJECT_DIR` through the agent's environment or SSH configuration if remote execution is expected. The package has no secrets and does not hardcode server addresses.

## Expected final artifacts

- `reports/final_report.md` — numerical evidence, failures, and verdict.
- `reports/decision.json` — formal machine-readable decision for Gates A/B/C.
- `reports/run_manifest.json` — MLflow IDs, source revision, package lock, model revisions, and split hashes.
- `reports/figures/` and `reports/tables/` — reproducible evidence.
- `uv.lock`, production code, unit/integration tests and invocation instructions.

A negative final verdict is an acceptable and valuable outcome.

---

## Execution record and exact reproduction (added by the executing agent)

**Status:** all three gates were implemented and executed on the remote RTX A4000 host. The formal verdict,
per-criterion adjudication and MLflow run IDs are in [`reports/final_report.md`](reports/final_report.md),
[`reports/decision.json`](reports/decision.json) and [`reports/run_manifest.json`](reports/run_manifest.json).
Deviations and operational specifications are logged (append-only) in
[`docs/07_DECISIONS_AND_DEVIATIONS.md`](docs/07_DECISIONS_AND_DEVIATIONS.md).

### Package layout (as built)

```text
src/bayescircuits/
  cli.py               doctor | prepare | gate-a | gate-b | gate-c | diagnostics | report
  config.py provenance.py tracking.py          configuration hashes, provenance, MLflow helpers
  masks.py                                     mask semantics (1 = ablate), designs, feature dictionary
  oracle/{base,synthetic,transformer}.py       budget-accounted oracle, Gate A worlds, TL hook_z head ablation
  planted.py                                   Gate B direct-weight planted XOR transformers + validation
  models/{bayes,surrogates,proxyspex}.py       spike-and-slab sampler, Lasso/pair-Lasso/LightGBM, shapiq ProxySPEX
  acquisition.py circuits.py discovery.py      pools, uncertainty & decision-focused acquisition, shared circuit search, loop
  evaluate.py criteria.py criteria_c.py        held-out metrics, gate adjudication, hierarchical bootstrap
  gate_a.py gate_b.py gate_c.py diagnostics.py gate runners
  report.py report_md.py                       recomputation of all criteria from saved artifacts
```

### Reproduce on a fresh host (single NVIDIA GPU, Linux, uv)

```bash
git clone <this repo> bayesian-circuit-poc && cd bayesian-circuit-poc
uv sync --frozen                                   # Python 3.11, torch 2.8.0+cu126 on Linux (explicit index)
bash scripts/start_mlflow.sh                       # tmux bccd-mlflow, sqlite + local artifacts, 127.0.0.1:5001
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
uv run python -m bayescircuits.cli doctor          # GPU/driver/torch + MLflow artifact round trip
uv run pytest -q && uv run ruff check .
# Gate A development worlds -> frozen misfit threshold (configs/gate_a_misfit_threshold.json is committed)
tmux new-session -d -s bccd-prep-a 'bash scripts/run_gate.sh prepare --stage a'
tmux new-session -d -s bccd-a 'bash scripts/run_gate.sh gate-a'          # CPU, 80 worlds x 6 methods
tmux new-session -d -s bccd-b 'bash scripts/run_gate.sh gate-b'          # GPU, construction + 10 models
tmux new-session -d -s bccd-prep-c 'bash scripts/run_gate.sh prepare --stage c'  # EAP-IG screening, frozen pool/splits/dev reference
tmux new-session -d -s bccd-c 'bash scripts/run_gate.sh gate-c'          # GPU, 5 seeds x 6 methods x B in {64,128,256}
bash scripts/run_gate.sh diagnostics                                      # ProxySPEX diagnostics (not a gate criterion)
bash scripts/run_gate.sh report                                           # recompute criteria -> reports/
# MLflow UI from a laptop: ssh -N -L 127.0.0.1:5001:127.0.0.1:5001 <host>  then open http://127.0.0.1:5001
```

Each `run_gate.sh` call writes `logs/<cmd>.log`, `logs/<cmd>.status.json` (state, parent MLflow run ID) and
`logs/<cmd>.exit.json` (exit code, UTC times). Units (worlds / planted models / seeds) are checkpointed in
`runs/<gate>/.../status.json`; re-running a command skips completed units. Confirmatory commands refuse a dirty
git worktree and shortened world/seed lists (`--max-worlds`, `--max-seeds`, `--dry-run` are exploratory only and
log to `bccd/dev-smoke`).
