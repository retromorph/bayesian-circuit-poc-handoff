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
