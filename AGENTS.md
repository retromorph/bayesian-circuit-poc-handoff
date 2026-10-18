# AGENTS.md — Bayesian Causal Circuit Discovery pilot

## Mission

Implement, execute, and independently audit **all three pre-registered Go/No-Go gates** for Bayesian causal circuit discovery in transformers. Produce code, reproducible MLflow runs, a machine-readable verdict, and a candid research report. **Negative results are successful completion of the task**; never optimize for obtaining a GO verdict.

Read, in order: `README.md`, `docs/01_PROTOCOL.md`, `docs/02_METHODS_AND_BASELINES.md`, `docs/03_INFRA_AND_OPERATIONS.md`, `docs/04_REPRODUCIBILITY_AND_QA.md`, and `docs/05_REPORT_TEMPLATE.md`. Treat `configs/preregistration.yaml` as the machine-readable source of truth for gates and budgets; the prose protocol defines semantics if a YAML field is ambiguous.

## Non-negotiable research invariants

1. Never modify primary gates, evaluation metrics, split identities, seed lists, or budget definitions after seeing confirmatory results. Record any necessary correction in `docs/07_DECISIONS_AND_DEVIATIONS.md` and in MLflow; mark affected results exploratory or inconclusive. Do not quietly re-label a failed experiment as a pass.
2. Treat `m[j] = 1` as **ablated**; `m[j] = 0` as preserved, at every level. `retain(C)` means ablate all candidate components *outside* `C`, leaving the noncandidate model scaffold unchanged.
3. Intervention targets are **measured effects of running the real frozen computation under masks**, never gradients, attribution scores, or surrogate predictions disguised as ground truth.
4. Never expose blind test responses to model fitting, hyperparameter selection, candidate-interaction screening, posterior updates, acquisition functions, circuit search, or early stopping. Independently evaluate saved circuit decisions only after discovery snapshots are frozen.
5. Count every unique mask–prompt model evaluation **used for discovery** toward the discovery budget, including candidate-circuit verifications used to adjust a decision. Account for blind auditing separately and uniformly. Report physical and logical model evaluations, screening costs, wall time, CPU time, and peak GPU memory.
6. Baselines must be credible: first-order Lasso, pairwise Lasso, LightGBM/GBDT surrogate, static Bayesian model, generic uncertainty acquisition, and decision-focused Bayesian acquisition; include official ProxySPEX where specified. Use a shared candidate universe and identical outcome/ablation semantics.
7. Do not conflate a pairwise nonadditive intervention effect with a directed causal edge. Do not claim recovery of neural semantic features or full model circuits from head-level mask tests.
8. Report posterior predictive calibration and failures (including overconfidence), not just predictive accuracy. Bayesian credible probabilities are not frequentist coverage guarantees.
9. All methods receive equal access to discovery data under the specified budget. Be explicit where an external algorithm uses its own intervention mask distribution or budget accounting.
10. Do not invent successful experimental runs, repository integrations, citations, wall-time measurements, or model artifacts. Use `BLOCKED`, `INCONCLUSIVE`, or `NOT RUN` with exact causes when appropriate.

## Operational constraints

- Target machine: **single NVIDIA RTX A4000, 16 GB VRAM**. Do not assume multiple GPUs, paid services, containers, or root access. CPU-only Gate A must run before expensive gates.
- Python 3.11 preferred; dependency management through **uv** (`pyproject.toml`, committed `uv.lock`, `uv sync --frozen` after locking). Record NVIDIA driver, CUDA runtime, PyTorch build, Python version, and all external repository SHAs.
- Use **local MLflow tracking hosted on the remote machine** with SQLite and local artifacts. Bind its server to `127.0.0.1:5000`, access the UI through an SSH port forward. Do **not** run an unauthenticated MLflow endpoint on `0.0.0.0`, enable permissive CORS, open port 5000 in UFW, or change firewall rules without explicit authorization.
- Run long-lived MLflow and experiment jobs in separate named **tmux** sessions. Capture stdout/stderr to `logs/`; preserve status files and MLflow run IDs. SSH must use existing credentials/configuration, never invented secrets.
- Use `torch.inference_mode()` for intervention evaluation, short context lengths, low microbatch sizes, and targeted hooks rather than collecting every activation. Check GPU memory with `nvidia-smi`; avoid silent CUDA OOM retries that change the protocol.
- No unsolicited deletion or host-level changes. Never commit or log SSH keys, MLflow credentials, Hugging Face tokens, private paths, or user-sensitive content. Respect model/data licenses.
- Fetch outside repositories into `third_party/` (ignored or tracked as submodules) and pin SHAs. Do not vendor whole projects into our source tree unless essential; prefer wrappers.

## Implementation contract

Develop a package at `src/bayescircuits/` with typed modules for: configuration and provenance; oracle interface; synthetic worlds; frozen-transformer masking; split generation; Bayesian inference; non-Bayesian baselines; experiment design; circuit optimization; evaluation; and MLflow/reporting.

Provide these commands (the agent implements them):

```bash
uv run python -m bayescircuits.cli doctor
uv run python -m bayescircuits.cli prepare --config configs/preregistration.yaml
uv run python -m bayescircuits.cli gate-a --config configs/preregistration.yaml
uv run python -m bayescircuits.cli gate-b --config configs/preregistration.yaml
uv run python -m bayescircuits.cli gate-c --config configs/preregistration.yaml
uv run python -m bayescircuits.cli report --config configs/preregistration.yaml
uv run pytest -q
uv run ruff check .
```

Commands must be idempotent or safely resumable. `--dry-run`, `--max-worlds`, or `--max-seeds` are allowed for development smoke tests but must visibly label the resulting runs **exploratory** and never count them as confirmatory passes. Implement checkpoint/resume at mask-budget boundaries, deterministic random states, and a machine-readable manifest per run.

## Build-and-run order

1. Run a preflight of SSH connectivity, A4000 GPU, driver compatibility, available disk, `uv`, MLflow, and model download rights/access. If remote access variables are absent, perform local CPU/unit work, provide copy/paste remote commands, and mark remote gates blocked; never claim they ran.
2. Implement tests first for masking semantics, pairwise interaction recovery, held-out split integrity, fixed-seed replication, posterior interval construction, and MLflow logging.
3. Run Gate A in full. Continue implementation of Gates B and C even if A fails unless further experiments are scientifically invalid or blocked by infrastructure; the **overall GO remains impossible** when a gate fails.
4. Validate Gate B planted-model construction before any localization attempts. If fewer than 10 valid models can be produced, report a benchmark-construction failure; do not count construction failures as successful model-discovery runs.
5. Freeze Gate C GPT-2 candidate indices using *screening-only* data, and freeze all blind mask/prompt splits before any agent tuning on Gate C validation data.
6. Run matched baselines and Bayesian variants, each across the frozen budget checkpoints and seeds. Preserve all negative results and intermediate metrics in MLflow.
7. Generate report, figures, tables, audit artifacts, and a final machine-readable Go/No-Go verdict. Include exact commands needed for external replication.

## Required deliverables

- `src/`, `tests/`, runnable `pyproject.toml`, committed `uv.lock`, reproducible configuration and split hashes.
- MLflow experiment runs tagged with gate, method, seed, protocol hash, git SHA, split hash, model ID, intervention semantics and data regime; logged predictions and circuits as artifacts.
- `reports/final_report.md` following the report template, `reports/decision.json`, figures/data tables, `reports/limitations.md` (can be included in the report), and `reports/run_manifest.json`.
- A final `README.md` section with run IDs and exact remote reproduction commands.
- Explicit distinction among `PASS`, `FAIL`, `INCONCLUSIVE`, `BLOCKED`, and `NOT_RUN` for each gate.

## Decision authority

Only preregistered Gate A + Gate B + Gate C conditions can yield **GO**. Every other outcome is **NO-GO** or **INCONCLUSIVE/BLOCKED** as appropriate. No discretionary optimistic override. In the final handoff, lead with the actual verdict, major numerical results, uncertainties, deviations, and MLflow run IDs.
