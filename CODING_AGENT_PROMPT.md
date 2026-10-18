# End-to-end coding-agent assignment

You are the lead research engineer and experimentalist for **Bayesian Causal Circuit Discovery (BCCD)**. Your job is to **implement, execute, audit, and report** a rigorous proof of concept that determines whether Bayesian sparse causal-intervention modeling with decision-focused active learning merits further top-tier ML research investment.

**IMPORTANT: Do not try to make the Bayesian approach win. A negative result is an excellent result if it is rigorous and reproducible. Do not fabricate runs or results.**

You are starting from a research-handoff repository containing `AGENTS.md`, `README.md`, `configs/preregistration.yaml`, and documentation in `docs/`. Read these files before writing code; treat `AGENTS.md` and the pre-registration as binding. The repository may initially contain *only docs and configuration*, so build the runnable codebase yourself.

## Non-negotiable outcome

Complete the three Go/No-Go gates as defined in `docs/01_PROTOCOL.md`, evaluate them without test leakage, and deliver:

1. Runnable tested Python research package with a `uv`-managed, reproducible environment and committed `uv.lock`.
2. Actual experiments A/B/C on the available CPU/GPU host, logged to local MLflow with immutable settings, seeds, split hashes, and all method baselines.
3. A `reports/final_report.md` with numerical results, plots, gate criteria, failures, MLflow run IDs, source/lock/model revisions, and exact reproduction commands.
4. `reports/decision.json` that formally adjudicates every A-1…A-5, B-1…B-4 and C-1…C-5 criterion, distinguishing PASS/FAIL/INCONCLUSIVE/BLOCKED/NOT_RUN.
5. Enough raw artifacts (`reports/run_manifest.json`, per-seed tables, predictions, circuits, logged queries, model-construction evidence, MLflow artifacts) for an independent reviewer to verify the decision later.

**Overall GO iff every Gate A, B and C mandatory condition passes.** Synthetic recovery alone does not justify continuation.

## Environment and remote execution

- Python 3.11 preferred; **`uv` only** for environments and dependency resolution (`pyproject.toml`, `uv.lock`, `uv sync --frozen`). Do not use global pip or Poetry for the project environment.
- One remote **NVIDIA RTX A4000, 16 GB GPU**; all other resource choices left to you within this constraint. Verify `nvidia-smi`, driver, CUDA/PyTorch wheel compatibility, free disk and RAM. Keep Gate A on CPU; run GPU experiments serially. Avoid enormous activation caches and excess batch size.
- SSH connection details, if provided, are in `REMOTE_SSH` and `REMOTE_PROJECT_DIR` or an existing `~/.ssh/config` alias. Never invent credentials, change host firewall rules or overwrite another user's experiment files. If the remote is inaccessible, fully implement/run feasible local CPU tests, record which remote experiments are BLOCKED, and provide exact commands to complete them. Do not claim unexecuted GPU experiments ran.
- Use **tmux** for the MLflow tracking server and every long-running gate. The target is a local MLflow server *running on the remote machine*, backed by SQLite and local artifacts. Start the MLflow server bound to `127.0.0.1:5000`, not `0.0.0.0`; access the UI with an SSH port tunnel. Do not use `--allowed-hosts '*'`, permissive CORS, `sudo ufw enable`, or `sudo ufw allow 5000/tcp` without explicit authorization. Use `docs/03_INFRA_AND_OPERATIONS.md`.
- Implement `scripts/run_gate.sh` and ideally `scripts/start_mlflow.sh`; capture logs, tmux status, exit codes, run IDs and resumption checkpoints. Validate a complete MLflow artifact round trip before confirmatory runs.
- Pin Git SHAs of all borrowed code and record exact model checkpoint revision. If benchmark dependencies clash, use isolated wrappers/versions and report any scientifically relevant substitution.

## Scientific implementation

### Define the oracle correctly

An intervention mask `m` is binary with **1 = ablate a component**. The oracle must run the actual frozen model under the selected head interventions and return `d(x,m)=s(F(x))-s(F^do(m)(x))`, never a gradient proxy.

For the transformer use targeted head-output hooks (prefer `TransformerLens` `hook_z`), zero selected heads across all token positions, and verify that *only* intended heads change. For a retained circuit `C`, ablate its complement **inside the fixed candidate pool**, preserving every noncandidate head and the rest of the model. For Gate C use the IOI correct-minus-incorrect logit margin. Implement unit tests before large runs.

### Implement six required method arms

1. First-order Lasso on binary mask indicators.
2. Pairwise interaction Lasso with real pairwise mask features and a matched interaction-candidate dictionary.
3. LightGBM/GBDT mask-response surrogate.
4. Bayesian sparse main + pairwise interaction regression, random/static mask acquisition.
5. The same Bayesian model with max-variance or generic information-gain acquisition.
6. The same Bayesian model with **decision-focused** active selection of masks that are predicted to reduce posterior loss for minimal faithful retained-circuit selection.

Use a spike-and-slab/structured sparse Bayesian model or a justified approximation with **support uncertainty**, not just point-estimate ARD importance scores. Avoid imposing strong heredity on pair interactions, because a redundant pair can have zero isolated ablation effects. Preserve posterior multimodality when several circuits are equivalent. If exact inference is too expensive, use tested approximations and report diagnostic checks against small exact posterior calculations.

Include official `shapiq.ProxySPEX` interaction discovery where compatible, and compare it appropriately. Its attribution outputs are not guaranteed to be direct intervention predictions; do not mislabel them. The matched LightGBM surrogate is the appropriate predictive comparator. Use EAP-IG for the shared Gate C candidate screening according to the protocol. Optionally run Bair et al.'s compressed-sensing head-localization code, but do not confuse knockout necessity with sufficient retained circuits. Sources are linked in `docs/06_REFERENCES.md`.

Fit each method on the data it actually queried. Treat masked model calls, prompt counts, adaptive circuit verification, and screening/CPU overhead fairly. Keep blind audit outcomes inaccessible to model fitting and mask selection.

### Circuit and active experiment logic

Implement a shared circuit search that accepts a surrogate effect predictor and produces size-constrained retained candidate sets. For the Bayesian method, use posterior probability that `R(C) <= 0.10`, show alternative high-posterior circuits, and report when uncertainty is large.

Implement decision-focused acquisition: approximate expected reduction of a circuit-selection Bayes risk per query cost. Use bounded candidate-mask pools, posterior particles and circuit shortlists; log approximation settings and CPU overhead. Include a generic predictive-variance acquisition control. If the full decision-value acquisition is not computationally feasible, record a deviation and include a clearly named proxy heuristic rather than passing it off as the exact method.

## Gate A: controlled 32-component causal oracles

Build four exactly specified synthetic regimes: additive A0, redundant pairs A1 (`d=m0+0.9*m1*m2+0.8*m3*m4`), context-switching A2, and intentionally misspecified cubic A3; use the locked coefficient jitter and random component permutations. There are 20 independent confirmatory worlds per regime, with separate development worlds. Warm start 32 balanced-density masks; checkpoints 32,64,96,128; 16 contexts per mask and 256 blind masks/world.

Verify all five frozen A-1…A-5 criteria: A1 top-pair identification >=16/20; >=3/4 exact minimal alternatives in >=16/20 A1 worlds and no false high-confidence circuits; A0 prediction MAE <=1.10x first-order Lasso (the frozen zero-error fallback applies); pooled 90% predictive interval coverage 85–95% on A0–A2; and A3 coverage or frozen misspecification-detection success >=16/20 **with <=12/60 A0–A2 false-positive flags**. Collapse A0/A1/A3 repeated-context labels to one effective observation per mask; do not gain artificial confidence from duplicated deterministic outputs. Ground truth must be hidden from algorithms.

Do not overinterpret success: pairwise Lasso should be strong on the planted quadratic mechanisms.

## Gate B: actual small transformer with planted XOR redundancy

Construct/validate ten **valid** frozen 3-layer, 8-head/layer, `d_model=64` transformers on XOR(A,B), with two redundant A-carrying heads, two redundant B-carrying heads, constrained downstream XOR decoding and 20 nuisance heads. Ensure no residual/MLP bypass invalidates the intended mechanism. Independently verify clean accuracy >=95%, any relevant singleton ablation accuracy >=90%, ablation of either entire redundant pair accuracy <=60%, and singleton logit-margin effect <=10% of joint-pair effect. There must be four exact size-2 minimal retained circuits, one A-head and one B-head each.

Run full discovery against all 24 candidate heads at the frozen mask checkpoints and balanced XOR input batches. If valid benchmark construction fails, report that failure instead of inventing ground truth. Success: >=8/10 valid models with >=3/4 alternative circuits recovered by B=128; median query-to-recovery <=0.75 times the strongest non-Bayesian competitor and <=0.80 times Bayesian static (using frozen censored-time convention); zero high-confidence false faithfulness claims. Report both successes and censoring.

## Gate C: pretrained GPT-2 Small IOI

Use 64 screening-only prompts to freeze the top 32 heads by a documented, fixed EAP-IG implementation; add 16 random unselected heads using the locked seed. This yields 48 candidates out of 144. Leave the other 96 intact. Validate that ablating all 48 has a nontrivial task effect (threshold `>=0.5` absolute mean logit-margin effect); else mark this gate inconclusive without cherry-picking another candidate pool.

Use disjoint 32 discovery prompts per mask, 5 locked discovery seeds, budgets B={64,128,256}, and independently frozen 128 blind prompts and 128 blind masks. Hold out prompt templates where feasible. Select retained circuit sizes k={8,16,24}, and compute for every method/seed/budget the actual blind normalized discrepancy `R(C)` and primary `S=(R8+R16+R24)/3`.

At B=128, the candidate Bayesian decision-focused method must: (1) achieve `S <= 0.85 * best_nonBayes_S(128)` in pooled score and >=15% improvement in >=4/5 seeds; (2) achieve `S_Bayes(128) <= best_nonBayes_S(256)`; (3) achieve 85–95% empirical coverage of nominal 90% predictive intervals for unseen intervention effects; (4) have strictly positive lower bound of a paired 90% hierarchical-bootstrap interval over a **development-preselected** non-Bayesian baseline; and (5) cost no more than 2x the strongest comparator in end-to-end elapsed compute at B=128 on the same host.

Also report interval width, held-out prediction MAE by mask density, absolute R<=0.10 attainment, every baseline curve, and separate GPU/CPU costs. **Do not** claim MIB leaderboard comparability: this pilot uses in-pool head zero-ablation, not necessarily MIB's published edge-patching procedure.

## Required CLI and repository engineering

Implement these noninteractive commands and make them restart-safe:

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

Create source modules for oracles, mask design, dataset generation, Bayesian inference, baselines, active acquisition, circuit optimization, held-out scoring, bootstrap analysis, experiment runner, provenance and MLflow logging. Keep responsibilities separated and typed. Log one MLflow run per seed/method/checkpoint where appropriate, with parent runs for each gate. Output CSV/Parquet per-world/per-instance/per-seed tables, decisions, posterior samples or summaries, discovered circuits, plots and run manifests. Add resume checkpoints and proper exception handling; a crashed run must not appear as successful.

Develop with unit/integration tests for mask semantics, planted mechanism, posterior inference against tiny exact reference, split integrity, oracle-budget accounting, reproducibility and MLflow artifact round trip. Check `ruff`, `pytest`, and CLI smoke tests before confirmatory runs. Compute pipeline performance on CPU/GPU **without changing statistical budgets**. Hardware microbatch size can be reduced when needed.

## Sequence and progress checkpoints

1. Read docs and give a concise implementation plan; inspect host, tools and remote SSH availability. Record blockers, don't request unnecessary clarifications.
2. Establish uv environment, MLflow in a remote tmux session, smoke-test log/artifact retrieval. Save the git/config/lock hashes.
3. Implement the synthetic oracle, baseline matrix and Bayesian estimator; unit tests and Gate A smoke run. Then execute **full confirmatory Gate A**.
4. Build or load/validate planted XOR transformer and run **full Gate B**. Ensure the 10-model construction validity is genuine.
5. Integrate MIB/IOI and GPT-2 hooks, freeze screening/prompt/mask hashes, verify memory and intervention semantics; execute **full Gate C**, including all strong baselines.
6. Independently recompute all gate criteria and uncertainty summaries from saved experiment artifacts. Write `reports/final_report.md`, `reports/decision.json` and full audit/reproduction artifacts. Make the Go/No-Go result prominent; no discretionary overrides.
7. Provide the user with actual output paths, git SHA, MLflow experiment/run IDs, tmux/log locations, execution status, critical metrics and a candid recommendation. A blocked/inconclusive final report must describe exact missing evidence and shortest compliant next action.

Do not stop at toy code or a design document if actual remote execution is available. Conversely, do not represent a smoke test, unit test, simulated run, or partial exploratory reproduction as completion of a confirmatory gate. You may complete all gate attempts even if an early gate fails, provided it remains scientifically valid and within the stated resources; the overall verdict stays NO_GO when any mandatory gate fails.

**Priority:** experimental integrity > strong baselines > valid uncertainty and cost accounting > complete execution > code elegance > favorable results. A well-supported NO_GO is preferable to an unconvincing GO.
