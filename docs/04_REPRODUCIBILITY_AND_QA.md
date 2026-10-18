# 04 — Reproducibility, testing and statistical audit

## 1. Before confirmatory experiments

Create a signed/hashed run manifest containing:

- Git commit SHA and whether the worktree is dirty; `uv.lock` SHA-256; Python, uv, PyTorch, MLflow, CUDA runtime/driver versions; A4000 memory and host OS.
- Main protocol and `configs/preregistration.yaml` SHA-256.
- Remote benchmark/repository pinned Git SHAs and model checkpoint ID/revision/hash (where feasible).
- Gate A synthetic world seeds; Gate B model-construction and discovery seeds; Gate C screening, discovery, test masks/prompt split seeds.
- Candidate head indices, hook points/positions, intervention baseline, tokenizer configuration, and scores.
- Exact computational cost accounting protocol and thresholds before any confirmatory test outcome is inspected.

Store the manifest as a regular file and as an MLflow artifact. Do not overwrite the original after a run; append deviations to the deviation log.

## 2. Minimum automated test suite

### Mask semantics and intervention oracle

- `m=0` returns `d=0` (within floating-point tolerance) for both synthetic and transformer oracles.
- `m=1 on head j` zeros the intended head's complete `hook_z` across token positions and no other head; adjacent heads and layers are unchanged.
- Joint masks affect exactly the union of selected heads; mask-index order does not matter.
- `retain(C)` is the complement of an ablation mask **only within the fixed candidate pool**; no masking outside the candidate universe.
- For synthetic A1, d({1})=0, d({2})=0, d({1,2})=0.9 (subject to jitter); all 4 exact minimal circuits are faithful.
- Gate B planted XOR construction passes validation without model leakage or a hidden bypass.

### Data leakage and budgets

- Screening/discovery/blind prompt IDs are pairwise disjoint; Gate C blind templates do not overlap discovery templates when feasible.
- All blind intervention masks are held out from discovery; if a random collision occurs, deterministically resample it before freezing the split.
- Acquisition code receives only the training oracle and training-history table; block test-oracle imports at the API level where feasible.
- Every queried mask has a unique ID, prompt-batch ID, outcome, budget debit and timestamps.
- Circuit verification used for policy updates debits the method's discovery budget. Blind evaluator emits score files only after decisions have been frozen.
- Re-running a completed checkpoint is idempotent and cannot create duplicate logical interventions or conflicting MLflow run IDs.

### Bayesian inference and active learning

- For small `p<=6`, compare approximate posterior inclusion probabilities and predicted intervals with exact support enumeration or high-quality reference MCMC; report maximum discrepancies.
- Changing a seed changes randomized worlds/masks but not the semantics of the intervention model.
- Posterior draws can represent both members of a redundant circuit equivalence class (do not deterministically collapse to first lexicographic solution).
- The acquisition function is deterministic given frozen data, posterior state and selection RNG seed.
- Missing context interactions in A2 or missing cubic interactions in A3 cause measurable errors; A3 ablation demonstrates the predeclared misspecification detector rather than a tuned post hoc threshold.

### Tracking and reports

- End-to-end MLflow test: create experiment, log param + metric + artifact, fetch artifact, validate contents, and end run with proper status.
- Full Gate A smoke test with 2 worlds and low budget uses an `exploratory` run tag; confirmatory mode refuses shortened world lists or budgets.
- Every row in a report table has a source run ID and matching raw metric artifact.
- Report builder independently recomputes the GO decision from immutable gate-summary artifacts, and refuses to report GO if required metrics are missing.

## 3. Split design and calibration

Gate A: draw 20 independent confirmatory coefficient worlds for each regime, separate from development worlds. Blind mask IDs are independent and masked results are computed only for final evaluation. Coverage is averaged over the agreed repeated-world distribution; it is not a proof of posterior calibration conditional on a single world. A0/A1/A3 duplicate context responses must be collapsed to one effective observation per mask, while A2 uses one outcome per context group. The A3 flagger must show <=12/60 false positives on A0–A2 worlds if used to satisfy A-5.

Gate B: planted-model-construction tests may use ground truth, but discovery must not. Hold out independent XOR examples for circuit validation and final blind accuracy. The benchmark creation log must state attempted and valid constructions.

Gate C: screening uses 64 prompts; discovery uses a separate fixed 32-prompt batch; final evaluation uses 128 prompts from disjoint template families where feasible. Use 128 blind masks for prediction/calibration, stratified by mask density. Store text/template identifiers and hashes, but avoid logging private data or model licenses violations.

For 90% interval calibration, the primary target is the **mean intervention effect on a defined prompt distribution**. Distinguish uncertainty in model coefficients from variability due to sampling prompts. Report actual effect target, interval definition and inference approximation. A posterior interval can fail coverage without contradicting Bayesian probability theory if its likelihood/model assumptions are violated.

## 4. Statistical analysis

### Gate A

Use exactly 20 independent worlds per regime. Report binary success counts and binomial Wilson intervals for pair/circuit recovery and A3 detection, plus test MAE and empirical 90% coverage. Pooled coverage across A0-A2 is the preregistered gate threshold; provide per-regime and per-world summaries to expose clustering.

### Gate B

Define one event time per instance/method: the first predefined discovery checkpoint at which the method has recovered >=3 of the 4 exact planted minimal circuits and at least one is independently faithful. At B=128 failure, set `event_time=160` and `censored=true` (strictly a ranking convention, not a measured recovery time). Compute the median across 10 valid instances; compare Bayesian decision-focused medians against **every** non-Bayesian baseline, and use the smallest median among them.

Report the full censoring and success table as well as ratios. A test that wins only because all competing baselines never recover is informative but should be described as a lower bound on relative efficiency, not an exact speedup.

### Gate C

For each seed and k in {8,16,24}, freeze circuit decisions at B=64,128,256. On blind prompts, compute R for each circuit; average over k to obtain S for each seed and method. Report the 5 seed-specific S values and the prespecified pooled average (mean of the 5 values).

- **Primary comparator:** the smallest pooled `S_M(128)` among preregistered non-Bayesian baselines, conservatively chosen from blind outcomes for the raw hurdle. Repeat at B=256.
- **Inferential comparator:** a *single* non-Bayesian method selected using development data before blind outcome inspection. Use this for the 90% paired bootstrap interval, avoiding selection-on-test inflation.
- **Bootstrap:** resample discovery seeds and, nested within each seed, sample independent held-out prompt families (then prompts inside families) with replacement; retain paired predictions of Bayesian and reference baseline on the same sampled prompts. Compute the paired improvement statistic per replicate. Clearly state the number of families and acknowledge limited precision when there are only five seeds.
- **Success in >=4/5 seeds:** within each seed, compare against the strongest non-Bayesian seed-specific score at B=128 and require >=15% relative reduction. Report both paired pooled and per-seed comparisons.
- **Calibration:** pooled empirical 90% predictive coverage in [85%,95%] on 128 unseen masks (and prompt variation), with additional mask-density-stratified coverage and widths.
- **Compute:** profile end-to-end wall-clock and GPU time under identical host occupancy assumptions; report screening cost separately and as part of full method. A stronger result on mask budget is not necessarily faster wall-clock if Bayesian acquisition is expensive.

The bootstrap is a pilot confidence diagnostic, not a guarantee of broad generalization across models, datasets or tasks.

## 5. Reporting negatives and edge cases

- If baseline prediction MAE is approximately zero, use the absolute fallback tolerance in the frozen protocol; don't divide by ~0.
- If the GPT-2 48-head pool has negligible all-ablated effect, Gate C is scientifically invalid/inconclusive: don't silently replace it after looking at blind test data.
- If the planted model fails to enforce four actual pathways, Gate B is a benchmark failure, not a recovery success.
- If a method makes no high-confidence claims, report that fact; it must **also** satisfy recovery and efficiency tests, so vacuous abstention cannot yield PASS.
- Treat any missing mandatory experiment as BLOCKED or NOT_RUN, not zero error or perfect coverage.
- If external code is unavailable or incompatible, report precisely which baseline is replaced and whether the replacement is equivalent for the target outcome. If a required scientific comparison is invalidated, Gate C cannot PASS.
- If the report presents a speculative theorem, label it a conjecture; do not invent proof results.

## 6. Required output tables

At minimum, generate:

1. `tables/gate_a_worlds.csv`: world/regime, method, B, pair rank, circuits recovered, errors, 90% coverage, detector flag.
2. `tables/gate_b_instances.csv`: planted instance, validity tests, discovered alternatives, B-to-recovery, censored, incorrect high-confidence claims.
3. `tables/gate_c_faithfulness.csv`: seed/method/B/k, raw numerator/denominator, R, S, circuit size, candidate-set hash.
4. `tables/gate_c_prediction.csv`: seed/method/mask density, held-out MAE, 90% coverage and width.
5. `tables/costs.csv`: method/gate/seed, logical masks, mask-prompt oracle calls, cache hits, GPU seconds, CPU seconds, elapsed time, peak VRAM, screening overhead.
6. `tables/gates.csv`: criterion ID, threshold, observed value, PASS/FAIL/INCONCLUSIVE/BLOCKED, underlying run IDs.

Plots: faithfulness–size frontier, score versus intervention budget, prediction-vs-truth with uncertainty, coverage/width by mask density, causal pair retrieval/confusion in A/B, query cost breakdown, and an all-gates verdict plot/table.

## 7. Final provenance check

Before declaring completion, rebuild the final tables from stored artifacts **without rerunning experiments**. Ensure hashes match, each metric has a corresponding run ID, failed runs remain visible, all gates are adjudicated against the original version, and any deviations are clearly highlighted in the report.
