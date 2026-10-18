# 01 — Frozen Go/No-Go experimental protocol

**Protocol:** BCCD-PILOT-1.0  
**Draft freeze date:** 2026-10-18  
**Scientific status:** Proposed experiment, not performed.  
**Primary decision:** GO only if Gate A, Gate B and Gate C each pass every mandatory condition.

This is the authoritative explanation of the pilot's scientific design. Machine-readable thresholds and default parameters live in `../configs/preregistration.yaml`. Any discrepancy must be resolved **before** confirmatory results are observed, with an entry in `07_DECISIONS_AND_DEVIATIONS.md`. Do not retroactively change this protocol to produce a favorable verdict.

## 1. Research hypothesis

Under a fixed budget of actual frozen-model interventions, a sparse Bayesian model with pair interactions and decision-focused acquisition will identify compact, functionally faithful head-level circuits more efficiently than first-order sparse regression, pairwise sparse regression, an interaction-aware GBDT/ProxySPEX approach, and Bayesian regression with passive mask selection. Its held-out intervention predictions will be calibrated, and it will report ambiguity rather than falsely assert a unique mechanism.

A counterexample on real GPT-2 is a valid reason to stop, regardless of synthetic results.

## 2. Definitions applicable to every gate

- `m_j=1`: **ablate** component j. `m_j=0`: leave it intact.
- `F^do(m)(x)`: run frozen model F with exactly the candidate components indicated by mask m ablated. Do not change any noncandidate component.
- `s(F(x))`: target scalar score (Gate A: synthetic scalar; Gate B: logit/score difference or predeclared XOR class margin; Gate C: correct minus incorrect IOI logit margin).
- `d(x,m)=s(F(x))-s(F^do(m)(x))`. An unmasked run must satisfy `d(x,0)=0` up to floating-point tolerance.
- `retain(C)`: inside the fixed candidate universe only, ablate every component not in C and retain every component in C. Noncandidate heads, MLPs, embedding and residual paths remain unchanged.
- `R(C) = mean_test |s(F(x))-s(F^retain(C)(x))| / mean_test |s(F(x))-s(F^ablate_all_candidates(x))|`. Calculate ratio of two means, not mean of per-example ratios. The denominator must pass the specified validity check before computing scientific metrics. All circuits in a comparison use identical intervention semantics.
- `faithful(C)`: `R(C) <= 0.10` on an independent, specified population/split; `posterior_faithful(C)` denotes `P(R(C)<=0.10 | D) >= 0.95`. These are distinct.
- `discovery budget B`: number of *distinct mask configurations* sent to the intervention oracle, with a fixed discovery prompt batch per mask. Report actual `mask × prompt` forward evaluations too. If an algorithm requests a new prompt set or repeated probes, count the extra evaluations and normalize the comparison; do not grant free information.
- `blind test`: never accessed by discovery, acquisition, hyperparameter selection, checkpoint choice, circuit choice, or stopping policy. Evaluating frozen outputs at blind checkpoints is an audit, not a feedback channel.
- `physical_cost`: actual elapsed wall time, GPU seconds where applicable, CPU time, memory peak and forward evaluations, including shared screening and fitting overhead. Cache reuse may reduce physical cost; also report counterfactual logical evaluations per method.
- `independent world`: freshly instantiated ground-truth synthetic mechanism or fully validated planted transformer, not another mask draw from the same world.

**Comparability caveat:** these are *node-level zero-ablation* results over a fixed candidate set. MIB's default edge-level patching metrics and published leaderboard numbers are not automatically comparable.

## 3. Common data and statistical rules

- Freeze seed lists, algorithm hyperparameters, mask-design distributions, prompt templates, tokenizer restrictions, and split construction before confirmatory evaluations.
- A development split (disjoint from blind test) can tune hyperparameters; its oracle calls are logged and counted as development cost. Prefer to tune once using development worlds, then keep the tuned configuration constant across confirmatory worlds.
- Algorithms at the same B may choose different masks. Their oracle interface, candidate components, score function, and discovery prompt budget must be identical.
- Include the non-Bayesian baselines specified below, and use the **best performing comparable baseline**, not an arbitrary easy reference.
- Report all individual world/seed results, including failures, censoring at maximum budget, and zero-effect regimes.
- Interventions are deterministic conditional on model/input/mask unless explicitly stated; predictive uncertainty must not be artificially improved by adding free output noise.
- Model fitting, support screening, training-data reuse, and circuit selection must not use held-out test information.

## 4. Gate A — Controlled causal-intervention oracle (CPU)

### 4.1 Mechanisms

Let p=32 and randomly permute nonzero component indices in each world, keeping permutation hidden from all discovery procedures.

- **A0 additive negative control:** `d(m)=1.0 m0 + 0.8 m1 - 0.7 m2` with 20% coefficient jitter. All other terms zero. Bayesian interactions must not reduce performance materially.
- **A1 redundant pair recovery:** `d(m)=1.0 m0 + 0.9 m1 m2 + 0.8 m3 m4`, with independent 20% coefficient jitter. True minimal faithful retained sets, in planted coordinates, are `{0,1,3}`, `{0,1,4}`, `{0,2,3}`, `{0,2,4}`. No other components contribute. These remain four minimal sets at tolerance `R<=0.10` under the jitter regime.
- **A2 context switching:** for observed context g in {0,1}, `d(g,m)=m0 + (1-g)*m1*m2 + g*m3*m4`, with 20% coefficient jitter and balanced contexts.
- **A3 misspecification:** `d(m)=1.0*m0+0.9*m1*m2*m3+0.8*m4*m5` with independent 20% coefficient jitter. This includes an important cubic term absent from the primary quadratic surrogate. Test whether the method abstains or recognizes inadequate fit.

Use 20 confirmatory independent worlds **per** regime; create separate tuning worlds with disjoint seeds. Do not construct training labels by reusing a surrogate.

### 4.2 Mask sampling and budget

- Initial warm start: 32 masks, balanced across four intervention densities: 10%, 30%, 60%, 90% (nearest feasible integer counts).
- Evaluate at discovery checkpoints 32, 64, 96, 128 masks. Active policies select subsequent masks; static policies draw predefined seeded masks from the same proposal family.
- For context regimes, 16 discovery contexts per mask (8 per context class for A2); fixed context batch across methods at the same gate and seed. In A0/A1/A3, where the intervention function is independent of prompt context, collapse repeated equal outcomes to **one effective observation per unique mask**; never treat 16 duplicate responses as 16 independent noisy measurements. In A2, use one aggregate outcome for each context group per mask.
- Blind evaluation: 256 additional masks per world, with disjoint mask IDs and independently sampled contexts. Never fit to blind results.
- For identifiability checks, enumerate only the 5 planted relevant indices; do **not** enumerate `2^32` masks.

### 4.3 Mandatory Gate A conditions (all must pass)

**A-1.** By B=128, in >=16/20 A1 worlds, the top two inferred pairwise nonadditivity scores correspond to the two planted pairs. Infer pairwise nonadditivity from predicted interventions, not just raw inclusion probabilities. Evaluate at B=128.

**A-2.** In >=16/20 A1 worlds, recover >=3 of the 4 exact, distinct, cardinality-3 minimal retained circuits; report no invalid circuit as `posterior_faithful` with >=95% posterior probability. The search must not receive ground-truth pair identities.

**A-3.** In A0, held-out prediction MAE of the full Bayesian model is <=1.10 times the tuned first-order Lasso MAE. If baseline error is numerically ~0, use an absolute tolerance of 0.01 after normalizing the synthetic output scale, locked before testing.

**A-4.** On the *pooled independent blind interventions from A0–A2*, empirical coverage of nominal 90% posterior-predictive intervals lies in [0.85, 0.95]. Also report per-world and per-regime coverage; pooled coverage cannot conceal a missing regime.

**A-5.** For A3, either blind 90% interval coverage is in [0.85, 0.95], **or** the frozen model-misfit/abstention test signals a problem in >=16/20 A3 worlds **while giving at most 12 false-positive world flags among the 60 A0–A2 confirmatory worlds**. Set the misfit rule using separate development worlds before confirmatory labels are inspected. This specificity condition prevents a trivial always-abstain strategy.

### 4.4 Gate A outputs

CSV/Parquet: one row per method/world/checkpoint; top inferred interactions; four proposed minimal circuits and posterior weights; MAE/RMSE; empirical 90% coverage and mean width; misspecification flags; query and CPU cost. Include exact truth for synthetic worlds in **evaluation-only** files with restricted path, never supplied to model fitting.

A success on A1 alone is **not** evidence of algorithmic superiority: pairwise sparse regression has been explicitly designed to solve such problems.

## 5. Gate B — Small planted redundant transformer

### 5.1 Architecture and task

Build ten **valid** independent model instances, using a 3-layer causal transformer with 8 attention heads/layer, `d_model=64`, short synthetic input sequences, and a binary XOR target `Y=A XOR B` evaluated at the last position.

Create two redundant attention-head pathways carrying A, two carrying B, and a downstream fixed or constrained MLP decoding XOR. The remaining attention heads are distractors. Use training constraints, fixed attention routes, or direct weight construction to prevent an untracked residual/MLP shortcut from bypassing the four heads. Vary seeds, distractors and benign nuisance computation across valid instances.

A planned 24-head candidate universe is all attention heads; MLPs are part of the unchanged scaffold. Use head-output intervention (e.g., TransformerLens `hook_z`) applied to every token position, unless direct construction requires a documented alternative that preserves uniform semantics.

### 5.2 Ground-truth validation *before* discovery

For each candidate instance on independent examples:

- Clean accuracy >=95%.
- Ablating any one of the four planted relevant heads leaves accuracy >=90%.
- Ablating both heads of either redundant pair drops accuracy <=60%.
- Single-head effect on the designated class-margin measure is <=10% of the corresponding paired effect.
- Confirm four exact minimal two-head faithful circuits: one head from A-pair and one from B-pair, for the frozen `R<=0.10` tolerance. If a model lacks this property, regenerate *before* discovery. Record failures and number of construction attempts.

The validation phase is benchmark construction; it may use known planted head identities but must be isolated from discovery code and its inputs. If 10 valid instances cannot be obtained within the allocated resource limit, mark Gate B **BLOCKED_BY_BENCHMARK_CONSTRUCTION**, not PASS.

### 5.3 Discovery

- 10 valid model instances; p=24.
- Same 32-mask warm start and 32/64/96/128 checkpoints as Gate A; 16–32 (freeze at 16 by default) balanced XOR inputs per mask; blind test is independent.
- Compare first-order Lasso, pairwise Lasso, LightGBM surrogate, static Bayesian, Bayesian variance-based active, Bayesian decision-focused active; include an official ProxySPEX diagnostic if compatible, without incorrectly equating interaction scoring to response prediction.
- Search and freeze proposed **exact size-2** retained circuits at each checkpoint before audit. Run blind evaluation only after snapshot is sealed.

### 5.4 Mandatory Gate B conditions

**B-1.** >=8/10 valid instances contain a discovered faithful retained circuit by B=128, and the algorithm recovers >=3/4 of the exact planted size-2 alternative circuits in each of those successful instances.

**B-2.** Median discovery mask budget to meet B-1 per-instance recovery is >=25% lower than the *best* non-Bayesian comparator (i.e., Bayesian budget <=0.75 comparator budget). If no recovery by B=128, set the event time to a predeclared censored value 160 for ranking and label censored values; do not infer exact speedups beyond the horizon.

**B-3.** The same median is >=20% lower than the Bayesian static/random-mask method. The decision-focused policy must add value beyond Bayesian priors.

**B-4.** Zero invalid high-confidence circuit claims: every reported `posterior_faithful(C)` with posterior >=0.95 must satisfy R(C)<=0.10 in exact blinded evaluation.

If baseline failure is widespread, describe results as censored/suggestive and include uncensored success fractions; do not hide baseline weaknesses or claim precise speedups from censored values.

## 6. Gate C — Real pretrained transformer: GPT-2 Small IOI

### 6.1 Fixed task and candidate pool

- Model: pretrained GPT-2 Small (12 layers x 12 heads = 144 heads); record model checkpoint revision and license.
- Task: indirect-object identification (IOI), use standardized MIB prompts/splits when available. Only use prompt instances with unambiguous single-token candidate names under the chosen tokenizer; log exclusion policy and the counts.
- Score: correct-name minus incorrect-name logit at the target position.
- Screening split: 64 prompts used **only** for candidate screening. Shared candidate pool of **top 32 heads by one fixed EAP-IG configuration + 16 uniformly sampled heads from remaining 112**, frozen with a seed and stored in `candidate_heads.json`. Noncandidate heads remain untouched.
- Candidate pool validity: mean absolute logit-margin difference between clean model and ablate-all-48 must be >=0.5 (predeclared sanity threshold). If this fails, mark Gate C INCONCLUSIVE; do not select a new pool using blind data.
- Discovery split: fixed batch of 32 prompts per mask. Mask ID and data examples are distinct from screening and blind test.
- Blind test: 128 held-out prompts from disjoint template families where feasible, plus 128 held-out masks across prespecified mask densities. Precompute and seal split hashes before fitting anything to Gate C discovery results.
- Intervention: set selected heads' post-attention head outputs to zero at all token positions, preserving noncandidate heads. Do not silently substitute clean-to-corrupted patching for zero ablation.
- Discovery seeds: 5 fixed seeds; checkpoints B in {64,128,256}, with the first 32 warm-start masks.
- Candidate retained sizes: k in {8,16,24}. Apply the same circuit search and fixed noncandidate scaffold to all methods.

This is an **in-pool head localization benchmark**, not a full causal circuit recovery claim and not numerically comparable to MIB's default edge-patching leaderboard.

### 6.2 Primary metric

For every seed, method M, budget B and k, select circuit `C_M,B,k` without using blind labels. Compute independent R(C) on the blind prompts. Define

`S_M(B) = (R(C_M,B,8) + R(C_M,B,16) + R(C_M,B,24)) / 3`.

Smaller is better. Precompute clean and all-pool-ablated denominator from the exact shared test prompts, without giving algorithms the corresponding test responses.

Let `S_best_nonBayes(B)` be the **minimum** primary score among all preregistered comparable non-Bayesian baselines at B. Report the name of the winner, and additionally use a single baseline preselected on development data for paired uncertainty analysis to avoid post-test selection bias.

### 6.3 Mandatory Gate C conditions (all)

**C-1 (quality).** At B=128, pooled across the five discovery seeds, `S_Bayesian(128) <= 0.85*S_best_nonBayes(128)`; the >=15% improvement holds in >=4/5 discovery seeds. If denominator/baseline score is ~0, the ratio test is not meaningful and Gate C cannot pass by this measure; report INCONCLUSIVE/NO-GO.

**C-2 (budget).** At B=128, pooled `S_Bayesian(128) <= S_best_nonBayes(256)`. This tests the concrete twofold **mask-budget** comparison; it does not by itself prove twofold wall-clock savings.

**C-3 (calibration).** Nominal 90% Bayesian posterior-predictive intervals for mean held-out intervention effects cover between 85% and 95% of blind mask targets. Also report each density stratum, widths and a proper score; an enormous but nominally calibrated interval is not a practical win.

**C-4 (independent evidence).** Paired hierarchical bootstrap of prompt families and discovery seeds gives a 90% interval for improvement over the **development-preselected** non-Bayesian baseline with strictly positive lower bound; report estimator and resampling structure.

**C-5 (compute).** At the same B=128 budget, end-to-end elapsed compute (screening + discovery model calls + CPU fitting/acquisition + validation used by algorithms), on the same single-GPU machine, is <=2x that of the strongest comparable non-Bayesian baseline. Report the GPU-only and CPU-only components separately.

The final Gate C scientific validity condition is that the held-out score and metrics were obtained from actual GPT-2 head interventions, not surrogate predictions or cached synthetic outcomes.

### 6.4 Supplementary analyses (never override failed main conditions)

- Compare mean absolute held-out mask-prediction error, posterior calibration, interval width and subgroup coverage.
- Report all `R(C)` at k=8,16,24; report whether any circuits meet the absolute `R<=0.10` faithfulness criterion, even though Gate C's primary improvement condition uses the continuous score.
- Ablation by intervention density (including large retain-only deletions), prompt template, and score sign; sensitivity to alternative zero/mean ablation semantics is exploratory and not pooled into primary evidence.
- All baseline checkpoint curves and successful/failed masked evaluations.

## 7. Go/No-Go decision table

| Outcome | Formal classification | Scientific action |
|---|---|---|
| A, B, C all satisfy every mandatory condition | `GO` | Invest in full-scale research, directed path validation and broader models/tasks. |
| Any completed gate fails a mandatory condition | `NO_GO` | Stop this formulation for top-tier-paper development; report possible narrower pivots separately. |
| A gate's scientific validity check fails (invalid candidate pool/benchmark) | `INCONCLUSIVE` | Repair and preregister a follow-up; do not call it GO. |
| Hardware, access, package or model unavailable | `BLOCKED` | State precise blockers and what ran. |
| A gate not executed | `NOT_RUN` | Never infer results from earlier gates. |

For completeness, implement and *attempt* all three gates even if early gates fail, unless a scientific validity blocker makes later results uninterpretable or remote resources are unavailable. Do not conceal that overall GO has already become impossible.

## 8. Immutable audit artifacts

At first confirmatory execution, hash and log: `configs/preregistration.yaml`, this protocol, Git SHA/dirty state, uv.lock SHA, random seed lists, screening/discovery/blind split manifests, model weight/revision identifiers, head intervention code version, and benchmark-external dependency SHAs. Record any later change as a new **protocol version** and do not merge across versions in a single confirmatory verdict.
