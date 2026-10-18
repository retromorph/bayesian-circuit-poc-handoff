# 02 — Algorithms, baseline fidelity and computational plan

## 1. Primary estimator

Given an observed (or prompt-averaged) intervention target `d(x,m)` and ablation mask `m` (`1 = ablated`), fit a sparse quadratic response model:

```math
\widehat d(x,m)=\sum_{j=1}^{p}m_j\beta_j(x)
+\sum_{(i,j)\in E_0}m_im_j\gamma_{ij}(x).
```

For Gate A2, context `g(x)` is observed and the coefficient model must include explicit interactions between `g` and mask features. A0/A1/A3 respond identically to repeated contexts within a mask; aggregate identical outcomes rather than falsely treating duplicates as independent evidence. For Gates B/C, a practical first version models the **mean intervention response over a fixed discovery-prompt batch**, with a prompt-variation noise term and independent blind-test prompt evaluation. Do not claim the surrogate predicts individual prompt-specific mechanisms if it only models batch means.

Use sparse Bayesian variable selection (spike-and-slab preferred; ARD or horseshoe acceptable as a fully logged comparator). For a support S, with mask design matrix `Phi_S`, a Gaussian conditional posterior can be updated by sufficient statistics. Account for both support uncertainty and coefficient uncertainty by posterior particles / Gibbs or a trustworthy approximate alternative.

Critical: **do not require strong heredity** (an interaction may have a nonzero effect even when both singleton intervention effects vanish). Also do not interpret a pair interaction coefficient as a directed edge.

## 2. Posterior target and circuit decision

For a retained component set `C` in the fixed candidate pool, define its complement mask `m(C)_j = 1[j not in C]`. A posterior sample of the response function predicts the discrepancy of a retained circuit. Approximate:

```math
P(R(C)\leq 0.10\mid D)
```

where D is discovery-only data, and report exact test R(C) separately.

Generate at least a short list of alternative circuits, not just one MAP circuit. Use a shared size-constrained optimizer (beam search, stochastic local search, or exact enumeration where feasible) that can operate on **each** surrogate's predicted intervention-response function. For Gate A/B planted cases, preserve exact minimum-cardinality outputs for the known small support, without giving algorithms access to planted indices.

When the probability model is ill-conditioned, multimodal or poorly calibrated, report ambiguity/abstention. Avoid choosing a single support from a collapsed approximate posterior and declaring it identified.

## 3. Active intervention selection

**Warm start:** 32 randomized masks shared where method design permits, drawn across the preregistered densities. After warm start, compare:

- **Static randomized:** spend remaining budget on seeded independent masks, with balanced intervention density.
- **Generic uncertainty:** choose highest posterior predictive variance (or coefficient information gain) per query cost.
- **Decision-focused:** choose the mask with largest approximate expected reduction in Bayes risk for retained-circuit selection.

For a posterior state `theta` and a circuit C, define

```math
L(C;\theta)=\frac{|C|}{p}+\lambda\max(R(C;\theta)-0.10,0),
```

and `V(D)=min_C E[L(C;theta)|D]`. Define the acquisition score

```math
A(a)=\frac{V(D)-\mathbb{E}_{y_a\mid D,a}[V(D\cup\{(a,y_a)\})]}
{cost(a)}.
```

Approximate nested expectations with a fixed, logged number of posterior particles and simulated outcomes. Default computational guardrails: maximum 128–256 candidate masks per round, <=32 posterior samples used for acquisition, <=20 decision-candidate circuits, and acquisition batches of at most 16 masks. These values are **implementation defaults** that can be tuned on development data, not gate thresholds. Report acquisition overhead and whether approximation noise changes selected actions.

If the exact policy is computationally infeasible on the A4000 host, implement a clearly labeled cheaper **decision-disagreement heuristic**: choose masks where posterior particles predict different acceptance/rejection outcomes for competing circuits. Evaluate this as a distinct method, not an unannounced substitution.

Candidate masks must span intervention densities relevant to *both* weak ablation and retain-only circuit testing. Repeated identical masks are not free new evidence.

## 4. Required baseline matrix

| Key | Baseline | Required result |
|---|---|---|
| `lasso_1st` | L1 sparse regression on main mask indicators | Held-out intervention prediction, selected circuits, costs. |
| `lasso_pair` | L1 sparse regression on main and pairwise indicators, same pair dictionary as Bayesian method where feasible | Required strong non-Bayesian interaction baseline. |
| `gbdt` | Tuned LightGBM/gradient-boosted trees trained on observed masks and effects | Held-out predictions and matched circuit-search output. |
| `bayes_static` | Exactly the Bayesian response model with seeded random acquisition after warm start | Isolates effect of Bayesian representation. |
| `bayes_uncertainty` | Same posterior model with max predictive variance / generic EIG acquisition | Isolates effect of decision-focused criterion. |
| `bayes_decision` | Full decision-focused policy | Candidate method and Gate B/C treatment. |
| `proxyspex` | Official `shapiq.ProxySPEX` if compatible | Pairwise interaction attribution; also log its internally learned surrogate predictions only if API exposes them reliably. |

**ProxySPEX output caution:** `shapiq.ProxySPEX` returns interaction-attribution values; it is not guaranteed to expose an interchangeable fitted predictor or a minimal-circuit optimizer. Do not present a ProxySPEX attribution matrix as though it were a directly comparable posterior predictive function. The matched GBDT surrogate handles predictive comparison. If adapting ProxySPEX attributions into a circuit ranking, explicitly define and validate the adapter, count its extra evaluations, and report limitations.

**Bair et al. (ICML 2026) compressed sensing:** implement/call as an optional head-knockout localization diagnostic, if licensing, dependencies, and objective compatibility allow. It finds capability-critical subsets, which is not necessarily the same as finding *sufficient retained* head circuits. Do not quietly force different objectives into one score. Inability to reproduce this optional diagnostic does not waive pairwise Lasso and GBDT baselines.

**EAP-IG:** shared Gate C *screening* on 64 screening-only prompts. Screening time, backward evaluations and choice of integration steps must be logged. It is not a substitute for a fair interaction-aware baseline on discovery effects. If its implementation is incompatible, report a protocol deviation and Gate C INCONCLUSIVE unless equivalence of a substitute is demonstrated prior to observing confirmatory results.

## 5. Matching intervention budgets

- The actual oracle must return `d(x,m)` via model execution. A surrogate may never be used as a hidden substitute for an oracle evaluation.
- Every method's query table contains `(gate, model/world, seed, mask_id, mask_bits, prompt_batch_id, output, time_ms, reused_cache)`.
- Count distinct masks, mask-prompt evaluations, and physical executions separately. If two baselines query the same mask and prompt batch, caching may avoid duplicate physical work but both methods incur the same *logical* budget charge.
- Baseline hyperparameter tuning uses the same access rules as Bayesian tuning and must be frozen before confirmatory blind evaluations.
- If a method uses feedback from testing a proposed circuit during discovery, those tests count toward its budget. The blind evaluator must not feed results back to the method.
- Use the identical circuit-search interface, allowed component universe, circuit sizes, and validation metric for all predictive surrogates. Explain any exception.

## 6. Uncertainty and calibration

For each held-out mask, predict a distribution for the mean causal intervention effect over the **specified prompt population**, not a raw point sample if the model fits batch means. Split epistemic uncertainty (unknown response surface) from variation due to sampling test prompts. If posterior interval computation includes a model-discrepancy term, learn its scale without using blind labels.

Measure empirical coverage of nominal 90% intervals, average interval width, negative log predictive density/CRPS where feasible, and the percentage of invalid `P(R(C)<=0.10)>=0.95` claims. Report by world, density, prompt family, and method. For the GBDT control, uncertainty can be estimated via a bootstrap ensemble (trained only on discovery data), with identical held-out targets.

An interval with coverage 90% but width spanning the entire observed effect range is not a useful result. Report width relative to an independently trained ensemble.

## 7. Computation and fallbacks

Target: one NVIDIA A4000 with 16 GB. Gate A runs CPU, Gate B is small, Gate C uses GPT-2 Small with limited sequence length and `torch.inference_mode()`. Prefer `TransformerLens` hooks on `blocks.{l}.attn.hook_z` or an explicitly verified equivalent. Ablate at **all token positions** and test the hook tensor layout with one-head examples before the full run.

Avoid retaining full activation caches between masks. Pre-tokenize prompts and batch modestly. Use model revision pinning. For deterministic repeats, set RNGs, use evaluation mode, disable sampling/dropout, and report non-deterministic kernel limitations.

Allowed fallback: reduce **runtime implementation microbatch size** while preserving the exact prompts/masks and budget. Not allowed: reduce the number of worlds, seeds, prompts, or checkpoints and still claim a confirmatory PASS.

## 8. Implementation checkpoints

1. Verify `d(x,0)==0`; known single and paired masks produce the expected numerical responses.
2. On Gate A1, first-order Lasso should fail to represent pure interactions; pairwise Lasso should succeed with enough informative masks. If this is not true, inspect intervention mask coding and feature construction before tuning Bayes.
3. Validate Bayesian posterior calculations on p<=6 synthetic worlds by exact enumeration of support configurations (or reference sampling) where tractable.
4. Demonstrate active policy reproducibility: same state/seed -> same next mask batch.
5. Run a fake-oracle smoke test end-to-end with MLflow before any expensive remote experiment.
6. Freeze and run all confirmatory gates; keep calibration and decision reports immutable afterward.
