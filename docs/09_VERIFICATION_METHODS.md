# 09 — BCCD-VIL-1.0 methods, budget accounting and computation

Implementation: `src/bayescircuits/vil.py` (methods), `src/bayescircuits/vil_run.py` (splits, sealing, audit,
MLflow) and `src/bayescircuits/vil_report.py` (criteria recomputation). Settings:
`configs/verification_implementation_v1.yaml`. Tests: `tests/test_vil.py`.

## 1. Shared components (reused from BCCD-PILOT-1.0 without modification)

- **Oracle.** `HeadAblationRunner` zeroes `hook_z` at all positions for every ablated pool head and scores the IOI
  logit difference from the final normalised residual. Methods only receive a `DiscoveryOracle`: one budget unit per
  distinct mask on the fixed 32-prompt batch, no duplicates, no reserved (blind) masks, at most 128 masks, every query
  logged.
- **Response target.** Surrogates are fit to the batch mean `y(m) = mean_x |d(x,m)|`, i.e. the numerator u of R on the
  discovery prompts (as in the pilot).
- **Pairwise Lasso** (`LassoSurrogate(order=2)`). Main plus all 1 128 pairwise mask features (1 176 total),
  RMS-scaled, no intercept, `LassoCV` (60 alphas, 5 seeded folds, tol 1e-4, 10 000 iterations).
- **Spike-and-slab** (`BayesSurrogate`). The same main + pairwise dictionary with a conjugate slab (τ = 5), an
  InvGamma(1, 1e-4) noise prior and Beta(1,1) beta-binomial group priors, without heredity. Two collapsed-MH chains
  (informed Barker flips + informed swaps) give 200 particles. Warm starts are used across rounds (600/200 iterations);
  cold fits use 1 500/500.
- **Predicted R.** `R̂(C) = |f(retain C)| / |f(ablate all 48)|`. For Bayes this is the mean over 64 evenly spaced
  particles of the per-particle ratio, i.e. the posterior expected R used in the pilot.
- **Shared circuit search** `search_fixed_k`. Greedy backward elimination from all 48 heads down to 8, recording the
  circuits at 24 and 16 on the way, then best-improvement single swaps per k (≤ 12 iterations). It returns the best
  circuit and up to 5 near-best alternatives.

## 2. Mask designs (per seed; hashes frozen by `vil-prepare`)

- `warm` = 32 balanced-density masks, `rng([seed, 22])`. `static` = the next 224 balanced-density masks,
  `rng([seed, 33])`. Both exclude the 128 reserved blind masks and repeats, and both are byte-identical to the pilot's
  Gate C streams (unit-tested).
- `d104 = warm + static[:72]` (shared by every method).
- `d128 = warm + static[:96]` (M0, M5; identical to the pilot's 128-mask static design).
- `density_matched` = 24 random retain masks, `rng([seed, 44])`: 8 retaining a uniformly random set of 8 heads,
  8 retaining 16 and 8 retaining 24 (ablating 40/32/24). All are distinct from `d104` and from the reserved masks.

## 3. Shared candidate generator (verifying methods)

`generate_candidates(objective, n_random_starts)`, for each k:

1. Run the shared search. Its best circuit (`search_top`) and its alternatives become candidates.
2. Run `n_random_starts` best-improvement swap local searches, each from a uniformly random k-subset (≤ 30
   iterations). Each local optimum and the 6 best circuits in its explored swap neighbourhood become candidates.

Each candidate carries its objective value R̂. A circuit is *available* if its retain mask has not yet been queried by
the method and is not reserved. If too few available candidates exist, `_top_up` adds the best remaining candidates
and then uniformly random circuits, and every such event is logged in the decision record and reported. This is a
safeguard only: zero events occurred for M3/M4 in development, and none occurred for M2 after D-VIL-1.

## 4. Methods

### M0 `lasso_static_128` (primary non-verified baseline)
Query `d128` → fit pairwise Lasso → shared search → for each k, the surrogate-optimal circuit. Identical to the pilot's
`lasso_pair@128` decision procedure. On the bridge seeds it must reproduce the pilot's circuits.

### M1 `lasso_density_matched`
Query `d104 + density_matched` → fit pairwise Lasso on all 128 → shared search → surrogate optimum. M1 gets 24
observations in exactly the regime that circuit verification probes (retaining 8/16/24 heads), but at random circuits.
It tests whether any VIL gain is just "more data in the aggressive-ablation regime".

### M2 `lasso_random_verify` (random-verification control)
Query `d104` → fit Lasso → candidate generator with 16 random starts → per k, a **diverse shortlist** of 16 circuits.
The shortlist is built greedily by ascending R̂ over **every circuit the surrogate scored during the k-specific
searches**: search alternatives, local optima and all explored swap neighbourhoods. A circuit is skipped if its
symmetric difference from an already shortlisted circuit is < 4 (i.e. one-swap neighbours). Choose 8 shortlist
circuits per k uniformly at random (seeded) and verify all 24 in one batch. Final circuit per k = the lowest measured
discovery u. There is no surrogate revision. (Deviation D-VIL-1 is a development-phase fix. Building the shortlist
from only the per-start top-6 candidates left fewer than 16 distinct circuits for k = 8, and random circuits were
padded in.)

### M3 `lasso_vil` (H1 candidate: error-aware revision, non-Bayesian)
Query `d104`, then for each of `rounds` rounds (8 by default, one circuit per k per round):

1. Refit pairwise Lasso on **all** observations so far: design masks plus earlier verification outcomes. The refit is
   the error-aware revision: a verified retain mask whose measured u exceeded its prediction becomes a training point
   exactly where the surrogate was overoptimistic.
2. Run the candidate generator (6 random starts) under the refit surrogate.
3. For each k, verify the available candidate with the lowest R̂ (`per_round` per k).

Final circuit per k = the lowest measured discovery u among its 8 verified circuits. The first-round proposals equal
the surrogate optimum at 104 masks over the candidate set, so M3's measured incumbent can never be worse on discovery
prompts than its first proposal.

### M4 `bayes_vil` (H2 candidate: uncertainty-aware revision)
Query `d104`, then for each round:

1. Update the spike-and-slab posterior on all observations (warm-started chains).
2. Run the candidate generator under the posterior-expected-R objective. Add `n_thompson = 4` **Thompson** searches,
   each the shared search under a single randomly drawn posterior particle, contributing its best circuit and
   alternatives.
3. For each k: if nothing of size k has been verified yet, verify the candidate with the lowest posterior expected R.
   Otherwise verify the candidate with the largest **expected improvement** over the incumbent measured u\*,
   `EI(C) = E_post[max(u* − |f(retain C)|, 0)]` over all 200 particles, with ties broken by lower expected R.

EI is the one-step Bayes-optimal acquisition for the final rule "keep the best measured circuit". Final circuit per
k = the lowest measured discovery u among verified circuits.

### M5 `bayes_static_128` (Bayesian non-verified reference)
Query `d128` → cold spike-and-slab fit → shared search minimising the posterior expected R → per k, the optimum.

## 5. Recorded per decision (sealed before evaluation)

For every proposed, verified and final circuit the record holds the surrogate's predicted R and u at proposal time.
For Bayes it also holds the 5–95% posterior quantiles of R, P(R ≤ 0.10) and the 90% posterior predictive interval of
the batch-mean target. Verified circuits additionally record their round, measured discovery u and (Bayes) whether the
measurement fell inside the interval. Each verifying method also records:

- `surrogate_at_104`: the shared-search circuit of the 104-mask surrogate. This is the counterfactual "no
  verification, 104 masks" choice.
- `refit_128`: a diagnostic only, not selectable. A fresh surrogate is fit on all 128 observations and the shared
  search is run on it. Its compute is logged separately and excluded from end-to-end cost.

## 6. Cost accounting

- **Logical:** 128 masks × 32 prompts = 4 096 forward passes per method and seed (asserted). Verifying methods record
  104 discovery + 24 verification masks.
- **Physical:** within a seed, oracle outputs are cached across methods (the 104 shared design masks are computed
  once). `physical_mask_evals` and `cache_hits` are reported. `logical_oracle_seconds` charges every mask its measured
  per-mask GPU time even when it was served from the cache, so caching cannot create a computational advantage.
- **CPU:** wall and process time for surrogate fitting, circuit search / candidate generation and acquisition (EI
  scoring, shortlist sampling).
- **End-to-end (logical):** logical oracle seconds + fit + search + acquisition wall seconds. Peak VRAM per method.
  The blind audit and the discovery-prompt audit are reported separately and are identical for all methods.
- Execution: seeds are partitioned over 4 processes on the single A4000 (`--shard i --n-shards 4`). Each process has its
  own model copy. Every method of a seed runs in one process, so the sharing of compute between methods is identical
  across seeds. Per-seed results are independent of sharding: all randomness is seeded by (seed, method).

## 7. What is *not* claimed

These are head-level, in-pool zero-ablation results on one model and task. Lower S means the retained heads reproduce
the clean IOI logit difference better under ablation of the other pool heads. It is not a directed causal path, a
feature-level mechanism or a complete GPT-2 circuit. Verification-selected circuits are chosen on discovery-prompt
measurements and evaluated on blind prompts from disjoint template families. The residual winner's curse that comes
from selecting the best of 8 noisy measurements is quantified, not assumed away.
