## 3. Gate A — Controlled causal worlds (CPU, 80 confirmatory worlds, 0 failed runs)

All six methods were run on 20 confirmatory worlds per regime (A0–A3; seeds 1001–4020) with the shared 32-mask
warm start, checkpoints 32/64/96/128 and 256 blind masks per world. Ground truth was held only by the world
object; discovery code received a budget-accounted oracle. The A3 misfit threshold (t = 0.0162) was frozen from
16 development worlds by the pre-declared rule D-A-3 before any confirmatory world ran.

| Criterion | Observed (bayes_decision, B=128) | Status |
|---|---|---|
| A-1 | top-2 nonadditivity pairs correct in 20/20 A1 worlds (Wilson 95% [0.84, 1.00]) | PASS |
| A-2 | ≥3/4 exact minimal size-3 circuits in 20/20 A1 worlds; 0 invalid P≥0.95 claims at any checkpoint | PASS |
| A-3 | A0 MAE 0.00093 vs first-order Lasso 0.00018 (**raw ratio 5.2×**); passes only through the pre-declared zero-error fallback (normalised MAE 0.0014 ≤ 0.01) | PASS (fallback) |
| A-4 | pooled A0–A2 coverage **1.000** (A0 1.000, A1 1.000, A2 1.000); mean 90% width ≈ 0.08–0.09 vs MAE ≈ 0.0005–0.0009 | **FAIL** |
| A-5 | A3 coverage 0.857 (in [0.85, 0.95]); detector flagged 20/20 A3 worlds with 0/60 false positives | PASS |

**Gate A verdict: FAIL (A-4).** The Bayesian posterior predictive intervals are grossly over-conservative on the
deterministic, exactly-representable regimes: intervals are ~100× wider than the actual errors, so every blind
target is covered. The cause is structural, not a sampler bug: with the conjugate slab
β ~ N(0, σ²τ²), the marginal posterior of σ² includes the ridge term ||β||²/τ², which places a noise floor of
roughly ||β||/(τ√n) on noiseless data. Tuning τ on development worlds to hit 90% coverage would have meant
calibrating to an artefact, so τ was left at its pre-set value (5.0).

**What Gate A does and does not show.** Pairwise Lasso is equally successful on the planted quadratic
mechanisms: it recovers ≥3/4 circuits in 12/20 A1 worlds at B=32 (Bayes: 13/20) and 20/20 from B=64 on, with
top-2 pairs correct in 20/20 worlds, at ~0.2 s CPU per world versus ~6 s for `bayes_decision`. On A1 the
decision-focused, uncertainty and static Bayesian variants are indistinguishable (all 20/20 from B=64). Gate A
therefore provides no evidence of a Bayesian or active-learning advantage; it does confirm that the
implementation recovers planted interactions without heredity and that the CV-based misfit detector separates
A3 from A0–A2 cleanly.

## 4. Gate B — Planted redundant XOR transformers (GPU)

**Construction.** Direct-weight TransformerLens models (3 layers × 8 heads, d_model = 64, d_head = 8, d_mlp = 32,
ReLU, no LayerNorm). Two redundant A heads and two redundant B heads, placed at random among the 16 heads of
layers 0–1, copy the bit signs from positions 0/1 to the query position; MLP1 saturates each sum to {−1, 0, +1};
MLP2 computes the XOR margin; 20 nuisance heads act in a separate residual subspace with a small calibrated
read-out (1–4% of the margin scale). Ten candidates (seeds 20000–20009) were built and **all 10 passed every
validity check on 256 independent inputs** (10 attempts, 0 rejections): clean accuracy 1.00; every singleton
relevant-head ablation 1.00; either full redundant pair ablated 0.50; singleton/pair margin-effect ratio 0.0;
exactly the four planted size-2 circuits faithful (R ≈ 0.016–0.045) and every other circuit of size ≤ 2 at
R ≈ 1.0. Ablating either pair removes all task information, which verifies that there is no residual/MLP bypass.

| Criterion | Observed | Status |
|---|---|---|
| B-1 | `bayes_decision` recovered ≥3/4 planted circuits with a blind-faithful proposal in **1/10** models by B=128 | **FAIL** |
| B-2 | median masks-to-recovery 160 (censored) vs best non-Bayesian 160 → ratio 1.00 | **FAIL** |
| B-3 | median 160 vs `bayes_static` 160 → ratio 1.00 | **FAIL** |
| B-4 | **78 of 84** high-confidence (P ≥ 0.95) claims were invalid on blind inputs (78 at B=32) | **FAIL** |

Successes by B=128 (of 10): bayes_static 2, bayes_uncertainty 2, bayes_decision 1, gbdt 1, lasso_pair 0
(1 at B=64, then lost), lasso_1st 0. Most event times are censored for every method, so no speed-up can be
claimed either way.

**Gate B verdict: FAIL (B-1–B-4).**

**Failure analysis.**
1. *The benchmark's response is not pairwise.* Removing either redundant pathway destroys XOR, and removing
   both does not destroy it "more". The intervention response is therefore close to κ·OR(AND(A₁,A₂), AND(B₁,B₂))
   = κ(m_{A1}m_{A2} + m_{B1}m_{B2} − m_{A1}m_{A2}m_{B1}m_{B2}), which contains a 4th-order term that no main+pairwise
   surrogate can represent. This is intrinsic to redundant-pathway XOR measured on a task margin, not a feature
   of this particular construction.
2. *Finding the interacting pairs is easy; turning them into faithful circuits is not.* Official
   `shapiq.ProxySPEX` (FBII, its own 128 uniformly-sampled coalitions) placed the two planted pairs as its top-2
   order-2 interactions in **10/10** models, as did pairwise Lasso, GBDT and `bayes_uncertainty` (10/10 at
   B=128). `bayes_decision` did so in only 3/10, since its circuit-test masks carry little information about
   global interaction structure. Under the misspecified surrogate, predicted R(C) for the true circuits is
   inaccurate, so proposals are wrong or absent.
3. *Overconfidence.* The invalid high-confidence claims are concentrated in two models (20004: 17, 20006: 61) at
   the shared 32-mask warm start (identical for the three Bayesian variants). Across all Bayesian snapshots,
   204 of the 255 invalid claims occurred where the two MCMC chains *agreed* on supports, so this is posterior
   overconfidence under misspecification at small n, not only poor mixing. (Post-hoc observation, not used to
   change any rule.)
