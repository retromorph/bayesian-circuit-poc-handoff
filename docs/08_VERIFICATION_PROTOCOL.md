# 08 — BCCD-VIL-1.0: Verification-in-the-Loop protocol (preregistered)

**Protocol:** BCCD-VIL-1.0 (follow-up to BCCD-PILOT-1.0; the pilot's protocol, configs and reports are not modified)  
**Drafted:** 2026-10-18 (UTC), before any confirmatory BCCD-VIL-1.0 seed was run  
**Machine-readable source of truth:** `configs/preregistration_verification_v1.yaml` (criteria, seeds, budgets) and
`configs/verification_implementation_v1.yaml` (implementation settings). Methods: `docs/09_VERIFICATION_METHODS.md`.  
**Freeze record:** `reports/verification_v1/preregistration_hashes.json` (SHA-256 of this document, doc 09, both configs,
`uv.lock`, the source revision and all frozen split/design artifacts), committed before the first confirmatory seed.

## 0. Provenance of this specification

The research brief for this follow-up arrived truncated: it specified the motivation, H1/H2, the definitions, the
128 = 104 + 24 budget, and methods M0, M1 and the start of M2. The text stopped mid-sentence in M2 ("Randomly select 24
distinct candidate circuits or …"). The executing agent wrote everything after that point **before any confirmatory
data existed**: M2's remaining details, M3–M5, every acceptance threshold, the seed counts and the decision table.
Read those parts as the agent's preregistered design choices, not as the original brief.

## 1. Motivation and hypotheses

In BCCD-PILOT-1.0 Gate C (GPT-2 Small, IOI, 48 frozen candidate heads), every surrogate predicted much better
faithfulness for its own selected circuits than was measured, even on the discovery prompts. At B = 128,
decision-focused Bayes predicted R̂ ≈ 0.10 and measured R ≈ 0.51; pairwise Lasso reached S = 0.385. This is the signature
of surrogate exploitation under combinatorial search (optimiser's / winner's curse).

- **H1.** Spending part of a fixed real-intervention budget on directly verifying proposed circuits during discovery,
  and revising the search with what verification reveals (error- or uncertainty-aware), gives more faithful circuits
  than purely surrogate-based selection *at the same total number of intervention masks*.
- **H2 (separate).** Bayesian inference adds value beyond a non-Bayesian verification-in-the-loop procedure.

H1 can pass while H2 fails; the recommendation then is verification-based optimisation without Bayesian machinery.
The goal is falsification. Negative results are a valid completion.

## 2. Fixed setting and intervention semantics (unchanged from the pilot)

- Model: frozen GPT-2 Small (`gpt2`, TransformerLens `from_pretrained_no_processing`, fp32). Weights hash logged.
- Task: MIB IOI (`mib-bench/ioi@e5f3468`); score s = correct-name minus incorrect-name logit at the final position.
- Candidate pool: the pilot's frozen 48 heads (`runs/gate_c/frozen/candidate_heads.json`, candidate-set SHA-256
  `25568c18…`). No re-screening.
- Mask semantics: `m_j = 1` ablates head j, `m_j = 0` preserves it. Ablation zeroes `blocks.{l}.attn.hook_z` for that
  head at **all token positions**. Heads outside the pool, MLPs, embeddings and residual paths are untouched.
- `d(x,m) = s(F(x)) − s(F^do(m)(x))`, measured by running the real model.
- `u(C) = E_x |d(x, m_retain(C))|`; `R(C) = u(C)/u(∅)`, with u(∅) the discrepancy when all 48 pool heads are ablated on
  the same prompts (ratio of means).
- Primary score for method M: `S_M = (R(C_M,8) + R(C_M,16) + R(C_M,24)) / 3`. Lower is better.
- `R(C) ≤ 0.10` is a **secondary** outcome. No circuit is called faithful unless its blind R ≤ 0.10.

This is an in-pool, node-level zero-ablation benchmark. It is not MIB's edge-patching metric and does not recover a
full model circuit.

## 3. Budget parity (binding)

- Every method receives **exactly 128 logical intervention masks** for discovery and circuit selection. One mask is
  one evaluation on the seed's fixed 32 discovery prompts (32 forward passes).
- A **verification** evaluates `retain(C)` for a proposed circuit C on those same discovery prompts. It is a real
  intervention and debits the same budget.
- Verifying methods (M2, M3, M4): **104 discovery masks + 24 verification masks**, eight per circuit size
  k ∈ {8, 16, 24}. The 104 masks are the 32-mask balanced warm start plus the first 72 masks of the pilot's
  seeded static balanced stream (densities 0.1/0.3/0.6/0.9).
- Non-verifying methods: M0 and M5 use the same 104 masks plus the next 24 static-stream masks. That 128-mask design is
  identical to the pilot's static 128. M1 uses the 104 plus 24 density-matched random retain masks.
- The budget is enforced by the pilot's `DiscoveryOracle`. It refuses repeated masks, refuses the 128 reserved blind
  masks, refuses queries beyond 128, and logs every query. Each method also asserts at the end that it used exactly
  128 masks.
- Within a seed, methods may share a physical cache of oracle outputs. Each method is still charged the logical cost
  of every mask it requests. Physical evaluations, cache hits, logical oracle seconds, CPU fit, search and acquisition
  seconds, and wall time are reported separately.
- Not charged to any method (applied identically to all): the blind audit, and a post-sealing *discovery-prompt
  audit*. That audit re-measures every sealed circuit and the all-ablated denominator on the discovery prompts, for the
  optimism analysis only. Neither ever feeds back into a decision.

## 4. Splits and seeds

| Set | Seeds | Discovery prompts (32 per seed) | Evaluation prompts | Role |
|---|---|---|---|---|
| development | 49001–49006 | fresh MIB *train* batches | 128 MIB *dev* prompts (template-balanced; disjoint from the pilot's dev-eval prompts) | exploratory only |
| confirmatory | 41001–41020 | fresh MIB *train* batches | the pilot's frozen 128 **blind** MIB *test* prompts (9 template families disjoint from train) | primary evidence |
| bridge | 31001–31005 | the pilot's Gate C discovery batches | the same blind 128 | secondary replication link to the pilot |

Fresh batches come from MIB train with every pilot-used prompt removed (screening, discovery, dev discovery), shuffled
with seed 41777. The batches are pairwise disjoint and disjoint from both evaluation sets. The blind split is rebuilt
from `runs/gate_c/frozen/splits.json` and must hash-match the pilot's frozen blind split. The 128 pilot blind masks
stay reserved and are never queried. Development never touches blind prompts. `vil-prepare` writes and hashes every
prompt id list and every per-seed mask design (`runs/verification_v1/frozen/`).

Bridge-seed outcomes for the pilot's lasso_pair@128 were seen in the pilot report. The bridge set therefore never
contributes to a criterion. It checks only that `lasso_static_128` reproduces the pilot's lasso_pair@128 circuits and
reports all methods on the original seeds.

## 5. Methods (details in doc 09)

| Code | Key | Surrogate | Masks | Final circuit per k |
|---|---|---|---|---|
| M0 | `lasso_static_128` | pairwise Lasso | 128 static | surrogate optimum (shared search) |
| M1 | `lasso_density_matched` | pairwise Lasso | 104 + 24 random retains (8 per k) | surrogate optimum after refit |
| M2 | `lasso_random_verify` | pairwise Lasso | 104 + 24 verifications, random from a diverse surrogate shortlist | lowest measured discovery u among verified |
| M3 | `lasso_vil` | pairwise Lasso | 104 + 24 verifications in rounds; refit after each round; propose best predicted unverified circuit | lowest measured discovery u among verified |
| M4 | `bayes_vil` | spike-and-slab | as M3, proposals by posterior expected improvement over the incumbent | lowest measured discovery u among verified |
| M5 | `bayes_static_128` | spike-and-slab | 128 static | posterior-expected-R optimum |

Lasso and spike-and-slab settings are copied unchanged from the pilot. The circuit search is the pilot's shared
greedy elimination + swap search. Verifying methods add a shared candidate generator (random-start swap searches).

## 6. Evaluation and sealing

For each seed, every method runs to completion and its decision record is written to
`runs/verification_v1/<set>/seed-*/decisions_sealed.json`. That file's SHA-256 is computed **before** any evaluation
of that seed. The record holds the final circuits and, for verifying methods, all 24 verified circuits with their
measured discovery u, the `surrogate_at_104` circuits and the `refit_128` diagnostic circuits. Evaluation then runs
real interventions on the evaluation prompts (blind for confirmatory and bridge, MIB dev for development) and the
discovery-prompt audit. Evaluation outputs never reach a method. During confirmatory execution the runner prints no
outcome metrics.

## 7. Primary analysis and acceptance criteria (fixed before confirmatory data)

Pooled S = the mean of the 20 per-seed S values. Bootstrap = paired hierarchical bootstrap with 2 000 replicates and
RNG seed 4242. It resamples seeds; within each resampled seed it resamples blind template families, then prompts within
families. Both methods are evaluated on the same draws. Statistic: the mean over resampled seeds of S_ref − S_cand.

**H1** (candidate M3 `lasso_vil`; comparators = **all** non-verifying methods {M0, M1, M5}):

- **V-1** pooled S(M3) ≤ 0.85 × min over comparators of pooled S (≥ 15% relative improvement over the best
  non-verified method; same hurdle as the pilot's C-1).
- **V-2** for **each** comparator, the 90% bootstrap CI of S_comp − S_M3 has a lower bound > 0. This is an
  intersection–union test, so no post-hoc comparator selection is involved.
- **V-3** S(M3) is below the seed-wise best comparator S in ≥ 15 of 20 seeds.
- **V-4** budget integrity holds: every method × seed used 128 logical masks (verifying 104 + 24), with zero
  reserved-mask queries, and decisions were sealed before evaluation.

H1 = PASS iff V-1…V-4 all pass. A V-4 failure, a missing seed, or u(∅) < 0.5 on blind makes the result INCONCLUSIVE.

**H2** (M4 `bayes_vil` vs M3 `lasso_vil`):

- **H2-1** pooled S(M4) ≤ 0.90 × pooled S(M3).
- **H2-2** the 90% bootstrap CI of S_M3 − S_M4 has a lower bound > 0.
- **H2-3** S(M4) < S(M3) in ≥ 13 of 20 seeds.

H2 = PASS iff V-4 passes and H2-1…H2-3 all pass. H2 is adjudicated regardless of the H1 outcome, but an H2 PASS alone
cannot produce a GO.

**Decision table**

| H1 | H2 | Recommendation |
|---|---|---|
| PASS | FAIL | **GO** for verification-in-the-loop circuit optimisation *without* Bayesian inference |
| PASS | PASS | **GO** for Bayesian verification-in-the-loop |
| FAIL | any | **NO_GO** for VIL as preregistered (secondary evidence reported; any rescue needs a new protocol) |
| INCONCLUSIVE | — | INCONCLUSIVE (state cause) |

## 8. Secondary analyses (never override the primary verdicts)

1. M3 vs M2: does active revision matter beyond verification-based selection? Reports the ratio, bootstrap CI and
   seed wins. Also M4 vs M2, M2 vs M0 and M1 vs M0. M1 vs M0 tests whether data from the aggressive-ablation regime
   alone explains any gain.
2. The H1 criteria applied to M4 (reported, not adjudicated).
3. Absolute faithfulness: the number of selected circuits with blind R ≤ 0.10, by method and k.
4. Optimism (winner's curse): blind R − predicted R for surrogate-selected circuits, blind R − measured discovery R
   for verification-selected circuits, and correlations.
5. Diagnostics for verifying methods: the blind S of `surrogate_at_104` (what the surrogate alone would pick before
   verification) and of `refit_128` (what the surrogate refit on all 128 observations would pick). These separate the
   gain from "selecting by measurement" from the gain from "a better surrogate".
6. Bayesian circuit-level calibration: coverage of 90% posterior predictive intervals for the measured discovery u of
   proposed and selected circuits, and invalid P(R ≤ 0.10) ≥ 0.95 claims.
7. Compute: logical oracle seconds, CPU fit, search and acquisition seconds, end-to-end logical wall time, peak VRAM.
8. Bridge seeds: replication of the pilot's lasso_pair@128 circuits by M0, and all methods on the pilot seeds.
9. Per-k R, per-seed S, verification trajectories.

## 9. Development phase

The development seeds are used for smoke tests, timing, bug fixing, and **one** preregistered development-dependent
setting: `vil.rounds ∈ {8, 4}`. Eight rounds verify one circuit per k per round; four rounds verify two per k per round.
The value applies to M3 and M4 alike. The rule picks the value with the lower mean development S of `lasso_vil`; if the
difference is < 0.005, it picks 8. The chosen value is written into the implementation config before the freeze record.
Development results are exploratory and never presented as confirmatory evidence.

## 10. Provenance, tracking and deviations

- MLflow (loopback server on the GPU host, `127.0.0.1:5001`, reached through an SSH tunnel). Experiments:
  `bccd/verification-v1` (confirmatory), `bccd/verification-v1-dev`, `bccd/verification-v1-bridge`. One parent run per
  shard and one child run per seed × method, tagged with protocol, hashes, git SHA, seed, method, split hashes and
  sealed-decision hash.
- Confirmatory runs refuse a dirty worktree. They also refuse to start if any hash in the freeze record differs from
  the current protocol docs, configs, `uv.lock` or frozen manifest.
- After execution, `vil-report` recomputes every criterion from the sealed per-seed artifacts and writes
  `reports/verification_v1/`.
- Any deviation is appended to `docs/07_DECISIONS_AND_DEVIATIONS.md` under a BCCD-VIL-1.0 heading, with timing
  relative to confirmatory outcome inspection.
