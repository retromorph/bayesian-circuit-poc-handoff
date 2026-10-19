# Verification-in-the-Loop for Causal Circuit Discovery — BCCD-VIL-1.0 Report

**Protocol:** BCCD-VIL-1.0, a follow-up to BCCD-PILOT-1.0. The pilot's protocol, configs and reports are unchanged.  
**Preregistration frozen:** 2026-10-18T23:24:43Z at source revision `a553a70`. The freeze record is committed in `9e2f365`
(`reports/verification_v1/preregistration_hashes.json`).  
**Confirmatory execution:** 2026-10-18T23:25Z–23:36Z UTC at git `9e2f365` (all 20 seeds, clean worktree).  
**Hashes (SHA-256):** protocol doc 08 `3c938257…`, methods doc 09 `845aacbc…`, preregistration YAML `1bf81016…`,
implementation YAML `565d085b…`, `uv.lock` `f05cad05…` (unchanged from the pilot).  
**Host:** NVIDIA RTX A4000 (16 376 MiB, driver 595.71.05); 8 CPU cores, 31 GiB RAM; Linux 7.0.0-15; Python 3.11.13,
torch 2.8.0+cu126, transformers 4.57.6, TransformerLens 2.15.4, scikit-learn 1.9.1, MLflow 3.17.1.  
**Model / data:** GPT-2 Small `gpt2@607a30d7…` (weights SHA-256 `2c06fd4a…`, identical to the pilot);
MIB IOI `mib-bench/ioi@e5f3468`.  
**MLflow** (`127.0.0.1:5001` on the GPU host, via SSH tunnel): experiment `bccd/verification-v1`. Confirmatory parents:
`798565ab26f246ceb9aeeec1e6fed9c4`, `2e3a300bf50f4a8ba652f13957276085`, `2b6a1154e3354e0d9a87b3312d3ebc82` and
`2015380686934ead965067f087fcc407`. Aggregate report run: `641fb9c806ce451a9cb2c9d5d9cebbae`. Bridge parent:
`f6e8bc161663426490b00de5255ecdd6` (`bccd/verification-v1-bridge`). Development parents are in
`bccd/verification-v1-dev`; exploratory runs are in `bccd/verification-v1-exploratory`.

## Executive decision

| Hypothesis | Verdict | Key evidence (20 confirmatory seeds, 128 blind prompts) |
|---|---|---|
| **H1** — verification-in-the-loop beats purely surrogate-based selection at equal budget | **PASS** (V-1 · V-2 · V-3 · V-4 all PASS) | `lasso_vil` S = **0.342** vs best non-verified `lasso_density_matched` 0.419 (ratio **0.818**; ≥ 15% required). 90% bootstrap CIs of the improvement are > 0 against **every** non-verified method. Wins 16/20 seeds (≥ 15 required). |
| **H2** — Bayesian inference adds value beyond non-Bayesian VIL | **FAIL** (H2-1 · H2-2 · H2-3 all FAIL) | `bayes_vil` S = **0.339** vs `lasso_vil` 0.342 (ratio 0.990; ≤ 0.90 required). 90% CI of the difference [−0.017, +0.023]. Wins 10/20 seeds. |

**Recommendation (preregistered decision table): GO for verification-in-the-loop circuit optimisation *without*
Bayesian inference** (`GO_VIL_WITHOUT_BAYES`).

What this result does **not** show:

- **No circuit is faithful in the absolute sense.** Across all 6 methods × 20 seeds × 3 sizes, no selected circuit
  reaches blind R ≤ 0.10. The best is R = 0.194. VIL makes the retained circuits *less unfaithful* (mean R at k = 24
  falls from 0.33 to 0.26), but at k ≤ 24 heads of this 48-head pool, IOI behaviour is not reproduced to the 10% level.
- The gain is relative, in-pool, node-level and zero-ablation, on one model and one task.
- More than half of the gain (53–63%, depending on whether M1 or M0 is the reference) comes from *selecting
  circuits by measured outcome* rather than by surrogate prediction: random verification of surrogate-shortlisted
  circuits (M2) already captures it. Active, error-aware revision (M3) adds a further, statistically supported ~9%
  (secondary analysis).

## 1. Question and design

In the pilot (Gate C), every surrogate predicted far better faithfulness for its own selected circuits than was
measured (decision-focused Bayes: R̂ ≈ 0.10 predicted vs 0.51 measured). That pattern indicates surrogate exploitation
under combinatorial search. BCCD-VIL-1.0 asks whether spending part of the *same* intervention budget on directly
verifying proposed circuits fixes this.

**Fixed setting.** The pilot's frozen GPT-2 Small, its 48-head candidate pool (SHA-256 `25568c18…`) and its zero
ablation of `hook_z` at all positions for pool heads (1 = ablated). The score is the IOI correct − incorrect logit
difference. `R(C) = mean|d(retain C)| / mean|d(ablate all 48)|`, with blind denominator 4.955.
`S = mean over k ∈ {8, 16, 24} of R`.

**Budget parity.** Every method gets exactly **128 logical masks** on its seed's 32 discovery prompts. Verifying methods
spend **104 discovery + 24 verification** masks (8 per k). A verification is a real `retain(C)` intervention on the
discovery prompts and is charged to the same budget. Enforcement and audit, from the raw query logs:
128/128 masks for every method × seed, verification masks exactly the 24/32/40-ablated retain masks of the 24 verified
circuits, 0 duplicates, 0 queries of the 128 reserved blind masks. Methods share a physical oracle cache within a seed,
but each is charged logically, and logical oracle seconds are reported per method.

**Methods** (details in `docs/09_VERIFICATION_METHODS.md`):

| Code | Method | Masks | Final circuit per k |
|---|---|---|---|
| M0 | `lasso_static_128` — pairwise Lasso, shared search | 128 static | surrogate optimum |
| M1 | `lasso_density_matched` — pairwise Lasso | 104 + 24 random retain masks (8 each retaining 8/16/24) | surrogate optimum |
| M2 | `lasso_random_verify` — random verification control | 104 + 24 random picks from a diverse surrogate shortlist | best *measured* |
| M3 | `lasso_vil` — **H1 candidate**: 8 rounds of refit-and-repropose (Lasso refit on all data incl. verifications; verify best predicted unverified circuit per k) | 104 + 24 | best *measured* |
| M4 | `bayes_vil` — **H2 candidate**: spike-and-slab posterior; verify the circuit with maximal posterior expected improvement over the incumbent (Thompson-augmented candidates) | 104 + 24 | best *measured* |
| M5 | `bayes_static_128` — spike-and-slab, posterior-expected-R search | 128 static | surrogate optimum |

**Splits and seeds.** 20 confirmatory seeds (41001–41020), each with a fresh 32-prompt MIB-train batch disjoint from
everything the pilot used. Evaluation is on the pilot's frozen blind split: 128 MIB-test prompts from 9 template
families, none present in any discovery batch (hash-verified against the pilot). There are also 6 development seeds
(MIB-dev evaluation only) and 5 bridge seeds (the pilot's Gate C seeds and discovery batches; secondary).

**Provenance of the specification.** The research brief arrived truncated inside the description of M2. M2's
remaining details, M3–M5, every threshold and the decision table were written by the executing agent **before any
confirmatory data existed** (D-VIL-0 in `docs/07_DECISIONS_AND_DEVIATIONS.md`). The H1 hurdle reuses the pilot's 15%.

## 2. Preregistration timeline

| UTC (2026-10-18) | Event |
|---|---|
| 23:16 | `vil-prepare`: VIL prompt batches, evaluation sets and per-seed designs frozen and hashed. Blind split hash-matches the pilot. Bridge batches match the pilot's 5/5. |
| 23:16–23:23 | Development runs (6 dev seeds; MIB-dev prompts only). The preregistered rule picks `rounds = 8` (dev S 0.374 vs 0.393 for 4). Bug fix D-VIL-1 (M2 shortlist padding with random circuits); M2 re-run on dev confirms 0 top-ups. |
| 23:24:43 | `vil-freeze`: hashes of docs 08/09, both configs, `uv.lock`, frozen manifests, source `a553a70`. Committed as `9e2f365`. |
| 23:25–23:37 | Confirmatory (20 seeds) and bridge (5 seeds). The runner refuses a dirty tree or hash drift and prints no confirmatory metrics. |
| 23:37 | `vil-report` recomputes all criteria from the sealed artifacts. Committed as `b078735`. |
| after 23:38 | Exploratory analyses only (Section 8). |

The development data hinted that H1 might fall short of the 15% hurdle: dev ratio vs M1 was 0.878. The threshold was
not changed.

## 3. Primary results (confirmatory, 20 seeds)

| Code | Method | **pooled S** | R k=8 | R k=16 | R k=24 | #R ≤ 0.10 |
|---|---|---:|---:|---:|---:|---:|
| M0 | lasso_static_128 | 0.4378 | 0.5734 | 0.4094 | 0.3308 | 0/60 |
| M1 | lasso_density_matched | 0.4188 | 0.5292 | 0.4028 | 0.3243 | 0/60 |
| M2 | lasso_random_verify | 0.3781 | 0.4969 | 0.3562 | 0.2810 | 0/60 |
| **M3** | **lasso_vil** | **0.3424** | 0.4593 | **0.3094** | 0.2587 | 0/60 |
| M4 | bayes_vil | 0.3392 | **0.4392** | 0.3253 | **0.2530** | 0/60 |
| M5 | bayes_static_128 | 0.4427 | 0.5817 | 0.4233 | 0.3231 | 0/60 |

| Criterion | Rule | Observed | Status |
|---|---|---|---|
| V-1 | pooled S(M3) ≤ 0.85 × best non-verified | 0.3424 / 0.4188 (M1) = **0.818** | PASS |
| V-2 | 90% paired hierarchical bootstrap CI of S_c − S_M3 > 0 for **each** of M0, M1, M5 | vs M0 [+0.066, +0.129]; vs M1 [+0.055, +0.100]; vs M5 [+0.072, +0.131] | PASS |
| V-3 | M3 beats the seed-wise best non-verified method in ≥ 15/20 seeds | 16/20 | PASS |
| V-4 | budget integrity (128 = 104 + 24; no reserved masks; sealed before evaluation) | holds for all 120 method × seed units | PASS |
| H2-1 | pooled S(M4) ≤ 0.90 × S(M3) | 0.3392 / 0.3424 = 0.990 | FAIL |
| H2-2 | 90% CI of S_M3 − S_M4 > 0 | [−0.017, +0.023] | FAIL |
| H2-3 | M4 < M3 in ≥ 13/20 seeds | 10/20 | FAIL |

Bootstrap: 2 000 paired replicates, resampling seeds, then blind template families (9), then prompts within families.
The relative-improvement CIs of M3 are [16%, 28%] vs M0, [14%, 23%] vs M1 and [17%, 28%] vs M5. Per-seed S values are in
`tables/vil_S_by_seed.csv`. In the 4 seeds where M3 does not beat every non-verified method (41003, 41007, 41010, 41012), it
trails the seed-wise best one by 0.012–0.028.

![S by method](figures/vil_confirmatory_S_by_method.png)

## 4. Where the gain comes from (secondary analyses; never override the verdicts)

| Comparison | Ratio of pooled S | 90% CI of S_ref − S_cand | Seed wins |
|---|---:|---|---:|
| M1 density-matched vs M0 static (more aggressive-ablation data, no verification) | 0.956 | [−0.010, +0.047] | 14/20 |
| M2 random verification vs M0 | 0.863 | [+0.037, +0.087] | 18/20 |
| M3 active VIL vs M2 random verification | 0.906 | [+0.012, +0.061] | 14/20 |
| M4 Bayesian VIL vs M2 | 0.897 | [+0.024, +0.054] | 17/20 |

Diagnostic circuits of the verifying methods (blind-evaluated, never selectable):

| Method | surrogate-only choice at 104 masks | surrogate refit on all 128 (incl. verifications) | **final (best measured)** |
|---|---:|---:|---:|
| lasso_random_verify | 0.448 | 0.427 | 0.378 |
| lasso_vil | 0.448 | 0.410 | 0.342 |
| bayes_vil | 0.455 | 0.414 | 0.339 |

Interpretation:

1. **More data in the relevant regime does not explain the gain.** M1 receives 24 random masks exactly at the
   retained-circuit densities and improves on M0 only by an insignificant 4%.
2. **Measurement-based selection is the main mechanism.** Verifying 24 random plausible circuits and keeping the best
   measured one (M2) beats every surrogate-only selection. The surrogate refit on the verification data (the
   "refit@128" column) improves only modestly on the surrogate at 104. A revised surrogate alone would *not* have
   produced the H1 effect.
3. **Active, error-aware revision adds a real but smaller increment.** M3 vs M2 is ~9% with a CI above 0 (secondary;
   not multiplicity-adjusted). The verification trajectory (figure below, right) shows the best measured discovery R
   still falling at round 8. However, the exploratory allocation runs in Section 8 found no benefit from shifting
   more of a fixed 128-mask budget into verification.

**Winner's curse.** For surrogate-selected circuits the optimism is large and of the same kind for every surrogate.
Mean blind R − predicted R is +0.31 (M0), +0.29 (M1) and +0.34 (M5). Predicted R averages 0.10–0.13; measured R is
0.42–0.44. For verification-selected circuits, the measured discovery-prompt R transfers to blind prompts from unseen
template families. Mean blind − discovery is −0.013 (M3), −0.018 (M4) and −0.013 (M2), with correlation 0.89–0.90 over
all 1 440 verified circuits. Selecting the best of 8 noisy measurements on 32 prompts therefore did not produce a
detectable second-order winner's curse at this scale.

![optimism](figures/vil_confirmatory_optimism.png)

## 5. H2: does Bayesian inference help?

No. With identical budgets, rounds and final selection rule, the spike-and-slab posterior with expected-improvement
proposals (M4) is statistically indistinguishable from the Lasso refit-and-repropose loop (M3). The pooled difference is
0.003, the CI spans 0 and each wins 10/20 seeds. M4 is slightly better at k = 8 (0.439 vs 0.459) and slightly worse at
k = 16 (0.325 vs 0.309). Without verification, the Bayesian surrogate (M5, 0.443) is no better than the Lasso (M0,
0.438), as in the pilot.

**Circuit-level calibration (secondary).** The Bayesian circuit-level uncertainty remains badly overconfident:

- `bayes_static_128`: 90% posterior predictive intervals for the selected circuits' discovery-prompt u cover 8% of
  measurements, and 92% of measurements fall above the upper limit.
- `bayes_vil`: coverage is 48% over all proposed, verified and final circuits, and only 12% for its 104-mask
  surrogate choice.
- Every P(R ≤ 0.10) ≥ 0.95 claim was false on blind prompts: 13/13 for M5 and 12/12 for M4.

The posterior's uncertainty is therefore not usable as a substitute for verification, and EI built on it buys nothing
over greedy Lasso exploitation here.

## 6. Compute (mean per seed; logical accounting)

| Method | logical oracle s (128 masks) | fit s | search s | acquisition s | end-to-end s (logical) |
|---|---:|---:|---:|---:|---:|
| M0 lasso_static_128 | 11.0 | 2.2 | 0.04 | — | 13.2 |
| M1 lasso_density_matched | 10.8 | 2.0 | 0.04 | — | 12.9 |
| M2 lasso_random_verify | 10.3 | 1.5 | 2.2 | 1.0 | 15.0 |
| M3 lasso_vil | 10.0 | 16.3 | 7.4 | 0.01 | 33.7 |
| M4 bayes_vil | 10.2 | 6.6 | 11.5 | 0.06 | 28.3 |
| M5 bayes_static_128 | 11.0 | 1.7 | 0.05 | — | 12.8 |

- Logical oracle seconds charge every mask its measured per-mask GPU time, including masks served from the shared
  cache. They were measured while 5 processes shared the GPU. Single-process, 128 masks × 32 prompts take ≈ 2.9 s
  (pilot), so absolute GPU times here are inflated about 3.5–4×, roughly equally for all methods.
- M3 costs about 2.5× M0 end-to-end, almost entirely CPU: 8 `LassoCV` refits on 1 176 pairwise features and the
  random-start searches.
- Peak PyTorch-allocated VRAM ≤ 1.34 GB per process.
- Blind and discovery-prompt audits are excluded from the table and identical for all methods.
- In this setting a real intervention mask costs ~23 ms of GPU time (single process). The CPU-side surrogate work is
  therefore the dominant cost of VIL, not the interventions.

## 7. Bridge to the pilot (secondary; pilot seeds 31001–31005)

`lasso_static_128` selects **exactly** the pilot's `lasso_pair@128` circuits on 5/5 seeds, and its blind S is 0.3848,
identical to the pilot's pooled value. This validates the shared pipeline. On these seeds:

| Method | M0 | M1 | M2 | M3 | M4 | M5 |
|---|---:|---:|---:|---:|---:|---:|
| pooled S | 0.3848 | 0.4152 | 0.3824 | **0.3346** | 0.3663 | 0.3674 |

On the same five seeds, the pilot's `lasso_pair` at **256** masks reached 0.3628 and its decision-focused Bayes at 128
masks reached 0.5108. M3 at 128 masks does better than both. With n = 5 this is descriptive only.

## 8. Exploratory analyses (post-verdict, not preregistered)

These runs were launched **after** the H1/H2 verdicts were computed and committed (`b078735`). They use the 20
confirmatory seeds and the blind prompts, are logged to `bccd/verification-v1-exploratory`, and are hypothesis-generating
only. Budget variants reuse prefixes of the same frozen seeded mask streams. For more verifications, M2 draws from a
shortlist twice the size of its picks (D-VIL-4).

| Configuration | masks | verifications | pooled S | R k=8 / 16 / 24 |
|---|---:|---:|---:|---|
| static Lasso | 64 | 0 | 0.4845 | 0.640 / 0.469 / 0.345 |
| static Lasso (M0) | 128 | 0 | 0.4378 | 0.573 / 0.409 / 0.331 |
| static Lasso | 256 | 0 | 0.3875 | 0.525 / 0.357 / 0.281 |
| Lasso VIL | 64 (40 + 24) | 24 | 0.3920 | 0.534 / 0.365 / 0.277 |
| **Lasso VIL (M3)** | 128 (104 + 24) | 24 | **0.3424** | 0.459 / 0.309 / 0.259 |
| Lasso VIL | 128 (80 + 48) | 48 | 0.3549 | 0.465 / 0.328 / 0.271 |
| Lasso VIL | 128 (56 + 72) | 72 | 0.3420 | 0.445 / 0.320 / 0.261 |
| random verification (M2) | 128 (104 + 24) | 24 | 0.3781 | 0.497 / 0.356 / 0.281 |
| random verification | 128 (80 + 48) | 48 | 0.3699 | 0.495 / 0.345 / 0.270 |
| random verification | 128 (56 + 72) | 72 | 0.3935 | 0.539 / 0.359 / 0.283 |

| Paired comparison (20 seeds) | ratio | 90% CI of S_ref − S_cand | wins |
|---|---:|---|---:|
| Lasso VIL @ 64 vs static Lasso @ 128 | 0.895 | [+0.020, +0.075] | 14/20 |
| Lasso VIL @ 128 vs static Lasso @ **256** | 0.884 | [+0.023, +0.068] | 16/20 |
| static Lasso @ 256 vs @ 128 | 0.885 | [+0.017, +0.088] | 13/20 |
| VIL 80 + 48 vs 104 + 24 | 1.036 | [−0.031, +0.005] | 7/20 |
| VIL 56 + 72 vs 104 + 24 | 0.999 | [−0.022, +0.022] | 10/20 |
| VIL 56 + 72 vs random verification 56 + 72 | 0.869 | [+0.027, +0.074] | 16/20 |

Observations (exploratory):

- **Mask efficiency better than 2×.** Lasso VIL with half the masks (64) beats static pairwise Lasso with 128.
  With 128 it beats static Lasso with 256. Both comparisons have CIs above 0.
- **The allocation hardly matters between 24 and 72 verifications.** Moving discovery masks into verification neither
  helps nor hurts the active method. Random verification gets *worse* at 72, plausibly because its shortlist comes from a
  weaker 56-mask surrogate (not tested). The still-falling verification trajectory (Section 4) does not translate into a benefit at fixed
  B = 128. Presumably what is gained by more verification is lost in surrogate quality.
- No configuration produced a circuit with blind R ≤ 0.10.

![exploratory](figures/vil_exploratory_budget_allocation.png)

## 9. Development results (exploratory; MIB-dev evaluation prompts)

| | M0 | M1 | M2 (after D-VIL-1) | M3 | M4 | M5 |
|---|---:|---:|---:|---:|---:|---:|
| mean dev S (6 seeds) | 0.447 | 0.426 | 0.413 | 0.374 | 0.388 | 0.431 |

The ordering on development matched confirmatory: VIL < random verification < non-verified. The margin of M3 over the
best non-verified method was smaller on dev (12%) than on confirmatory (18%).

## 10. Deviations, caveats and limitations

- **D-VIL-0:** the brief was truncated. M2's details after the truncation, M3–M5, all thresholds and the decision table
  were authored by the agent before confirmatory data existed.
- **D-VIL-1:** a development-phase bug fix to M2's shortlist, made before the freeze.
- **D-VIL-2:** the dev rule selected `rounds = 8`.
- **D-VIL-3:** the pilot `DiscoveryOracle` writes an incorrect `query_index` value into its log rows. The field is
  computed after the per-query loop has already incremented the seen-set. It is cosmetic: budget enforcement uses the
  seen-set size, and log rows are stored in true chronological order. The defect is unchanged, so the pilot's oracle
  code stays as it ran.
- **D-VIL-4:** exploratory runs only. The first exploratory launch had 8 concurrent GPU processes and hit CUDA OOM in
  one shard. A second crash came from M2's shortlist being smaller than the requested picks at large verification
  counts. Partial outputs were moved to `runs/verification_v1/_aborted_exploratory/` and the runs were re-executed
  with fewer processes and the shortlist sized at twice the picks; V48/V72 were later re-sharded (addendum). No
  confirmatory run was affected.
- **Multiplicity.** H1 and H2 are separate preregistered hypotheses with fixed decision rules. Secondary comparisons
  are not multiplicity-adjusted.
- **Scope.** One model (GPT-2 Small), one task (IOI) and a fixed 48-head pool chosen by the pilot's EAP-IG screening
  plus 16 random heads. Head-level zero ablation only. Retained "circuits" are sets of heads, not directed paths. No
  circuit is faithful at R ≤ 0.10. The relative improvements do not show that VIL recovers *the* IOI circuit.
- **Statistics.** 20 seeds and 9 blind template families. The hierarchical bootstrap treats seeds and families as
  exchangeable. The 15% hurdle is a project convention, not a cost–benefit optimum.
- **Compute claims** are for this host under concurrent execution (Section 6). The CPU cost of M3 could likely be cut
  (warm-started Lasso paths, fewer random starts); this was not tested.
- **Bayesian credible probabilities are not frequentist guarantees.** The calibration failures above are empirical
  over these seeds and circuits.

## 11. Recommendation

**GO for verification-based circuit optimisation without Bayesian inference.** Concretely:

1. Make *measurement-based selection* the default. Never report a surrogate-optimal circuit without measuring it, and
   choose among measured candidates. This alone (M2) removes most of the surrogate optimism seen in the pilot.
2. Use a cheap surrogate (pairwise Lasso) as a *proposal generator*, revised with the verification outcomes (M3). The
   active loop adds a further, smaller gain.
3. Drop the spike-and-slab machinery for this purpose. It adds compute and miscalibrated certainty but no faithfulness.
4. Exploratory evidence suggests better than 2× mask efficiency: VIL at 64 masks beats static Lasso at 128, and VIL
   at 128 beats static Lasso at 256. It also suggests the discovery/verification split is not critical between 24 and
   72 verifications out of 128. A follow-up protocol should confirm the efficiency claim prospectively, ideally on a
   second task or model.
5. Target the absolute faithfulness gap. No k ≤ 24 circuit reaches R ≤ 0.10 under any method or budget tested, so
   larger k, a different candidate pool, or different ablation semantics are needed before any claim of a *faithful*
   circuit.

## 12. Reproduction

```bash
git checkout 9e2f365            # frozen protocol + code used for the confirmatory run (report code: see later commits)
uv sync --frozen
bash scripts/start_mlflow.sh    # loopback 127.0.0.1:5001, tmux bccd-mlflow
# requires the pilot's frozen Gate C artifacts in runs/gate_c/frozen/ (candidate pool, blind split/masks)
uv run python -m bayescircuits.cli vil-prepare
for i in 0 1 2 3; do tmux new-session -d -s vil-conf-$i "bash scripts/run_vil.sh vil-confirmatory-$i vil-run --set confirmatory --shard $i --n-shards 4"; done
bash scripts/run_vil.sh vil-bridge vil-run --set bridge
uv run python -m bayescircuits.cli vil-report --mlflow
```

Artifacts: `reports/verification_v1/{decision.json, run_manifest.json, generated_results.md, tables/, figures/}`,
`reports/verification_v1/preregistration_hashes.json`, `reports/verification_v1/frozen_artifacts/`, and
`reports/verification_v1/audit/vil_runs_artifacts.tgz`. The archive holds every sealed decision, query log, blind and
discovery audit, per-prompt discrepancies and run log. `vil-report` recomputes every number in this report from
those artifacts without re-running any experiment.
