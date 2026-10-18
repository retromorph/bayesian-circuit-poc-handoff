# 07 — Decisions, clarifications and append-only deviations

## Initial pre-registration clarifications (2026-10-18; before any implementation)

The prior conversation provided a high-level Gate A/B/C plan. The handoff package makes the following operational choices **before data collection**:

1. **Masks:** 1 means ablated throughout, retain(C) ablates the complement inside a fixed pool. Attributions alone are not interventions.
2. **Gate A:** confirmed the four minimal circuits in A1 are cardinality 3; used 20 worlds/regime, 256 blind masks, and a 90% interval definition for the prompt-averaged causal response. Exact top-pair ranking and 16/20 recovery thresholds are fixed.
3. **Gate B:** exact alternative minimal circuits for the two redundant A and B pathways are **four size-2 retained sets**, distinct from Gate A's size-3 alternatives. The model construction must verify the desired behavior before discovery. This prevents treating a failed planted mechanism as a negative algorithm result.
4. **Gate C:** frozen 48-head candidate pool, EAP-IG screening split (64), discovery prompts (32), 128 blind test prompts and 128 blind test masks; score is mean of k={8,16,24} retained-circuit normalized discrepancies. MIB's edge-patching leaderboard is not equated to our head-zeroing experiment.
5. **Budget fairness:** masked-circuit queries that feed back into active learning count toward discovery; independent blind auditing does not. Logical model evaluations and wall time are reported separately.
6. **Competitor compatibility:** ProxySPEX returns interaction attributions; the common LightGBM surrogate and shared circuit optimizer are used for directly comparable intervention prediction and circuit selection, while official ProxySPEX is a distinct interaction-attribution diagnostic.
7. **Research decision:** a synthetic-only pass cannot produce overall GO; all A/B/C mandatory conditions are required. Scientific benchmark invalidity or missing computational resources must be labeled INCONCLUSIVE/BLOCKED, not silently PASS.
8. **Infrastructure:** local MLflow on the remote SSH host is bound to loopback and accessed using SSH port forwarding. The user's alternative public `0.0.0.0`/wildcard/CORS/UFW example is intentionally not the default because it exposes unauthenticated services.
9. **Additional validity clarity:** an informative Gate C candidate set requires mean clean-vs-all-pool-ablate logit-margin discrepancy >=0.5. This threshold is a preregistered operational sanity check; failure makes the experiment inconclusive, not negative evidence against the Bayesian method.
10. **Computational simplification:** Gate C models mean intervention effects over a fixed prompt batch; individualized context effects are a secondary analysis. This prevents inventing strong prompt-conditional inference from only 32 discovery prompts per mask.
11. **A3 false-alarm safeguard:** A3 detector success counts toward A-5 only if it has at most 12 false-positive world flags among all 60 A0–A2 worlds. This excludes trivial always-abstain strategies.
12. **Synthetic duplicates:** A0/A1/A3 replicated context responses are not independent statistical observations; collapse to one observation per mask (A2 uses one per context group).

## Deviation entry template (append, never erase existing entries)

### [UTC timestamp] — [short title]

- **Affected experiment / criterion:**
- **What changed:**
- **Reason:**
- **When relative to blind-outcome inspection:** before / after / unknown
- **Original protocol / config SHA-256:**
- **Updated protocol / config SHA-256 (if applicable):**
- **Which run IDs and data are affected:**
- **Statistical classification:** confirmatory / exploratory / invalidated / blocked
- **Impact on formal Go/No-Go verdict:**
- **Remediation / new preregistration needed:**

## Deviations during execution

### 2026-10-18T21:30Z — Operational specifications fixed before any confirmatory run

These entries fix implementation-level choices that the protocol leaves open. They were written
**before** any confirmatory world/seed was run (only the development world A1-90001 had been run as a
smoke test). They do not change any gate threshold, seed list, split identity or budget.

- **D-INF-1 (infrastructure).** The shared remote host already runs another project's MLflow server on
  `127.0.0.1:5000` (tmux session `mlflow`, project `geodesic-ig-pilot`). To avoid touching it, the BCCD
  server runs on **`127.0.0.1:5001`** (tmux `bccd-mlflow`, SQLite `mlruns/mlflow.db`, artifacts
  `mlruns/artifacts`), loopback only. Tunnel: `ssh -N -L 127.0.0.1:5001:127.0.0.1:5001 <host>`.
  Classification: infrastructure only; no scientific impact.
- **D-ENV-1.** `transformer-lens` pinned to 2.15.4 (`HookedTransformer` hook API) and `transformers<5`
  because TL 2.x is incompatible with transformers 5.x; torch 2.8.0+cu126 from an explicit uv index on Linux.
- **D-A-1 (jitter).** "20% coefficient jitter" is implemented as an independent multiplicative factor
  `1+U(-0.2, 0.2)` per nonzero coefficient; hidden component permutation per world.
- **D-A-2 (candidate method).** Every gate criterion is adjudicated for `bayes_decision` (the full
  method); `bayes_static` and `bayes_uncertainty` are reported as ablations only.
- **D-A-3 (A-5 misfit detector, frozen rule).** Statistic: 4-fold cross-validated RMSE of posterior-mean
  predictions on the method's *own discovery masks* at B=128 divided by the RMS of the observed responses
  (no blind data). Flag if statistic > t. **Threshold rule (pre-declared):** run the 16 development worlds
  (seeds 90001-90004 x A0..A3); let `c = max` statistic over the 12 development A0-A2 worlds and
  `a = min` over the 4 development A3 worlds. If `c < a`, `t = sqrt(c*a)` (geometric midpoint); otherwise
  t is the value among development statistics that minimises development misclassifications, ties broken
  toward the larger t (more specific). t is written to `configs/gate_a_misfit_threshold.json` and hashed
  before confirmatory Gate A.
- **D-A-4 (A-3 zero-error fallback).** MAE is normalised per world by the SD of the blind truth and
  averaged over worlds. If the pooled first-order Lasso normalised MAE is < 1e-3 ("numerically ~0"), A-3
  passes iff the Bayesian normalised MAE <= 0.01; otherwise iff pooled raw MAE ratio <= 1.10.
- **D-A-5 (circuits & claims).** Proposals = up to 4 circuits of the minimal cardinality at which a
  circuit is predicted faithful (non-Bayes: R_hat<=0.10, ranked by R_hat; Bayes: P(R<=0.10|D) >= 0.5,
  ranked by P). Gate A enumerates all circuits of size <= 3 exactly. *High-confidence claims* = every
  enumerated circuit with P >= 0.95 (not only the proposals). Pair nonadditivity is
  `NA_ij = f(e_i+e_j)-f(e_i)-f(e_j)+f(0)` from predicted interventions (Bayes: |posterior mean|), base mask 0.
- **D-A-6 (A-4/A-5 coverage).** Coverage is over all blind targets (A2: two context-group targets per
  mask) of the posterior predictive for a new observation (function + learned noise term).
- **D-B-1 (planted models).** Gate B models are built by direct weight construction in a TransformerLens
  `HookedTransformer` (3 layers, 8 heads, d_model=64, no LayerNorm, ReLU MLPs). Response target for
  surrogate fitting and circuit selection is the batch mean of |d(x,m)| (matching the numerator of R);
  proposals are the top-4 exact size-2 circuits (non-Bayes R_hat<=0.10; Bayes P>=0.5). Event time = first
  checkpoint with >=3 of 4 planted circuits among the proposals and >=1 proposal blind-faithful; else 160
  (censored).
- **D-C-1 (prompts).** MIB IOI (`mib-bench/ioi`, revision e5f3468) is used: screening and discovery
  prompts from the MIB *train* split, development baseline-preselection prompts from MIB *dev*, blind
  prompts from MIB *test*, whose template families are disjoint from train/dev. Only prompts whose two
  names are single GPT-2 tokens (with leading space) and whose ABC counterfactual has equal token length
  are used; exclusions are counted.
- **D-C-2 (EAP-IG).** Screening uses an in-repo, node-level implementation of EAP-IG (Hanna et al. 2024)
  — integrated gradients over input embeddings from the ABC counterfactual to the clean prompt, 5 steps,
  node score = sum over positions of (z_clean - z_corrupt) . mean_alpha dL/dz at `hook_z`
  (equivalent to the sum of the head's outgoing EAP edge scores), metric = IO-S logit difference, ranked
  by |mean score| over the 64 screening prompts. This is a *replica*, not the MIB package.
- **D-C-3 (decision losses).** Gate C circuit selection minimises the posterior expected R(C) (Bayes) or
  predicted R (non-Bayes) for each k in {8,16,24} with the shared greedy-elimination + swap search. The
  decision-focused acquisition uses L(C)=|C|/p+10*max(R-0.1,0) per k and V(D)=mean_k min_C E[L].
  Surrogates are fit to the batch mean of |d(x,m)| (R's numerator). C-3 calibration uses a separate
  Bayesian model of the signed mean effect fit on the same discovery data, plus a prompt-resampling
  variance term var_x(d)/32 + var_x(d)/128 predicted by a ridge model of per-mask discovery prompt
  variance (discovery data only).
- **D-C-4 (development reference baseline).** Before blind evaluation, lasso_1st, lasso_pair and gbdt are
  run on a development discovery run (seed 31900, 32 MIB-train prompts disjoint from all others, warm
  start + static masks to B=128) and scored on 64 MIB-dev prompts; the lowest development S is frozen as the
  C-4 reference in `runs/gate_c/dev_reference.json`.
- **D-C-5 (compute).** End-to-end compute at B=128 = shared screening + the method's own oracle calls +
  all fitting / acquisition / circuit-search work needed to produce its B=128 decision (passive methods:
  one fit + search at 128; active methods: all rounds). No cross-method oracle cache in Gates B/C.
  The extra calibration (signed-effect) model is reported separately.


### 2026-10-18T21:55Z — D-C-6: Lasso solver tolerance for Gate C (before any Gate C run)

- **Affected experiment / criterion:** Gate C baselines `lasso_1st`, `lasso_pair` (and therefore C-1, C-2, C-4, C-5 comparators).
- **What changed:** LassoCV coordinate-descent tolerance for Gate C is sklearn's default `tol=1e-4`
  (`max_iter=10000`) instead of the `1e-7`/50000 used in Gates A/B. Regularisation path, CV folds,
  feature dictionary and scaling are unchanged.
- **Reason:** on a local CPU smoke test (48 masks, 1176 features, GPT-2 effects; no blind data) a single
  pairwise LassoCV fit at tol=1e-7 took ~56 s and >3 min on a 256x1176 synthetic benchmark. This is a
  solver-precision setting chosen for noiseless synthetic targets; on noisy GPT-2 targets it would
  inflate the comparator's compute and bias C-5 *toward* the Bayesian method.
- **When relative to blind-outcome inspection:** before any Gate C screening, discovery or blind run.
- **Statistical classification:** confirmatory (implementation setting, no threshold change).
- **Impact on verdict:** none by construction; Gates A/B results unaffected.

### 2026-10-18T22:25Z — D-PROV-1: `src/bayescircuits/models/` was missing from git (found after all gates ran)

- **Affected experiment / criterion:** provenance of Gates A, B, C (no criterion value).
- **What happened:** the handoff `.gitignore` contained the unanchored pattern `models/` (intended for model
  checkpoint caches). It also matched `src/bayescircuits/models/` (`bayes.py`, `surrogates.py`,
  `proxyspex.py`, `__init__.py`), so these files were never committed. Code was deployed to the GPU host with
  rsync (which ignores `.gitignore`), so the runs executed the files, but the git SHAs recorded at run time
  (A `1658055`, B `f8b232f`, C `6b43b7f`) do not contain them, and `git_dirty=False` could not detect it.
- **Executed versions (from remote file mtimes, UTC):** `bayes.py` last modified 21:18 (before Gate A started at
  21:33; unchanged since); `proxyspex.py` 21:34 (before Gate B; unchanged since); `surrogates.py` modified at
  21:52 only to expose the Lasso `tol`/`max_iter` as parameters whose defaults (1e-7 / 50000) equal the values
  previously hard-coded (D-C-6) — Gate B had already imported the earlier version; Gate C used the new one.
  SHA-256 of the committed files: bayes.py `de541aa4…`, surrogates.py `e55eae8c…`, proxyspex.py `4398f9ce…`.
- **Fix:** pattern anchored to `/models/`; package committed. A deterministic re-execution of one Gate A world
  per regime family and one Gate B model from the committed code is compared with the sealed confirmatory
  outputs in `runs/diagnostics/replication.json` (see final report).
- **Statistical classification:** provenance defect; no change to any data, rule or result.
- **Impact on verdict:** none.
## BCCD-VIL-1.0 (verification-in-the-loop follow-up) — decisions and deviations

The BCCD-PILOT-1.0 entries above are unchanged. The follow-up protocol is `docs/08_VERIFICATION_PROTOCOL.md`, with
methods in `docs/09_VERIFICATION_METHODS.md` and configs `configs/preregistration_verification_v1.yaml` and
`configs/verification_implementation_v1.yaml`. All entries below were written **before any confirmatory BCCD-VIL-1.0
seed was run**.

### 2026-10-18T23:24Z — D-VIL-0: truncated research brief; agent-specified design elements

- **Affected:** methods M2 (partly), M3–M5, every acceptance threshold, seed counts, decision table.
- **What:** the brief arrived truncated mid-way through method M2. The executing agent specified the remaining design
  before any confirmatory data existed. Choices: M3 = Lasso refit-and-repropose verification loop; M4 = spike-and-slab
  posterior with expected-improvement proposals; M5 = Bayesian non-verified reference. H1 thresholds V-1…V-3 reuse the
  pilot's 15% hurdle, with an intersection–union bootstrap against every non-verifying method and ≥ 15/20 seed wins.
  H2 uses a ≥ 10% hurdle with bootstrap and ≥ 13/20 seed wins. 20 fresh confirmatory seeds; the 5 pilot seeds are a
  secondary bridge.
- **Classification:** preregistered design (agent-authored). **Impact:** defines the confirmatory analysis.

### 2026-10-18T23:24Z — D-VIL-1: M2 shortlist source (development-phase bug fix)

- **Affected:** `lasso_random_verify` (M2) only.
- **What happened:** in the first development run (seeds 49001–49006), M2's diverse 16-circuit shortlist was built from
  the candidate set that keeps only each search's best circuit plus its top-6 alternatives. For k = 8 (sometimes 16)
  the random-start searches converge on the same basin, so fewer than 16 distinct circuits survived the diversity rule,
  and the safeguard `_top_up` padded the shortlist with 4–10 *uniformly random* circuits per seed. M2 then sometimes
  spent verifications on random circuits, which contradicts its definition ("diverse shortlist of plausible circuits")
  and would have made it an artificially weak control.
- **Fix:** the shortlist is now drawn from every circuit the surrogate scored during the k-specific searches. These are
  all plausible, with surrogate-scored R̂, and the same diversity rule applies. M3/M4 are unchanged; they had zero
  top-up events.
- **When:** after development outcomes, before any confirmatory run. M2 was re-run on the development seeds
  (`runs/verification_v1/dev_rerun-m2fix/`) to confirm zero top-ups. The original development outputs are kept.
- **Classification:** development fix. **Impact on verdict:** none by construction (M2 is a secondary control).

### 2026-10-18T23:24Z — D-VIL-2: development-dependent setting resolved by the preregistered rule

- **Rule** (prereg `decision.development_dependent_setting`): `vil.rounds ∈ {8, 4}`, chosen by lower mean development S
  of `lasso_vil`; ties (< 0.005) go to 8.
- **Observed (MIB dev evaluation prompts, 6 dev seeds):** rounds = 8 gives 0.3739 and rounds = 4 gives 0.3932, so the
  rule selects **8** (the default already in the config). For reference, bayes_vil scored 0.3877 (8) vs 0.3822 (4);
  it does not enter the rule.
- **Classification:** preregistered development selection. No other setting was changed after development.

### 2026-10-18T23:24Z — D-VIL-INF: operations

- The pilot's MLflow server is reused on the GPU host (`127.0.0.1:5001`, loopback, tmux `bccd-mlflow`), with new
  experiments `bccd/verification-v1`, `-dev` and `-bridge`. No firewall or host-level changes were made.
- Seeds run as 4 parallel shard processes on the single A4000, each with its own model copy. Determinism is per
  (seed, method), and the sharding assignment cannot affect any decision.
- Code is synced to the host with rsync (including `.git`). Confirmatory runs refuse a dirty worktree and a changed
  freeze record.

### 2026-10-18T23:39Z — D-VIL-3: cosmetic `query_index` defect in the pilot oracle log (found after confirmatory runs)

- **What:** `DiscoveryOracle.query` (pilot code, unchanged) writes `query_index = n_masks - len(masks) + i` after the
  loop has already added earlier masks of the same batch to the seen-set. The logged index is therefore wrong (it can
  even be negative). The pilot's logs carry the same defect.
- **Impact:** none on any decision or criterion. Budget enforcement uses the seen-set size, which is correct; the
  independent audit counts exactly 128 distinct masks per method × seed. Parquet rows are stored in true chronological
  order. The audit also re-derived the discovery/verification split from ablation counts: verification = 24/32/40
  ablated; design = 5/14/29/43.
- **Action:** documented, not fixed, so the oracle code stays identical to what both protocols executed.

### 2026-10-18T23:45Z — D-VIL-4: exploratory (post-verdict) run issues

- **Context:** exploratory budget/allocation variants (`--set exploratory`, `bccd/verification-v1-exploratory`) were
  launched only after the confirmatory H1/H2 verdicts were computed and committed (`b078735`).
- **Issue 1:** the first launch started 8 GPU processes concurrently. Shard 0 of B128V48 hit CUDA OOM (8 × ~2 GB on
  16 GB). Nothing was retried silently.
- **Issue 2:** for 72 verifications, M2 requested 24 picks per k from its 16-circuit shortlist and crashed.
- **Fix:** for exploratory budget variants, M2's shortlist size is set to twice the per-k picks, the preregistered
  8-from-16 ratio. All partial V48/V72 outputs (both shards, including one still running with the old shortlist size)
  were moved to `runs/verification_v1/_aborted_exploratory/` and their logs to `logs/aborted_exploratory/`. Both
  variants were re-run from scratch, one process each.
- **Classification:** exploratory only; no confirmatory or bridge artifact is affected.

### 2026-10-18T23:48Z — D-VIL-4 addendum: exploratory re-sharding

- To finish sooner, the single-process V48/V72 exploratory runs were stopped after seed 41001 and relaunched as 3
  shards each. Seed 41001 was kept as complete; seeds without a completed `status.json` are re-run from scratch.
  The stopped processes' MLflow parent runs in `bccd/verification-v1-exploratory` stay un-terminated (killed) and
  contain no seed results beyond 41001. Exploratory only.
