# 05 — Final research report template

> This document is a template. The coding agent must create `reports/final_report.md` with actual measured results. Never fill missing numerical cells with estimates or invented values.

# Bayesian Causal Circuit Discovery — Go/No-Go Pilot Report

**Date:** [UTC completion date]  
**Protocol:** BCCD-PILOT-1.0 [SHA-256]  
**Git revision / uv lock:** [SHA / SHA-256]  
**Host / GPU / driver / torch:** [observed details]  
**Overall verdict:** GO / NO_GO / INCONCLUSIVE / BLOCKED  
**MLflow tracking:** [localhost UI via SSH tunnel, run IDs]

## Executive decision (maximum 250 words)

- What was implemented and really executed?
- Did all three gates pass? If not, identify the first failing criterion and the strongest evidence against the method.
- At fixed B=128 on GPT-2, what was Bayesian S and what was the strongest non-Bayesian S, with the actual percent difference?
- How did Bayesian B=128 compare with best baseline B=256?
- Were 90% posterior predictive intervals calibrated? What was their typical width?
- Is continued research justified, a narrower pivot justified, or should the project stop?

## 1. Environment and reproducibility

| Property | Observed value |
|---|---|
| GPU and VRAM | ... |
| NVIDIA driver / torch CUDA build | ... |
| Python / uv / PyTorch / TransformerLens / MLflow | ... |
| Git / lockfile / protocol SHA | ... |
| MIB / shapiq / external repository SHAs | ... |
| GPT-2 Small weights revision | ... |
| MLflow experiment names and run IDs | ... |

Describe SSH/tmux/MLflow environment, artifact location, exact commands, and any run failures or OOM events.

## 2. Method summary

Document mask meaning and hook location, Bayesian sparse model, posterior approximation, acquisition function, candidate circuit generation, and comparison methods. Explain approximate steps that departed from the theoretical design. Mention whether context-specific modeling was implemented on A2 and whether the A3 misspecification rule was frozen.

## 3. Gate A — Controlled causal worlds

Table by A0/A1/A2/A3 and method. Include 20-world recovery fractions, true interaction recovery, number of alternative minimal circuits, held-out prediction metrics, nominal 90% interval coverage and interval width, A3 detection, and all 5 criterion outcomes.

**Gate A verdict:** ...; why ...

**Failure analysis:** Which response patterns or masks caused failure? Was the posterior genuinely multimodal or simply diffuse?

## 4. Gate B — Planted XOR transformer

Describe actual architecture, construction methodology, 10 valid models, construction attempts/rejections, clean and intervention accuracies, checks against shortcuts, and masks/experiment budgets.

Provide per-instance recovery by B=32/64/96/128, all alternative circuits, median time-to-recovery and censoring for each baseline, high-confidence mistakes and resource costs.

**Gate B verdict:** ...; why ...

**Failure analysis:** Were effects not sparse, the active policy too expensive, or the planted mechanism too easy for non-Bayesian baselines?

## 5. Gate C — GPT-2 Small IOI

State model checkpoint, screening method, frozen 48 candidates and hash, screening/discovery/test split hashes, prompt/token policy and clean vs ablate-all candidate effect sanity check. Describe all masking semantics.

### Primary results

| Method | S(64) | S(128) | S(256) | GPU seconds @128 | End-to-end elapsed @128 |
|---|---:|---:|---:|---:|---:|
| First-order Lasso | ... | ... | ... | ... | ... |
| Pairwise Lasso | ... | ... | ... | ... | ... |
| GBDT | ... | ... | ... | ... | ... |
| Bayesian static | ... | ... | ... | ... | ... |
| Bayesian variance acquisition | ... | ... | ... | ... | ... |
| Bayesian decision-focused | ... | ... | ... | ... | ... |

Also provide faithfulness–size graphs for k={8,16,24}, individual 5-seed results, and the statistical confidence interval on the paired improvement.

### Held-out intervention prediction

Table held-out mask MAE, 90% coverage, interval width, proper scoring rule and density-stratified results. Examine whether circuit predictions generalize to aggressive masks and held-out template families.

### Scientific limitations

Explicitly distinguish node-level masked head localization from full directed causal circuit discovery and from MIB edge-patching leaderboard metrics.

**Gate C verdict:** ...; include all C-1 to C-5 results and failures.

## 6. Ablations and scientific interpretation

- Bayesian static vs pairwise Lasso: is Bayesian sparsity actually useful?
- Bayesian active vs Bayesian static: does active query selection save interventions?
- Decision-focused vs variance-based active: is the proposed acquisition rule doing real work?
- ProxySPEX interaction attribution/GBDT: are the same interactions already accessible more cheaply?
- Additive control vs redundancy vs context switching vs misspecification: is the method robust or only winning on its generative assumptions?
- Are alternative minimal circuit explanations distinguishable under the queried intervention family?

Provide concrete counterexamples if the method fails; negative results are scientifically important.

## 7. Mandatory Go/No-Go matrix

| Criterion | Frozen threshold | Observed | Status | Supporting MLflow run IDs |
|---|---|---|---|---|
| A-1 | >=16/20 A1 pair recovery | ... | ... | ... |
| A-2 | >=16/20 recover >=3/4 alternatives and no false high-conf circuits | ... | ... | ... |
| A-3 | A0 MAE <=1.10x first-order Lasso | ... | ... | ... |
| A-4 | 90% predictive coverage within [85%,95%] | ... | ... | ... |
| A-5 | A3 coverage or detection >=16/20 with <=12/60 false alarms | ... | ... | ... |
| B-1 | Recovery >=8/10 & >=3/4 alternatives | ... | ... | ... |
| B-2 | Median mask budget <=0.75x strongest non-Bayesian | ... | ... | ... |
| B-3 | Median mask budget <=0.80x Bayesian static | ... | ... | ... |
| B-4 | Zero invalid >=95%-confidence circuit claims | ... | ... | ... |
| C-1 | >=15% better S@128 and >=4/5 seed wins | ... | ... | ... |
| C-2 | Bayesian S@128 <= best non-Bayes S@256 | ... | ... | ... |
| C-3 | 90% predictive coverage within [85%,95%] | ... | ... | ... |
| C-4 | Positive lower bound of 90% paired bootstrap interval | ... | ... | ... |
| C-5 | <=2x end-to-end compute at B=128 | ... | ... | ... |

**Formal overall verdict:** ...

## 8. Caveats, deviations and confidence in the result

Every pre/post-registration change, incompatible benchmark, failed run, missing baseline, hyperparameter deviation, seed exclusion, reconstruction attempt, inability to reproduce prior work, GPU memory intervention, and change to test prompt composition belongs here. Point to `docs/07_DECISIONS_AND_DEVIATIONS.md` and relevant MLflow artifacts.

## 9. Recommendation for the next step

Choose exactly one:

1. **GO:** support a larger study, with specific method refinement and next experiment.
2. **NO_GO:** stop this algorithmic direction, state the substantive failure and whether a narrow uncertainty-only or acquisition-only pivot remains interesting.
3. **INCONCLUSIVE/BLOCKED:** specify precisely which missing scientific/infrastructure evidence prevents a decision and the smallest valid fix.

## 10. Reproduction appendix

Provide exact `uv sync --frozen`, MLflow, tmux and `python -m bayescircuits.cli ...` commands, git revision and data/model download commands, plus known compatibility caveats. Supply pointers to manifests, figures, raw prediction artifacts, and final decision JSON.
