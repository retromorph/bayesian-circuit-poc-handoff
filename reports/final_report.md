# Bayesian Causal Circuit Discovery — Go/No-Go Pilot Report

**Date (UTC):** 2026-10-18T22:14:57Z  
**Protocol:** BCCD-PILOT-1.0 (protocol doc SHA-256 `fe2a3d4f77aaa69f2684267c54980119524e99c8f9a04db456f4d325f9d57986`; config SHA-256 `c2102e71fc7e1cfe2b922103bb5e4c35f643dc2e6b5b1a6a48132453782890ec`)  
**Git revision / uv lock:** `3ebde28f7fe62e5684063a1cd1c7d1b9dc73ae31` (dirty=False) / `f05cad05dc522932c1a36df061d6970970852bb48f1162bd2769de8284a6a759`  
**Host / GPU / driver / torch:** Linux-7.0.0-15-generic-x86_64-with-glibc2.43 / NVIDIA RTX A4000, 595.71.05, 16376 MiB, 4 MiB / torch 2.8.0+cu126 (CUDA build 12.6)  
**Overall verdict:** **NO_GO** (Gate A FAIL, Gate B FAIL, Gate C FAIL)  
**MLflow tracking:** remote `http://127.0.0.1:5001` via SSH tunnel; parent runs {"gate_a_parent": "d6ebe797a95d48c0ba72674736cdfb36", "gate_b_parent": "221e31a4735542b6bdf8320d6a947b7f", "gate_c_parent": "c8c20c5ea1804702a85b57b49552c364", "gate_c_prepare_parent": "69acae6453904987aa681553f90bf9fc", "gate_a_dev_parent": "3c2a5918fca143269f369ee682219ab2"}


**Run-time provenance per gate** (each gate's summary records the code revision it actually ran with):

| Gate | git SHA at run | dirty | impl-config SHA-256 | uv.lock SHA-256 | parent MLflow run |
|---|---|---|---|---|---|
| A | `165805538ba41139336cc17563c8d4f14568f913` | False | `7e48e9b5eadb4d72…` | `f05cad05dc522932…` | `d6ebe797a95d48c0ba72674736cdfb36` |
| B | `f8b232f2d88a538711561ec7981104f569df1c4a` | False | `7e48e9b5eadb4d72…` | `f05cad05dc522932…` | `221e31a4735542b6bdf8320d6a947b7f` |
| C | `6b43b7f37a19c2d1729f21b4d1ebc7c72fe5076b` | False | `177e80fcf1f24483…` | `f05cad05dc522932…` | `c8c20c5ea1804702a85b57b49552c364` |


## Appendix A — Gate A numerical results (B=128, 20 confirmatory worlds per regime)

| Regime | Method | MAE | norm. MAE | 90% coverage | mean width | top-2 pairs (frac) | worlds >=3/4 circuits | invalid claims | misfit flags | CPU s (e2e) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | bayes_decision | 0.0009 | 0.0014 | 1.000 | 0.083 | — | 0 | 0 | 0 | 5.69 |
| A0 | bayes_static | 0.0007 | 0.0011 | 1.000 | 0.083 | — | 0 | 0 | 0 | 0.64 |
| A0 | bayes_uncertainty | 0.0005 | 0.0007 | 1.000 | 0.083 | — | 0 | 0 | 0 | 5.38 |
| A0 | gbdt | 0.0102 | 0.0155 | — | — | — | 0 | — | — | 0.67 |
| A0 | lasso_1st | 0.0002 | 0.0003 | — | — | — | 0 | — | — | 0.05 |
| A0 | lasso_pair | 0.0002 | 0.0003 | — | — | — | 0 | — | — | 0.20 |
| A1 | bayes_decision | 0.0005 | 0.0005 | 1.000 | 0.093 | 1.00 | 20 | 0 | 0 | 5.72 |
| A1 | bayes_static | 0.0005 | 0.0005 | 1.000 | 0.093 | 1.00 | 20 | 0 | 0 | 0.65 |
| A1 | bayes_uncertainty | 0.0003 | 0.0003 | 1.000 | 0.093 | 1.00 | 20 | 0 | 0 | 5.44 |
| A1 | gbdt | 0.0774 | 0.0771 | — | — | 1.00 | 4 | — | — | 0.73 |
| A1 | lasso_1st | 0.2817 | 0.2802 | — | — | 0.00 | 0 | — | — | 0.09 |
| A1 | lasso_pair | 0.0003 | 0.0003 | — | — | 1.00 | 20 | — | — | 0.23 |
| A2 | bayes_decision | 0.0007 | 0.0009 | 1.000 | 0.081 | — | 0 | 0 | 0 | 8.03 |
| A2 | bayes_static | 0.0007 | 0.0009 | 1.000 | 0.081 | — | 0 | 0 | 0 | 0.97 |
| A2 | bayes_uncertainty | 0.0004 | 0.0005 | 1.000 | 0.081 | — | 0 | 0 | 0 | 7.75 |
| A2 | gbdt | 0.0831 | 0.1038 | — | — | — | 0 | — | — | 0.83 |
| A2 | lasso_1st | 0.2085 | 0.2605 | — | — | — | 0 | — | — | 0.27 |
| A2 | lasso_pair | 0.0003 | 0.0003 | — | — | — | 0 | — | — | 0.98 |
| A3 | bayes_decision | 0.0947 | 0.0992 | 0.857 | 0.358 | — | 0 | 0 | 20 | 5.94 |
| A3 | bayes_static | 0.0855 | 0.0892 | 0.867 | 0.343 | — | 0 | 0 | 20 | 0.67 |
| A3 | bayes_uncertainty | 0.0848 | 0.0889 | 0.957 | 0.385 | — | 0 | 0 | 20 | 5.54 |
| A3 | gbdt | 0.1083 | 0.1129 | — | — | — | 0 | — | — | 0.74 |
| A3 | lasso_1st | 0.2998 | 0.3133 | — | — | — | 0 | — | — | 0.09 |
| A3 | lasso_pair | 0.1003 | 0.1048 | — | — | — | 0 | — | — | 13.41 |

Misfit threshold (frozen from development worlds): `{"threshold": 0.016224898824084153, "rule": "geometric midpoint", "dev_control_max": 0.0026257347819119, "dev_A3_min": 0.10025663812858}`

![Gate A](figures/gate_a_overview.png)


## Appendix B — Gate B numerical results

Event time = first checkpoint with >=3/4 planted size-2 circuits among the (<=4) proposals and >=1 blind-faithful proposal; 160 = censored.

| Model seed | lasso_1st | lasso_pair | gbdt | bayes_static | bayes_uncertainty | bayes_decision |
|---|---:|---:|---:|---:|---:|---:|
| 20000 | 160 | 160 | 160 | 160 | 160 | 128 |
| 20001 | 160 | 160 | 160 | 160 | 96 | 160 |
| 20002 | 160 | 160 | 160 | 160 | 160 | 160 |
| 20003 | 160 | 160 | 160 | 160 | 160 | 160 |
| 20004 | 160 | 160 | 160 | 160 | 160 | 160 |
| 20005 | 160 | 64 | 160 | 96 | 128 | 160 |
| 20006 | 160 | 160 | 160 | 160 | 160 | 160 |
| 20007 | 160 | 160 | 128 | 128 | 160 | 160 |
| 20008 | 160 | 160 | 160 | 160 | 128 | 160 |
| 20009 | 160 | 160 | 160 | 160 | 160 | 160 |
| **median** | **160** | **160** | **160** | **160** | **160** | **160** |
| censored | 10 | 9 | 9 | 8 | 7 | 9 |

Construction: {"n_attempts": 10, "n_valid": 10, "valid_seeds": [20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009], "status": "ok"}

ProxySPEX (official shapiq, FBII, own 128 coalitions) top-2 pair retrieval per model: {"model-20000": {"top2_pairs_correct": true, "oracle_calls": 128}, "model-20001": {"top2_pairs_correct": true, "oracle_calls": 128}, "model-20002": {"top2_pairs_correct": true, "oracle_calls": 128}, "model-20003": {"top2_pairs_correct": true, "oracle_calls": 128}, "model-20004": {"top2_pairs_correct": true, "oracle_calls": 128}, "model-20005": {"top2_pairs_correct": true, "oracle_calls": 128}, "model-20006": {"top2_pairs_correct": true, "oracle_calls": 128}, "model-20007": {"top2_pairs_correct": true, "oracle_calls": 128}, "model-20008": {"top2_pairs_correct": true, "oracle_calls": 128}, "model-20009": {"top2_pairs_correct": true, "oracle_calls": 128}}

![Gate B](figures/gate_b_overview.png)


## Appendix C — Gate C numerical results

| Method | S(64) | S(128) | S(256) | GPU oracle s @128 | CPU fit s @128 | CPU acq s @128 | End-to-end s @128 (excl. screening) |
|---|---:|---:|---:|---:|---:|---:|---:|
| lasso_1st | 0.4974 | 0.4208 | 0.3885 | 2.93 | 0.18 | 0.00 | 3.01 |
| lasso_pair | 0.5059 | 0.3848 | 0.3628 | 2.94 | 2.91 | 0.00 | 5.04 |
| gbdt | 0.4189 | 0.4540 | 0.4180 | 2.93 | 1.35 | 0.00 | 3.80 |
| bayes_static | 0.5121 | 0.3969 | 0.4111 | 2.93 | 2.51 | 0.00 | 3.73 |
| bayes_uncertainty | 0.5276 | 0.4777 | 0.3707 | 2.92 | 8.74 | 0.55 | 12.38 |
| bayes_decision | 0.5011 | 0.5108 | 0.3920 | 2.91 | 8.73 | 1.02 | 12.88 |

Per-seed S (rows: method, budget):

| method | B | 31001 | 31002 | 31003 | 31004 | 31005 |
|---|---:|---:|---:|---:|---:|---:|
| bayes_decision | 64 | 0.3927 | 0.5041 | 0.5021 | 0.5134 | 0.5933 |
| bayes_decision | 128 | 0.4950 | 0.4599 | 0.5669 | 0.5209 | 0.5112 |
| bayes_decision | 256 | 0.4210 | 0.3937 | 0.3629 | 0.3187 | 0.4637 |
| bayes_static | 64 | 0.4548 | 0.4640 | 0.5593 | 0.4460 | 0.6364 |
| bayes_static | 128 | 0.3753 | 0.3528 | 0.5290 | 0.3576 | 0.3699 |
| bayes_static | 256 | 0.4344 | 0.4176 | 0.3523 | 0.4801 | 0.3712 |
| bayes_uncertainty | 64 | 0.4582 | 0.4278 | 0.5327 | 0.7122 | 0.5070 |
| bayes_uncertainty | 128 | 0.5204 | 0.4174 | 0.5128 | 0.4460 | 0.4921 |
| bayes_uncertainty | 256 | 0.4671 | 0.3751 | 0.3613 | 0.3566 | 0.2934 |
| gbdt | 64 | 0.4207 | 0.4539 | 0.4106 | 0.4866 | 0.3226 |
| gbdt | 128 | 0.3924 | 0.5596 | 0.4814 | 0.3834 | 0.4534 |
| gbdt | 256 | 0.3850 | 0.3709 | 0.4323 | 0.4801 | 0.4216 |
| lasso_1st | 64 | 0.3598 | 0.5009 | 0.5424 | 0.4168 | 0.6670 |
| lasso_1st | 128 | 0.3087 | 0.5167 | 0.4588 | 0.3294 | 0.4905 |
| lasso_1st | 256 | 0.3616 | 0.4185 | 0.3750 | 0.4640 | 0.3235 |
| lasso_pair | 64 | 0.5188 | 0.5041 | 0.4782 | 0.4633 | 0.5651 |
| lasso_pair | 128 | 0.3679 | 0.4368 | 0.4090 | 0.3287 | 0.3818 |
| lasso_pair | 256 | 0.4043 | 0.4100 | 0.3696 | 0.3141 | 0.3159 |

![Gate C](figures/gate_c_overview.png)

![Calibration](figures/gate_c_calibration.png)

![Costs](figures/gate_c_costs.png)


## Mandatory Go/No-Go matrix (recomputed from saved artifacts)

| Criterion | Frozen threshold | Observed | Status | Supporting MLflow run IDs |
|---|---|---|---|---|
| A-1 | >=16/20 A1 worlds: top-2 predicted nonadditivity pairs = planted pairs (B=128) | 20/20 A1 worlds with top-2 nonadditivity pairs = planted pairs | **PASS** | parent `d6ebe797a95d48c0ba72674736cdfb36` |
| A-2 | >=16/20 A1 worlds recover >=3/4 exact size-3 minimal circuits; 0 invalid P>=0.95 claims | 20/20 worlds recover >=3/4 minimal circuits; invalid high-confidence claims at B=128: 0 (all checkpoints: 0) | **PASS** | parent `d6ebe797a95d48c0ba72674736cdfb36` |
| A-3 | A0 Bayes MAE <= 1.10 x first-order Lasso (zero-error fallback: normalised MAE <= 0.01) | A0 MAE bayes=0.00093 lasso_1st=0.00018 ratio=5.162; normalised 0.00143 vs 0.00028; rule: zero-error fallback (normalised MAE <= 0.01) | **PASS** | parent `d6ebe797a95d48c0ba72674736cdfb36` |
| A-4 | pooled A0-A2 90% predictive coverage in [0.85, 0.95] | pooled A0-A2 coverage 1.0000; per regime A0=1.000, A1=1.000, A2=1.000, A3=0.857 | **FAIL** | parent `d6ebe797a95d48c0ba72674736cdfb36` |
| A-5 | A3 coverage in [0.85,0.95] OR misfit flags >=16/20 with <=12/60 A0-A2 false positives | A3 coverage 0.8568; detector flags A3 20/20, false positives 0/60 (t=0.01622) | **PASS** | parent `d6ebe797a95d48c0ba72674736cdfb36` |
| B-1 | >=8/10 valid models: >=3/4 planted size-2 circuits + a blind-faithful proposal by B=128 | 1/10 models recovered >=3/4 planted size-2 circuits with a blind-faithful proposal by B=128 | **FAIL** | parent `221e31a4735542b6bdf8320d6a947b7f` |
| B-2 | median masks-to-recovery <= 0.75 x best non-Bayesian median (censored at 160) | median event time bayes_decision=160.0 vs best non-Bayes lasso_1st=160.0 (ratio 1.000); censored counts {'bayes_decision': 9, 'bayes_static': 8, 'bayes_uncertainty': 7, 'gbdt': 9, 'lasso_1st': 10, 'lasso_pair': 9} | **FAIL** | parent `221e31a4735542b6bdf8320d6a947b7f` |
| B-3 | median masks-to-recovery <= 0.80 x bayes_static median | median bayes_decision=160.0 vs bayes_static=160.0 (ratio 1.000) | **FAIL** | parent `221e31a4735542b6bdf8320d6a947b7f` |
| B-4 | 0 invalid high-confidence (P>=0.95) circuit claims | 78 invalid of 84 high-confidence (P>=0.95) claims across all checkpoints/models | **FAIL** | parent `221e31a4735542b6bdf8320d6a947b7f` |
| C-1 | S_Bayes(128) <= 0.85 x best non-Bayes S(128) pooled AND >=15% gain in >=4/5 seeds | S_bayes(128)=0.5108, best non-Bayes lasso_pair S(128)=0.3848, ratio=1.327 (need <=0.85); seeds with >=15% gain: 0/5 | **FAIL** | parent `c8c20c5ea1804702a85b57b49552c364` |
| C-2 | S_Bayes(128) <= best non-Bayes S(256) | S_bayes(128)=0.5108 vs best non-Bayes S(256)=0.3628 (lasso_pair) | **FAIL** | parent `c8c20c5ea1804702a85b57b49552c364` |
| C-3 | 90% predictive coverage of blind mean intervention effects in [0.85, 0.95] | coverage 0.809 (mean width 2.584; CRPS 0.5579); by #ablated: 2:0.88/w=2.02, 10:0.72/w=2.41, 24:0.79/w=2.91, 43:0.86/w=3.00 | **FAIL** | parent `c8c20c5ea1804702a85b57b49552c364` |
| C-4 | paired hierarchical bootstrap 90% CI lower bound > 0 vs dev-preselected baseline | reference=lasso_pair (dev-preselected); paired 90% CI of S_ref-S_bayes = [-0.1701, -0.0758] | **FAIL** | parent `c8c20c5ea1804702a85b57b49552c364` |
| C-5 | end-to-end elapsed at B=128 <= 2 x strongest non-Bayesian baseline | end-to-end wall at B=128 (incl. 1.5s screening): bayes_decision 14.4s vs lasso_pair 6.5s (ratio 2.20) | **FAIL** | parent `c8c20c5ea1804702a85b57b49552c364` |


**Formal overall verdict: NO_GO** — rule: GO iff Gate A, B and C all PASS.


Runtime-summary consistency check (recomputed status == status written at run time): `{"A": {"A-1": true, "A-2": true, "A-3": true, "A-4": true, "A-5": true}, "B": {"B-1": true, "B-2": true, "B-3": true, "B-4": true}, "C": {"C-1": true, "C-2": true, "C-3": true, "C-4": true, "C-5": true}}`
