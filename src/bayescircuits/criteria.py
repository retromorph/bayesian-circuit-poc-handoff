"""Formal adjudication of the pre-registered criteria from saved per-unit tables.

These functions only read evaluation tables produced after discovery snapshots were sealed.
They never fill missing values with zero: missing evidence yields NOT_RUN / BLOCKED.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .evaluate import wilson

CANDIDATE = "bayes_decision"


def _status(ok: bool | None) -> str:
    if ok is None:
        return "INCONCLUSIVE"
    return "PASS" if ok else "FAIL"


def gate_a_criteria(rows: pd.DataFrame, cfg: dict, misfit_threshold: float | None) -> dict:
    ga = cfg["gate_a"]
    cr = ga["criteria"]
    lo, hi = cfg["shared"]["accepted_coverage_interval"]
    B = cr["A-1"]["checkpoint_masks"]
    r = rows[(rows.split == "confirmatory") & (rows.budget == B)]
    out: dict = {}
    n_expected = ga["n_confirmatory_worlds_per_regime"]

    def sub(reg, method=CANDIDATE):
        return r[(r.regime == reg) & (r.method == method)]

    complete = all(len(sub(reg)) == n_expected for reg in ga["regimes"])
    if not complete:
        for c in ["A-1", "A-2", "A-3", "A-4", "A-5"]:
            out[c] = {"status": "NOT_RUN", "observed": "incomplete confirmatory worlds"}
        return out
    a1 = sub("A1")
    k1 = int(a1.top2_pairs_correct.astype(bool).sum())
    out["A-1"] = {
        "status": _status(k1 >= cr["A-1"]["required_successful_A1_worlds"]),
        "observed": f"{k1}/{len(a1)} A1 worlds with top-2 nonadditivity pairs = planted pairs",
        "value": k1,
        "wilson95": wilson(k1, len(a1)),
        "threshold": f">= {cr['A-1']['required_successful_A1_worlds']}/20",
    }
    k2 = int((a1.circuits_recovered >= cr["A-2"]["min_true_alternative_circuits"]).sum())
    inv = int(a1.invalid_high_conf_claims.sum())
    inv_all = int(rows[(rows.split == "confirmatory") & (rows.regime == "A1") & (rows.method == CANDIDATE)].invalid_high_conf_claims.sum())
    out["A-2"] = {
        "status": _status(k2 >= cr["A-2"]["required_successful_A1_worlds"] and inv <= cr["A-2"]["max_invalid_high_confidence_claims"]),
        "observed": f"{k2}/20 worlds recover >=3/4 minimal circuits; invalid high-confidence claims at B=128: {inv} (all checkpoints: {inv_all})",
        "value": k2,
        "invalid_claims_B128": inv,
        "invalid_claims_all_checkpoints": inv_all,
        "wilson95": wilson(k2, len(a1)),
        "threshold": ">= 16/20 and 0 invalid claims",
    }
    a0b, a0l = sub("A0"), sub("A0", "lasso_1st")
    mae_b, mae_l = float(a0b.test_mae.mean()), float(a0l.test_mae.mean())
    nmae_b, nmae_l = float(a0b.test_mae_normalized.mean()), float(a0l.test_mae_normalized.mean())
    fb = cfg["_impl"]["gate_a"]
    if nmae_l < fb["a3_zero_mae_fallback_lasso_normalized_mae"]:
        ok3 = nmae_b <= fb["a3_zero_mae_fallback_abs_tolerance"]
        rule = "zero-error fallback (normalised MAE <= 0.01)"
    else:
        ok3 = mae_b <= cr["A-3"]["additive_mae_max_ratio_to_first_order_lasso"] * mae_l
        rule = "ratio <= 1.10"
    out["A-3"] = {
        "status": _status(ok3),
        "observed": f"A0 MAE bayes={mae_b:.5f} lasso_1st={mae_l:.5f} ratio={mae_b / mae_l if mae_l > 0 else float('inf'):.3f}; normalised {nmae_b:.5f} vs {nmae_l:.5f}; rule: {rule}",
        "mae_bayes": mae_b,
        "mae_lasso_1st": mae_l,
        "ratio": mae_b / mae_l if mae_l > 0 else None,
        "rule": rule,
    }
    pooled = pd.concat([sub(reg) for reg in cr["A-4"]["pooled_regimes"]])
    cov = float((pooled.coverage_90 * pooled.n_targets).sum() / pooled.n_targets.sum())
    per_reg = {reg: float((sub(reg).coverage_90 * sub(reg).n_targets).sum() / sub(reg).n_targets.sum()) for reg in ga["regimes"]}
    out["A-4"] = {
        "status": _status(lo <= cov <= hi),
        "observed": f"pooled A0-A2 coverage {cov:.4f}; per regime " + ", ".join(f"{k}={v:.3f}" for k, v in per_reg.items()),
        "value": cov,
        "per_regime": per_reg,
        "mean_width": float(pooled.interval_width_90.mean()),
        "threshold": f"[{lo}, {hi}]",
    }
    a3 = sub("A3")
    cov3 = per_reg["A3"]
    cov_ok = lo <= cov3 <= hi
    if misfit_threshold is None or a3.misfit_stat.isna().any():
        det = None
        det_txt = "detector unavailable"
    else:
        flags3 = int((a3.misfit_stat > misfit_threshold).sum())
        ctrl = pd.concat([sub(reg) for reg in ["A0", "A1", "A2"]])
        fp = int((ctrl.misfit_stat > misfit_threshold).sum())
        det = flags3 >= cr["A-5"]["alternative_min_detector_success_worlds"] and fp <= cr["A-5"]["max_misfit_false_positive_worlds_on_A0_to_A2"]
        det_txt = f"detector flags A3 {flags3}/20, false positives {fp}/60 (t={misfit_threshold:.4g})"
    out["A-5"] = {
        "status": _status(cov_ok or bool(det)),
        "observed": f"A3 coverage {cov3:.4f}; {det_txt}",
        "coverage_A3": cov3,
        "detector_pass": det,
        "threshold": "coverage in [0.85,0.95] OR (>=16/20 flags AND <=12/60 false positives)",
    }
    return out


def gate_verdict(criteria: dict) -> str:
    st = [v["status"] for v in criteria.values()]
    if not st:
        return "NOT_RUN"
    if all(s == "PASS" for s in st):
        return "PASS"
    if any(s == "FAIL" for s in st):
        return "FAIL"
    for s in ("BLOCKED", "INCONCLUSIVE", "NOT_RUN"):
        if s in st:
            return s
    return "INCONCLUSIVE"


def overall_verdict(gates: dict[str, str]) -> str:
    vals = list(gates.values())
    if all(v == "PASS" for v in vals):
        return "GO"
    if any(v == "FAIL" for v in vals):
        return "NO_GO"
    if any(v == "BLOCKED" for v in vals):
        return "BLOCKED"
    return "INCONCLUSIVE"


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def dumps(o) -> str:
    return json.dumps(jsonable(o), indent=2)
