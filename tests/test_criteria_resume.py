import json

import numpy as np
import pandas as pd

from bayescircuits.config import load_config
from bayescircuits.criteria import gate_a_criteria, gate_verdict, overall_verdict
from bayescircuits.criteria_c import hierarchical_bootstrap


def test_missing_evidence_never_go():
    assert overall_verdict({"A": "PASS", "B": "PASS", "C": "NOT_RUN"}) != "GO"
    assert overall_verdict({"A": "PASS", "B": "BLOCKED", "C": "PASS"}) == "BLOCKED"
    assert overall_verdict({"A": "FAIL", "B": "PASS", "C": "PASS"}) == "NO_GO"
    assert overall_verdict({"A": "PASS", "B": "PASS", "C": "PASS"}) == "GO"
    assert gate_verdict({"x": {"status": "PASS"}, "y": {"status": "NOT_RUN"}}) == "NOT_RUN"
    assert gate_verdict({"x": {"status": "PASS"}, "y": {"status": "FAIL"}}) == "FAIL"


def test_gate_a_criteria_refuses_incomplete_tables():
    cfg = load_config("configs/preregistration.yaml")
    rows = pd.DataFrame({"split": ["confirmatory"], "budget": [128], "regime": ["A1"], "method": ["bayes_decision"]})
    crit = gate_a_criteria(rows, cfg, 0.01)
    assert all(v["status"] == "NOT_RUN" for v in crit.values())


def test_hierarchical_bootstrap_detects_paired_improvement():
    rng = np.random.default_rng(0)
    seeds = [1, 2, 3, 4, 5]
    fam = np.repeat(np.arange(8), 16)
    pp = {}
    for s in seeds:
        den = 1.0 + rng.random(128)
        d = {"__denominator__": den}
        for k in (8, 16, 24):
            ref = den * (0.5 + 0.05 * rng.random(128))
            d[f"bayes_decision__128__{k}"] = 0.7 * ref
            d[f"lasso_pair__128__{k}"] = ref
        pp[s] = d
    out = hierarchical_bootstrap(pp, seeds, fam, "lasso_pair", 128, [8, 16, 24], 300, 1)
    assert out["ci90"][0] > 0 and out["n_families"] == 8


def test_world_rerun_is_idempotent(tmp_path):
    from bayescircuits.gate_a import run_world

    cfg = load_config("configs/preregistration.yaml")
    d = tmp_path / "smoke" / "A0-90001"
    d.mkdir(parents=True)
    st = {"status": "complete", "regime": "A0", "seed": 90001, "dir": str(d)}
    (d / "status.json").write_text(json.dumps(st))
    assert run_world("A0", 90001, cfg, str(tmp_path), "smoke") == st  # no recomputation, no duplicate queries
