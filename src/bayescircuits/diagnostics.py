"""Supplementary diagnostics (never part of a gate criterion).

* Official shapiq.ProxySPEX interaction attribution on the 20 confirmatory A1 worlds
  (own 128 coalitions each): are the two planted pairs its top-2 order-2 interactions?
* ProxySPEX on GPT-2 IOI (own 128 coalitions x 32 discovery prompts per Gate C seed):
  top pairwise interactions, compared with the pairwise features carrying the highest
  posterior inclusion probability in bayes_decision@128.
"""

from __future__ import annotations

import json
import time

import numpy as np

from .config import PROJECT_ROOT
from .models.proxyspex import run_proxyspex
from .oracle.synthetic import SyntheticWorld
from .provenance import provenance, write_json

OUT = PROJECT_ROOT / "runs" / "diagnostics"


def proxyspex_a1(cfg: dict) -> dict:
    res = []
    for seed in cfg["seeds"]["gate_a_confirmatory"]["A1"]:
        w = SyntheticWorld.create("A1", seed)
        px = run_proxyspex(32, lambda M, w=w: w.evaluate(M), 128, seed)
        top2 = {tuple(sorted(x["pair"])) for x in px["top_pairs"][:2]}
        res.append({"seed": seed, "top2_correct": top2 == set(w.true_pairs()), "oracle_calls": px["oracle_calls"],
                    "wall_seconds": px["wall_seconds"], "top_pairs": px["top_pairs"][:4]})
    return {"worlds": res, "n_top2_correct": int(sum(r["top2_correct"] for r in res)), "n": len(res)}


def proxyspex_c(cfg: dict) -> dict:
    import torch

    from . import gate_c as G

    if not torch.cuda.is_available():
        return {"status": "BLOCKED", "reason": "no GPU"}
    model, _ = G.load_model("cuda")
    ps, _ = G.build_splits(model, cfg, "cuda")
    cand = json.loads((G.FROZEN / "candidate_heads.json").read_text())
    n = cfg["gate_c"]["discovery"]["n_prompts_per_mask"]
    out = []
    for si, seed in enumerate(cfg["seeds"]["gate_c_discovery"]):
        dps = G.to_promptset(f"disc-{seed}", ps["discovery"].items[si * n : (si + 1) * n], "cuda")
        runner = G.make_runner(model, cand["heads"], dps, cfg["_impl"])
        px = run_proxyspex(48, lambda M, r=runner: np.mean(np.abs(r.effects(M)), 1), 128, seed)
        post = json.loads((PROJECT_ROOT / "runs" / "gate_c" / f"seed-{seed}" / "posterior_summaries.json").read_text())
        bayes_pairs = [x for x in post.get("bayes_decision@128", {}).get("top_inclusion", []) if "*" in x["feature"]]
        out.append({"seed": seed, "oracle_calls": px["oracle_calls"], "wall_seconds": px["wall_seconds"], "top_pairs": px["top_pairs"][:8],
                    "bayes_decision_top_pair_features": bayes_pairs[:8]})
    return {"seeds": out}


def discovery_vs_blind_R(cfg: dict) -> dict:
    """Exploratory: for every sealed Gate C circuit, measure R on the seed's own 32 discovery prompts
    (real GPT-2 interventions) and compare with blind R. Separates surrogate error from
    discovery-to-blind prompt/template transfer. Never feeds back into any decision."""
    import pandas as pd
    import torch

    from . import gate_c as G
    from .masks import retain_mask

    if not torch.cuda.is_available():
        return {"status": "BLOCKED", "reason": "no GPU"}
    model, _ = G.load_model("cuda")
    ps, _ = G.build_splits(model, cfg, "cuda")
    cand = json.loads((G.FROZEN / "candidate_heads.json").read_text())
    n = cfg["gate_c"]["discovery"]["n_prompts_per_mask"]
    rows = []
    for si, seed in enumerate(cfg["seeds"]["gate_c_discovery"]):
        dps = G.to_promptset(f"disc-{seed}", ps["discovery"].items[si * n : (si + 1) * n], "cuda")
        runner = G.make_runner(model, cand["heads"], dps, cfg["_impl"])
        den = float(np.mean(np.abs(runner.effects(np.ones((1, 48), np.uint8))[0])))
        f = pd.read_csv(PROJECT_ROOT / "runs" / "gate_c" / f"seed-{seed}" / "faithfulness.csv")
        for _, r in f.iterrows():
            C = json.loads(r.circuit)
            num = float(np.mean(np.abs(runner.effects(retain_mask(48, C)[None])[0])))
            rows.append({"seed": seed, "method": r.method, "budget": int(r.budget), "k": int(r.k), "R_blind": float(r.R),
                         "R_discovery_prompts": num / den, "pred_R": float(r.pred_R)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gate_c_discovery_vs_blind_R.csv", index=False)
    summ = df.groupby(["method", "budget"])[["pred_R", "R_discovery_prompts", "R_blind"]].mean().reset_index()
    return {"summary": summ.to_dict(orient="records"),
            "corr_disc_blind": float(np.corrcoef(df.R_discovery_prompts, df.R_blind)[0, 1])}


def run_diagnostics(cfg: dict, which: str = "all") -> int:
    from . import tracking
    from .cli import _finish_parent, _gate_parent

    prov = provenance(cfg)
    pid = _gate_parent("bccd/aggregate", "diagnostics" if which == "all" else f"diagnostics-{which}", prov, True, "parent_run_id")
    t0 = time.time()
    if which in ("all", "proxyspex"):
        res = {"proxyspex_A1": proxyspex_a1(cfg)}
        try:
            res["proxyspex_C"] = proxyspex_c(cfg)
        except Exception as e:  # diagnostic only; record failure faithfully
            res["proxyspex_C"] = {"status": "FAILED", "error": repr(e)}
        res["seconds"] = time.time() - t0
        write_json(OUT / "proxyspex.json", res)
        _finish_parent(pid, "FINISHED", {"proxyspex.json": res}, metrics={"proxyspex_A1_top2_correct": res["proxyspex_A1"]["n_top2_correct"]})
        print(json.dumps({"A1_top2_correct": f"{res['proxyspex_A1']['n_top2_correct']}/{res['proxyspex_A1']['n']}"}))
    if which in ("all", "transfer"):
        tr = discovery_vs_blind_R(cfg)
        write_json(OUT / "gate_c_transfer.json", tr)
        _finish_parent(pid, "FINISHED", {"gate_c_transfer.json": tr})
        print(json.dumps({"corr_disc_blind": tr.get("corr_disc_blind")}))
    _ = tracking
    return 0
