"""Gate B: planted redundant XOR transformers (construction, validation, discovery, audit)."""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import PROJECT_ROOT, run_root
from .discovery import METHODS, NON_BAYES, DecisionSpec, Problem, end_to_end_cost, run_method
from .evaluate import interval_metrics, point_metrics
from .gate_a import acq_spec, bayes_cfg
from .masks import balanced_masks, mask_key, masks_sha256, retain_mask
from .models.proxyspex import run_proxyspex
from .oracle.base import DiscoveryOracle, QueryResult
from .oracle.transformer import HeadAblationRunner, last_token_margin
from .planted import SEQ, build_instance, label_tokens, make_inputs, validate_instance
from .provenance import provenance, write_json


def model_checksum(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def construct_models(cfg: dict, device: str, out: Path, n_needed: int) -> dict:
    gb = cfg["gate_b"]
    v = dict(gb["construction_validity"])
    v["rmax"] = cfg["shared"]["faithful_R_max"]
    log = {"attempts": [], "valid_seeds": []}
    s0, s1 = cfg["seeds"]["gate_b_candidate_model_seed_start"], cfg["seeds"]["gate_b_candidate_model_seed_limit"]
    for seed in range(s0, s1):
        if len(log["valid_seeds"]) >= n_needed or len(log["attempts"]) >= gb["candidate_model_attempt_limit"]:
            break
        inst = build_instance(seed, device)
        res = validate_instance(inst, v, n_inputs=cfg["_impl"]["gate_b"]["n_validation_inputs"], device=device)
        res["model_sha256"] = model_checksum(inst.model)
        res["a_heads"], res["b_heads"] = inst.a_heads, inst.b_heads
        log["attempts"].append(res)
        if res["valid"]:
            log["valid_seeds"].append(seed)
    log["n_attempts"] = len(log["attempts"])
    log["n_valid"] = len(log["valid_seeds"])
    log["status"] = "ok" if log["n_valid"] >= n_needed else "BLOCKED_BY_BENCHMARK_CONSTRUCTION"
    write_json(out / "construction_log.json", log)
    return log


class _BlindEvaluator:
    """Evaluation-only access to blind inputs (never passed to discovery)."""

    def __init__(self, inst, seed: int, n_inputs: int, device: str):
        toks, y = make_inputs(np.random.default_rng([seed, 2]), n_inputs, device)
        c, w = label_tokens(y)
        self.runner = HeadAblationRunner(inst.model, inst.candidate_heads, toks, last_token_margin(c, w), torch.full((len(toks),), SEQ - 1, device=device))
        self.p = 24
        self.den = float(np.mean(np.abs(self.runner.effects(np.ones((1, 24), np.uint8))[0])))
        self.input_sha = hashlib.sha256(toks.cpu().numpy().tobytes()).hexdigest()
        self._cache: dict = {}

    def R(self, C) -> float:
        C = tuple(sorted(C))
        if C not in self._cache:
            d = self.runner.effects(retain_mask(self.p, C)[None])[0]
            self._cache[C] = float(np.mean(np.abs(d)) / self.den)
        return self._cache[C]

    def mask_targets(self, masks: np.ndarray) -> np.ndarray:
        d = self.runner.effects(masks)
        return np.mean(np.abs(d), axis=1)


def run_model(seed: int, cfg: dict, out: Path, device: str) -> dict:
    impl = cfg["_impl"]
    gb = cfg["gate_b"]
    udir = out / f"model-{seed}"
    st_path = udir / "status.json"
    if st_path.exists() and json.loads(st_path.read_text()).get("status") == "complete":
        return json.loads(st_path.read_text())
    udir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    inst = build_instance(seed, device)
    dseed = seed + cfg["seeds"]["gate_b_discovery_seed_offset"]
    p = 24
    dens = cfg["shared"]["intervention_density_mixture"]
    blind_masks = balanced_masks(np.random.default_rng([dseed, 11]), p, impl["gate_b"]["n_blind_masks"], dens)
    reserved = frozenset(mask_key(m) for m in blind_masks)
    warm = balanced_masks(np.random.default_rng([dseed, 22]), p, cfg["shared"]["initial_warm_start_masks"], dens, exclude=set(reserved))
    excl = set(reserved) | {mask_key(m) for m in warm}
    bmax = max(gb["checkpoints_mask_count"])
    static = balanced_masks(np.random.default_rng([dseed, 33]), p, bmax - len(warm), dens, exclude=excl)
    dtoks, dy = make_inputs(np.random.default_rng([dseed, 1]), gb["discovery_contexts_per_mask"], device)
    dc, dw = label_tokens(dy)
    disc_runner = HeadAblationRunner(inst.model, inst.candidate_heads, dtoks, last_token_margin(dc, dw), torch.full((len(dtoks),), SEQ - 1, device=device))

    def fn(M: np.ndarray) -> QueryResult:
        d = disc_runner.effects(M)
        return QueryResult(masks=M, y={"abs": np.mean(np.abs(d), 1)[:, None], "effect": np.mean(d, 1)[:, None]}, per_prompt=d)

    problem = Problem(
        unit_id=f"B-{seed}", p=p, contexts=[None], target="abs", warm_start=warm, static_masks=static,
        checkpoints=list(gb["checkpoints_mask_count"]), densities=list(dens),
        decision=DecisionSpec(mode="minimal", enum_sizes=list(impl["gate_b"]["enum_sizes"]), propose_sizes=list(impl["gate_b"]["propose_sizes"]),
                              rmax=cfg["shared"]["faithful_R_max"], lam=impl["decision"]["lam"], max_props=impl["decision"]["max_props"]),
        acq=acq_spec(impl), reserved_keys=reserved,
    )
    blind = _BlindEvaluator(inst, dseed, impl["gate_b"]["n_blind_inputs"], device)
    blind_T = blind.mask_targets(blind_masks)
    true_c = set(inst.true_circuits())
    true_pairs = set(inst.true_pairs())
    rows, decisions, queries = [], {}, []
    bcfg = bayes_cfg(impl)
    for method in METHODS:
        oracle = DiscoveryOracle(fn=fn, p=p, budget=bmax, method=method, unit_id=problem.unit_id,
                                 prompt_batch_id=f"xor16-{dseed}", reserved_keys=reserved, n_prompts_per_mask=len(dtoks))
        snaps = run_method(problem, method, oracle, dseed, bcfg)
        event = None
        for snap in snaps:  # snapshots are sealed; blind audit happens only now
            dec = snap.decision
            props = [tuple(sorted(c)) for c in dec.get("proposals", [])]
            props_R = [blind.R(c) for c in props]
            rec = len(true_c & set(props))
            any_faithful = any(r <= cfg["shared"]["faithful_R_max"] for r in props_R)
            success = rec >= gb["criteria"]["B-1"]["true_alternatives_required_each_success"] and any_faithful
            if success and event is None:
                event = snap.budget
            bayes = getattr(snap.surrogate, "is_bayesian", False)
            claims = [tuple(sorted(c)) for c in dec.get("high_conf_claims", [])]
            claims_R = [blind.R(c) for c in claims]
            row = {"model_seed": seed, "method": method, "budget": snap.budget, "circuits_recovered": rec,
                   "any_proposal_blind_faithful": any_faithful, "success": success, "n_proposals": len(props),
                   "proposals": json.dumps([list(c) for c in props]), "proposal_blind_R": json.dumps(props_R),
                   "n_high_conf_claims": len(claims) if bayes else None,
                   "invalid_high_conf_claims": int(sum(r > cfg["shared"]["faithful_R_max"] for r in claims_R)) if bayes else None,
                   "top2_pairs_correct": {tuple(x["pair"]) for x in dec.get("top_pairs", [])[:2]} == true_pairs}
            if bayes:
                pr = snap.surrogate.predictive(blind_masks, 0.90)
                row.update(interval_metrics(blind_T, pr["lo"][:, 0], pr["hi"][:, 0]))
                pred = pr["mean"][:, 0]
            else:
                pred = snap.surrogate.predict(blind_masks)[:, 0]
            row.update(point_metrics(blind_T, pred))
            row.update({f"cost_{k}": v for k, v in end_to_end_cost(snap).items()})
            row["discovery_masks"] = snap.costs["n_masks"]
            row["discovery_forward_evals"] = snap.costs["n_prompt_evals"]
            rows.append(row)
            decisions[f"{method}@{snap.budget}"] = {"decision": dec, "model_summary": snap.model_summary,
                                                     "acq_log": snap.costs.get("acq_log", [])}
        for r in rows:
            if r["method"] == method:
                r["event_time"] = event if event is not None else gb["censored_event_time_masks"]
                r["censored"] = event is None
        queries.extend(oracle.log)
    # ProxySPEX interaction-attribution diagnostic (own 128 coalitions, counted separately)
    try:
        px = run_proxyspex(p, lambda M: np.mean(np.abs(disc_runner.effects(M)), 1), 128, dseed)
        px["top2_pairs_correct"] = {tuple(sorted(x["pair"])) for x in px["top_pairs"][:2]} == true_pairs
    except Exception as e:  # diagnostic only
        px = {"error": repr(e)}
    pd.DataFrame(rows).to_csv(udir / "rows.csv", index=False)
    pd.DataFrame(queries).to_parquet(udir / "queries.parquet", index=False)
    write_json(udir / "decisions.json", decisions)
    write_json(udir / "proxyspex.json", px)
    write_json(out / "evaluation_only_truth" / f"model-{seed}.json",
               {"seed": seed, "a_heads": inst.a_heads, "b_heads": inst.b_heads, "true_circuits": inst.true_circuits(),
                "true_pairs": inst.true_pairs(), "kappa": inst.kappa})
    status = {"status": "complete", "seed": seed, "seconds": time.time() - t0, "dir": str(udir),
              "blind_masks_sha256": masks_sha256(blind_masks), "warm_masks_sha256": masks_sha256(warm),
              "blind_inputs_sha256": blind.input_sha, "model_sha256": model_checksum(inst.model),
              "discovery_inputs_sha256": hashlib.sha256(dtoks.cpu().numpy().tobytes()).hexdigest()}
    write_json(st_path, status)
    return status


def gate_b_criteria(rows: pd.DataFrame, cfg: dict, n_valid: int) -> dict:
    gb = cfg["gate_b"]
    cr = gb["criteria"]
    out: dict = {}
    if n_valid < gb["n_valid_models"]:
        for c in ["B-1", "B-2", "B-3", "B-4"]:
            out[c] = {"status": "BLOCKED", "observed": f"BLOCKED_BY_BENCHMARK_CONSTRUCTION ({n_valid} valid)"}
        return out
    ev = rows.groupby(["method", "model_seed"]).agg(event_time=("event_time", "first"), censored=("censored", "first")).reset_index()
    if ev.model_seed.nunique() < gb["n_valid_models"]:
        for c in ["B-1", "B-2", "B-3", "B-4"]:
            out[c] = {"status": "NOT_RUN", "observed": "incomplete"}
        return out
    med = ev.groupby("method").event_time.median().to_dict()
    succ = ev[~ev.censored].groupby("method").size().reindex(ev.method.unique(), fill_value=0).to_dict()
    cand = "bayes_decision"
    ns = int(succ.get(cand, 0))
    out["B-1"] = {"status": "PASS" if ns >= cr["B-1"]["min_successful_models"] else "FAIL",
                  "observed": f"{ns}/10 models recovered >=3/4 planted size-2 circuits with a blind-faithful proposal by B=128",
                  "value": ns, "successes_by_method": succ}
    best_nb = min(NON_BAYES, key=lambda m: med[m])
    r2 = med[cand] / med[best_nb]
    cens_note = {m: int(ev[(ev.method == m)].censored.sum()) for m in med}
    out["B-2"] = {"status": "PASS" if r2 <= cr["B-2"]["median_budget_ratio_max_vs_best_non_bayesian"] else "FAIL",
                  "observed": f"median event time {cand}={med[cand]} vs best non-Bayes {best_nb}={med[best_nb]} (ratio {r2:.3f}); censored counts {cens_note}",
                  "value": r2, "medians": med, "best_non_bayes": best_nb}
    r3 = med[cand] / med["bayes_static"]
    out["B-3"] = {"status": "PASS" if r3 <= cr["B-3"]["median_budget_ratio_max_vs_bayes_static"] else "FAIL",
                  "observed": f"median {cand}={med[cand]} vs bayes_static={med['bayes_static']} (ratio {r3:.3f})", "value": r3}
    inv = int(rows[rows.method == cand].invalid_high_conf_claims.fillna(0).sum())
    ncl = int(rows[rows.method == cand].n_high_conf_claims.fillna(0).sum())
    out["B-4"] = {"status": "PASS" if inv <= cr["B-4"]["max_invalid_high_confidence_circuits"] else "FAIL",
                  "observed": f"{inv} invalid of {ncl} high-confidence (P>=0.95) claims across all checkpoints/models", "value": inv}
    return out


def cmd_gate_b(args, cfg: dict) -> int:
    from . import tracking
    from .cli import _finish_parent, _gate_parent, _require_clean, _write_status
    from .criteria import gate_verdict

    prov = provenance(cfg)
    confirmatory = args.max_seeds is None and not args.dry_run
    _require_clean(prov, confirmatory, args.allow_dirty)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and confirmatory:
        raise SystemExit("Gate B confirmatory run requires the GPU host (BLOCKED otherwise)")
    out = run_root(confirmatory) / "gate_b"
    out.mkdir(parents=True, exist_ok=True)
    gate = "gate-b" if confirmatory else "gate-b-smoke"
    exp = "bccd/gate-b" if confirmatory else "bccd/dev-smoke"
    _write_status(gate, state="running", pid=__import__("os").getpid())
    pid = _gate_parent(exp, gate, prov, confirmatory, "parent_run_id")
    try:
        n_needed = cfg["gate_b"]["n_valid_models"] if confirmatory else max(1, args.max_seeds or 1)
        log = construct_models(cfg, device, out, n_needed)
        print(json.dumps({"construction": {k: log[k] for k in ("n_attempts", "n_valid", "valid_seeds", "status")}}), flush=True)
        results = []
        for seed in log["valid_seeds"][:n_needed]:
            try:
                res = run_model(seed, cfg, out, device)
            except Exception:
                res = {"status": "failed", "seed": seed, "traceback": traceback.format_exc()}
                print(res["traceback"], flush=True)
            results.append(res)
            print(json.dumps({k: res.get(k) for k in ("status", "seed", "seconds")}), flush=True)
            if res.get("status") == "complete":
                rows = pd.read_csv(Path(res["dir"]) / "rows.csv")
                for method, g in rows.groupby("method"):
                    tags = tracking.base_tags(prov, "B", confirmatory, world_id=f"B-{seed}", seed=seed, method=method,
                                              task="planted-xor", model_id=f"planted-xor-{seed}", model_revision=res["model_sha256"],
                                              split_sha256=res["blind_masks_sha256"], device=device, budget_masks="32,64,96,128")
                    with tracking.run(exp, f"B:{seed}:{method}", tags, pid):
                        for _, row in g.iterrows():
                            tracking.log_metrics_step({k: row[k] for k in row.index if not isinstance(row[k], str) and k not in ("model_seed",)},
                                                      step=int(row["budget"]))
                        dec = json.loads((Path(res["dir"]) / "decisions.json").read_text())
                        tracking.log_json({k: v for k, v in dec.items() if k.startswith(method + "@")}, "decisions.json")
        frames = [pd.read_csv(Path(r["dir"]) / "rows.csv") for r in results if r.get("status") == "complete"]
        rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        rows.to_csv(out / "rows_all.csv", index=False)
        failed = [r for r in results if r.get("status") != "complete"]
        crit = gate_b_criteria(rows, cfg, log["n_valid"]) if confirmatory else {}
        summary = {"confirmatory": confirmatory, "construction": {k: log[k] for k in ("n_attempts", "n_valid", "valid_seeds", "status")},
                   "criteria": crit, "verdict": gate_verdict(crit) if crit else "EXPLORATORY", "n_failed": len(failed),
                   "provenance": prov, "parent_run_id": pid}
        write_json(out / "summary.json", summary)
        _finish_parent(pid, "FAILED" if failed else "FINISHED", {"gate_b_summary.json": summary})
        _write_status(gate, state="failed" if failed else "complete", exit_code=1 if failed else 0, verdict=summary["verdict"])
        print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2, default=str))
        return 1 if failed else 0
    except BaseException as e:
        _write_status(gate, state="crashed", error=repr(e), traceback=traceback.format_exc())
        try:
            _finish_parent(pid, "FAILED", {"error.json": {"error": repr(e)}})
        except Exception:
            pass
        raise


_ = PROJECT_ROOT
