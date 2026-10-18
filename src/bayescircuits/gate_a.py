"""Gate A: controlled 32-component causal oracles (CPU)."""

from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .config import run_root
from .discovery import METHODS, AcqSpec, DecisionSpec, Problem, end_to_end_cost, run_method
from .evaluate import crps_from_samples, interval_metrics, point_metrics
from .masks import balanced_masks, expand_contexts, mask_key, masks_sha256
from .models.bayes import SpikeSlabConfig
from .models.surrogates import BayesSurrogate
from .oracle.base import DiscoveryOracle, QueryResult
from .oracle.synthetic import SyntheticWorld, verify_a1_minimal_circuits
from .provenance import write_json

REGIMES = ["A0", "A1", "A2", "A3"]


def bayes_cfg(impl: dict) -> SpikeSlabConfig:
    return SpikeSlabConfig(**impl["bayes"])


def acq_spec(impl: dict) -> AcqSpec:
    return AcqSpec(**impl["acquisition"])


def world_designs(world: SyntheticWorld, cfg: dict) -> dict:
    ga = cfg["gate_a"]
    dens = cfg["shared"]["intervention_density_mixture"]
    p = world.p
    rng = np.random.default_rng([world.seed, 11])
    blind = balanced_masks(rng, p, ga["n_blind_masks_per_world"], dens)
    reserved = {mask_key(m) for m in blind}
    warm = balanced_masks(np.random.default_rng([world.seed, 22]), p, cfg["shared"]["initial_warm_start_masks"], dens, exclude=reserved)
    excl = reserved | {mask_key(m) for m in warm}
    bmax = max(ga["checkpoints_mask_count"])
    static = balanced_masks(np.random.default_rng([world.seed, 33]), p, bmax - len(warm), dens, exclude=excl)
    return {"blind": blind, "reserved": frozenset(reserved), "warm": warm, "static": static}


def world_oracle_fn(world: SyntheticWorld):
    """Context batch per mask: 16 contexts; A0/A1/A3 collapse to one effective observation per
    mask, A2 to one aggregate outcome per context group (8 contexts each)."""

    def fn(M: np.ndarray) -> QueryResult:
        if world.has_context:
            y = np.stack([world.evaluate(M, np.zeros(len(M))), world.evaluate(M, np.ones(len(M)))], axis=1)
        else:
            y = world.evaluate(M)[:, None]
        return QueryResult(masks=M, y={"d": y})

    return fn


def make_problem(world: SyntheticWorld, des: dict, cfg: dict, impl: dict) -> Problem:
    ga = cfg["gate_a"]
    return Problem(
        unit_id=f"{world.regime}-{world.seed}",
        p=world.p,
        contexts=[0, 1] if world.has_context else [None],
        target="d",
        warm_start=des["warm"],
        static_masks=des["static"],
        checkpoints=list(ga["checkpoints_mask_count"]),
        densities=list(cfg["shared"]["intervention_density_mixture"]),
        decision=DecisionSpec(
            mode="minimal",
            enum_sizes=list(impl["gate_a"]["enum_sizes"]),
            rmax=cfg["shared"]["faithful_R_max"],
            lam=impl["decision"]["lam"],
            max_props=impl["decision"]["max_props"],
            compute_pairs=True,
        ),
        acq=acq_spec(impl),
        reserved_keys=des["reserved"],
    )


def misfit_statistic(snap, problem: Problem, bcfg: SpikeSlabConfig, seed: int, folds: int) -> float:
    """Frozen misspecification statistic: K-fold CV RMSE of posterior-mean predictions on the
    method's own discovery masks, divided by the RMS of the observed responses. Deterministic
    oracles that the quadratic model represents exactly give ~0; omitted higher-order terms give
    a clearly positive value."""
    M, Y = snap.masks, snap.Y["d"]
    rng = np.random.default_rng([seed, 999])
    fid = rng.permutation(np.arange(len(M)) % folds)
    pred = np.zeros_like(Y)
    for f in range(folds):
        tr, te = fid != f, fid == f
        sur = BayesSurrogate(problem.p, problem.contexts, bcfg, seed + 17 * f, name="cv")
        sur.fit(M[tr], Y[tr], warm=False)
        pred[te] = sur.predict(M[te])
    return float(np.sqrt(np.mean((pred - Y) ** 2)) / max(np.sqrt(np.mean(Y**2)), 1e-12))


def evaluate_snapshot(world: SyntheticWorld, snap, method: str, des: dict, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    blind = des["blind"]
    if world.has_context:
        T = np.stack([world.evaluate(blind, np.zeros(len(blind))), world.evaluate(blind, np.ones(len(blind)))], axis=1)
    else:
        T = world.evaluate(blind)[:, None]
    sur = snap.surrogate
    row: dict = {"budget": snap.budget}
    if getattr(sur, "is_bayesian", False):
        pr = sur.predictive(blind, level=0.90)
        pred, lo, hi = pr["mean"], pr["lo"], pr["hi"]
        row.update(interval_metrics(T, lo, hi))
        row["crps"] = float(np.mean(crps_from_samples(pr["samples"], T.reshape(-1))))
        row["pred_sd_mean"] = float(np.mean(pr["pred_sd"]))
    else:
        pred = sur.predict(blind)
        lo = hi = np.full_like(pred, np.nan)
    row.update(point_metrics(T, pred))
    row["truth_sd"] = float(np.std(T))
    dec = snap.decision
    true_c = {tuple(c) for c in world.true_circuits()}
    props = [tuple(c) for c in dec.get("proposals", [])]
    row["n_proposals"] = len(props)
    row["proposal_size"] = dec.get("proposal_size")
    row["proposal_true_R"] = json.dumps([world.true_R(c) for c in props])
    if world.regime == "A1":
        row["circuits_recovered"] = len(true_c & set(props))
        tp = {tuple(x["pair"]) for x in dec.get("top_pairs", [])[:2]}
        row["top2_pairs_correct"] = bool(tp == set(world.true_pairs()))
        ranks = [i for i, x in enumerate(dec.get("top_pairs", [])) if tuple(x["pair"]) in set(world.true_pairs())]
        row["true_pair_ranks"] = json.dumps(ranks)
    claims = [tuple(c) for c in dec.get("high_conf_claims", [])]
    claim_R = [world.true_R(c) for c in claims]
    row["n_high_conf_claims"] = len(claims) if getattr(sur, "is_bayesian", False) else None
    row["invalid_high_conf_claims"] = int(sum(r > 0.10 for r in claim_R)) if getattr(sur, "is_bayesian", False) else None
    cst = end_to_end_cost(snap)
    row.update({f"cost_{k}": v for k, v in cst.items()})
    row["discovery_masks"] = snap.costs["n_masks"]
    row["discovery_context_evals"] = snap.costs["n_prompt_evals"]
    M, g = expand_contexts(blind, [0, 1] if world.has_context else [None])
    pf = pd.DataFrame(
        {
            "mask_idx": np.repeat(np.arange(len(blind)), T.shape[1]),
            "ctx": np.tile(np.arange(T.shape[1]), len(blind)),
            "n_ablated": np.repeat(blind.sum(1), T.shape[1]),
            "truth": T.reshape(-1),
            "pred": pred.reshape(-1),
            "lo": np.asarray(lo).reshape(-1),
            "hi": np.asarray(hi).reshape(-1),
        }
    )
    pf["budget"] = snap.budget
    pf["method"] = method
    return row, pf


def run_world(regime: str, seed: int, cfg: dict, out_dir: str, split: str) -> dict:
    """Run all methods on one world; persist results. Executed in a worker process."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    impl = cfg["_impl"]
    out = Path(out_dir) / split / f"{regime}-{seed}"
    done = out / "status.json"
    if done.exists() and json.loads(done.read_text()).get("status") == "complete":
        return json.loads(done.read_text())
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    world = SyntheticWorld.create(regime, seed, p=cfg["gate_a"]["p"], jitter=cfg["gate_a"]["nonzero_coefficient_relative_jitter"])
    des = world_designs(world, cfg)
    problem = make_problem(world, des, cfg, impl)
    bcfg = bayes_cfg(impl)
    fn = world_oracle_fn(world)
    rows, preds, decisions, queries = [], [], {}, []
    rng = np.random.default_rng([seed, 5])
    for method in METHODS:
        oracle = DiscoveryOracle(
            fn=fn,
            p=world.p,
            budget=max(problem.checkpoints),
            method=method,
            unit_id=problem.unit_id,
            prompt_batch_id=f"ctx16-{problem.unit_id}",
            reserved_keys=des["reserved"],
            n_prompts_per_mask=cfg["gate_a"]["context_batch_per_mask"],
        )
        snaps = run_method(problem, method, oracle, seed, bcfg)
        for snap in snaps:
            row, pf = evaluate_snapshot(world, snap, method, des, rng)
            row.update({"regime": regime, "world_seed": seed, "split": split, "method": method})
            if method.startswith("bayes") and snap.budget == max(problem.checkpoints):
                tm = time.process_time()
                row["misfit_stat"] = misfit_statistic(snap, problem, bcfg, seed, impl["gate_a"]["misfit_cv_folds"])
                row["misfit_cpu_seconds"] = time.process_time() - tm
            rows.append(row)
            preds.append(pf)
            decisions[f"{method}@{snap.budget}"] = {"decision": snap.decision, "model_summary": snap.model_summary,
                                                     "acq_log": snap.costs.get("acq_log", [])}
        queries.extend(oracle.log)
    pd.DataFrame(rows).to_csv(out / "rows.csv", index=False)
    pd.concat(preds).to_parquet(out / "blind_predictions.parquet", index=False)
    pd.DataFrame(queries).to_parquet(out / "queries.parquet", index=False)
    write_json(out / "decisions.json", decisions)
    truth_dir = Path(out_dir) / "evaluation_only_truth"
    tr = world.truth_record()
    if regime == "A1":
        tr["verified_minimal"] = verify_a1_minimal_circuits(world)
    write_json(truth_dir / f"{split}-{regime}-{seed}.json", tr)
    status = {
        "status": "complete",
        "regime": regime,
        "seed": seed,
        "split": split,
        "seconds": time.time() - t0,
        "blind_masks_sha256": masks_sha256(des["blind"]),
        "warm_masks_sha256": masks_sha256(des["warm"]),
        "static_masks_sha256": masks_sha256(des["static"]),
        "dir": str(out),
    }
    write_json(done, status)
    return status


def _safe_run_world(*a):
    try:
        return run_world(*a)
    except Exception:
        return {"status": "failed", "args": [str(x) for x in a[:2]], "traceback": traceback.format_exc()}


def world_list(cfg: dict, split: str) -> list[tuple[str, int]]:
    if split == "development":
        return [(r, s) for r in REGIMES for s in cfg["seeds"]["gate_a_development_worlds"]]
    return [(r, s) for r in REGIMES for s in cfg["seeds"]["gate_a_confirmatory"][r]]


def run_gate_a(cfg: dict, split: str, confirmatory: bool, max_worlds: int | None = None, workers: int | None = None, on_done=None) -> list[dict]:
    worlds = world_list(cfg, split)
    if max_worlds is not None:
        if confirmatory:
            raise ValueError("confirmatory mode refuses shortened world lists")
        per = max(1, max_worlds)
        worlds = [w for r in REGIMES for w in [x for x in worlds if x[0] == r][:per]]
    out_dir = str(run_root(confirmatory) / "gate_a")
    workers = workers or cfg["_impl"]["gate_a"]["workers"]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_safe_run_world, r, s, cfg, out_dir, split): (r, s) for r, s in worlds}
        for fu in as_completed(futs):
            res = fu.result()
            results.append(res)
            if on_done is not None:
                on_done(res)
            print(json.dumps({k: res.get(k) for k in ("status", "regime", "seed", "seconds")}), flush=True)
            if res.get("status") == "failed":
                print(res["traceback"], flush=True)
    return results
