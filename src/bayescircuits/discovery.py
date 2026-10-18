"""Shared discovery loop: one method, one unit (world / planted model / seed).

The loop only sees a :class:`DiscoveryOracle`; no blind data are passed in. At each
checkpoint a frozen :class:`Snapshot` is produced (selected circuits, posterior claims,
interaction ranking, fitted surrogate, costs). Blind evaluation happens elsewhere, after
the snapshot is sealed.
"""

from __future__ import annotations

import copy
import time
import zlib
from dataclasses import dataclass, field

import numpy as np

from .acquisition import candidate_pool, select_decision, select_uncertainty
from .circuits import (
    enumerate_circuits,
    loss_matrix,
    predicted_R,
    propose_minimal,
    search_fixed_k,
)
from .masks import mask_key
from .models.bayes import SpikeSlabConfig
from .models.surrogates import BayesSurrogate, make_surrogate, pair_nonadditivity
from .oracle.base import DiscoveryOracle

METHODS = ["lasso_1st", "lasso_pair", "gbdt", "bayes_static", "bayes_uncertainty", "bayes_decision"]
NON_BAYES = ["lasso_1st", "lasso_pair", "gbdt"]
ACTIVE = {"bayes_uncertainty", "bayes_decision"}


@dataclass
class DecisionSpec:
    mode: str  # "minimal" (Gates A/B: enumerate small circuits) or "fixed" (Gate C: k-constrained search)
    enum_sizes: list[int] = field(default_factory=list)  # minimal mode: sizes enumerated exactly
    propose_sizes: list[int] | None = None  # minimal mode: restrict proposals (Gate B: [2])
    fixed_sizes: list[int] = field(default_factory=list)  # fixed mode: k values
    rmax: float = 0.10
    lam: float = 10.0
    max_props: int = 4
    compute_pairs: bool = True


@dataclass
class AcqSpec:
    batch: int = 16
    n_random_pool: int = 64
    n_circuit_pool: int = 64
    n_particles: int = 32
    n_shortlist: int = 20
    search_particles: int = 64


@dataclass
class Problem:
    unit_id: str
    p: int
    contexts: list
    target: str
    warm_start: np.ndarray
    static_masks: np.ndarray
    checkpoints: list[int]
    densities: list[float]
    decision: DecisionSpec
    acq: AcqSpec
    reserved_keys: frozenset
    extra_targets: list[str] = field(default_factory=list)


@dataclass
class Snapshot:
    budget: int
    masks: np.ndarray
    Y: dict
    per_prompt: np.ndarray | None
    surrogate: object
    extra_surrogates: dict
    decision: dict
    model_summary: dict
    costs: dict


class _Clock:
    def __init__(self) -> None:
        self.wall: dict[str, float] = {}
        self.cpu: dict[str, float] = {}

    def add(self, key: str, wall: float, cpu: float) -> None:
        self.wall[key] = self.wall.get(key, 0.0) + wall
        self.cpu[key] = self.cpu.get(key, 0.0) + cpu

    def timed(self, key: str):
        clock = self

        class _T:
            def __enter__(self_inner):
                self_inner.w = time.perf_counter()
                self_inner.c = time.process_time()

            def __exit__(self_inner, *a):
                clock.add(key, time.perf_counter() - self_inner.w, time.process_time() - self_inner.c)

        return _T()


def _bayes_particle_R(sur: BayesSurrogate, p: int, circuits, idx=None) -> np.ndarray:
    return predicted_R(lambda M: sur.predict_particles(M, idx), p, circuits)


def make_decision(sur, problem: Problem, need_shortlist: bool) -> dict:
    """Circuit decisions from a fitted surrogate (shared logic for every method)."""
    dec = problem.decision
    p = problem.p
    out: dict = {}
    bayes = getattr(sur, "is_bayesian", False)
    if dec.mode == "minimal":
        circuits = enumerate_circuits(p, dec.enum_sizes)
        sizes = np.array([len(c) for c in circuits], dtype=float)
        if bayes:
            Rp = _bayes_particle_R(sur, p, circuits)  # (S, nC)
            P = (Rp <= dec.rmax).mean(0)
            EL = loss_matrix(Rp, sizes, p, dec.lam, dec.rmax).mean(0)
            score, ER = P, Rp.mean(0)
        else:
            R = predicted_R(sur.predict, p, circuits)
            score, ER, P, EL = R, R, None, sizes / p + dec.lam * np.maximum(R - dec.rmax, 0)
        if dec.propose_sizes is not None:
            keep = [i for i, c in enumerate(circuits) if len(c) in dec.propose_sizes]
            props = propose_minimal([circuits[i] for i in keep], score[keep], bayes, dec.max_props, dec.rmax)
        else:
            props = propose_minimal(circuits, score, bayes, dec.max_props, dec.rmax)
        out["proposals"] = [list(c) for c in props["circuits"]]
        out["proposal_scores"] = props["scores"]
        out["proposal_size"] = props["size"]
        idx_of = {c: i for i, c in enumerate(circuits)}
        out["proposal_pred_R"] = [float(ER[idx_of[tuple(c)]]) for c in props["circuits"]]
        if bayes:
            hc = np.where(P >= 0.95)[0]
            out["high_conf_claims"] = [list(circuits[i]) for i in hc]
            out["high_conf_probs"] = P[hc].tolist()
            out["proposal_P"] = [float(P[idx_of[tuple(c)]]) for c in props["circuits"]]
            # alternative high-posterior circuits (top 8 by posterior probability)
            top = np.argsort(-P, kind="stable")[:8]
            out["top_posterior_circuits"] = [{"circuit": list(circuits[i]), "P": float(P[i])} for i in top]
        if need_shortlist:
            order = np.argsort(EL, kind="stable")[: problem.acq.n_shortlist]
            out["_shortlist"] = [circuits[i] for i in order]
            out["_shortlist_groups"] = [list(range(len(order)))]
        if dec.compute_pairs:
            if bayes:
                na = pair_nonadditivity(lambda M: sur.predict_particles(M), p)  # (S, n_pairs)
                na_score = np.abs(na.mean(0))
                na_sd = na.std(0)
            else:
                na_score = np.abs(pair_nonadditivity(sur.predict, p))
                na_sd = None
            I, J = np.triu_indices(p, 1)
            order = np.argsort(-na_score, kind="stable")[:10]
            out["top_pairs"] = [
                {
                    "pair": [int(I[i]), int(J[i])],
                    "score": float(na_score[i]),
                    **({"sd": float(na_sd[i])} if na_sd is not None else {}),
                }
                for i in order
            ]
    else:  # fixed-k search (Gate C)
        if bayes:
            idx = sur.particle_subset(problem.acq.search_particles)

            def obj(M):
                F = sur.predict_particles(M, idx)  # (S, n, C)
                full = sur.predict_particles(np.ones((1, p), np.uint8), idx)
                R = np.abs(F).mean(-1) / np.maximum(np.abs(full).mean(-1), 1e-9)
                return R.mean(0)
        else:

            def obj(M):
                F = sur.predict(M)
                full = sur.predict(np.ones((1, p), np.uint8))
                return np.abs(F).mean(-1) / max(float(np.abs(full).mean()), 1e-9)

        res = search_fixed_k(obj, p, dec.fixed_sizes)
        out["selected"] = {str(k): list(res[k][0][0]) for k in dec.fixed_sizes}
        out["selected_pred_R"] = {str(k): float(res[k][0][1]) for k in dec.fixed_sizes}
        out["alternatives"] = {str(k): [{"circuit": list(c), "pred_R": float(v)} for c, v in res[k][1:]] for k in dec.fixed_sizes}
        if bayes:
            sel = [tuple(res[k][0][0]) for k in dec.fixed_sizes]
            Rp = _bayes_particle_R(sur, p, sel)
            out["selected_P_faithful"] = {str(k): float((Rp[:, i] <= dec.rmax).mean()) for i, k in enumerate(dec.fixed_sizes)}
            out["selected_R_q05_q95"] = {
                str(k): [float(np.quantile(Rp[:, i], 0.05)), float(np.quantile(Rp[:, i], 0.95))]
                for i, k in enumerate(dec.fixed_sizes)
            }
        if need_shortlist:
            sl, groups = [], []
            per = max(1, problem.acq.n_shortlist // len(dec.fixed_sizes))
            for k in dec.fixed_sizes:
                g = []
                for c, _ in res[k][:per]:
                    g.append(len(sl))
                    sl.append(tuple(c))
                groups.append(g)
            out["_shortlist"] = sl
            out["_shortlist_groups"] = groups
    return out


def run_method(
    problem: Problem,
    method: str,
    oracle: DiscoveryOracle,
    seed: int,
    bayes_cfg: SpikeSlabConfig,
    extra_hook=None,
) -> list[Snapshot]:
    """Run discovery for one method; returns sealed snapshots at every checkpoint.

    extra_hook(snapshot_masks, Y, per_prompt) -> dict of extra surrogates (fit only on
    discovery data), e.g. Gate C effect-target model + prompt-variance model."""
    rng = np.random.default_rng([seed, zlib.crc32(method.encode())])
    p = problem.p
    clock = _Clock()
    sur = make_surrogate(method, p, problem.contexts, seed, bayes_cfg)
    masks_list: list[np.ndarray] = []
    Y: dict[str, list] = {}
    PP: list = []

    def do_query(M: np.ndarray) -> None:
        res = oracle.query(M)
        masks_list.extend(list(res.masks))
        for k, v in res.y.items():
            Y.setdefault(k, []).extend(list(v))
        if res.per_prompt is not None:
            PP.extend(list(res.per_prompt))

    def Yarr(name: str) -> np.ndarray:
        a = np.asarray(Y[name], dtype=float)
        return a.reshape(len(a), -1)

    t_or = oracle.oracle_seconds
    with clock.timed("oracle"):
        do_query(problem.warm_start)
    snapshots: list[Snapshot] = []
    static_ptr = 0
    acq_logs: list[dict] = []
    cps = sorted(problem.checkpoints)
    bmax = cps[-1]
    while True:
        n = len(masks_list)
        is_cp = n in cps
        M_now = np.stack(masks_list)
        fitted_for_round = False
        fit_w = fit_c = 0.0
        if method in ACTIVE or is_cp:
            w0, c0 = time.perf_counter(), time.process_time()
            with clock.timed("fit"):
                sur.fit(M_now, Yarr(problem.target))
            fit_w, fit_c = time.perf_counter() - w0, time.process_time() - c0
            fitted_for_round = True
        if is_cp:
            w0, c0 = time.perf_counter(), time.process_time()
            with clock.timed("decision"):
                dec = make_decision(sur, problem, need_shortlist=method in ACTIVE)
            dec_w, dec_c = time.perf_counter() - w0, time.process_time() - c0
            extras = {}
            if extra_hook is not None:
                with clock.timed("extra_models"):
                    extras = extra_hook(M_now, {k: Yarr(k) for k in Y}, np.asarray(PP) if PP else None, seed)
            summary = sur.summary() if hasattr(sur, "summary") else {}
            costs = {
                "wall": dict(clock.wall),
                "cpu": dict(clock.cpu),
                "oracle_seconds_measured": oracle.oracle_seconds - t_or,
                "n_masks": oracle.n_masks,
                "n_prompt_evals": oracle.n_masks * oracle.n_prompts_per_mask,
                "acq_rounds": len(acq_logs),
                "active": method in ACTIVE,
                "checkpoint_fit": {"wall": fit_w, "cpu": fit_c},
                "checkpoint_decision": {"wall": dec_w, "cpu": dec_c},
            }
            snapshots.append(
                Snapshot(
                    budget=n,
                    masks=M_now.copy(),
                    Y={k: Yarr(k).copy() for k in Y},
                    per_prompt=np.asarray(PP).copy() if PP else None,
                    surrogate=copy.deepcopy(sur),
                    extra_surrogates=extras,
                    decision={k: v for k, v in dec.items() if not k.startswith("_")},
                    model_summary=summary,
                    costs=costs,
                )
            )
            if n >= bmax:
                break
        else:
            dec = None
        next_cp = min(c for c in cps if c > n)
        nb = min(problem.acq.batch, next_cp - n)
        if method in ACTIVE:
            if dec is None:
                with clock.timed("acq_shortlist"):
                    dec = make_decision(sur, problem, need_shortlist=True)
            with clock.timed("acquisition"):
                unavailable = {mask_key(m) for m in masks_list} | set(problem.reserved_keys)
                pool = candidate_pool(
                    rng, p, unavailable, problem.densities, dec["_shortlist"], problem.acq.n_random_pool, problem.acq.n_circuit_pool
                )
                idx = sur.particle_subset(problem.acq.n_particles)
                F = sur.predict_particles(pool, idx)
                s2 = sur.post.sigma2[idx]
                if method == "bayes_uncertainty":
                    chosen, info = select_uncertainty(F, s2, nb, rng)
                else:
                    shortlist = dec["_shortlist"]
                    Rs = _bayes_particle_R(sur, p, shortlist, idx)  # (S, J)
                    sizes = np.array([len(c) for c in shortlist], dtype=float)
                    Lfull = loss_matrix(Rs, sizes, p, problem.decision.lam, problem.decision.rmax)
                    Lmats = [Lfull[:, g] for g in dec["_shortlist_groups"] if len(g)]
                    chosen, info = select_decision(F, s2, Lmats, nb, rng)
            info.update({"round_start_n": n, "pool_size": len(pool), "batch": nb, "fitted": fitted_for_round})
            acq_logs.append(info)
            with clock.timed("oracle"):
                do_query(pool[chosen])
        else:
            with clock.timed("oracle"):
                do_query(problem.static_masks[static_ptr : static_ptr + nb])
            static_ptr += nb
    for s in snapshots:
        s.costs["acq_log"] = [a for a in acq_logs if a["round_start_n"] < s.budget]
    return snapshots


def end_to_end_cost(snap: Snapshot) -> dict:
    """Compute needed to produce the decision at this budget.

    passive methods: oracle(cumulative) + fit at this checkpoint + circuit decision at this checkpoint
                     (fits/decisions at earlier reporting checkpoints are not needed to reach B)
    active methods:  oracle + all round fits + shortlist/acquisition work + all decisions (cumulative);
                     earlier-checkpoint decisions double as acquisition shortlists, so they count.
    Extra reporting models (Gate C effect/calibration model) are reported separately."""
    c = snap.costs
    out = {}
    for nm in ("wall", "cpu"):
        cum = c[nm]
        if c["active"]:
            e2e = sum(cum.get(k, 0.0) for k in ("oracle", "fit", "acq_shortlist", "acquisition", "decision"))
        else:
            e2e = cum.get("oracle", 0.0) + c["checkpoint_fit"][nm] + c["checkpoint_decision"][nm]
        out[f"{nm}_e2e"] = e2e
        out[f"{nm}_oracle"] = cum.get("oracle", 0.0)
        out[f"{nm}_fit_cum"] = cum.get("fit", 0.0)
        out[f"{nm}_acquisition_cum"] = cum.get("acquisition", 0.0) + cum.get("acq_shortlist", 0.0)
        out[f"{nm}_extra_models_cum"] = cum.get("extra_models", 0.0)
    return out
