"""Verification-in-the-loop (VIL) circuit discovery for protocol BCCD-VIL-1.0.

Every method sees only a budget-accounted :class:`DiscoveryOracle` bound to its seed's fixed
32-prompt discovery batch. Discovery-design masks and circuit-verification masks are sent to
the same oracle and debit the same 128-mask budget; a verification is simply the retain mask
``retain(C)`` of a proposed circuit, evaluated on the discovery prompts. Blind prompts are
never visible here (they are evaluated by ``vil_run`` after decisions are sealed).

Methods (docs/09_VERIFICATION_METHODS.md):

* ``lasso_static_128``      M0  pairwise Lasso on 128 design masks, shared circuit search
* ``lasso_density_matched`` M1  104 design masks + 24 random retain masks (8 per k), refit, search
* ``lasso_random_verify``   M2  104 design masks, diverse surrogate shortlist, 24 random verifications
* ``lasso_vil``             M3  104 design masks, 24 verifications in rounds; refit after each round,
                                propose the best predicted unverified circuit per k (error-aware revision)
* ``bayes_vil``             M4  as M3 with the spike-and-slab posterior; proposals by expected
                                improvement over the incumbent measured discrepancy (uncertainty-aware)
* ``bayes_static_128``      M5  spike-and-slab posterior on 128 design masks, posterior-expected-R search

Verifying methods select, for each k, the verified circuit with the lowest *measured*
discovery-prompt discrepancy (the denominator u(empty) is a per-seed constant, so ranking the
numerator mean|d| ranks R). Non-verifying methods select the surrogate optimum.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass

import numpy as np

from .circuits import circuit_masks, search_fixed_k
from .masks import balanced_masks, mask_key, retain_mask
from .models.bayes import SpikeSlabConfig
from .models.surrogates import make_surrogate
from .oracle.base import DiscoveryOracle

METHODS_VIL = [
    "lasso_static_128",
    "lasso_density_matched",
    "lasso_random_verify",
    "lasso_vil",
    "bayes_vil",
    "bayes_static_128",
]
METHOD_CODE = {
    "lasso_static_128": "M0",
    "lasso_density_matched": "M1",
    "lasso_random_verify": "M2",
    "lasso_vil": "M3",
    "bayes_vil": "M4",
    "bayes_static_128": "M5",
}
VERIFYING = ("lasso_random_verify", "lasso_vil", "bayes_vil")
NON_VERIFYING = ("lasso_static_128", "lasso_density_matched", "bayes_static_128")


def surrogate_kind(method: str) -> str:
    return "bayes_static" if method.startswith("bayes") else "lasso_pair"


@dataclass(frozen=True)
class VILSpec:
    p: int = 48
    ks: tuple = (8, 16, 24)
    budget: int = 128
    n_warm: int = 32
    n_discovery: int = 104
    n_verify: int = 24
    rounds: int = 8  # verification rounds for lasso_vil / bayes_vil (n_verify / rounds / |ks| per k per round)
    max_swap_iters: int = 12  # shared search (unchanged from BCCD-PILOT-1.0)
    n_alternatives: int = 6
    n_random_starts: int = 6  # candidate generator: random-start local searches per k
    rs_max_iters: int = 30
    shortlist_starts: int = 16  # M2: random starts used to build its diverse shortlist
    shortlist_size: int = 16
    shortlist_min_symdiff: int = 4  # circuits in M2's shortlist differ by more than one swap
    n_thompson: int = 4  # M4: single-particle searches added to the candidate set per round
    search_particles: int = 64  # Bayes: particles inside the circuit search objective

    @property
    def per_round(self) -> int:
        q, r = divmod(self.n_verify, self.rounds * len(self.ks))
        if r or q < 1:
            raise ValueError("n_verify must be a positive multiple of rounds * |ks|")
        return q

    @classmethod
    def from_dict(cls, d: dict) -> VILSpec:
        d = dict(d)
        if "ks" in d:
            d["ks"] = tuple(d["ks"])
        return cls(**d)


# ----------------------------------------------------------------------------- designs
def seed_designs(seed: int, reserved: set, spec: VILSpec, densities: list[float], static_total: int = 224) -> dict:
    """Mask designs for one seed (identical to BCCD-PILOT-1.0 Gate C warm start / static stream).

    d104 = warm(32) + static[:72]; d128 = warm(32) + static[:96] (= the pilot's 128-mask static design);
    density_matched = 24 random retain masks, 8 retaining each k (ablating p-k heads)."""
    p = spec.p
    warm = balanced_masks(np.random.default_rng([seed, 22]), p, spec.n_warm, densities, exclude=set(reserved))
    excl = set(reserved) | {mask_key(m) for m in warm}
    static = balanced_masks(np.random.default_rng([seed, 33]), p, static_total, densities, exclude=excl)
    n_more = spec.n_discovery - spec.n_warm
    d104 = np.concatenate([warm, static[:n_more]])
    d128 = np.concatenate([warm, static[: spec.budget - spec.n_warm]])
    rng = np.random.default_rng([seed, 44])
    used = set(reserved) | {mask_key(m) for m in d104}
    per_k = spec.n_verify // len(spec.ks)
    dm = []
    for k in spec.ks:
        got = 0
        while got < per_k:
            C = rng.choice(p, size=k, replace=False)
            m = retain_mask(p, C)
            if mask_key(m) in used:
                continue
            used.add(mask_key(m))
            dm.append(m)
            got += 1
    return {"warm": warm, "d104": d104, "d128": d128, "density_matched": np.stack(dm)}


# ----------------------------------------------------------------------------- objectives
class _Objective:
    """Predicted normalised discrepancy R_hat(C) = |f(retain C)| / |f(ablate all)| (posterior mean for Bayes)."""

    def __init__(self, sur, p: int, idx: np.ndarray | None = None):
        self.sur = sur
        self.p = p
        self.bayes = bool(getattr(sur, "is_bayesian", False))
        self.idx = idx
        ones = np.ones((1, p), np.uint8)
        if self.bayes:
            self.full = np.maximum(np.abs(sur.predict_particles(ones, idx)[:, 0, 0]), 1e-9)  # (S,)
        else:
            self.full = max(float(abs(sur.predict(ones)[0, 0])), 1e-9)
        self.n_calls = 0

    def __call__(self, M: np.ndarray) -> np.ndarray:
        self.n_calls += len(M)
        if self.bayes:
            F = np.abs(self.sur.predict_particles(M, self.idx)[..., 0])  # (S, n)
            return (F / self.full[:, None]).mean(0)
        return np.abs(self.sur.predict(M)[:, 0]) / self.full


def _local_search(obj, p: int, init, max_iters: int) -> tuple[tuple[int, ...], float, dict]:
    C = tuple(sorted(int(c) for c in init))
    cur = float(obj(circuit_masks(p, [C]))[0])
    pool: dict[tuple[int, ...], float] = {C: cur}
    for _ in range(max_iters):
        inside = list(C)
        outside = [j for j in range(p) if j not in C]
        cands = [tuple(sorted([x for x in inside if x != i] + [j])) for i in inside for j in outside]
        if not cands:
            break
        vals = obj(circuit_masks(p, cands))
        for c, v in zip(cands, vals):
            pool.setdefault(c, float(v))
        b = int(np.argmin(vals))
        if vals[b] < cur - 1e-12:
            C, cur = cands[b], float(vals[b])
        else:
            break
    return C, cur, pool


def generate_candidates(obj, spec: VILSpec, rng: np.random.Generator, n_random_starts: int, keep_pool: bool = False) -> dict:
    """Shared candidate generator: the shared greedy-elimination + swap search (BCCD-PILOT-1.0)
    plus ``n_random_starts`` random-start best-improvement swap searches per k.

    Returns {k: {"cands": {circuit: obj}, "optima": [circuits], "search_top": circuit, "search_top_val": v}};
    with ``keep_pool`` also "pool": every circuit scored during the k-specific searches (used by M2's shortlist)."""
    p = spec.p
    res = search_fixed_k(obj, p, list(spec.ks), spec.max_swap_iters, spec.n_alternatives)
    out = {}
    for k in spec.ks:
        cands: dict[tuple[int, ...], float] = {}
        for c, v in res[k]:
            cands[tuple(c)] = float(v)
        optima = [tuple(res[k][0][0])]
        explored: dict[tuple[int, ...], float] = dict(cands)
        for _ in range(n_random_starts):
            init = rng.choice(p, size=k, replace=False)
            C, v, pool = _local_search(obj, p, init, spec.rs_max_iters)
            optima.append(C)
            cands[C] = v
            for c, vv in sorted(pool.items(), key=lambda t: t[1])[: spec.n_alternatives]:
                cands.setdefault(c, vv)
            if keep_pool:
                for c, vv in pool.items():
                    explored.setdefault(c, vv)
        out[k] = {"cands": cands, "optima": optima, "search_top": tuple(res[k][0][0]), "search_top_val": float(res[k][0][1])}
        if keep_pool:
            out[k]["pool"] = explored
    return out


def diverse_shortlist(cands: dict, size: int, min_symdiff: int) -> list[tuple[int, ...]]:
    """Greedy diverse top-``size`` by predicted R: skip circuits within ``min_symdiff`` of a chosen one."""
    ranked = sorted(cands.items(), key=lambda t: (t[1], t[0]))
    chosen: list[tuple[int, ...]] = []
    sets: list[set] = []
    for c, _ in ranked:
        s = set(c)
        if all(len(s ^ t) >= min_symdiff for t in sets):
            chosen.append(c)
            sets.append(s)
        if len(chosen) >= size:
            break
    for c, _ in ranked:  # fill (only if the diverse pass ran out)
        if len(chosen) >= size:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen


def _top_up(chosen: list, cands: dict, size: int, available, rng: np.random.Generator, p: int, k: int, out: dict) -> list:
    """Ensure ``size`` distinct available circuits: best remaining candidates first, then uniformly random
    circuits (never needed at p=48 in development; every top-up is logged in the decision record)."""
    chosen = list(chosen)
    for c, _ in sorted(cands.items(), key=lambda t: (t[1], t[0])):
        if len(chosen) >= size:
            break
        if c not in chosen and available(c):
            chosen.append(c)
            out.setdefault("top_up_events", []).append({"k": k, "kind": "candidate"})
    tries = 0
    while len(chosen) < size and tries < 10000:
        tries += 1
        c = tuple(sorted(int(x) for x in rng.choice(p, size=k, replace=False)))
        if c not in chosen and available(c):
            chosen.append(c)
            out.setdefault("top_up_events", []).append({"k": k, "kind": "random"})
    if len(chosen) < size:
        raise RuntimeError("could not find enough available circuits to verify")
    return chosen


# ----------------------------------------------------------------------------- clock
class _Clock:
    def __init__(self) -> None:
        self.wall: dict[str, float] = {}
        self.cpu: dict[str, float] = {}

    def timed(self, key: str):
        clock = self

        class _T:
            def __enter__(self_inner):
                self_inner.w = time.perf_counter()
                self_inner.c = time.process_time()

            def __exit__(self_inner, *a):
                clock.wall[key] = clock.wall.get(key, 0.0) + time.perf_counter() - self_inner.w
                clock.cpu[key] = clock.cpu.get(key, 0.0) + time.process_time() - self_inner.c

        return _T()


E2E_KEYS = ("oracle_discovery", "oracle_verification", "fit", "search", "acquisition")


# ----------------------------------------------------------------------------- predictions
def _circuit_record(sur, p: int, C: tuple[int, ...], obj_val: float | None = None) -> dict:
    """Surrogate prediction for one circuit at the time it is proposed / selected."""
    M = retain_mask(p, C)[None]
    ones = np.ones((1, p), np.uint8)
    rec: dict = {"circuit": [int(c) for c in C], "size": len(C)}
    if getattr(sur, "is_bayesian", False):
        F = sur.predict_particles(M)[:, 0, 0]
        full = np.maximum(np.abs(sur.predict_particles(ones)[:, 0, 0]), 1e-9)
        R = np.abs(F) / full
        pr = sur.predictive(M, 0.90)
        rec.update({
            "pred_u": float(np.abs(F).mean()),
            "pred_R": float(R.mean()) if obj_val is None else float(obj_val),
            "pred_R_all_particles": float(R.mean()),
            "pred_R_q05": float(np.quantile(R, 0.05)),
            "pred_R_q95": float(np.quantile(R, 0.95)),
            "P_faithful": float((R <= 0.10).mean()),
            "pred_y_lo90": float(pr["lo"][0, 0]),
            "pred_y_hi90": float(pr["hi"][0, 0]),
        })
    else:
        f = float(abs(sur.predict(M)[0, 0]))
        full = max(float(abs(sur.predict(ones)[0, 0])), 1e-9)
        rec.update({"pred_u": f, "pred_R": f / full if obj_val is None else float(obj_val)})
    return rec


def expected_improvement(sur, p: int, circuits: list[tuple[int, ...]], incumbent_u: float) -> np.ndarray:
    """EI(C) = E_post[max(u* - |f(retain C)|, 0)] over all posterior particles (u on the target scale)."""
    F = np.abs(sur.predict_particles(circuit_masks(p, circuits))[..., 0])  # (S, n)
    return np.maximum(incumbent_u - F, 0.0).mean(0)


# ----------------------------------------------------------------------------- one method
def run_vil_method(
    method: str,
    oracle: DiscoveryOracle,
    designs: dict,
    spec: VILSpec,
    seed: int,
    bayes_cfg: SpikeSlabConfig,
    solver: dict,
) -> dict:
    """Run one method for one seed; returns its sealed decision record (no blind information)."""
    if method not in METHODS_VIL:
        raise KeyError(method)
    p = spec.p
    rng = np.random.default_rng([seed, zlib.crc32(method.encode()), 7])
    clock = _Clock()
    M_all: list[np.ndarray] = []
    y_all: list[float] = []
    kinds: list[str] = []
    PP: list[np.ndarray] = []

    def query(M: np.ndarray, kind: str) -> np.ndarray:
        with clock.timed(f"oracle_{kind}"):
            res = oracle.query(M)
        M_all.extend(list(res.masks))
        y = np.asarray(res.y["abs"], float).reshape(-1)
        y_all.extend(y.tolist())
        kinds.extend([kind] * len(M))
        if res.per_prompt is not None:
            PP.extend(list(res.per_prompt))
        return y

    def fit(sur, warm: bool = True) -> None:
        M = np.stack(M_all)
        Y = np.asarray(y_all, float)[:, None]
        with clock.timed("fit"):
            if getattr(sur, "is_bayesian", False):
                sur.fit(M, Y, warm=warm)
            else:
                sur.fit(M, Y)

    def make_obj(sur, idx=None) -> _Objective:
        if getattr(sur, "is_bayesian", False) and idx is None:
            idx = sur.particle_subset(spec.search_particles)
        return _Objective(sur, p, idx)

    sur = make_surrogate(surrogate_kind(method), p, [None], seed, bayes_cfg, solver)
    out: dict = {"method": method, "code": METHOD_CODE[method], "seed": seed, "surrogate": surrogate_kind(method)}

    # ------------------------------------------------------------- non-verifying methods
    if method in NON_VERIFYING:
        design = designs["d128"] if method != "lasso_density_matched" else np.concatenate([designs["d104"], designs["density_matched"]])
        query(design, "discovery")
        fit(sur, warm=False)
        with clock.timed("search"):
            obj = make_obj(sur)
            res = search_fixed_k(obj, p, list(spec.ks), spec.max_swap_iters, spec.n_alternatives)
            final = {}
            for k in spec.ks:
                C, v = tuple(res[k][0][0]), float(res[k][0][1])
                final[str(k)] = {**_circuit_record(sur, p, C, v), "source": "surrogate"}
        out["final"] = final
        out["verified"] = []
    # ------------------------------------------------------------- verifying methods
    else:
        query(designs["d104"], "discovery")
        verified: list[dict] = []
        seen_keys = {mask_key(m) for m in M_all}

        def available(C) -> bool:
            m = retain_mask(p, C)
            return oracle.is_available(m) and mask_key(m) not in seen_keys

        def verify(batch: list[tuple[int, tuple]], recs: list[dict], rnd: int) -> None:
            M = np.stack([retain_mask(p, C) for _, C in batch])
            y = query(M, "verification")
            for (k, C), yy, rec in zip(batch, y, recs):
                seen_keys.add(mask_key(retain_mask(p, C)))
                verified.append({**rec, "k": int(k), "round": rnd, "measured_u": float(yy),
                                 "covered90": (None if "pred_y_lo90" not in rec else bool(rec["pred_y_lo90"] <= yy <= rec["pred_y_hi90"]))})

        if method == "lasso_random_verify":
            fit(sur)
            with clock.timed("search"):
                obj = make_obj(sur)
                gen = generate_candidates(obj, spec, rng, spec.shortlist_starts, keep_pool=True)
            out["surrogate_at_104"] = {str(k): {**_circuit_record(sur, p, gen[k]["search_top"], gen[k]["search_top_val"]), "source": "surrogate"} for k in spec.ks}
            batch, recs = [], []
            per_k = spec.n_verify // len(spec.ks)
            out["shortlists"] = {}
            with clock.timed("acquisition"):
                for k in spec.ks:
                    # diverse shortlist over every circuit the surrogate scored during the k-specific searches
                    # (D-VIL-1: the 6-per-start candidate set alone held < 16 diverse circuits for small k)
                    pool = {c: v for c, v in gen[k]["pool"].items() if available(c)}
                    sl = diverse_shortlist(pool, spec.shortlist_size, spec.shortlist_min_symdiff)
                    sl = _top_up(sl, pool, spec.shortlist_size, available, rng, p, k, out)
                    out["shortlists"][str(k)] = [list(c) for c in sl]
                    pick = rng.choice(len(sl), size=per_k, replace=False)
                    for i in sorted(pick.tolist()):
                        batch.append((k, sl[i]))
                        recs.append(_circuit_record(sur, p, sl[i], pool.get(sl[i])))
            verify(batch, recs, 0)
        else:
            per = spec.per_round
            for rnd in range(spec.rounds):
                fit(sur, warm=True)
                with clock.timed("search"):
                    obj = make_obj(sur)
                    gen = generate_candidates(obj, spec, rng, spec.n_random_starts)
                    if method == "bayes_vil":
                        for _ in range(spec.n_thompson):
                            s = int(rng.integers(sur.post.n_particles))
                            ts = search_fixed_k(make_obj(sur, np.array([s])), p, list(spec.ks), spec.max_swap_iters, spec.n_alternatives)
                            for k in spec.ks:
                                for c, _ in ts[k]:
                                    gen[k]["cands"].setdefault(tuple(c), float(obj(circuit_masks(p, [tuple(c)]))[0]))
                if rnd == 0:
                    out["surrogate_at_104"] = {str(k): {**_circuit_record(sur, p, gen[k]["search_top"], gen[k]["search_top_val"]), "source": "surrogate"} for k in spec.ks}
                batch, recs, acq_info = [], [], {}
                with clock.timed("acquisition"):
                    for k in spec.ks:
                        cl = [c for c in gen[k]["cands"] if available(c)]
                        vals = np.array([gen[k]["cands"][c] for c in cl])
                        inc = [v["measured_u"] for v in verified if v["k"] == k]
                        if method == "bayes_vil" and inc:
                            score = expected_improvement(sur, p, cl, min(inc))
                            order = sorted(range(len(cl)), key=lambda i: (-score[i], vals[i]))
                            acq_info[str(k)] = {"incumbent_u": min(inc), "ei_max": float(score.max()), "n_cands": len(cl)}
                        else:
                            order = sorted(range(len(cl)), key=lambda i: (vals[i], cl[i]))
                            acq_info[str(k)] = {"n_cands": len(cl)}
                        picks = [cl[i] for i in order[:per]]
                        picks = _top_up(picks, {}, per, available, rng, p, k, out)
                        for c in picks:
                            i = cl.index(c) if c in gen[k]["cands"] and c in cl else None
                            batch.append((k, c))
                            rec = _circuit_record(sur, p, c, vals[i] if i is not None else None)
                            if method == "bayes_vil" and inc and i is not None:
                                rec["EI"] = float(score[i])
                            recs.append(rec)
                verify(batch, recs, rnd)
                out.setdefault("acq_log", []).append({"round": rnd, "n_obs_before": len(y_all) - len(batch), **acq_info})
        final = {}
        for k in spec.ks:
            vk = [v for v in verified if v["k"] == k]
            best = min(vk, key=lambda v: (v["measured_u"], v["circuit"]))
            final[str(k)] = {**{kk: vv for kk, vv in best.items() if kk not in ("k",)}, "source": "verified"}
        out["final"] = final
        out["verified"] = verified
        # diagnostic only (not part of the method, excluded from end-to-end cost): refit on all 128
        # observations and select by the surrogate, i.e. what the revised surrogate alone would choose
        with clock.timed("diagnostic"):
            M = np.stack(M_all)
            Y = np.asarray(y_all, float)[:, None]
            dsur = make_surrogate(surrogate_kind(method), p, [None], seed + 7, bayes_cfg, solver)
            if getattr(dsur, "is_bayesian", False):
                dsur.fit(M, Y, warm=False)
            else:
                dsur.fit(M, Y)
            res = search_fixed_k(make_obj(dsur), p, list(spec.ks), spec.max_swap_iters, spec.n_alternatives)
            out["refit_128"] = {str(k): {**_circuit_record(dsur, p, tuple(res[k][0][0]), float(res[k][0][1])), "source": "surrogate"} for k in spec.ks}
    # ------------------------------------------------------------- accounting
    n_disc = sum(1 for k in kinds if k == "discovery")
    n_ver = sum(1 for k in kinds if k == "verification")
    out["n_masks"] = int(oracle.n_masks)
    out["n_discovery_masks"] = n_disc
    out["n_verification_masks"] = n_ver
    out["n_prompt_evals_logical"] = int(oracle.n_masks * oracle.n_prompts_per_mask)
    if oracle.n_masks != spec.budget:
        raise AssertionError(f"{method}: budget parity violated ({oracle.n_masks} != {spec.budget})")
    out["costs"] = {
        "wall": dict(clock.wall),
        "cpu": dict(clock.cpu),
        "wall_e2e": float(sum(clock.wall.get(k, 0.0) for k in E2E_KEYS)),
        "cpu_e2e_excl_oracle": float(sum(clock.cpu.get(k, 0.0) for k in ("fit", "search", "acquisition"))),
    }
    out["model_summary"] = sur.summary() if hasattr(sur, "summary") else {}
    out["_data"] = {"masks": np.stack(M_all), "y": np.asarray(y_all), "kinds": kinds, "per_prompt": np.asarray(PP) if PP else None}
    return out
