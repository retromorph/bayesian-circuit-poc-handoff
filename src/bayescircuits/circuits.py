"""Shared, size-constrained retained-circuit search used by every surrogate.

A circuit C is a retained subset of the fixed candidate pool; its intervention is
``retain_mask(C)`` (ablate the complement inside the pool). Predicted normalized
discrepancy for a surrogate prediction f:

    R_hat(C) = mean_ctx |f(retain(C))| / mean_ctx |f(ablate all)|

For Bayesian surrogates the same quantity is computed per posterior particle, giving
P(R(C) <= 0.10 | D) and posterior expected losses.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .masks import retain_mask

EPS = 1e-9


def enumerate_circuits(p: int, sizes: list[int]) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for k in sizes:
        out.extend(combinations(range(p), k))
    return out


def circuit_masks(p: int, circuits: list[tuple[int, ...]]) -> np.ndarray:
    return np.stack([retain_mask(p, C) for C in circuits]) if circuits else np.zeros((0, p), np.uint8)


def predicted_R(predict_fn, p: int, circuits: list[tuple[int, ...]], chunk: int = 2048) -> np.ndarray:
    """predict_fn(masks)->(..., n, n_ctx). Returns R with shape (..., n_circuits)."""
    full = np.ones((1, p), dtype=np.uint8)
    den = np.abs(predict_fn(full)).mean(-1)[..., 0]
    nums = []
    for s in range(0, len(circuits), chunk):
        M = circuit_masks(p, circuits[s : s + chunk])
        nums.append(np.abs(predict_fn(M)).mean(-1))
    num = np.concatenate(nums, axis=-1)
    return num / np.maximum(den[..., None], EPS)


def loss_matrix(R: np.ndarray, sizes: np.ndarray, p: int, lam: float, rmax: float = 0.10) -> np.ndarray:
    """L(C; theta) = |C|/p + lam * max(R(C;theta) - rmax, 0)."""
    return sizes[None, :] / p + lam * np.maximum(R - rmax, 0.0)


def propose_minimal(
    circuits: list[tuple[int, ...]],
    score: np.ndarray,
    bayesian: bool,
    max_props: int = 4,
    rmax: float = 0.10,
    p_min: float = 0.5,
) -> dict:
    """Minimal-cardinality faithful proposals.

    Non-Bayesian: smallest size with some R_hat <= rmax; up to ``max_props`` circuits of that
    size with R_hat <= rmax, ranked by R_hat. Bayesian: ``score`` is P(R<=rmax|D); smallest size
    with some P >= p_min; up to ``max_props`` circuits with P >= p_min, ranked by P."""
    sizes = np.array([len(c) for c in circuits])
    for k in sorted(set(sizes.tolist())):
        idx = np.where(sizes == k)[0]
        s = score[idx]
        ok = (s >= p_min) if bayesian else (s <= rmax)
        if ok.any():
            cand = idx[ok]
            order = np.argsort(-score[cand] if bayesian else score[cand], kind="stable")[:max_props]
            return {"size": int(k), "circuits": [circuits[i] for i in cand[order]], "scores": score[cand[order]].tolist()}
    return {"size": None, "circuits": [], "scores": []}


def search_fixed_k(
    objective,
    p: int,
    ks: list[int],
    max_swap_iters: int = 12,
    n_alternatives: int = 6,
) -> dict[int, list[tuple[tuple[int, ...], float]]]:
    """Greedy backward elimination + best-improvement swap local search.

    objective(masks (n,p)) -> (n,) values to minimise (e.g. predicted R or posterior expected loss).
    Returns, for each k, a list of (circuit, objective) sorted ascending, the first entry being
    the selected circuit and the rest near-best alternatives found during the search."""
    retained = list(range(p))
    by_k: dict[int, tuple[int, ...]] = {}
    kset = set(ks)
    while len(retained) > min(ks):
        cands = [tuple(c for c in retained if c != r) for r in retained]
        vals = objective(circuit_masks(p, cands))
        best = int(np.argmin(vals))
        retained = list(cands[best])
        if len(retained) in kset:
            by_k[len(retained)] = tuple(retained)
    if p in kset:
        by_k[p] = tuple(range(p))
    out: dict[int, list[tuple[tuple[int, ...], float]]] = {}
    for k in ks:
        C = by_k[k]
        cur = float(objective(circuit_masks(p, [C]))[0])
        pool: dict[tuple[int, ...], float] = {C: cur}
        for _ in range(max_swap_iters):
            inside = list(C)
            outside = [j for j in range(p) if j not in C]
            cands = [tuple(sorted([x for x in inside if x != i] + [j])) for i in inside for j in outside]
            if not cands:
                break
            vals = objective(circuit_masks(p, cands))
            for c, v in zip(cands, vals):
                if c not in pool or v < pool[c]:
                    pool[c] = float(v)
            b = int(np.argmin(vals))
            if vals[b] < cur - 1e-12:
                C, cur = cands[b], float(vals[b])
            else:
                break
        ranked = sorted(pool.items(), key=lambda t: t[1])
        sel = [(C, cur)] + [(c, v) for c, v in ranked if c != C][: n_alternatives - 1]
        out[k] = sel
    return out
