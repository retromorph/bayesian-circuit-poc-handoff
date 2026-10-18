"""Mask acquisition policies.

* static:       seeded balanced-density masks fixed before discovery (shared by passive methods)
* uncertainty:  greedy batch maximisation of (fantasised) posterior function variance
* decision:     greedy batch maximisation of the approximate expected reduction of the
                circuit-selection Bayes risk  V(D) = min_C E[L(C; theta) | D]  per unit cost

Both active policies use the *same* candidate pool, the same posterior particles and the
same "hypothetical truth" fantasy: for particle s* acting as the truth, outcomes
y_a = f_{s*}(a) + sigma_{s*} z (common random numbers z), and the posterior after the
batch is approximated by importance re-weighting of the particle set.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .masks import balanced_masks, mask_key, retain_mask


def candidate_pool(
    rng: np.random.Generator,
    p: int,
    unavailable: set[bytes],
    densities: list[float],
    shortlist: list[tuple[int, ...]],
    n_random: int,
    n_circuit: int,
) -> np.ndarray:
    """Bounded pool: balanced random masks + retain-circuit tests and their 1-edit neighbours."""
    pool: list[np.ndarray] = []
    keys: set[bytes] = set(unavailable)

    def add(m: np.ndarray) -> None:
        k = mask_key(m)
        if k not in keys and m.any():
            keys.add(k)
            pool.append(m)

    circ: list[np.ndarray] = []
    for C in shortlist:
        circ.append(retain_mask(p, C))
    order = list(range(len(shortlist)))
    tries = 0
    while len(circ) < 4 * n_circuit and tries < 20 * n_circuit and shortlist:
        tries += 1
        C = list(shortlist[order[tries % len(order)]])
        r = rng.random()
        outside = [j for j in range(p) if j not in C]
        if r < 0.34 and len(C) > 1:
            C2 = [c for c in C if c != C[rng.integers(len(C))]]
        elif r < 0.67 and outside:
            C2 = C + [outside[rng.integers(len(outside))]]
        elif outside and C:
            C2 = [c for c in C if c != C[rng.integers(len(C))]] + [outside[rng.integers(len(outside))]]
        else:
            continue
        circ.append(retain_mask(p, C2))
    before = len(pool)
    for m in circ:
        if len(pool) - before >= n_circuit:
            break
        add(m)
    rnd = balanced_masks(rng, p, n_random + (n_circuit - (len(pool) - before)), densities, exclude=keys)
    for m in rnd:
        add(m)
    return np.stack(pool)


def _loglik(Y: np.ndarray, F: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    """Y: (S*, N, C) fantasised outcomes; F: (S, N, C); sig2: (S,). -> (N, S*, S)."""
    diff = Y[:, None, :, :] - F[None, :, :, :]  # (S*, S, N, C)
    ll = -0.5 * diff**2 / sig2[None, :, None, None] - 0.5 * np.log(2 * np.pi * sig2)[None, :, None, None]
    return ll.sum(-1).transpose(2, 0, 1)


def _fantasy(F: np.ndarray, sig2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal(F.shape)
    return F + np.sqrt(sig2)[:, None, None] * z


def select_uncertainty(
    F: np.ndarray, sig2: np.ndarray, batch: int, rng: np.random.Generator
) -> tuple[list[int], dict]:
    """F: (S, N, C) particle predictions at pool masks."""
    S, N, _ = F.shape
    Y = _fantasy(F, sig2, rng)
    LL = _loglik(Y, F, sig2)  # (N, S*, S)
    acc = np.zeros((S, S))
    chosen: list[int] = []
    scores_first = None
    for _ in range(min(batch, N)):
        W = np.exp(acc - logsumexp(acc, axis=1, keepdims=True))  # (S*, S)
        m1 = np.einsum("ts,snc->tnc", W, F)
        m2 = np.einsum("ts,snc->tnc", W, F**2)
        var = np.maximum(m2 - m1**2, 0.0).mean(-1).mean(0)  # (N,)
        var[chosen] = -np.inf
        if scores_first is None:
            scores_first = var.copy()
        best = _argmax_tiebreak(var, rng)
        chosen.append(best)
        acc = acc + LL[best]
    return chosen, {"score_first": _finite_stats(scores_first)}


def bayes_risk(W: np.ndarray, Lmats: list[np.ndarray]) -> np.ndarray:
    """W: (..., S) particle weights; Lmats: list of (S, J_g). -> (...,) mean_g min_j E_w[L]."""
    vals = [np.min(W @ L, axis=-1) for L in Lmats]
    return np.mean(vals, axis=0)


def select_decision(
    F: np.ndarray, sig2: np.ndarray, Lmats: list[np.ndarray], batch: int, rng: np.random.Generator, cost: np.ndarray | None = None
) -> tuple[list[int], dict]:
    """Greedy batch maximisation of [V(D) - E V(D + batch + a)] / cost(a)."""
    S, N, _ = F.shape
    cost = np.ones(N) if cost is None else cost
    Y = _fantasy(F, sig2, rng)
    LL = _loglik(Y, F, sig2)  # (N, S*, S)
    acc = np.zeros((S, S))
    W0 = np.full(S, 1.0 / S)
    V0 = float(bayes_risk(W0[None, :], Lmats)[0])
    chosen: list[int] = []
    gains_first = None
    n_ties = 0
    v_cur = V0
    for _ in range(min(batch, N)):
        A = acc[None, :, :] + LL  # (N, S*, S)
        W = np.exp(A - logsumexp(A, axis=2, keepdims=True))
        V = bayes_risk(W, Lmats).mean(1)  # (N,)
        gain = (v_cur - V) / cost
        gain[chosen] = -np.inf
        if gains_first is None:
            gains_first = gain.copy()
        finite = gain[np.isfinite(gain)]
        if finite.size and np.ptp(finite) < 1e-12:
            n_ties += 1
        best = _argmax_tiebreak(gain, rng)
        chosen.append(best)
        acc = acc + LL[best]
        v_cur = float(V[best])
    return chosen, {
        "V0": V0,
        "V_after_batch_expected": v_cur,
        "gain_first": _finite_stats(gains_first),
        "n_tied_steps": n_ties,
    }


def _argmax_tiebreak(v: np.ndarray, rng: np.random.Generator, tol: float = 1e-12) -> int:
    m = np.max(v)
    ids = np.where(v >= m - tol * max(1.0, abs(m)))[0]
    return int(ids[rng.integers(len(ids))])


def _finite_stats(v: np.ndarray | None) -> dict:
    if v is None:
        return {}
    f = v[np.isfinite(v)]
    if not f.size:
        return {}
    return {"max": float(f.max()), "mean": float(f.mean()), "min": float(f.min())}
