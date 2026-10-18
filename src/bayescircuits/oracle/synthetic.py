"""Gate A synthetic causal-intervention worlds (A0-A3).

The planted mechanism is defined in *planted coordinates* 0..5 and mapped to hidden
component indices by a random permutation. Ground truth lives only on the world object;
discovery code only receives a ``DiscoveryOracle`` wrapping ``world.evaluate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

BASE_TERMS: dict[str, list[tuple[float, tuple[int, ...], int | None]]] = {
    # (base coefficient, planted indices, context-gate: None | 0 -> active when g=0 | 1 -> active when g=1)
    "A0": [(1.0, (0,), None), (0.8, (1,), None), (-0.7, (2,), None)],
    "A1": [(1.0, (0,), None), (0.9, (1, 2), None), (0.8, (3, 4), None)],
    "A2": [(1.0, (0,), None), (1.0, (1, 2), 0), (1.0, (3, 4), 1)],
    "A3": [(1.0, (0,), None), (0.9, (1, 2, 3), None), (0.8, (4, 5), None)],
}
CONTEXT_REGIMES = {"A2"}
A1_PLANTED_CIRCUITS = [(0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4)]
A1_PLANTED_PAIRS = [(1, 2), (3, 4)]


@dataclass
class SyntheticWorld:
    regime: str
    seed: int
    p: int
    jitter: float
    perm: np.ndarray = field(repr=False)
    terms: list[tuple[float, tuple[int, ...], int | None]] = field(repr=False)

    @classmethod
    def create(cls, regime: str, seed: int, p: int = 32, jitter: float = 0.20) -> SyntheticWorld:
        rng = np.random.default_rng([seed, 7919])
        perm = rng.permutation(p)
        terms = []
        for coef, idx, gate in BASE_TERMS[regime]:
            c = coef * (1.0 + rng.uniform(-jitter, jitter))
            terms.append((float(c), tuple(int(perm[i]) for i in idx), gate))
        return cls(regime=regime, seed=seed, p=p, jitter=jitter, perm=perm, terms=terms)

    @property
    def has_context(self) -> bool:
        return self.regime in CONTEXT_REGIMES

    def evaluate(self, masks: np.ndarray, g: np.ndarray | None = None) -> np.ndarray:
        """d(m) (or d(g, m)) = s(F(x)) - s(F^do(m)(x)); d(0) == 0 exactly."""
        M = np.atleast_2d(np.asarray(masks, dtype=float))
        out = np.zeros(M.shape[0])
        for c, idx, gate in self.terms:
            term = c * np.prod(M[:, list(idx)], axis=1)
            if gate is not None:
                if g is None:
                    raise ValueError("context regime requires g")
                gg = np.asarray(g, dtype=float)
                term = term * (gg if gate == 1 else (1.0 - gg))
            out += term
        return out

    # ---- evaluation-only ground truth ------------------------------------------------
    def true_circuits(self) -> list[tuple[int, ...]]:
        if self.regime != "A1":
            return []
        return [tuple(sorted(int(self.perm[i]) for i in c)) for c in A1_PLANTED_CIRCUITS]

    def true_pairs(self) -> list[tuple[int, int]]:
        if self.regime != "A1":
            return []
        return [tuple(sorted(int(self.perm[i]) for i in pr)) for pr in A1_PLANTED_PAIRS]

    def true_R(self, circuit: tuple[int, ...]) -> float:
        from ..masks import retain_mask

        m = retain_mask(self.p, circuit)[None, :]
        full = np.ones((1, self.p))
        if self.has_context:
            num = np.mean([abs(self.evaluate(m, np.array([g]))[0]) for g in (0, 1)])
            den = np.mean([abs(self.evaluate(full, np.array([g]))[0]) for g in (0, 1)])
        else:
            num = abs(self.evaluate(m)[0])
            den = abs(self.evaluate(full)[0])
        return float(num / den)

    def truth_record(self) -> dict:
        return {
            "regime": self.regime,
            "seed": self.seed,
            "perm": self.perm.tolist(),
            "terms": [{"coef": c, "components": list(i), "context_gate": g} for c, i, g in self.terms],
            "true_circuits": [list(c) for c in self.true_circuits()],
            "true_pairs": [list(pq) for pq in self.true_pairs()],
        }


def verify_a1_minimal_circuits(world: SyntheticWorld, tol: float = 0.10) -> dict:
    """Exhaustive check over the 5 planted relevant indices (never 2^32 masks)."""
    rel = [int(world.perm[i]) for i in range(5)]
    faithful_by_size: dict[int, list[tuple[int, ...]]] = {}
    for bits in product([0, 1], repeat=5):
        C = tuple(sorted(r for r, b in zip(rel, bits) if b))
        if world.true_R(C) <= tol:
            faithful_by_size.setdefault(len(C), []).append(C)
    kmin = min(faithful_by_size)
    return {"min_size": kmin, "minimal": sorted(faithful_by_size[kmin])}
