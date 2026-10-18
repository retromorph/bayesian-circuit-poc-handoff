"""Mask semantics, mask designs and surrogate feature construction.

Convention (binding, AGENTS.md invariant 2): ``mask[j] == 1`` means component j is
**ablated**; ``mask[j] == 0`` means it is preserved. ``retain_mask(C)`` ablates every
candidate component outside C (and nothing outside the candidate universe).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import numpy as np

MAIN, PAIR, CTX_MAIN, CTX_PAIR = 0, 1, 2, 3


def n_ablated(p: int, density: float) -> int:
    """Nearest feasible integer count of ablated components for a density."""
    return int(min(p, max(1, round(density * p))))


def mask_key(m: np.ndarray) -> bytes:
    return np.packbits(np.asarray(m, dtype=np.uint8)).tobytes() + bytes([len(m) % 256])


def mask_hex(m: np.ndarray) -> str:
    return np.packbits(np.asarray(m, dtype=np.uint8)).tobytes().hex()


def masks_sha256(masks: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(masks, dtype=np.uint8)).tobytes()).hexdigest()


def random_mask(rng: np.random.Generator, p: int, k: int) -> np.ndarray:
    m = np.zeros(p, dtype=np.uint8)
    m[rng.choice(p, size=k, replace=False)] = 1
    return m


def balanced_masks(
    rng: np.random.Generator,
    p: int,
    n: int,
    densities: Sequence[float],
    exclude: set[bytes] | None = None,
    max_tries: int = 100000,
) -> np.ndarray:
    """n distinct masks, cycling through densities (balanced counts), avoiding ``exclude``."""
    exclude = set() if exclude is None else exclude
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    tries = 0
    i = 0
    while len(out) < n:
        k = n_ablated(p, densities[i % len(densities)])
        m = random_mask(rng, p, k)
        key = mask_key(m)
        tries += 1
        if tries > max_tries:
            raise RuntimeError("could not draw enough distinct masks")
        if key in exclude or key in seen:
            continue  # deterministic resample (same RNG stream)
        seen.add(key)
        out.append(m)
        i += 1
    return np.stack(out)


def retain_mask(p: int, C: Iterable[int]) -> np.ndarray:
    m = np.ones(p, dtype=np.uint8)
    idx = list(C)
    if idx:
        m[np.asarray(idx, dtype=int)] = 0
    return m


def pair_index(p: int) -> tuple[np.ndarray, np.ndarray]:
    I, J = np.triu_indices(p, k=1)
    return I, J


def design_matrix(masks: np.ndarray, contexts: np.ndarray | None = None, order: int = 2) -> np.ndarray:
    """Binary feature matrix: main indicators (+ all pairwise products) (+ context interactions).

    If ``contexts`` (0/1 per row) is given, features are [phi, g * phi] so that every
    main/pair coefficient may differ by context (Gate A2)."""
    M = np.asarray(masks, dtype=np.float64)
    if M.ndim == 1:
        M = M[None, :]
    feats = [M]
    if order >= 2:
        I, J = pair_index(M.shape[1])
        feats.append(M[:, I] * M[:, J])
    phi = np.concatenate(feats, axis=1)
    if contexts is None:
        return phi
    g = np.asarray(contexts, dtype=np.float64)[:, None]
    return np.concatenate([phi, g * phi], axis=1)


def feature_groups(p: int, order: int = 2, with_context: bool = False) -> np.ndarray:
    n_pair = p * (p - 1) // 2 if order >= 2 else 0
    base = np.concatenate([np.full(p, MAIN), np.full(n_pair, PAIR)])
    if not with_context:
        return base
    return np.concatenate([base, np.where(base == MAIN, CTX_MAIN, CTX_PAIR)])


def feature_names(p: int, order: int = 2, with_context: bool = False) -> list[str]:
    names = [f"m{j}" for j in range(p)]
    if order >= 2:
        I, J = pair_index(p)
        names += [f"m{i}*m{j}" for i, j in zip(I, J)]
    if with_context:
        names = names + [f"g*{n}" for n in names]
    return names


def expand_contexts(masks: np.ndarray, contexts: Sequence[int | None]) -> tuple[np.ndarray, np.ndarray | None]:
    """Repeat each mask once per context value (context-free problems: contexts == [None])."""
    if len(contexts) == 1 and contexts[0] is None:
        return np.asarray(masks), None
    M = np.repeat(np.asarray(masks), len(contexts), axis=0)
    g = np.tile(np.asarray(contexts, dtype=float), len(masks))
    return M, g
