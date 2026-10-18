"""Official ``shapiq.ProxySPEX`` interaction-attribution diagnostic.

ProxySPEX samples its *own* coalitions (retained sets); every coalition evaluation is a real
oracle call and is counted. It returns interaction attributions (FBII by default), **not** an
intervention-response predictor, so it is only compared on pairwise-interaction retrieval.
Game: v(T) = response of the frozen system when only coalition T (retained components) is kept,
i.e. mask m = 1 - T. For a pure pair term c*m_i*m_j the order-2 attribution has magnitude |c|.
"""

from __future__ import annotations

import time

import numpy as np


def run_proxyspex(n: int, value_fn, budget: int, seed: int, index: str = "FBII", top: int = 10) -> dict:
    import shapiq

    calls = {"n": 0}

    def game(coalitions: np.ndarray) -> np.ndarray:
        T = np.asarray(coalitions, dtype=bool)
        calls["n"] += len(T)
        return np.asarray(value_fn((~T).astype(np.uint8)), dtype=float)

    t0, c0 = time.perf_counter(), time.process_time()
    ap = shapiq.ProxySPEX(n=n, index=index, max_order=2, random_state=int(seed))
    iv = ap.approximate(budget=budget, game=game)
    dv = iv.dict_values
    pairs = sorted(((k, float(v)) for k, v in dv.items() if len(k) == 2), key=lambda t: -abs(t[1]))
    singles = sorted(((k, float(v)) for k, v in dv.items() if len(k) == 1), key=lambda t: -abs(t[1]))
    return {
        "index": index,
        "budget": budget,
        "oracle_calls": calls["n"],
        "wall_seconds": time.perf_counter() - t0,
        "cpu_seconds": time.process_time() - c0,
        "top_pairs": [{"pair": [int(a) for a in k], "value": v} for k, v in pairs[:top]],
        "top_singles": [{"comp": int(k[0]), "value": v} for k, v in singles[:top]],
        "baseline_value": float(iv.baseline_value),
    }
