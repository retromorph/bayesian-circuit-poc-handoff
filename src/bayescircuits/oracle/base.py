"""Budget-accounted discovery oracle.

Discovery methods only ever receive a :class:`DiscoveryOracle`. It

* charges one logical budget unit per *distinct* mask (fixed prompt batch per mask),
* refuses repeated masks (no free duplicate evidence) and reserved blind-test masks,
* refuses queries beyond the method's budget,
* logs every query (mask id, bits, prompt batch id, outputs, timing, cache use).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..masks import mask_hex, mask_key


class BudgetExceeded(RuntimeError):
    pass


class ReservedMaskError(RuntimeError):
    pass


@dataclass
class QueryResult:
    masks: np.ndarray  # (b, p)
    y: dict[str, np.ndarray]  # named per-mask targets, each (b, n_ctx) or (b,)
    per_prompt: np.ndarray | None = None  # (b, n_prompts) raw d(x, m) when applicable


@dataclass
class DiscoveryOracle:
    fn: Callable[[np.ndarray], QueryResult]
    p: int
    budget: int
    method: str
    unit_id: str
    prompt_batch_id: str
    reserved_keys: frozenset = frozenset()
    shared_cache: dict | None = None
    n_prompts_per_mask: int = 1
    log: list[dict] = field(default_factory=list)
    _seen: set = field(default_factory=set)
    oracle_seconds: float = 0.0

    @property
    def n_masks(self) -> int:
        return len(self._seen)

    def is_available(self, m: np.ndarray) -> bool:
        k = mask_key(m)
        return k not in self._seen and k not in self.reserved_keys

    def query(self, masks: np.ndarray) -> QueryResult:
        masks = np.atleast_2d(np.asarray(masks, dtype=np.uint8))
        keys = [mask_key(m) for m in masks]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate masks within a query batch")
        for k in keys:
            if k in self.reserved_keys:
                raise ReservedMaskError("attempted to query a reserved blind mask")
            if k in self._seen:
                raise ValueError("mask already queried by this method (no free duplicates)")
        if self.n_masks + len(masks) > self.budget:
            raise BudgetExceeded(f"{self.method}: {self.n_masks}+{len(masks)} > {self.budget}")
        # physical execution (optionally reusing a cache shared between methods;
        # the logical budget is charged regardless)
        cached = [self.shared_cache is not None and k in self.shared_cache for k in keys]
        todo = [i for i, c in enumerate(cached) if not c]
        t0 = time.perf_counter()
        fresh = self.fn(masks[todo]) if todo else None
        dt = time.perf_counter() - t0
        self.oracle_seconds += dt
        names = list(fresh.y) if fresh is not None else list(self.shared_cache[keys[0]]["y"])
        y = {nm: [None] * len(masks) for nm in names}
        pp = [None] * len(masks)
        for r, i in enumerate(todo):
            for nm in names:
                y[nm][i] = fresh.y[nm][r]
            if fresh.per_prompt is not None:
                pp[i] = fresh.per_prompt[r]
            if self.shared_cache is not None:
                self.shared_cache[keys[i]] = {
                    "y": {nm: fresh.y[nm][r] for nm in names},
                    "pp": None if fresh.per_prompt is None else fresh.per_prompt[r],
                    "ms": 1000 * dt / max(1, len(todo)),
                }
        for i, c in enumerate(cached):
            if c:
                ent = self.shared_cache[keys[i]]
                for nm in names:
                    y[nm][i] = ent["y"][nm]
                pp[i] = ent["pp"]
        out = QueryResult(
            masks=masks,
            y={nm: np.asarray(v) for nm, v in y.items()},
            per_prompt=None if any(v is None for v in pp) else np.asarray(pp),
        )
        now = time.time()
        for i, m in enumerate(masks):
            self._seen.add(keys[i])
            row = {
                "unit": self.unit_id,
                "method": self.method,
                "query_index": self.n_masks - len(masks) + i,
                "mask_id": f"{self.unit_id}:{mask_hex(m)}",
                "mask_bits": mask_hex(m),
                "n_ablated": int(m.sum()),
                "prompt_batch_id": self.prompt_batch_id,
                "n_prompt_evals": self.n_prompts_per_mask,
                "reused_cache": bool(cached[i]),
                "time_ms": (self.shared_cache[keys[i]]["ms"] if cached[i] else 1000 * dt / max(1, len(todo))),
                "timestamp": now,
            }
            for nm in names:
                v = np.atleast_1d(out.y[nm][i])
                for c, vv in enumerate(v):
                    row[f"y_{nm}_{c}"] = float(vv)
            self.log.append(row)
        return out
