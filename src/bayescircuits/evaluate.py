"""Held-out scoring helpers shared by all gates (evaluation side only)."""

from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def crps_from_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS per column. samples: (N, n); y: (n,). Sorted-sample estimator."""
    x = np.sort(samples, axis=0)
    N = x.shape[0]
    i = np.arange(1, N + 1)[:, None]
    ind = (y[None, :] < x).astype(float)
    return (2.0 / N**2) * np.sum((x - y[None, :]) * (N * ind - i + 0.5), axis=0)


def interval_metrics(truth: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    cov = (truth >= lo) & (truth <= hi)
    return {"coverage_90": float(cov.mean()), "interval_width_90": float(np.mean(hi - lo)), "n_targets": int(cov.size)}


def point_metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - truth
    sd = float(np.std(truth))
    return {
        "test_mae": float(np.mean(np.abs(err))),
        "test_rmse": float(np.sqrt(np.mean(err**2))),
        "test_mae_normalized": float(np.mean(np.abs(err)) / sd) if sd > 0 else float("nan"),
    }
