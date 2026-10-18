"""Common surrogate interface for all predictive methods.

Every surrogate is fit on (masks, Y) where Y has shape (n_masks, n_ctx); n_ctx == 1 for
context-free problems and 2 for Gate A2 (context groups g=0, g=1). All methods share the
same mask coding, feature dictionary (where applicable) and no-intercept convention
(d(x, 0) == 0 by definition).
"""

from __future__ import annotations

import warnings
from typing import Protocol

import numpy as np

from ..masks import design_matrix, expand_contexts, feature_groups
from .bayes import Posterior, SpikeSlabConfig, fit_spike_slab


class Surrogate(Protocol):
    name: str
    is_bayesian: bool

    def fit(self, masks: np.ndarray, Y: np.ndarray) -> None: ...

    def predict(self, masks: np.ndarray) -> np.ndarray: ...


def _ctx_arg(contexts: list) -> list:
    return contexts if contexts else [None]


def _flatten_y(Y: np.ndarray) -> np.ndarray:
    return np.asarray(Y, dtype=float).reshape(-1)


def _fold_ids(n_masks: int, n_ctx: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng([seed, 1234])
    f = rng.permutation(np.arange(n_masks) % k)
    return np.repeat(f, n_ctx)


class BayesSurrogate:
    """Spike-and-slab main + pairwise (+ context) regression with posterior particles."""

    is_bayesian = True

    def __init__(self, p: int, contexts: list, cfg: SpikeSlabConfig, seed: int, name: str = "bayes"):
        self.name = name
        self.p = p
        self.contexts = _ctx_arg(contexts)
        self.with_ctx = self.contexts[0] is not None
        self.cfg = cfg
        self.rng = np.random.default_rng([seed, 4242])
        self.groups = feature_groups(p, 2, self.with_ctx)
        self.post: Posterior | None = None
        self._init_states: list | None = None

    def _X(self, masks: np.ndarray) -> np.ndarray:
        M, g = expand_contexts(masks, self.contexts)
        return design_matrix(M, g, order=2)

    def fit(self, masks: np.ndarray, Y: np.ndarray, warm: bool = True) -> None:
        X = self._X(masks)
        y = _flatten_y(Y)
        self.post = fit_spike_slab(
            X, y, self.groups, self.cfg, self.rng, init_states=self._init_states if warm else None
        )
        self._init_states = self.post.last_states

    def predict_particles(self, masks: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        """Function samples, shape (S, n_masks, n_ctx)."""
        F = self.post.predict_f(self._X(masks), idx)
        return F.reshape(F.shape[0], len(masks), len(self.contexts))

    def predict(self, masks: np.ndarray) -> np.ndarray:
        return self.predict_particles(masks).mean(0)

    def predictive(self, masks: np.ndarray, level: float = 0.90, extra_var: np.ndarray | None = None) -> dict:
        X = self._X(masks)
        ev = None if extra_var is None else np.repeat(np.asarray(extra_var), len(self.contexts))
        out = self.post.predictive_summary(X, self.rng, level=level, extra_var=ev)
        shp = (len(masks), len(self.contexts))
        return {k: (v.reshape(shp) if k != "samples" else v) for k, v in out.items()}

    def particle_subset(self, n: int) -> np.ndarray:
        S = self.post.n_particles
        return np.linspace(0, S - 1, min(n, S)).round().astype(int)

    def summary(self, top: int = 20) -> dict:
        from ..masks import feature_names

        inc = self.post.inclusion_probs()
        names = feature_names(self.p, 2, self.with_ctx)
        order = np.argsort(-inc)[:top]
        return {
            "top_inclusion": [{"feature": names[i], "pip": float(inc[i])} for i in order],
            "sigma_mean": float(np.sqrt(self.post.sigma2).mean()),
            "support_size_mean": float(np.mean([len(s) for s in self.post.supports])),
            **{f"diag_{k}": v for k, v in self.post.diagnostics.items()},
        }


class LassoSurrogate:
    """L1 regression on main (order=1) or main+pairwise (order=2) mask indicators.

    Columns are RMS-scaled (no centering, no intercept), alpha chosen by K-fold CV on
    discovery data only (masks of a context pair are kept in the same fold)."""

    is_bayesian = False

    def __init__(self, p: int, contexts: list, order: int, seed: int, name: str | None = None, tol: float = 1e-7, max_iter: int = 50000):
        self.tol = tol
        self.max_iter = max_iter
        self.p = p
        self.contexts = _ctx_arg(contexts)
        self.with_ctx = self.contexts[0] is not None
        self.order = order
        self.seed = seed
        self.name = name or ("lasso_1st" if order == 1 else "lasso_pair")
        self.coef_: np.ndarray | None = None
        self.alpha_: float | None = None

    def _X(self, masks: np.ndarray) -> np.ndarray:
        M, g = expand_contexts(masks, self.contexts)
        return design_matrix(M, g, order=self.order)

    def fit(self, masks: np.ndarray, Y: np.ndarray) -> None:
        from sklearn.linear_model import LassoCV
        from sklearn.model_selection import PredefinedSplit

        X = self._X(masks)
        y = _flatten_y(Y)
        sc = np.sqrt(np.mean(X**2, axis=0))
        sc[sc == 0] = 1.0
        Xs = X / sc
        folds = _fold_ids(len(masks), len(self.contexts), 5, self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LassoCV(
                fit_intercept=False,
                cv=PredefinedSplit(folds),
                alphas=60,
                eps=1e-4,
                max_iter=self.max_iter,
                tol=self.tol,
                selection="cyclic",
            ).fit(Xs, y)
        self.coef_ = m.coef_ / sc
        self.alpha_ = float(m.alpha_)

    def predict(self, masks: np.ndarray) -> np.ndarray:
        return (self._X(masks) @ self.coef_).reshape(len(masks), len(self.contexts))

    def summary(self, top: int = 20) -> dict:
        from ..masks import feature_names

        names = feature_names(self.p, self.order, self.with_ctx)
        order = np.argsort(-np.abs(self.coef_))[:top]
        return {
            "alpha": self.alpha_,
            "n_nonzero": int(np.sum(self.coef_ != 0)),
            "top_coef": [{"feature": names[i], "coef": float(self.coef_[i])} for i in order if self.coef_[i] != 0],
        }


GBDT_GRID = [
    {"num_leaves": 4, "min_child_samples": 3, "n_estimators": 400, "learning_rate": 0.05},
    {"num_leaves": 8, "min_child_samples": 3, "n_estimators": 400, "learning_rate": 0.05},
    {"num_leaves": 8, "min_child_samples": 5, "n_estimators": 800, "learning_rate": 0.02},
    {"num_leaves": 16, "min_child_samples": 2, "n_estimators": 600, "learning_rate": 0.05},
]


class GBDTSurrogate:
    """LightGBM mask-response surrogate; hyper-parameters chosen by 4-fold CV on discovery data."""

    is_bayesian = False

    def __init__(self, p: int, contexts: list, seed: int, name: str = "gbdt", n_jobs: int = 1):
        self.p = p
        self.contexts = _ctx_arg(contexts)
        self.with_ctx = self.contexts[0] is not None
        self.seed = seed
        self.name = name
        self.n_jobs = n_jobs
        self.model = None
        self.params_: dict | None = None
        self.cv_mae_: list | None = None

    def _X(self, masks: np.ndarray) -> np.ndarray:
        M, g = expand_contexts(masks, self.contexts)
        M = np.asarray(M, dtype=float)
        return M if g is None else np.concatenate([M, g[:, None]], axis=1)

    def _make(self, params: dict):
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            **params,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_split_gain=0.0,
            random_state=self.seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=self.n_jobs,
            verbose=-1,
        )

    def fit(self, masks: np.ndarray, Y: np.ndarray) -> None:
        X = self._X(masks)
        y = _flatten_y(Y)
        folds = _fold_ids(len(masks), len(self.contexts), 4, self.seed)
        maes = []
        for params in GBDT_GRID:
            err = []
            for f in range(4):
                tr, te = folds != f, folds == f
                mdl = self._make(params).fit(X[tr], y[tr])
                err.append(np.mean(np.abs(mdl.predict(X[te]) - y[te])))
            maes.append(float(np.mean(err)))
        self.cv_mae_ = maes
        self.params_ = GBDT_GRID[int(np.argmin(maes))]
        self.model = self._make(self.params_).fit(X, y)

    def predict(self, masks: np.ndarray) -> np.ndarray:
        return self.model.predict(self._X(masks)).reshape(len(masks), len(self.contexts))

    def summary(self, top: int = 20) -> dict:
        return {"params": self.params_, "cv_mae": self.cv_mae_}


def make_surrogate(method: str, p: int, contexts: list, seed: int, bayes_cfg: SpikeSlabConfig, solver: dict | None = None):
    solver = solver or {}
    lkw = {k: solver[k] for k in ("tol", "max_iter") if k in solver}
    if method == "lasso_1st":
        return LassoSurrogate(p, contexts, 1, seed, **lkw)
    if method == "lasso_pair":
        return LassoSurrogate(p, contexts, 2, seed, **lkw)
    if method == "gbdt":
        return GBDTSurrogate(p, contexts, seed)
    if method.startswith("bayes"):
        return BayesSurrogate(p, contexts, bayes_cfg, seed, name=method)
    raise KeyError(method)


def pair_nonadditivity(predict_fn, p: int) -> np.ndarray:
    """NA_ij = f(e_i+e_j) - f(e_i) - f(e_j) + f(0) from *predicted interventions*.

    predict_fn(masks) -> (..., n_masks, n_ctx). Returns (..., n_pairs) averaged over ctx."""
    I, J = np.triu_indices(p, 1)
    eye = np.eye(p, dtype=np.uint8)
    zero = np.zeros((1, p), dtype=np.uint8)
    pair_masks = np.clip(eye[I] + eye[J], 0, 1)
    allm = np.concatenate([zero, eye, pair_masks])
    F = predict_fn(allm)
    f0 = F[..., 0:1, :]
    fs = F[..., 1 : 1 + p, :]
    fp = F[..., 1 + p :, :]
    na = fp - fs[..., I, :] - fs[..., J, :] + f0
    return na.mean(-1)
