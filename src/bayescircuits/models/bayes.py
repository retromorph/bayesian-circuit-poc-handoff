"""Sparse Bayesian main + pairwise interaction regression (spike-and-slab).

Model (all on the standardized response y/scale, no intercept because d(x, 0) = 0):

    y = X_S beta_S + eps,   eps ~ N(0, sigma^2 I)
    beta_S | sigma^2 ~ N(0, sigma^2 tau^2 I)          (conjugate slab)
    sigma^2 ~ InvGamma(a0, b0)
    gamma_g  ~ Beta-Binomial(a_pi, b_pi) independently within each feature group g
               (main effects, pairwise effects, context x main, context x pair)

No heredity constraint is imposed: a pairwise feature may enter with no main effect.
Supports are sampled with a collapsed (beta, sigma^2 integrated out) Metropolis-Hastings
sampler that combines (i) locally-balanced informed single-flip proposals
(Zanella 2020, sqrt balancing) and (ii) informed swap moves. Coefficients and noise are
then drawn exactly from their conditional conjugate posteriors, producing posterior
particles theta_s = (S_s, beta_s, sigma2_s) that carry both support and coefficient
uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import betaln, logsumexp


@dataclass(frozen=True)
class SpikeSlabConfig:
    tau: float = 5.0
    a0: float = 1.0
    b0: float = 1e-4
    a_pi: float = 1.0
    b_pi: float = 1.0
    n_iter: int = 1500
    burn: int = 500
    n_chains: int = 2
    n_keep: int = 200
    swap_prob: float = 0.3
    max_support: int = 64
    warm_iter: int = 600
    warm_burn: int = 200

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Posterior:
    """Posterior particles in original response units."""

    supports: list[np.ndarray]
    betas: list[np.ndarray]
    sigma2: np.ndarray  # noise variances (original units)
    n_features: int
    diagnostics: dict = field(default_factory=dict)
    last_states: list[np.ndarray] = field(default_factory=list)

    @property
    def n_particles(self) -> int:
        return len(self.supports)

    def coef_matrix(self, idx: np.ndarray | None = None) -> np.ndarray:
        """Dense (K, S) coefficient matrix for selected particles."""
        ids = np.arange(self.n_particles) if idx is None else np.asarray(idx)
        B = np.zeros((self.n_features, len(ids)))
        for c, s in enumerate(ids):
            if len(self.supports[s]):
                B[self.supports[s], c] = self.betas[s]
        return B

    def predict_f(self, X: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        """Function samples (S, n) without observation noise."""
        return (X @ self.coef_matrix(idx)).T

    def inclusion_probs(self) -> np.ndarray:
        inc = np.zeros(self.n_features)
        for s in self.supports:
            inc[s] += 1.0
        return inc / max(1, self.n_particles)

    def predictive_samples(self, X: np.ndarray, rng: np.random.Generator, n_noise: int = 10) -> np.ndarray:
        F = self.predict_f(X)  # (S, n)
        sd = np.sqrt(self.sigma2)[:, None, None]
        eps = rng.standard_normal((F.shape[0], n_noise, F.shape[1])) * sd
        return (F[:, None, :] + eps).reshape(-1, F.shape[1])

    def predictive_summary(
        self, X: np.ndarray, rng: np.random.Generator, level: float = 0.90, extra_var: np.ndarray | None = None
    ) -> dict:
        """Posterior predictive mean / central interval for new observations.

        extra_var: optional per-row additional variance (e.g. prompt-resampling variance
        estimated from discovery data only)."""
        F = self.predict_f(X)
        n_noise = 10
        var = self.sigma2[:, None] + (0.0 if extra_var is None else np.asarray(extra_var)[None, :])
        eps = rng.standard_normal((F.shape[0], n_noise, F.shape[1])) * np.sqrt(var)[:, None, :]
        Y = (F[:, None, :] + eps).reshape(-1, F.shape[1])
        lo, hi = np.quantile(Y, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
        return {
            "mean": F.mean(0),
            "f_sd": F.std(0),
            "pred_sd": Y.std(0),
            "lo": lo,
            "hi": hi,
            "samples": Y,
        }


class _Data:
    def __init__(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: SpikeSlabConfig):
        self.X = X
        self.n, self.K = X.shape
        self.groups = groups.astype(int)
        self.n_groups = int(self.groups.max()) + 1 if self.K else 0
        self.K_g = np.bincount(self.groups, minlength=self.n_groups)
        rms = float(np.sqrt(np.mean(y**2)))
        self.scale = rms if rms > 1e-12 else 1.0
        self.y = y / self.scale
        self.G = X.T @ X
        self.dG = np.diag(self.G).copy()
        self.u = X.T @ self.y
        self.yy = float(self.y @ self.y)
        self.cfg = cfg
        self.inv_tau2 = 1.0 / cfg.tau**2
        self.log_tau = math.log(cfg.tau)
        self.shape = cfg.a0 + 0.5 * self.n
        # columns that are identically zero carry no information: forbid them
        self.allowed = self.dG > 0


class _State:
    """Sufficient statistics for a support S."""

    __slots__ = ("S", "s_g", "logdet", "Q", "Pinv", "alpha", "lp")

    def __init__(self, d: _Data, S: np.ndarray):
        S = np.sort(np.asarray(S, dtype=int))
        self.S = S
        self.s_g = np.bincount(d.groups[S], minlength=d.n_groups) if len(S) else np.zeros(d.n_groups, int)
        if len(S) == 0:
            self.logdet = 0.0
            self.Q = d.yy
            self.Pinv = np.zeros((0, 0))
            self.alpha = np.zeros(0)
        else:
            A = d.G[np.ix_(S, S)] + d.inv_tau2 * np.eye(len(S))
            c, low = cho_factor(A, lower=True, check_finite=False)
            self.logdet = 2.0 * float(np.sum(np.log(np.diag(c))))
            self.Pinv = cho_solve((c, low), np.eye(len(S)), check_finite=False)
            bS = d.u[S]
            self.alpha = self.Pinv @ bS
            self.Q = max(d.yy - float(bS @ self.alpha), 0.0)
        self.lp = log_post_from(d, len(S), self.logdet, self.Q, self.s_g)


def log_prior_groups(d: _Data, s_g: np.ndarray) -> float:
    a, b = d.cfg.a_pi, d.cfg.b_pi
    return float(np.sum(betaln(a + s_g, b + d.K_g - s_g) - betaln(a, b)))


def log_post_from(d: _Data, s: int, logdet: float, Q: float, s_g: np.ndarray) -> float:
    return (
        -s * d.log_tau
        - 0.5 * logdet
        - d.shape * math.log(d.cfg.b0 + 0.5 * Q)
        + log_prior_groups(d, s_g)
    )


def flip_scores(d: _Data, st: _State, exclude: np.ndarray | None = None) -> np.ndarray:
    """Log posterior of every single-flip neighbour of st (add j not in S / delete j in S)."""
    cfg = d.cfg
    a, b = cfg.a_pi, cfg.b_pi
    s = len(st.S)
    out = np.full(d.K, -np.inf)
    # ---- additions
    if s == 0:
        delta = d.dG + d.inv_tau2
        r = d.u.copy()
    else:
        V = d.G[st.S, :]
        W = st.Pinv @ V
        delta = d.dG + d.inv_tau2 - np.einsum("ij,ij->j", V, W)
        r = d.u - W.T @ d.u[st.S]
    delta = np.maximum(delta, 1e-12)
    Q_add = np.maximum(st.Q - r**2 / delta, 0.0)
    g = d.groups
    room = d.K_g[g] - st.s_g[g] - 1.0
    cur_prior = log_prior_groups(d, st.s_g)
    # beta-binomial prior ratio for adding one feature of group g
    prior_add = np.log((a + st.s_g[g]) / np.maximum(b + room, 1e-300))
    lp_add = (
        -(s + 1) * d.log_tau
        - 0.5 * (st.logdet + np.log(delta))
        - d.shape * np.log(cfg.b0 + 0.5 * Q_add)
        + cur_prior
        + prior_add
    )
    ok = d.allowed & (room >= 0)
    if s >= cfg.max_support:
        ok[:] = False
    out[ok] = lp_add[ok]
    # ---- deletions
    if s > 0:
        Pd = np.maximum(np.diag(st.Pinv), 1e-300)
        logdet_del = st.logdet + np.log(Pd)
        Q_del = st.Q + st.alpha**2 / Pd
        gS = g[st.S]
        prior_del = np.log((b + d.K_g[gS] - st.s_g[gS]) / (a + st.s_g[gS] - 1.0))
        lp_del = (
            -(s - 1) * d.log_tau
            - 0.5 * logdet_del
            - d.shape * np.log(cfg.b0 + 0.5 * Q_del)
            + cur_prior
            + prior_del
        )
        out[st.S] = lp_del
    if exclude is not None and len(exclude):
        out[exclude] = -np.inf
    return out


def _log_barker(delta: np.ndarray) -> np.ndarray:
    """log g(e^delta) with g(t) = t / (1 + t); -inf stays -inf."""
    with np.errstate(over="ignore", invalid="ignore"):
        out = -np.logaddexp(0.0, -delta)
    out[~np.isfinite(delta)] = -np.inf
    return out


def _flip(S: np.ndarray, j: int) -> np.ndarray:
    if j in set(S.tolist()):
        return S[S != j]
    return np.sort(np.append(S, j))


def run_chain(d: _Data, S0: np.ndarray, n_iter: int, burn: int, rng: np.random.Generator) -> tuple[list, dict]:
    st = _State(d, S0)
    sc = flip_scores(d, st)
    trace: list[np.ndarray] = []
    lps: list[float] = []
    n_acc = {"flip": 0, "swap": 0}
    n_prop = {"flip": 0, "swap": 0}
    for it in range(n_iter):
        if len(st.S) > 0 and rng.random() < d.cfg.swap_prob:
            # ---- informed swap: delete i (uniform from S), add j informed given S-i
            n_prop["swap"] += 1
            i = int(st.S[rng.integers(len(st.S))])
            T = st.S[st.S != i]
            stT = _State(d, T)
            scT = flip_scores(d, stT, exclude=np.concatenate([T, [i]]).astype(int))
            # scT[j] = log post of T+j for j not in T and j != i
            if not np.isfinite(scT).any():
                continue
            half = 0.5 * scT
            logZ_f = logsumexp(half)
            pj = np.exp(half - logZ_f)
            j = int(rng.choice(d.K, p=pj / pj.sum()))
            # reverse normalizer: candidates not in T, excluding j (i becomes available)
            half_rev = half.copy()
            half_rev[j] = -np.inf
            half_rev[i] = 0.5 * st.lp
            logZ_r = logsumexp(half_rev)
            lp_new = scT[j]
            log_acc = 0.5 * (lp_new - st.lp) + logZ_f - logZ_r
            if math.log(rng.random() + 1e-300) < log_acc:
                st = _State(d, np.sort(np.append(T, j)))
                sc = flip_scores(d, st)
                n_acc["swap"] += 1
        else:
            # ---- informed single flip with Barker balancing g(t) = t / (1 + t)
            # q(S -> S_j) = g(pi_j / pi) / Z(S); acceptance min(1, Z(S) / Z(S_j)).
            # (sqrt balancing stalls in near-noiseless, extremely peaked landscapes.)
            n_prop["flip"] += 1
            lw = _log_barker(sc - st.lp)
            logZ = logsumexp(lw)
            if not np.isfinite(logZ):
                continue
            pj = np.exp(lw - logZ)
            j = int(rng.choice(d.K, p=pj / pj.sum()))
            st_new = _State(d, _flip(st.S, j))
            sc_new = flip_scores(d, st_new)
            logZ_new = logsumexp(_log_barker(sc_new - st_new.lp))
            if math.log(rng.random() + 1e-300) < logZ - logZ_new:
                st, sc = st_new, sc_new
                n_acc["flip"] += 1
        if it >= burn:
            trace.append(st.S.copy())
            lps.append(st.lp)
    diag = {
        "acc_flip": n_acc["flip"] / max(1, n_prop["flip"]),
        "acc_swap": n_acc["swap"] / max(1, n_prop["swap"]),
        "mean_support": float(np.mean([len(s) for s in trace])) if trace else float("nan"),
        "mean_lp": float(np.mean(lps)) if lps else float("nan"),
        "final_state": st.S.copy(),
    }
    return trace, diag


def greedy_init(d: _Data, max_steps: int = 40) -> np.ndarray:
    """Forward/backward greedy MAP search used as a second chain initialisation."""
    st = _State(d, np.zeros(0, int))
    for _ in range(max_steps):
        sc = flip_scores(d, st)
        j = int(np.argmax(sc))
        if sc[j] <= st.lp + 1e-9:
            break
        st = _State(d, _flip(st.S, j))
    return st.S


def fit_spike_slab(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: SpikeSlabConfig,
    rng: np.random.Generator,
    init_states: list[np.ndarray] | None = None,
) -> Posterior:
    d = _Data(np.asarray(X, float), np.asarray(y, float), np.asarray(groups), cfg)
    warm = init_states is not None and len(init_states) == cfg.n_chains
    n_iter = cfg.warm_iter if warm else cfg.n_iter
    burn = cfg.warm_burn if warm else cfg.burn
    traces, diags = [], []
    for c in range(cfg.n_chains):
        if warm:
            S0 = np.asarray(init_states[c], int)
            S0 = S0[S0 < d.K]
        elif c == 0:
            S0 = np.zeros(0, int)
        else:
            S0 = greedy_init(d)
        tr, dg = run_chain(d, S0, n_iter, burn, rng)
        traces.append(tr)
        diags.append(dg)
    # thin to n_keep particles, equally from each chain
    per = max(1, cfg.n_keep // cfg.n_chains)
    sel: list[np.ndarray] = []
    for tr in traces:
        idx = np.linspace(0, len(tr) - 1, per).round().astype(int)
        sel.extend(tr[i] for i in idx)
    supports, betas, sig2 = [], [], []
    cache: dict[bytes, _State] = {}
    for S in sel:
        key = S.astype(np.int32).tobytes()
        st = cache.get(key)
        if st is None:
            st = _State(d, S)
            cache[key] = st
        s2 = 1.0 / rng.gamma(d.shape, 1.0 / (cfg.b0 + 0.5 * st.Q))
        if len(S):
            Lc = np.linalg.cholesky(st.Pinv + 1e-15 * np.eye(len(S)))
            beta = st.alpha + math.sqrt(s2) * (Lc @ rng.standard_normal(len(S)))
        else:
            beta = np.zeros(0)
        supports.append(S)
        betas.append(beta * d.scale)
        sig2.append(s2 * d.scale**2)
    # convergence diagnostics: inclusion agreement between chains
    incs = []
    for tr in traces:
        inc = np.zeros(d.K)
        for S in tr:
            inc[S] += 1
        incs.append(inc / max(1, len(tr)))
    max_inc_diff = float(np.max(np.abs(incs[0] - incs[-1]))) if len(incs) > 1 else float("nan")
    diag = {
        "n_unique_supports": len(cache),
        "max_chain_inclusion_diff": max_inc_diff,
        "acc_flip": float(np.mean([g["acc_flip"] for g in diags])),
        "acc_swap": float(np.mean([g["acc_swap"] for g in diags])),
        "mean_support": float(np.mean([g["mean_support"] for g in diags])),
        "scale": d.scale,
        "n_obs": d.n,
        "n_features": d.K,
        "warm_started": bool(warm),
    }
    return Posterior(
        supports=supports,
        betas=betas,
        sigma2=np.asarray(sig2),
        n_features=d.K,
        diagnostics=diag,
        last_states=[g["final_state"] for g in diags],
    )


def exact_posterior_inclusion(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: SpikeSlabConfig) -> dict:
    """Exact enumeration over all 2^K supports (tests only; K <= ~14)."""
    d = _Data(np.asarray(X, float), np.asarray(y, float), np.asarray(groups), cfg)
    K = d.K
    lps = np.empty(2**K)
    supports = []
    for code in range(2**K):
        S = np.array([j for j in range(K) if code >> j & 1], dtype=int)
        supports.append(S)
        lps[code] = _State(d, S).lp
    w = np.exp(lps - logsumexp(lps))
    inc = np.zeros(K)
    for S, wi in zip(supports, w):
        inc[S] += wi
    return {"inclusion": inc, "log_post": lps, "weights": w, "supports": supports, "data": d}
