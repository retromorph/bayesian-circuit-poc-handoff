import numpy as np

from bayescircuits.acquisition import select_decision, select_uncertainty
from bayescircuits.masks import balanced_masks, design_matrix, feature_groups
from bayescircuits.models.bayes import SpikeSlabConfig, exact_posterior_inclusion, fit_spike_slab
from bayescircuits.models.surrogates import BayesSurrogate, LassoSurrogate, pair_nonadditivity


def test_posterior_matches_exact_enumeration_small_p():
    """p=4 -> 10 features -> 1024 supports enumerated exactly."""
    rng = np.random.default_rng(0)
    p = 4
    M = balanced_masks(rng, p, 12, [0.25, 0.5, 0.75])
    X, g = design_matrix(M), feature_groups(p)
    y = 1.0 * M[:, 0] + 0.9 * M[:, 1] * M[:, 2] + 0.05 * rng.standard_normal(len(M))
    cfg = SpikeSlabConfig(n_iter=5000, burn=1000, n_keep=4000, n_chains=2)
    ex = exact_posterior_inclusion(X, y, g, cfg)
    post = fit_spike_slab(X, y, g, cfg, np.random.default_rng(1))
    assert np.max(np.abs(ex["inclusion"] - post.inclusion_probs())) < 0.03


def test_no_heredity_pure_interaction_recovered():
    rng = np.random.default_rng(3)
    p = 16
    M = balanced_masks(rng, p, 64, [0.1, 0.3, 0.6, 0.9])
    y = 0.9 * M[:, 2] * M[:, 7]  # zero isolated effects
    s = BayesSurrogate(p, [None], SpikeSlabConfig(), seed=1)
    s.fit(M, y[:, None])
    na = pair_nonadditivity(lambda Z: s.predict_particles(Z), p).mean(0)
    I, J = np.triu_indices(p, 1)
    top = int(np.argmax(np.abs(na)))
    assert (I[top], J[top]) == (2, 7)
    # single ablations predicted ~0 (no heredity forced)
    eye = np.eye(p, dtype=np.uint8)
    assert np.all(np.abs(s.predict(eye[[2, 7]])) < 0.05)


def test_redundant_alternatives_all_represented():
    """Posterior particles must support all equivalent retained circuits, not collapse to one."""
    from bayescircuits.circuits import predicted_R

    rng = np.random.default_rng(5)
    p = 16
    M = balanced_masks(rng, p, 96, [0.1, 0.3, 0.6, 0.9])
    y = 1.0 * M[:, 0] + 0.9 * M[:, 1] * M[:, 2] + 0.8 * M[:, 3] * M[:, 4]
    s = BayesSurrogate(p, [None], SpikeSlabConfig(), seed=2)
    s.fit(M, y[:, None])
    circ = [(0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4)]
    R = predicted_R(lambda Z: s.predict_particles(Z), p, circ)
    assert np.all((R <= 0.1).mean(0) > 0.9)


def test_first_order_lasso_cannot_represent_pure_interaction():
    rng = np.random.default_rng(4)
    p = 16
    M = balanced_masks(rng, p, 96, [0.1, 0.3, 0.6, 0.9])
    y = (0.9 * M[:, 1] * M[:, 2])[:, None]
    l1, l2 = LassoSurrogate(p, [None], 1, 0), LassoSurrogate(p, [None], 2, 0)
    l1.fit(M, y)
    l2.fit(M, y)
    assert np.mean(np.abs(l2.predict(M) - y)) < 0.02 < np.mean(np.abs(l1.predict(M) - y))


def test_acquisition_deterministic_given_state_and_seed():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(8, 30, 1))
    s2 = np.full(8, 0.01)
    L = [rng.random((8, 5))]
    a = select_decision(F, s2, L, 4, np.random.default_rng(7))[0]
    b = select_decision(F, s2, L, 4, np.random.default_rng(7))[0]
    assert a == b and len(set(a)) == 4
    u1 = select_uncertainty(F, s2, 4, np.random.default_rng(7))[0]
    u2 = select_uncertainty(F, s2, 4, np.random.default_rng(7))[0]
    assert u1 == u2 and len(set(u1)) == 4


def test_decision_gain_prefers_discriminating_mask():
    """Two particle groups disagree on which circuit is faithful; only mask 3 separates them."""
    S, N = 8, 6
    F = np.zeros((S, N, 1))
    F[:4, 3, 0], F[4:, 3, 0] = 1.0, -1.0
    L = np.zeros((S, 2))
    L[:4, 1] = 1.0
    L[4:, 0] = 1.0
    chosen, info = select_decision(F, np.full(S, 1e-4), [L], 1, np.random.default_rng(0))
    assert chosen == [3] and info["V0"] > info["V_after_batch_expected"]
