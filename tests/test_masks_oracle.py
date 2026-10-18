import numpy as np
import pytest

from bayescircuits.masks import (
    balanced_masks,
    design_matrix,
    feature_groups,
    mask_key,
    n_ablated,
    retain_mask,
)
from bayescircuits.oracle.base import BudgetExceeded, DiscoveryOracle, QueryResult, ReservedMaskError
from bayescircuits.oracle.synthetic import SyntheticWorld, verify_a1_minimal_circuits


def test_retain_is_complement_within_pool():
    m = retain_mask(8, [1, 5])
    assert m.tolist() == [1, 0, 1, 1, 1, 0, 1, 1]
    assert retain_mask(4, []).tolist() == [1, 1, 1, 1]


def test_density_counts():
    assert [n_ablated(32, d) for d in (0.1, 0.3, 0.6, 0.9)] == [3, 10, 19, 29]
    assert [n_ablated(48, d) for d in (0.05, 0.2, 0.5, 0.9)] == [2, 10, 24, 43]


def test_balanced_masks_distinct_and_exclusion():
    rng = np.random.default_rng(0)
    blind = balanced_masks(rng, 32, 64, [0.1, 0.3, 0.6, 0.9])
    excl = {mask_key(m) for m in blind}
    disc = balanced_masks(np.random.default_rng(1), 32, 128, [0.1, 0.3, 0.6, 0.9], exclude=excl)
    keys = [mask_key(m) for m in disc]
    assert len(set(keys)) == 128 and not (set(keys) & excl)
    counts = sorted(set(disc.sum(1).tolist()))
    assert counts == [3, 10, 19, 29]


def test_design_matrix_pairs_and_context():
    m = np.array([[1, 1, 0]], dtype=np.uint8)
    X = design_matrix(m)
    assert X.tolist() == [[1, 1, 0, 1, 0, 0]]  # m0 m1 m2 m0m1 m0m2 m1m2
    Xc = design_matrix(np.repeat(m, 2, 0), np.array([0, 1]))
    assert Xc.shape == (2, 12) and Xc[0, 6:].sum() == 0 and Xc[1, 6:].sum() == 3
    assert feature_groups(3, 2, True).tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


@pytest.mark.parametrize("regime", ["A0", "A1", "A2", "A3"])
def test_synthetic_zero_mask_zero_effect(regime):
    w = SyntheticWorld.create(regime, 1)
    z = np.zeros((1, 32))
    if w.has_context:
        assert w.evaluate(z, np.array([0]))[0] == 0 and w.evaluate(z, np.array([1]))[0] == 0
    else:
        assert w.evaluate(z)[0] == 0


def test_a1_redundant_pairs_and_minimal_circuits():
    w = SyntheticWorld.create("A1", 2003)
    perm = w.perm
    e = lambda idx: np.isin(np.arange(32), [perm[i] for i in idx]).astype(float)[None]  # noqa: E731
    assert abs(w.evaluate(e([1]))[0]) < 1e-12 and abs(w.evaluate(e([2]))[0]) < 1e-12
    c12 = [c for c, idx, _ in w.terms if len(idx) == 2][0]
    assert abs(w.evaluate(e([1, 2]))[0] - c12) < 1e-12
    assert 0.9 * 0.8 <= c12 <= 0.9 * 1.2
    v = verify_a1_minimal_circuits(w)
    assert v["min_size"] == 3 and sorted(v["minimal"]) == sorted(w.true_circuits())
    for C in w.true_circuits():
        assert w.true_R(C) <= 0.10


def test_seed_changes_world_not_semantics():
    a, b = SyntheticWorld.create("A1", 2001), SyntheticWorld.create("A1", 2002)
    assert not np.array_equal(a.perm, b.perm)
    assert SyntheticWorld.create("A1", 2001).perm.tolist() == a.perm.tolist()


def _oracle(budget=4, reserved=frozenset()):
    fn = lambda M: QueryResult(masks=M, y={"d": M.sum(1, keepdims=True).astype(float)})  # noqa: E731
    return DiscoveryOracle(fn=fn, p=6, budget=budget, method="t", unit_id="u", prompt_batch_id="b", reserved_keys=reserved)


def test_oracle_budget_duplicates_reserved():
    o = _oracle()
    m = np.eye(6, dtype=np.uint8)
    r = o.query(m[:2])
    assert r.y["d"].ravel().tolist() == [1.0, 1.0] and o.n_masks == 2
    with pytest.raises(ValueError):
        o.query(m[:1])  # repeated mask is not free evidence
    with pytest.raises(BudgetExceeded):
        o.query(m[2:5])
    o2 = _oracle(reserved=frozenset({mask_key(m[0])}))
    with pytest.raises(ReservedMaskError):
        o2.query(m[:1])
    assert len(o.log) == 2 and {"mask_id", "mask_bits", "prompt_batch_id", "time_ms", "reused_cache"} <= set(o.log[0])
