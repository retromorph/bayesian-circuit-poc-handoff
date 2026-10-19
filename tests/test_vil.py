"""BCCD-VIL-1.0: budget parity, verification accounting, selection rules, determinism."""

import numpy as np
import pytest

from bayescircuits.masks import mask_key, retain_mask
from bayescircuits.models.bayes import SpikeSlabConfig
from bayescircuits.oracle.base import DiscoveryOracle, QueryResult
from bayescircuits.vil import METHODS_VIL, VERIFYING, VILSpec, diverse_shortlist, run_vil_method, seed_designs

P = 20
SPEC = VILSpec(p=P, ks=(4, 8), budget=64, n_warm=16, n_discovery=48, n_verify=16, rounds=4, n_random_starts=1, rs_max_iters=5,
               shortlist_starts=4, shortlist_size=10, n_thompson=1, search_particles=16)
FAST_BAYES = SpikeSlabConfig(n_iter=150, burn=50, n_keep=40, warm_iter=60, warm_burn=20, max_support=24)
DENS = [0.1, 0.3, 0.6, 0.9]


def fake_world(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, P) * (rng.random(P) < 0.4)
    B = np.triu(rng.normal(0, 0.6, (P, P)) * (rng.random((P, P)) < 0.05), 1)
    eps = 1 + 0.1 * rng.standard_normal(8)

    def u_true(M):
        M = np.asarray(M, float)
        return M @ a + np.einsum("ni,ij,nj->n", M, B, M) + 0.3 * M.prod(1)

    def fn(M):
        d = u_true(M)[:, None] * eps[None, :]
        return QueryResult(masks=M, y={"abs": np.mean(np.abs(d), 1)[:, None], "effect": np.mean(d, 1)[:, None]}, per_prompt=d)

    return fn


def _run(method, seed=3, reserved=frozenset(), cache=None):
    fn = fake_world()
    des = seed_designs(seed, set(reserved), SPEC, DENS)
    orc = DiscoveryOracle(fn=fn, p=P, budget=SPEC.budget, method=method, unit_id="t", prompt_batch_id="x",
                          reserved_keys=reserved, shared_cache=cache, n_prompts_per_mask=8)
    dec = run_vil_method(method, orc, des, SPEC, seed, FAST_BAYES, {"tol": 1e-4, "max_iter": 5000})
    return dec, orc, des


@pytest.mark.parametrize("method", METHODS_VIL)
def test_budget_parity_and_verification_accounting(method):
    dec, orc, des = _run(method)
    assert dec["n_masks"] == SPEC.budget == orc.n_masks
    if method in VERIFYING:
        assert dec["n_discovery_masks"] == SPEC.n_discovery and dec["n_verification_masks"] == SPEC.n_verify
        assert len(dec["verified"]) == SPEC.n_verify
        for k in SPEC.ks:
            vk = [v for v in dec["verified"] if v["k"] == k]
            assert len(vk) == SPEC.n_verify // len(SPEC.ks)
            assert all(len(v["circuit"]) == k for v in vk)
            # final = lowest measured discovery discrepancy among verified circuits of size k
            best = min(vk, key=lambda v: v["measured_u"])
            assert dec["final"][str(k)]["circuit"] == best["circuit"]
            assert dec["final"][str(k)]["source"] == "verified"
        # every verification is a real query on the oracle (logged), using the retain mask of the circuit
        logged = {r["mask_bits"] for r in orc.log}
        for v in dec["verified"]:
            assert np.packbits(retain_mask(P, v["circuit"])).tobytes().hex() in logged
    else:
        assert dec["n_verification_masks"] == 0 and dec["n_discovery_masks"] == SPEC.budget
        assert all(dec["final"][str(k)]["source"] == "surrogate" for k in SPEC.ks)
    for k in SPEC.ks:
        assert len(dec["final"][str(k)]["circuit"]) == k


def test_density_matched_design():
    des = seed_designs(5, set(), SPEC, DENS)
    dm = des["density_matched"]
    assert len(dm) == SPEC.n_verify
    counts = sorted((P - dm.sum(1)).tolist())
    per = SPEC.n_verify // len(SPEC.ks)
    assert counts == sorted([k for k in SPEC.ks for _ in range(per)])
    keys = {mask_key(m) for m in des["d104"]}
    assert not ({mask_key(m) for m in dm} & keys)
    assert np.array_equal(des["d128"][: SPEC.n_discovery], des["d104"])


def test_designs_match_pilot_stream():
    """d128 must equal the pilot Gate C warm start + first 96 static masks (bridge replication)."""
    from bayescircuits.gate_c import unit_designs

    cfg = {"shared": {"intervention_density_mixture": DENS},
           "gate_c": {"discovery": {"warm_start_masks": 32, "budget_mask_checkpoints": [64, 128, 256]}}}
    reserved = {mask_key(m) for m in np.random.default_rng(1).integers(0, 2, (10, 48)).astype(np.uint8)}
    warm, static = unit_designs(31001, cfg, set(reserved))
    des = seed_designs(31001, set(reserved), VILSpec(), DENS)
    assert np.array_equal(des["d128"], np.concatenate([warm, static[:96]]))


def test_deterministic_given_seed():
    for m in ("lasso_vil", "lasso_random_verify"):
        a, _, _ = _run(m)
        b, _, _ = _run(m)
        assert a["final"] == b["final"] and [v["circuit"] for v in a["verified"]] == [v["circuit"] for v in b["verified"]]


def test_reserved_masks_never_queried():
    # reserve some retain masks of plausible circuits and the full design's first masks are excluded by construction
    des = seed_designs(3, set(), SPEC, DENS)
    rng = np.random.default_rng(0)
    reserved = frozenset(mask_key(retain_mask(P, sorted(rng.choice(P, k, replace=False).tolist()))) for k in SPEC.ks for _ in range(30))
    reserved = frozenset(k for k in reserved if k not in {mask_key(m) for m in des["d128"]})
    for m in ("lasso_vil", "lasso_random_verify"):
        dec, orc, _ = _run(m, reserved=reserved)
        assert not ({bytes.fromhex(r["mask_bits"]) + bytes([P % 256]) for r in orc.log} & reserved)
        assert dec["n_masks"] == SPEC.budget


def test_shared_cache_charges_logical_budget():
    cache: dict = {}
    d1, o1, _ = _run("lasso_static_128", cache=cache)
    d2, o2, _ = _run("lasso_vil", cache=cache)
    assert o2.n_masks == SPEC.budget  # cache hits still debit the method's own budget
    assert sum(r["reused_cache"] for r in o2.log) >= SPEC.n_discovery


def test_diverse_shortlist_excludes_one_swap_neighbours():
    cands = {(0, 1, 2, 3): 0.1, (0, 1, 2, 4): 0.11, (5, 6, 7, 8): 0.2, (0, 1, 9, 10): 0.3}
    sl = diverse_shortlist(cands, 3, 4)
    assert sl[:3] == [(0, 1, 2, 3), (5, 6, 7, 8), (0, 1, 9, 10)]
