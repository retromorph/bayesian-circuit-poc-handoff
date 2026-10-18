import numpy as np

from bayescircuits.config import load_config
from bayescircuits.gate_a import make_problem, world_designs, world_oracle_fn
from bayescircuits.masks import mask_key
from bayescircuits.oracle.synthetic import SyntheticWorld


def test_gate_a_split_integrity_and_hidden_truth():
    cfg = load_config("configs/preregistration.yaml")
    w = SyntheticWorld.create("A1", 2001)
    des = world_designs(w, cfg)
    b = {mask_key(m) for m in des["blind"]}
    d = {mask_key(m) for m in np.concatenate([des["warm"], des["static"]])}
    assert len(b) == 256 and len(d) == 128 and not (b & d)
    prob = make_problem(w, des, cfg, cfg["_impl"])
    # the discovery problem carries no ground truth
    assert not any(hasattr(prob, a) for a in ("perm", "terms", "world"))
    r = world_oracle_fn(w)(np.zeros((1, 32), np.uint8))
    assert r.y["d"].ravel()[0] == 0.0


def test_confirmatory_refuses_short_world_list():
    import pytest

    from bayescircuits.gate_a import run_gate_a

    cfg = load_config("configs/preregistration.yaml")
    with pytest.raises(ValueError):
        run_gate_a(cfg, "confirmatory", confirmatory=True, max_worlds=2)
