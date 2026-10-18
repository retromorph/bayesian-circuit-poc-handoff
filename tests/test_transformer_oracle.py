import numpy as np
import torch

from bayescircuits.oracle.transformer import HeadAblationRunner, last_token_margin
from bayescircuits.planted import SEQ, build_instance, label_tokens, make_inputs, validate_instance


def _setup(seed=20000, n=16):
    inst = build_instance(seed, "cpu")
    toks, y = make_inputs(np.random.default_rng(1), n, "cpu")
    c, w = label_tokens(y)
    r = HeadAblationRunner(inst.model, inst.candidate_heads, toks, last_token_margin(c, w), torch.full((n,), SEQ - 1), microbatch=7)
    return inst, toks, r


def _z(model, toks, keep):
    """Collect hook_z of every layer under the runner's ablation hooks."""
    store = {}
    hooks = []
    for layer in range(3):
        def ab(z, hook, kl=keep[:, layer, :]):
            return z * kl[:, None, :, None]

        def cap(z, hook):
            store[hook.name] = z.clone()

        hooks += [(f"blocks.{layer}.attn.hook_z", ab), (f"blocks.{layer}.attn.hook_z", cap)]
    model.run_with_hooks(toks, fwd_hooks=hooks)
    return store


def test_zero_mask_zero_effect_and_microbatch_invariance():
    inst, toks, r = _setup()
    d = r.effects(np.zeros((3, 24), np.uint8))
    assert np.max(np.abs(d)) < 1e-5
    m = np.random.default_rng(0).integers(0, 2, (5, 24)).astype(np.uint8)
    r2 = HeadAblationRunner(inst.model, inst.candidate_heads, toks, r.score_fn, r.score_pos, microbatch=512)
    assert np.allclose(r.effects(m), r2.effects(m), atol=1e-5)


def test_single_head_ablation_zeroes_only_that_head_all_positions():
    inst, toks, r = _setup()
    j = 9  # layer 1, head 1
    m = np.zeros((1, 24), np.uint8)
    m[0, j] = 1
    keep = r.keep_tensor(m).repeat(len(toks), 1, 1)
    z_ab = _z(inst.model, toks, keep)
    z_cl = _z(inst.model, toks, torch.ones_like(keep))
    layer, head = inst.candidate_heads[j]
    assert torch.all(z_ab[f"blocks.{layer}.attn.hook_z"][:, :, head, :] == 0)
    # other heads of same layer and all heads of earlier layers unchanged
    others = [h for h in range(8) if h != head]
    assert torch.allclose(z_ab[f"blocks.{layer}.attn.hook_z"][:, :, others], z_cl[f"blocks.{layer}.attn.hook_z"][:, :, others])
    assert torch.allclose(z_ab["blocks.0.attn.hook_z"], z_cl["blocks.0.attn.hook_z"])


def test_joint_mask_is_union_and_order_invariant():
    inst, toks, r = _setup()
    k = r.keep_tensor(np.array([[1 if j in (2, 17) else 0 for j in range(24)]], np.uint8))[0]
    assert k.sum().item() == 24 - 2 and k[0, 2] == 0 and k[2, 1] == 0
    perm_heads = list(reversed(inst.candidate_heads))
    r_perm = HeadAblationRunner(inst.model, perm_heads, toks, r.score_fn, r.score_pos)
    m = np.zeros((1, 24), np.uint8)
    m[0, [2, 17]] = 1
    m_perm = np.zeros((1, 24), np.uint8)
    m_perm[0, [23 - 2, 23 - 17]] = 1
    assert np.allclose(r.effects(m), r_perm.effects(m_perm), atol=1e-6)


def test_retain_only_ablates_within_pool():
    """Candidate pool = subset of heads: retain(C) never touches non-candidate heads."""
    inst, toks, r = _setup()
    pool = inst.candidate_heads[:10]
    rp = HeadAblationRunner(inst.model, pool, toks, r.score_fn, r.score_pos)
    keep = rp.keep_tensor(np.ones((1, 10), np.uint8))[0]
    assert keep.flatten()[:10].sum() == 0 and keep.flatten()[10:].sum() == 14


def test_planted_instance_is_valid_and_has_no_bypass():
    inst = build_instance(20001, "cpu")
    v = {"clean_accuracy_min": 0.95, "singleton_relevant_head_ablated_accuracy_min": 0.9, "full_A_pair_ablated_accuracy_max": 0.6,
         "full_B_pair_ablated_accuracy_max": 0.6, "singleton_to_corresponding_pair_effect_ratio_max": 0.1,
         "exact_minimal_alternative_circuits": 4, "exact_minimal_circuit_size": 2, "rmax": 0.1}
    res = validate_instance(inst, v, n_inputs=64)
    assert res["valid"], res["checks"]
    assert res["pair"]["A"]["accuracy"] <= 0.6 and res["pair"]["B"]["accuracy"] <= 0.6
