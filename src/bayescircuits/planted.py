"""Gate B benchmark construction: planted redundant XOR transformers (direct weight construction).

Architecture: TransformerLens HookedTransformer, 3 layers x 8 heads, d_model=64, d_head=8,
d_mlp=32, ReLU, no LayerNorm. Input ``[A, B, N, N, N, QUERY]``; output = class-token logits
(Y0, Y1) at the last position, target Y = A XOR B.

Residual layout
  0-7   one-hot position           8  A sign (+/-1, token at pos 0)      9  B sign (pos 1)
  10    is-query flag              11 A-feature written by A heads      12 B-feature (B heads)
  13    clip(A)  (MLP1)            14 clip(B) (MLP1)                     15 XOR margin feature (MLP2)
  16-31 nuisance-token features    32-63 nuisance head / neuron output space

Mechanism: two redundant A heads and two redundant B heads (random positions among the 16
heads of layers 0-1) attend from the query position to position 0 (resp. 1) and copy the
bit sign into dim 11 (resp. 12) with strength w_h in [0.8, 1.2]. MLP1 saturates each sum to
{-1, 0, +1}; MLP2 computes the XOR margin. 20 nuisance heads (random QK/OV restricted to the
nuisance subspace) write to dims 32-63, which feed the class logits only through a small random
read-out (benign nuisance effects). No nuisance pathway reads or writes dims 8-15, so there is
no residual/MLP bypass around the planted heads; validation verifies this behaviourally.

The planted head identities are benchmark metadata used only by validation/evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import torch

A0, A1T, B0, B1T, QUERY = 0, 1, 2, 3, 4
NUIS = list(range(5, 15))
Y0, Y1 = 15, 16
D_VOCAB = 17
SEQ = 6


@dataclass
class PlantedInstance:
    seed: int
    model: object = field(repr=False)
    a_heads: list[tuple[int, int]]
    b_heads: list[tuple[int, int]]
    kappa: float
    nuisance_scale: float
    strengths: dict

    @property
    def candidate_heads(self) -> list[tuple[int, int]]:
        return [(layer, h) for layer in range(3) for h in range(8)]

    def planted_indices(self) -> dict:
        idx = {hd: i for i, hd in enumerate(self.candidate_heads)}
        return {"A": [idx[h] for h in self.a_heads], "B": [idx[h] for h in self.b_heads]}

    def true_circuits(self) -> list[tuple[int, int]]:
        pi = self.planted_indices()
        return sorted(tuple(sorted((a, b))) for a in pi["A"] for b in pi["B"])

    def true_pairs(self) -> list[tuple[int, int]]:
        pi = self.planted_indices()
        return sorted([tuple(sorted(pi["A"])), tuple(sorted(pi["B"]))])


def build_instance(seed: int, device: str = "cpu") -> PlantedInstance:
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    rng = np.random.default_rng([seed, 2718])
    cfg = HookedTransformerConfig(
        n_layers=3, d_model=64, n_ctx=8, d_head=8, n_heads=8, d_mlp=32, d_vocab=D_VOCAB,
        act_fn="relu", normalization_type=None, positional_embedding_type="standard", seed=int(seed),
        default_prepend_bos=False,
    )
    model = HookedTransformer(cfg)
    sd = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    W_E = np.zeros((D_VOCAB, 64))
    W_E[A0, 8], W_E[A1T, 8] = -1.0, 1.0
    W_E[B0, 9], W_E[B1T, 9] = -1.0, 1.0
    W_E[QUERY, 10] = 1.0
    for t in NUIS:
        W_E[t, 16:32] = rng.normal(0, 1, 16) / 4.0
    W_pos = np.zeros((8, 64))
    W_pos[np.arange(8), np.arange(8)] = 1.0
    planted = rng.choice(16, size=4, replace=False)
    heads = [(int(i // 8), int(i % 8)) for i in planted]
    a_heads, b_heads = heads[:2], heads[2:]
    strengths = {f"{h}": float(rng.uniform(0.8, 1.2)) for h in heads}
    q_scale = 40.0 * np.sqrt(8)
    Wq = np.zeros((3, 8, 64, 8))
    Wk = np.zeros((3, 8, 64, 8))
    Wv = np.zeros((3, 8, 64, 8))
    Wo = np.zeros((3, 8, 8, 64))
    for grp, src_pos, src_dim, dst_dim in ((a_heads, 0, 8, 11), (b_heads, 1, 9, 12)):
        for (layer, h) in grp:
            Wq[layer, h, 10, 0] = q_scale
            Wk[layer, h, src_pos, 0] = 1.0
            Wv[layer, h, src_dim, 0] = 1.0
            Wo[layer, h, 0, dst_dim] = strengths[f"{(layer, h)}"]
    planted_set = set(heads)
    nuis_scale = float(rng.uniform(0.6, 1.4))
    for layer in range(3):
        for h in range(8):
            if (layer, h) in planted_set:
                continue
            read_qk = np.r_[0:8, 16:32]
            Wq[layer, h][read_qk, :] = rng.normal(0, 1.0, (len(read_qk), 8))
            Wk[layer, h][read_qk, :] = rng.normal(0, 1.0, (len(read_qk), 8))
            read_v = np.r_[16:32, 32:64]
            Wv[layer, h][read_v, :] = rng.normal(0, 1.0 / np.sqrt(len(read_v)), (len(read_v), 8))
            Wo[layer, h][:, 32:64] = rng.normal(0, nuis_scale / np.sqrt(8), (8, 32))
    W_in = np.zeros((3, 64, 32))
    b_in = np.zeros((3, 32))
    W_out = np.zeros((3, 32, 64))
    b_out = np.zeros((3, 64))
    c = 1.0
    # MLP1: saturating clip of dims 11, 12 into 13, 14
    for k, (src, dst) in enumerate(((11, 13), (12, 14))):
        W_in[1, src, 2 * k], b_in[1, 2 * k] = 2.0 / c, 1.0
        W_in[1, src, 2 * k + 1], b_in[1, 2 * k + 1] = 2.0 / c, -1.0
        W_out[1, 2 * k, dst], W_out[1, 2 * k + 1, dst] = 1.0, -1.0
        b_out[1, dst] = -1.0
    kappa = float(rng.uniform(3.0, 5.0))
    W_in[2, 13, 0], W_in[2, 14, 0], b_in[2, 0] = 1.0, 1.0, -1.0
    W_in[2, 13, 1], W_in[2, 14, 1], b_in[2, 1] = -1.0, -1.0, -1.0
    W_out[2, 0, 15], W_out[2, 1, 15], b_out[2, 15] = -2.0 * kappa, -2.0 * kappa, kappa
    # benign nuisance neurons: read/write only the nuisance output subspace
    for layer in range(3):
        lo = 4 if layer == 1 else (2 if layer == 2 else 0)
        W_in[layer, 32:64, lo:] = rng.normal(0, 1 / np.sqrt(32), (32, 32 - lo))
        W_out[layer, lo:, 32:64] = rng.normal(0, 0.3 / np.sqrt(32), (32 - lo, 32))
    W_U = np.zeros((64, D_VOCAB))
    W_U[15, Y1], W_U[15, Y0] = 0.5, -0.5
    u = rng.normal(0, 1, 32)
    W_U[32:64, Y1], W_U[32:64, Y0] = u, -u
    vals = {"embed.W_E": W_E, "pos_embed.W_pos": W_pos, "unembed.W_U": W_U}
    for layer in range(3):
        vals[f"blocks.{layer}.attn.W_Q"] = Wq[layer]
        vals[f"blocks.{layer}.attn.W_K"] = Wk[layer]
        vals[f"blocks.{layer}.attn.W_V"] = Wv[layer]
        vals[f"blocks.{layer}.attn.W_O"] = Wo[layer]
        vals[f"blocks.{layer}.mlp.W_in"] = W_in[layer]
        vals[f"blocks.{layer}.mlp.b_in"] = b_in[layer]
        vals[f"blocks.{layer}.mlp.W_out"] = W_out[layer]
        vals[f"blocks.{layer}.mlp.b_out"] = b_out[layer]
    for k, v in vals.items():
        sd[k] = torch.tensor(v, dtype=sd[k].dtype)
    for k in sd:  # keep TL's causal-mask buffers / IGNORE constants
        if k.endswith(".mask") or k.endswith("IGNORE"):
            sd[k] = model.state_dict()[k]
    model.load_state_dict(sd)
    model.eval()
    # calibrate nuisance read-out so that removing all nuisance heads moves the margin by a
    # small, instance-specific fraction rho of kappa
    rho = float(rng.uniform(0.01, 0.04))
    model.to(device)
    inst = PlantedInstance(seed=seed, model=model, a_heads=a_heads, b_heads=b_heads, kappa=kappa,
                           nuisance_scale=rho, strengths=strengths)
    toks, _ = make_inputs(np.random.default_rng([seed, 99]), 64, device)
    with torch.inference_mode():
        res = model(toks, return_type="logits")[:, -1]
        nz = (res[:, Y1] - res[:, Y0]).cpu().numpy()
    # nuisance component of margin = margin - kappa*(+/-1); scale u so mean |.| == rho*kappa
    a, b = _bits(toks)
    xor = (a ^ b).cpu().numpy()
    nuis = nz - kappa * np.where(xor == 1, 1.0, -1.0)
    cur = float(np.mean(np.abs(nuis)))
    if cur > 0:
        f = rho * kappa / cur
        with torch.no_grad():
            model.unembed.W_U[32:64, :] *= f
    return inst


def _bits(toks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (toks[:, 0] == A1T).long(), (toks[:, 1] == B1T).long()


def make_inputs(rng: np.random.Generator, n: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """n balanced XOR inputs (n/4 per (A,B) combination) with random nuisance tokens."""
    combos = np.array([(a, b) for a in (0, 1) for b in (0, 1)])
    ab = combos[np.arange(n) % 4]
    ab = ab[rng.permutation(n)]
    toks = np.zeros((n, SEQ), dtype=np.int64)
    toks[:, 0] = np.where(ab[:, 0] == 1, A1T, A0)
    toks[:, 1] = np.where(ab[:, 1] == 1, B1T, B0)
    toks[:, 2:5] = rng.choice(NUIS, size=(n, 3))
    toks[:, 5] = QUERY
    y = ab[:, 0] ^ ab[:, 1]
    return torch.tensor(toks, device=device), torch.tensor(y, device=device)


def label_tokens(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    correct = torch.where(y == 1, Y1, Y0)
    wrong = torch.where(y == 1, Y0, Y1)
    return correct, wrong


def validate_instance(inst: PlantedInstance, cfgv: dict, n_inputs: int = 256, device: str = "cpu") -> dict:
    """Ground-truth validation on independent inputs (benchmark construction only)."""
    from .circuits import circuit_masks
    from .oracle.transformer import HeadAblationRunner, last_token_margin

    toks, y = make_inputs(np.random.default_rng([inst.seed, 31337]), n_inputs, device)
    correct, wrong = label_tokens(y)
    runner = HeadAblationRunner(inst.model, inst.candidate_heads, toks, last_token_margin(correct, wrong), torch.full((len(toks),), SEQ - 1, device=device))
    p = 24
    pi = inst.planted_indices()
    clean = runner.clean.cpu().numpy()

    def acc(scores):
        return float(np.mean(scores > 0))

    res: dict = {"seed": inst.seed, "clean_accuracy": acc(clean), "kappa": inst.kappa}
    singles = {}
    for j in pi["A"] + pi["B"]:
        m = np.zeros((1, p), np.uint8)
        m[0, j] = 1
        sc = runner.scores(m)[0]
        singles[j] = {"accuracy": acc(sc), "mean_effect": float(np.mean(clean - sc))}
    pairs = {}
    for nm in ("A", "B"):
        m = np.zeros((1, p), np.uint8)
        m[0, pi[nm]] = 1
        sc = runner.scores(m)[0]
        pairs[nm] = {"accuracy": acc(sc), "mean_effect": float(np.mean(clean - sc))}
    ratios = []
    for nm in ("A", "B"):
        for j in pi[nm]:
            ratios.append(abs(singles[j]["mean_effect"]) / max(abs(pairs[nm]["mean_effect"]), 1e-12))
    res["singleton"] = singles
    res["pair"] = pairs
    res["max_singleton_to_pair_ratio"] = float(max(ratios))
    # exact minimal faithful circuits among all size<=2 retained sets
    full = np.ones((1, p), np.uint8)
    den = float(np.mean(np.abs(clean - runner.scores(full)[0])))
    circs = [(j,) for j in range(p)] + list(combinations(range(p), 2))
    R = []
    M = circuit_masks(p, circs)
    for s in range(0, len(M), 64):
        sc = runner.scores(M[s : s + 64])
        R.extend(np.mean(np.abs(clean[None, :] - sc), axis=1) / den)
    R = np.asarray(R)
    faithful = [c for c, r in zip(circs, R) if r <= cfgv["rmax"]]
    res["denominator_mean_abs"] = den
    res["faithful_size_le2"] = [list(c) for c in faithful]
    res["true_circuits"] = [list(c) for c in inst.true_circuits()]
    res["true_circuit_R"] = [float(R[circs.index(c)]) for c in inst.true_circuits()]
    res["max_R_other_size2"] = float(max(r for c, r in zip(circs, R) if len(c) == 2 and c not in inst.true_circuits()))
    v = cfgv
    checks = {
        "clean_accuracy": res["clean_accuracy"] >= v["clean_accuracy_min"],
        "singleton_accuracy": all(s["accuracy"] >= v["singleton_relevant_head_ablated_accuracy_min"] for s in singles.values()),
        "pair_A_accuracy": pairs["A"]["accuracy"] <= v["full_A_pair_ablated_accuracy_max"],
        "pair_B_accuracy": pairs["B"]["accuracy"] <= v["full_B_pair_ablated_accuracy_max"],
        "singleton_effect_ratio": res["max_singleton_to_pair_ratio"] <= v["singleton_to_corresponding_pair_effect_ratio_max"],
        "four_exact_minimal_size2": sorted(tuple(c) for c in faithful) == inst.true_circuits()
        and len(faithful) == v["exact_minimal_alternative_circuits"],
    }
    res["checks"] = checks
    res["valid"] = bool(all(checks.values()))
    return res
