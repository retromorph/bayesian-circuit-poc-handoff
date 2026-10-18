"""Gate C: pretrained GPT-2 Small, IOI, 48 frozen candidate heads (in-pool head zero-ablation).

This is an in-pool head-localization benchmark, not a full circuit-recovery claim and not
numerically comparable to MIB's edge-patching leaderboard.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import PROJECT_ROOT, run_root, sha256_file
from .discovery import METHODS, NON_BAYES, DecisionSpec, Problem, end_to_end_cost, run_method
from .evaluate import crps_from_samples
from .gate_a import acq_spec, bayes_cfg
from .masks import balanced_masks, mask_key, masks_sha256, n_ablated, random_mask, retain_mask
from .models.surrogates import make_surrogate
from .oracle.base import DiscoveryOracle, QueryResult
from .oracle.transformer import HeadAblationRunner, resid_logit_diff
from .provenance import provenance, write_json

MIB_REV = "e5f3468f3af4c0883be35cd3bced8c711c95d286"
MIB_URL = "https://huggingface.co/datasets/mib-bench/ioi/resolve/{rev}/{split}.json"
CACHE = PROJECT_ROOT / "cache"
FROZEN = PROJECT_ROOT / "runs" / "gate_c" / "frozen"


# ----------------------------------------------------------------------------- data
def load_mib(split: str) -> tuple[list[dict], str]:
    path = CACHE / "mib" / f"{split}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MIB_URL.format(rev=MIB_REV, split=split), path)
    text = path.read_text().strip()
    data = json.loads(text)
    return data, sha256_file(path)


def load_model(device: str):
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained_no_processing("gpt2", device=device)
    model.eval()
    rev = None
    for d in [Path.home() / ".cache/huggingface/hub/models--gpt2", Path.home() / ".cache/huggingface/hub/models--openai-community--gpt2"]:
        ref = d / "refs" / "main"
        if ref.exists():
            rev = {"repo": d.name.replace("models--", "").replace("--", "/"), "revision": ref.read_text().strip()}
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return model, {"model_id": "gpt2", "hf_revision": rev, "weights_sha256": h.hexdigest(), "license": "MIT (openai-community/gpt2)",
                   "loader": "transformer_lens.HookedTransformer.from_pretrained_no_processing", "dtype": "float32"}


@dataclass
class PromptSet:
    name: str
    items: list[dict]
    tokens: torch.Tensor = field(repr=False)
    corrupt: torch.Tensor = field(repr=False)
    pos: torch.Tensor = field(repr=False)
    io: torch.Tensor = field(repr=False)
    s: torch.Tensor = field(repr=False)

    @property
    def templates(self) -> list[str]:
        return [it["template"] for it in self.items]

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps([(it["src"], it["idx"], it["prompt"]) for it in self.items]).encode()).hexdigest()


def eligible(model, data: list[dict], src: str) -> tuple[list[dict], dict]:
    out, excl = [], {"multi_token_name": 0, "counterfactual_length_mismatch": 0}
    for i, it in enumerate(data):
        io = it["choices"][it["answerKey"]]
        s = it["choices"][1 - it["answerKey"]]
        try:
            io_t = model.to_single_token(" " + io)
            s_t = model.to_single_token(" " + s)
        except Exception:
            excl["multi_token_name"] += 1
            continue
        clean = model.to_tokens(it["prompt"], prepend_bos=True)[0].tolist()
        corr = model.to_tokens(it["abc_counterfactual"]["prompt"], prepend_bos=True)[0].tolist()
        if len(clean) != len(corr):
            excl["counterfactual_length_mismatch"] += 1
            continue
        out.append({"src": src, "idx": i, "template": it["template"], "prompt": it["prompt"], "io": io, "s": s,
                    "io_tok": io_t, "s_tok": s_t, "clean": clean, "corrupt": corr})
    excl["eligible"] = len(out)
    excl["total"] = len(data)
    return out, excl


def to_promptset(name: str, items: list[dict], device: str, pad: int = 50256) -> PromptSet:
    L = max(len(it["clean"]) for it in items)
    T = torch.full((len(items), L), pad, dtype=torch.long)
    Cr = torch.full((len(items), L), pad, dtype=torch.long)
    for i, it in enumerate(items):
        T[i, : len(it["clean"])] = torch.tensor(it["clean"])
        Cr[i, : len(it["corrupt"])] = torch.tensor(it["corrupt"])
    pos = torch.tensor([len(it["clean"]) - 1 for it in items])
    io = torch.tensor([it["io_tok"] for it in items])
    s = torch.tensor([it["s_tok"] for it in items])
    return PromptSet(name, items, T.to(device), Cr.to(device), pos.to(device), io.to(device), s.to(device))


def build_splits(model, cfg: dict, device: str) -> tuple[dict[str, PromptSet], dict]:
    gc = cfg["gate_c"]
    impl = cfg["_impl"]["gate_c"]
    meta: dict = {"mib_revision": MIB_REV}
    pools = {}
    for split in ("train", "dev", "test"):
        data, sha = load_mib(split)
        el, excl = eligible(model, data, split)
        pools[split] = el
        meta[f"mib_{split}_sha256"] = sha
        meta[f"exclusions_{split}"] = excl
    rng = np.random.default_rng(cfg["seeds"]["gate_c_split"])
    tr = [pools["train"][i] for i in rng.permutation(len(pools["train"]))]
    n_scr, n_disc = gc["screening"]["n_prompts"], gc["discovery"]["n_prompts_per_mask"] * gc["discovery"]["n_seeds"]
    n_dd = impl["n_dev_discovery_prompts"]
    scr = tr[:n_scr]
    disc = tr[n_scr : n_scr + n_disc]
    devd = tr[n_scr + n_disc : n_scr + n_disc + n_dd]
    dv = [pools["dev"][i] for i in rng.permutation(len(pools["dev"]))][: impl["n_dev_eval_prompts"]]
    # blind: MIB test split (template families disjoint from train/dev), balanced over templates
    test = pools["test"]
    by_t: dict[str, list] = {}
    for it in [test[i] for i in rng.permutation(len(test))]:
        by_t.setdefault(it["template"], []).append(it)
    blind, k = [], 0
    tnames = sorted(by_t)
    while len(blind) < gc["blind"]["n_prompts"]:
        t = tnames[k % len(tnames)]
        if by_t[t]:
            blind.append(by_t[t].pop())
        k += 1
    sets = {"screening": scr, "discovery": disc, "dev_discovery": devd, "dev_eval": dv, "blind": blind}
    texts = {n: {(it["src"], it["idx"]) for it in v} for n, v in sets.items()}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (texts[names[i]] & texts[names[j]]), f"split overlap {names[i]}/{names[j]}"
    train_t = {it["template"] for n in ("screening", "discovery", "dev_discovery") for it in sets[n]}
    meta["blind_templates"] = sorted({it["template"] for it in blind})
    meta["blind_template_overlap_with_discovery"] = sorted(train_t & set(meta["blind_templates"]))
    ps = {n: to_promptset(n, v, device) for n, v in sets.items()}
    meta["split_sha256"] = {n: p.sha256() for n, p in ps.items()}
    meta["split_sizes"] = {n: len(v) for n, v in sets.items()}
    return ps, meta


# ----------------------------------------------------------------------------- screening
def eap_ig_head_scores(model, ps: PromptSet, steps: int, chunk: int = 16) -> tuple[np.ndarray, dict]:
    """Node-level EAP-IG (inputs) score per head: sum_pos (z_clean - z_corr) . mean_alpha dM/dz."""
    L, H = model.cfg.n_layers, model.cfg.n_heads
    scores = torch.zeros(L, H, device=ps.tokens.device)
    t0 = time.perf_counter()
    n_fwd = n_bwd = 0
    zname = [f"blocks.{layer}.attn.hook_z" for layer in range(L)]
    for s0 in range(0, len(ps.items), chunk):
        sl = slice(s0, s0 + chunk)
        toks, corr, pos, io, s = ps.tokens[sl], ps.corrupt[sl], ps.pos[sl], ps.io[sl], ps.s[sl]
        rows = torch.arange(len(toks), device=toks.device)
        with torch.no_grad():
            _, cc = model.run_with_cache(toks, names_filter=lambda n: n in zname or n == "hook_embed")
            _, cr = model.run_with_cache(corr, names_filter=lambda n: n in zname or n == "hook_embed")
            n_fwd += 2 * len(toks)
        e_clean, e_corr = cc["hook_embed"], cr["hook_embed"]
        grads = [torch.zeros_like(cc[z]) for z in zname]
        for k in range(1, steps + 1):
            alpha = k / steps
            store: dict = {}

            def emb_hook(e, hook, alpha=alpha):
                return e_corr + alpha * (e_clean - e_corr)

            def z_hook(z, hook):
                z.retain_grad()
                store[hook.name] = z
                return z

            hooks = [("hook_embed", emb_hook)] + [(z, z_hook) for z in zname]
            with torch.enable_grad():
                logits = model.run_with_hooks(toks, fwd_hooks=hooks)
                lg = logits[rows, pos]
                metric = (lg[rows, io] - lg[rows, s]).sum()
                metric.backward()
            n_fwd += len(toks)
            n_bwd += len(toks)
            for i, z in enumerate(zname):
                grads[i] += store[z].grad.detach() / steps
            model.zero_grad(set_to_none=True)
            del logits, store
        for i, z in enumerate(zname):
            dz = cc[z] - cr[z]  # (b, pos, h, d)
            scores[i] += (dz * grads[i]).sum(dim=(0, 1, 3))
        del cc, cr, grads
    scores = scores / len(ps.items)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return scores.detach().cpu().numpy(), {"wall_seconds": time.perf_counter() - t0, "forward_evals": n_fwd, "backward_evals": n_bwd,
                                           "ig_steps": steps, "corruption": "abc_counterfactual", "n_prompts": len(ps.items)}


def select_candidates(scores: np.ndarray, cfg: dict) -> dict:
    sc = cfg["gate_c"]["screening"]
    L, H = scores.shape
    flat = np.abs(scores).reshape(-1)
    order = np.lexsort((np.arange(L * H), -flat))  # by |score| desc, ties by index
    top = sorted(order[: sc["best_attribution_heads"]].tolist())
    rest = sorted(set(range(L * H)) - set(top))
    rng = np.random.default_rng(sc["frozen_screening_seed"])
    rnd = sorted(rng.choice(rest, size=sc["extra_random_heads"], replace=False).tolist())
    pool = sorted(top + rnd)
    heads = [[int(i // H), int(i % H)] for i in pool]
    return {"heads": heads, "eap_top_flat": top, "random_flat": rnd, "scores": scores.tolist(),
            "candidate_set_sha256": hashlib.sha256(json.dumps(heads).encode()).hexdigest()}


def make_runner(model, cand_heads, ps: PromptSet, impl: dict) -> HeadAblationRunner:
    return HeadAblationRunner(model, [tuple(h) for h in cand_heads], ps.tokens, resid_logit_diff(model, ps.io, ps.s), ps.pos,
                              microbatch=impl["gate_c"]["max_seq_microbatch"], final_resid=True)


def blind_masks(cfg: dict, p: int = 48) -> np.ndarray:
    bl = cfg["gate_c"]["blind"]
    rng = np.random.default_rng(cfg["seeds"]["gate_c_blind_masks"])
    out, seen = [], set()
    for d in bl["densities"]:
        k = n_ablated(p, d)
        got = 0
        while got < bl["masks_per_density"]:
            m = random_mask(rng, p, k)
            if mask_key(m) in seen:
                continue
            seen.add(mask_key(m))
            out.append(m)
            got += 1
    return np.stack(out)


def unit_designs(seed: int, cfg: dict, reserved: set, p: int = 48) -> tuple[np.ndarray, np.ndarray]:
    dens = cfg["shared"]["intervention_density_mixture"]
    warm = balanced_masks(np.random.default_rng([seed, 22]), p, cfg["gate_c"]["discovery"]["warm_start_masks"], dens, exclude=set(reserved))
    excl = set(reserved) | {mask_key(m) for m in warm}
    bmax = max(cfg["gate_c"]["discovery"]["budget_mask_checkpoints"])
    static = balanced_masks(np.random.default_rng([seed, 33]), p, bmax - len(warm), dens, exclude=excl)
    return warm, static


def make_problem(unit: str, warm, static, cfg: dict, reserved: frozenset, checkpoints=None) -> Problem:
    impl = cfg["_impl"]
    gc = cfg["gate_c"]
    return Problem(
        unit_id=unit, p=48, contexts=[None], target="abs", warm_start=warm, static_masks=static,
        checkpoints=list(checkpoints or gc["discovery"]["budget_mask_checkpoints"]),
        densities=list(cfg["shared"]["intervention_density_mixture"]),
        decision=DecisionSpec(mode="fixed", fixed_sizes=list(gc["discovery"]["retained_circuit_sizes"]), rmax=cfg["shared"]["faithful_R_max"],
                              lam=impl["decision"]["lam"], compute_pairs=False),
        acq=acq_spec(impl), reserved_keys=reserved,
    )


def oracle_fn(runner: HeadAblationRunner):
    def fn(M: np.ndarray) -> QueryResult:
        d = runner.effects(M)
        return QueryResult(masks=M, y={"abs": np.mean(np.abs(d), 1)[:, None], "effect": np.mean(d, 1)[:, None]}, per_prompt=d)

    return fn


def effect_hook_factory(method: str, cfg: dict):
    """Extra models fit on discovery data only: signed-effect surrogate (C-3 calibration/MAE)
    and a ridge model of log per-mask prompt variance (prompt-resampling term)."""
    from sklearn.linear_model import Ridge

    bcfg = bayes_cfg(cfg["_impl"])

    def hook(M, Y, PP, seed):
        sur = make_surrogate(method, 48, [None], seed + 101, bcfg)
        if method.startswith("bayes"):
            sur.fit(M, Y["effect"], warm=False)
        else:
            sur.fit(M, Y["effect"])
        v = np.var(PP, axis=1, ddof=1)
        rg = Ridge(alpha=1.0).fit(np.asarray(M, float), np.log(v + 1e-8))
        return {"effect": sur, "pvar": rg}

    return hook


def evaluate_unit(seed: int, snaps_by_method: dict, blind: HeadAblationRunner, bmasks: np.ndarray, bmask_targets: np.ndarray,
                  den: float, den_pp: np.ndarray, cfg: dict) -> tuple[list, list, dict]:
    """Blind audit of sealed snapshots: real GPT-2 interventions on blind prompts."""
    n_disc = cfg["gate_c"]["discovery"]["n_prompts_per_mask"]
    n_bl = cfg["gate_c"]["blind"]["n_prompts"]
    faith, pred_rows, pp_store = [], [], {}
    dens_of = bmasks.sum(1)
    rng = np.random.default_rng([seed, 77])
    for method, snaps in snaps_by_method.items():
        for snap in snaps:
            for k, C in snap.decision["selected"].items():
                d = blind.effects(retain_mask(48, C)[None])[0]
                num = float(np.mean(np.abs(d)))
                pp_store[f"{method}|{snap.budget}|{k}"] = np.abs(d).astype(np.float32)
                faith.append({"seed": seed, "method": method, "budget": snap.budget, "k": int(k), "circuit": json.dumps(C),
                              "numerator": num, "denominator": den, "R": num / den,
                              "pred_R": snap.decision["selected_pred_R"][k],
                              "P_faithful": (snap.decision.get("selected_P_faithful") or {}).get(k)})
            eff = snap.extra_surrogates["effect"]
            pv = np.exp(snap.extra_surrogates["pvar"].predict(np.asarray(bmasks, float)))
            if getattr(eff, "is_bayesian", False):
                pr = eff.predictive(bmasks, 0.90, extra_var=pv * (1.0 / n_disc + 1.0 / n_bl))
                mean, lo, hi = pr["mean"][:, 0], pr["lo"][:, 0], pr["hi"][:, 0]
                crps = crps_from_samples(pr["samples"], bmask_targets)
                # also intervals without the prompt-resampling term (diagnostic)
                pr0 = eff.predictive(bmasks, 0.90)
                lo0, hi0 = pr0["lo"][:, 0], pr0["hi"][:, 0]
            else:
                mean = eff.predict(bmasks)[:, 0]
                lo = hi = lo0 = hi0 = crps = np.full(len(bmasks), np.nan)
            for i in range(len(bmasks)):
                pred_rows.append({"seed": seed, "method": method, "budget": snap.budget, "mask_idx": i, "n_ablated": int(dens_of[i]),
                                  "truth": float(bmask_targets[i]), "pred": float(mean[i]), "lo": float(lo[i]), "hi": float(hi[i]),
                                  "lo_noprompt": float(lo0[i]), "hi_noprompt": float(hi0[i]), "crps": float(crps[i])})
    _ = rng
    pp_store["__denominator__"] = den_pp.astype(np.float32)
    return faith, pred_rows, pp_store


# ----------------------------------------------------------------------------- prepare
def prepare_gate_c(cfg: dict, prov: dict) -> dict:
    from . import tracking
    from .cli import _finish_parent, _gate_parent

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("Gate C preparation requires the GPU host")
    FROZEN.mkdir(parents=True, exist_ok=True)
    fz = FROZEN / "frozen_manifest.json"
    if fz.exists():
        print("Gate C frozen artifacts already exist:", fz.read_text()[:400])
        return json.loads(fz.read_text())
    pid = _gate_parent("bccd/gate-c", "prepare-c", prov, True, "parent_run_id")
    torch.manual_seed(0)
    model, minfo = load_model(device)
    ps, meta = build_splits(model, cfg, device)
    impl = cfg["_impl"]
    # --- screening (EAP-IG on screening prompts only)
    torch.cuda.reset_peak_memory_stats()
    scores, scost = eap_ig_head_scores(model, ps["screening"], impl["gate_c"]["eap_ig_steps"])
    scost["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
    cand = select_candidates(scores, cfg)
    cand["screening_cost"] = scost
    write_json(FROZEN / "candidate_heads.json", cand)
    # --- candidate-pool validity (screening prompts; never blind data)
    scr_runner = make_runner(model, cand["heads"], ps["screening"], impl)
    d_all = scr_runner.effects(np.ones((1, 48), np.uint8))[0]
    validity = {"mean_abs_all48_effect_screening": float(np.mean(np.abs(d_all))),
                "mean_clean_logit_diff_screening": float(scr_runner.clean.mean().item()),
                "threshold": cfg["gate_c"]["screening"]["min_abs_all_pool_ablation_logit_margin_change"]}
    validity["valid"] = validity["mean_abs_all48_effect_screening"] >= validity["threshold"]
    bm = blind_masks(cfg)
    np.save(FROZEN / "blind_masks.npy", bm)
    # --- development reference baseline preselection (dev prompts only; no blind data)
    dev_seed = impl["gate_c"]["dev_reference_seed"]
    reserved = frozenset(mask_key(m) for m in bm)
    warm, static = unit_designs(dev_seed, cfg, set(reserved))
    prob = make_problem(f"C-dev-{dev_seed}", warm, static, cfg, reserved, checkpoints=[cfg["gate_c"]["criterion_checkpoint_masks"]])
    dev_runner = make_runner(model, cand["heads"], ps["dev_discovery"], impl)
    ev_runner = make_runner(model, cand["heads"], ps["dev_eval"], impl)
    den = float(np.mean(np.abs(ev_runner.effects(np.ones((1, 48), np.uint8))[0])))
    dev_res = {}
    for method in NON_BAYES:
        orc = DiscoveryOracle(fn=oracle_fn(dev_runner), p=48, budget=128, method=method, unit_id=prob.unit_id,
                              prompt_batch_id=ps["dev_discovery"].sha256()[:12], reserved_keys=reserved, n_prompts_per_mask=32)
        snap = run_method(prob, method, orc, dev_seed, bayes_cfg(impl))[-1]
        Rs = {k: float(np.mean(np.abs(ev_runner.effects(retain_mask(48, C)[None])[0])) / den) for k, C in snap.decision["selected"].items()}
        dev_res[method] = {"R": Rs, "S": float(np.mean(list(Rs.values())))}
    ref = min(dev_res, key=lambda m: dev_res[m]["S"])
    dev_ref = {"reference_baseline": ref, "dev_results": dev_res, "dev_seed": dev_seed, "dev_eval_denominator": den,
               "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(FROZEN / "dev_reference.json", dev_ref)
    split_manifest = {n: [{"src": it["src"], "idx": it["idx"], "template": it["template"]} for it in p.items] for n, p in ps.items()}
    write_json(FROZEN / "splits.json", split_manifest)
    manifest = {"model": minfo, "splits_meta": meta, "candidate_set_sha256": cand["candidate_set_sha256"], "candidate_heads": cand["heads"],
                "eap_top": cand["eap_top_flat"], "random": cand["random_flat"], "screening_cost": scost, "validity": validity,
                "blind_masks_sha256": masks_sha256(bm), "dev_reference": ref,
                "files_sha256": {f.name: sha256_file(f) for f in sorted(FROZEN.glob("*")) if f.name != "frozen_manifest.json"},
                "provenance": prov, "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(fz, manifest)
    _finish_parent(pid, "FINISHED", {"gate_c_frozen_manifest.json": manifest, "candidate_heads.json": cand, "dev_reference.json": dev_ref},
                   metrics={"screening_wall_seconds": scost["wall_seconds"], "all48_effect_screening": validity["mean_abs_all48_effect_screening"]})
    print(json.dumps({k: manifest[k] for k in ("candidate_heads", "validity", "dev_reference", "screening_cost")}, indent=1, default=str))
    _ = tracking
    return manifest


# ----------------------------------------------------------------------------- confirmatory
def cmd_gate_c(args, cfg: dict) -> int:
    from . import tracking
    from .cli import _finish_parent, _gate_parent, _require_clean, _write_status

    prov = provenance(cfg)
    confirmatory = args.max_seeds is None and not args.dry_run
    _require_clean(prov, confirmatory, args.allow_dirty)
    if not torch.cuda.is_available():
        raise SystemExit("Gate C requires the GPU host (BLOCKED otherwise)")
    device = "cuda"
    fz = FROZEN / "frozen_manifest.json"
    if not fz.exists():
        raise SystemExit("run `prepare --stage c` first (frozen candidate pool, splits, dev reference)")
    manifest = json.loads(fz.read_text())
    for name, h in manifest["files_sha256"].items():
        if sha256_file(FROZEN / name) != h:
            raise SystemExit(f"frozen artifact {name} hash mismatch")
    out = run_root(confirmatory) / "gate_c"
    out.mkdir(parents=True, exist_ok=True)
    gate = "gate-c" if confirmatory else "gate-c-smoke"
    exp = "bccd/gate-c" if confirmatory else "bccd/dev-smoke"
    _write_status(gate, state="running", pid=os.getpid())
    pid = _gate_parent(exp, gate, prov, confirmatory, "parent_run_id")
    try:
        torch.manual_seed(0)
        model, minfo = load_model(device)
        ps, meta = build_splits(model, cfg, device)
        if meta["split_sha256"] != manifest["splits_meta"]["split_sha256"]:
            raise SystemExit("split hashes differ from frozen manifest")
        cand = json.loads((FROZEN / "candidate_heads.json").read_text())
        bm = np.load(FROZEN / "blind_masks.npy")
        reserved = frozenset(mask_key(m) for m in bm)
        impl = cfg["_impl"]
        seeds = cfg["seeds"]["gate_c_discovery"]
        eval_set = "blind"
        cps = None
        if not confirmatory:
            # exploratory smoke: never touch the blind split; use dev prompts and the dev seed
            seeds = [impl["gate_c"]["dev_reference_seed"]]
            eval_set = "dev_eval"
            cps = [64]
        blind = make_runner(model, cand["heads"], ps[eval_set], impl)
        d_all = blind.effects(np.ones((1, 48), np.uint8))[0]
        den = float(np.mean(np.abs(d_all)))
        bt_path = out / "blind_mask_targets.npy"
        bmask_eff = blind.effects(bm)  # evaluation only
        np.save(bt_path, bmask_eff.astype(np.float32))
        bmask_targets = bmask_eff.mean(1)
        n_disc = cfg["gate_c"]["discovery"]["n_prompts_per_mask"]
        bcfg = bayes_cfg(impl)
        status_all = []
        for si, seed in enumerate(seeds):
            udir = out / f"seed-{seed}"
            stp = udir / "status.json"
            if stp.exists() and json.loads(stp.read_text()).get("status") == "complete":
                status_all.append(json.loads(stp.read_text()))
                continue
            udir.mkdir(parents=True, exist_ok=True)
            idx = list(range(si * n_disc, (si + 1) * n_disc))
            src = ps["discovery"].items if confirmatory else ps["dev_discovery"].items
            dps = to_promptset(f"disc-{seed}", [src[i] for i in idx], device)
            runner = make_runner(model, cand["heads"], dps, impl)
            warm, static = unit_designs(seed, cfg, set(reserved))
            prob = make_problem(f"C-{seed}", warm, static, cfg, reserved, checkpoints=cps)
            snaps_by_method, queries, cost_rows, decisions = {}, [], [], {}
            for method in METHODS:
                torch.cuda.reset_peak_memory_stats()
                orc = DiscoveryOracle(fn=oracle_fn(runner), p=48, budget=max(prob.checkpoints), method=method, unit_id=prob.unit_id,
                                      prompt_batch_id=dps.sha256()[:12], reserved_keys=reserved, n_prompts_per_mask=n_disc)
                snaps = run_method(prob, method, orc, seed, bcfg, extra_hook=effect_hook_factory(method, cfg))
                snaps_by_method[method] = snaps
                queries.extend(orc.log)
                peak = torch.cuda.max_memory_allocated() / 2**20
                for snap in snaps:
                    c = end_to_end_cost(snap)
                    c.update({"seed": seed, "method": method, "budget": snap.budget, "peak_vram_mb": peak,
                              "discovery_masks": snap.costs["n_masks"], "discovery_forward_evals": snap.costs["n_prompt_evals"],
                              "screening_wall_seconds": manifest["screening_cost"]["wall_seconds"],
                              "screening_forward_evals": manifest["screening_cost"]["forward_evals"],
                              "screening_backward_evals": manifest["screening_cost"]["backward_evals"]})
                    cost_rows.append(c)
                    decisions[f"{method}@{snap.budget}"] = {"decision": snap.decision, "model_summary": snap.model_summary,
                                                             "acq_log": snap.costs.get("acq_log", [])}
            # seal decisions before any blind evaluation of this seed
            write_json(udir / "decisions_sealed.json", decisions)
            sealed_sha = sha256_file(udir / "decisions_sealed.json")
            faith, preds, pp = evaluate_unit(seed, snaps_by_method, blind, bm, bmask_targets, den, np.abs(d_all), cfg)
            pd.DataFrame(faith).to_csv(udir / "faithfulness.csv", index=False)
            pd.DataFrame(preds).to_parquet(udir / "blind_predictions.parquet", index=False)
            pd.DataFrame(cost_rows).to_csv(udir / "costs.csv", index=False)
            pd.DataFrame(queries).to_parquet(udir / "queries.parquet", index=False)
            np.savez_compressed(udir / "blind_abs_discrepancies.npz", **{k.replace("|", "__"): v for k, v in pp.items()})
            # posterior summaries (bayes particles: support/coef summary only)
            post = {}
            for m, snaps in snaps_by_method.items():
                for s in snaps:
                    post[f"{m}@{s.budget}"] = s.model_summary
            write_json(udir / "posterior_summaries.json", post)
            fdf = pd.DataFrame(faith)
            run_ids = {}
            for method in METHODS:
                tags = tracking.base_tags(prov, "C", confirmatory, world_id=f"C-{seed}", seed=seed, method=method, task="ioi",
                                          model_id="gpt2", model_revision=json.dumps(minfo["hf_revision"]), candidate_set_sha256=cand["candidate_set_sha256"],
                                          split_sha256=manifest["splits_meta"]["split_sha256"]["blind"], prompt_batch_id=dps.sha256()[:12],
                                          device=device, budget_masks="64,128,256", decisions_sealed_sha256=sealed_sha)
                with tracking.run(exp, f"C:{seed}:{method}", tags, pid) as r:
                    for B in prob.checkpoints:
                        g = fdf[(fdf.method == method) & (fdf.budget == B)]
                        mets = {f"circuit_R_{int(k)}": v for k, v in zip(g.k, g.R)}
                        mets["S"] = float(g.R.mean())
                        cr = [c for c in cost_rows if c["method"] == method and c["budget"] == B][0]
                        mets.update({k: v for k, v in cr.items() if isinstance(v, int | float) and k not in ("seed", "budget")})
                        pr = pd.DataFrame([p for p in preds if p["method"] == method and p["budget"] == B])
                        mets["test_mae"] = float(np.mean(np.abs(pr.pred - pr.truth)))
                        if method.startswith("bayes"):
                            mets["coverage_90"] = float(np.mean((pr.truth >= pr.lo) & (pr.truth <= pr.hi)))
                            mets["interval_width_90"] = float(np.mean(pr.hi - pr.lo))
                        tracking.log_metrics_step(mets, step=B)
                    tracking.log_json({k: v for k, v in decisions.items() if k.startswith(method + "@")}, "candidate_circuits.json")
                    run_ids[method] = r.info.run_id
            st = {"status": "complete", "seed": seed, "dir": str(udir), "decisions_sealed_sha256": sealed_sha, "mlflow_runs": run_ids,
                  "discovery_prompts_sha256": dps.sha256()}
            write_json(stp, st)
            status_all.append(st)
            print(json.dumps({"seed": seed, "status": "complete"}), flush=True)
        from .criteria_c import gate_c_criteria

        summary = gate_c_criteria(out, cfg, manifest, seeds) if confirmatory else {"verdict": "EXPLORATORY"}
        summary.update({"blind_denominator": den, "provenance": prov, "parent_run_id": pid, "model": minfo})
        write_json(out / "summary.json", summary)
        _finish_parent(pid, "FINISHED", {"gate_c_summary.json": summary})
        _write_status(gate, state="complete", exit_code=0, verdict=summary.get("verdict"))
        print(json.dumps({k: v for k, v in summary.items() if k not in ("provenance",)}, indent=1, default=str)[:6000])
        return 0
    except BaseException as e:
        _write_status(gate, state="crashed", error=repr(e), traceback=traceback.format_exc())
        try:
            _finish_parent(pid, "FAILED", {"error.json": {"error": repr(e), "traceback": traceback.format_exc()}})
        except Exception:
            pass
        raise
