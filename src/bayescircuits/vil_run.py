"""BCCD-VIL-1.0 runner: frozen splits, per-seed discovery with sealed decisions, blind audit,
MLflow logging. See docs/08_VERIFICATION_PROTOCOL.md.

Sets
* ``dev``          development seeds; evaluated on MIB *dev* prompts only (never blind)
* ``confirmatory`` confirmatory seeds; evaluated on the frozen 128-prompt blind split
* ``bridge``       the five BCCD-PILOT-1.0 Gate C seeds and their discovery batches (secondary;
                   lasso_static_128 must reproduce the pilot's lasso_pair@128 decisions)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .config import PROJECT_ROOT, sha256_file
from .masks import mask_key, masks_sha256, retain_mask
from .models.bayes import SpikeSlabConfig
from .oracle.base import DiscoveryOracle, QueryResult
from .oracle.transformer import HeadAblationRunner, resid_logit_diff
from .provenance import env_info, git_info, lock_info, write_json
from .vil import METHODS_VIL, VERIFYING, VILSpec, run_vil_method, seed_designs

PROTOCOL_ID = "BCCD-VIL-1.0"
PREREG = PROJECT_ROOT / "configs" / "preregistration_verification_v1.yaml"
IMPL = PROJECT_ROOT / "configs" / "verification_implementation_v1.yaml"
PROTOCOL_DOCS = [PROJECT_ROOT / "docs" / "08_VERIFICATION_PROTOCOL.md", PROJECT_ROOT / "docs" / "09_VERIFICATION_METHODS.md"]
VIL_ROOT = PROJECT_ROOT / "runs" / "verification_v1"
VIL_FROZEN = VIL_ROOT / "frozen"
PILOT_FROZEN = PROJECT_ROOT / "runs" / "gate_c" / "frozen"
REPORT_DIR = PROJECT_ROOT / "reports" / "verification_v1"
FREEZE_RECORD = REPORT_DIR / "preregistration_hashes.json"
EXPERIMENTS = {"confirmatory": "bccd/verification-v1", "dev": "bccd/verification-v1-dev", "bridge": "bccd/verification-v1-bridge",
               "exploratory": "bccd/verification-v1-exploratory"}
LOGS = PROJECT_ROOT / "logs"


# ----------------------------------------------------------------------------- config / provenance
def load_vil_config(prereg: Path = PREREG, impl: Path = IMPL) -> dict:
    cfg = yaml.safe_load(Path(prereg).read_text())
    cfg["_impl"] = yaml.safe_load(Path(impl).read_text())
    cfg["_hashes"] = protocol_hashes(prereg, impl)
    return cfg


def protocol_hashes(prereg: Path = PREREG, impl: Path = IMPL) -> dict:
    h = {"preregistration_yaml": sha256_file(prereg), "implementation_yaml": sha256_file(impl)}
    for d in PROTOCOL_DOCS:
        h[d.name] = sha256_file(d) if d.exists() else None
    h["uv_lock"] = lock_info()["uv_lock_sha256"]
    return h


def vil_provenance(cfg: dict) -> dict:
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "protocol_version": PROTOCOL_ID,
            "hashes": cfg["_hashes"], **git_info(), **lock_info(), **env_info()}


def spec_from_cfg(cfg: dict, rounds_override: int | None = None, budget_override: tuple[int, int] | None = None) -> VILSpec:
    d = dict(cfg["_impl"]["vil"])
    if rounds_override is not None:
        d["rounds"] = rounds_override
    if budget_override is not None:  # exploratory only: (budget, n_verify); n_discovery = budget - n_verify
        d["budget"], d["n_verify"] = budget_override
        d["n_discovery"] = d["budget"] - d["n_verify"]
    return VILSpec.from_dict(d)


def bayes_cfg(cfg: dict) -> SpikeSlabConfig:
    return SpikeSlabConfig(**cfg["_impl"]["bayes"])


def solver_cfg(cfg: dict) -> dict:
    return {"tol": cfg["_impl"]["lasso"]["tol"], "max_iter": cfg["_impl"]["lasso"]["max_iter"]}


# ----------------------------------------------------------------------------- splits
def build_vil_splits(model, cfg: dict) -> tuple[dict, dict]:
    """Prompt items for every VIL set. Returns ({"batches": {seed: items}, "eval": {name: items}}, meta)."""
    from .gate_c import eligible, load_mib

    pilot = json.loads((PILOT_FROZEN / "splits.json").read_text())
    pools, byidx, meta = {}, {}, {"mib_sha256": {}}
    for split in ("train", "dev", "test"):
        data, sha = load_mib(split)
        el, excl = eligible(model, data, split)
        pools[split] = el
        byidx[split] = {it["idx"]: it for it in el}
        meta["mib_sha256"][split] = sha
        meta[f"exclusions_{split}"] = excl
    used_train = {it["idx"] for n in ("screening", "discovery", "dev_discovery") for it in pilot[n]}
    used_dev = {it["idx"] for it in pilot["dev_eval"]}
    rng = np.random.default_rng(cfg["seeds"]["split_seed"])
    rest = [it for it in pools["train"] if it["idx"] not in used_train]
    rest = [rest[i] for i in rng.permutation(len(rest))]
    npm = cfg["budget"]["prompts_per_mask"]
    dev_seeds, conf_seeds, bridge_seeds = cfg["seeds"]["development"], cfg["seeds"]["confirmatory"], cfg["seeds"]["bridge"]
    batches: dict[int, list] = {}
    off = 0
    for s in dev_seeds + conf_seeds:
        batches[s] = rest[off : off + npm]
        off += npm
    pilot_disc = [byidx["train"][it["idx"]] for it in pilot["discovery"]]
    for i, s in enumerate(bridge_seeds):
        batches[s] = pilot_disc[i * npm : (i + 1) * npm]
    dev_rest = [it for it in pools["dev"] if it["idx"] not in used_dev]
    dev_rest = [dev_rest[i] for i in rng.permutation(len(dev_rest))]
    by_t: dict[str, list] = {}
    for it in dev_rest:
        by_t.setdefault(it["template"], []).append(it)
    dev_eval, k, tn = [], 0, sorted(by_t)
    while len(dev_eval) < cfg["splits"]["dev_eval_n_prompts"]:
        t = tn[k % len(tn)]
        if by_t[t]:
            dev_eval.append(by_t[t].pop())
        k += 1
    blind = [byidx["test"][it["idx"]] for it in pilot["blind"]]
    ev = {"blind": blind, "dev_eval": dev_eval}
    # disjointness checks (VIL batches vs each other, vs pilot screening, and evaluation sets)
    key = lambda it: (it["src"], it["idx"])  # noqa: E731
    seen: dict = {}
    for s in dev_seeds + conf_seeds:
        for it in batches[s]:
            assert key(it) not in seen, f"prompt reused across VIL batches: {key(it)}"
            seen[key(it)] = s
    scr = {("train", it["idx"]) for it in pilot["screening"]}
    assert not (set(seen) & scr), "VIL discovery overlaps pilot screening"
    for n, items in ev.items():
        assert not ({key(it) for it in items} & set(seen)), f"{n} overlaps VIL discovery"
    meta["blind_templates"] = sorted({it["template"] for it in blind})
    meta["discovery_templates"] = sorted({it["template"] for s in conf_seeds for it in batches[s]})
    meta["blind_template_overlap_with_confirmatory_discovery"] = sorted(set(meta["blind_templates"]) & set(meta["discovery_templates"]))
    return {"batches": batches, "eval": ev}, meta


def items_sha(items: list[dict]) -> str:
    return hashlib.sha256(json.dumps([(it["src"], it["idx"], it["prompt"]) for it in items]).encode()).hexdigest()


def make_runner(model, heads, ps, cfg: dict) -> HeadAblationRunner:
    return HeadAblationRunner(model, [tuple(h) for h in heads], ps.tokens, resid_logit_diff(model, ps.io, ps.s), ps.pos,
                              microbatch=cfg["_impl"]["runtime"]["max_seq_microbatch"], final_resid=True)


def oracle_fn(runner: HeadAblationRunner):
    def fn(M: np.ndarray) -> QueryResult:
        d = runner.effects(M)
        return QueryResult(masks=M, y={"abs": np.mean(np.abs(d), 1)[:, None], "effect": np.mean(d, 1)[:, None]}, per_prompt=d)

    return fn


# ----------------------------------------------------------------------------- prepare / freeze
def cmd_vil_prepare(args) -> int:
    """Freeze VIL prompt batches, evaluation sets and per-seed mask designs (no oracle calls)."""
    from .gate_c import load_model

    cfg = load_vil_config()
    VIL_FROZEN.mkdir(parents=True, exist_ok=True)
    fz = VIL_FROZEN / "vil_frozen_manifest.json"
    if fz.exists():
        print("VIL frozen manifest already exists:", fz)
        return 0
    pilot_manifest = json.loads((PILOT_FROZEN / "frozen_manifest.json").read_text())
    for name, h in pilot_manifest["files_sha256"].items():
        if sha256_file(PILOT_FROZEN / name) != h:
            raise SystemExit(f"pilot frozen artifact {name} hash mismatch")
    cand = json.loads((PILOT_FROZEN / "candidate_heads.json").read_text())
    if cand["candidate_set_sha256"] != cfg["candidate_pool"]["candidate_set_sha256"]:
        raise SystemExit("candidate pool hash differs from the preregistration")
    model, minfo = load_model("cuda" if torch.cuda.is_available() else "cpu")
    sp, meta = build_vil_splits(model, cfg)
    blind_sha = items_sha(sp["eval"]["blind"])
    if blind_sha != pilot_manifest["splits_meta"]["split_sha256"]["blind"]:
        raise SystemExit("reconstructed blind split differs from the pilot's frozen blind split")
    bridge_check = {}
    for s in cfg["seeds"]["bridge"]:
        st = PROJECT_ROOT / "runs" / "gate_c" / f"seed-{s}" / "status.json"
        if st.exists():
            bridge_check[s] = json.loads(st.read_text()).get("discovery_prompts_sha256") == items_sha(sp["batches"][s])
    bm = np.load(PILOT_FROZEN / "blind_masks.npy")
    reserved = {mask_key(m) for m in bm}
    spec = spec_from_cfg(cfg)
    dens = cfg["budget"]["design_densities"]
    designs = {}
    for s in sorted(sp["batches"]):
        d = seed_designs(s, reserved, spec, dens)
        designs[str(s)] = {k: masks_sha256(v) for k, v in d.items()}
    split_ids = {"batches": {str(s): [{"src": it["src"], "idx": it["idx"], "template": it["template"]} for it in v] for s, v in sp["batches"].items()},
                 "eval": {n: [{"src": it["src"], "idx": it["idx"], "template": it["template"]} for it in v] for n, v in sp["eval"].items()}}
    write_json(VIL_FROZEN / "vil_splits.json", split_ids)
    manifest = {"protocol": PROTOCOL_ID, "model": minfo, "splits_meta": meta,
                "batch_sha256": {str(s): items_sha(v) for s, v in sp["batches"].items()},
                "eval_sha256": {n: items_sha(v) for n, v in sp["eval"].items()},
                "blind_matches_pilot": True, "bridge_batches_match_pilot": bridge_check,
                "candidate_set_sha256": cand["candidate_set_sha256"], "pilot_frozen_files_sha256": pilot_manifest["files_sha256"],
                "blind_masks_sha256": masks_sha256(bm), "design_sha256": designs,
                "vil_splits_sha256": sha256_file(VIL_FROZEN / "vil_splits.json"),
                "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "provenance": vil_provenance(cfg)}
    write_json(fz, manifest)
    print(json.dumps({k: manifest[k] for k in ("batch_sha256", "eval_sha256", "bridge_batches_match_pilot")}, indent=1)[:3000])
    return 0


def cmd_vil_freeze(args) -> int:
    """Record protocol/config/lockfile/source/frozen-artifact hashes before confirmatory execution."""
    cfg = load_vil_config()
    if FREEZE_RECORD.exists() and not args.force:
        print("freeze record already exists:", FREEZE_RECORD.read_text()[:2000])
        return 0
    g = git_info()
    if g["git_dirty"]:
        raise SystemExit("commit protocol/config/code before freezing")
    fz = VIL_FROZEN / "vil_frozen_manifest.json"
    rec = {"protocol": PROTOCOL_ID, "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "source_git_sha": g["git_sha"], "hashes": cfg["_hashes"],
           "vil_frozen_manifest_sha256": sha256_file(fz) if fz.exists() else None,
           "vil_splits_sha256": sha256_file(VIL_FROZEN / "vil_splits.json") if (VIL_FROZEN / "vil_splits.json").exists() else None,
           "pilot_frozen_files_sha256": json.loads((PILOT_FROZEN / "frozen_manifest.json").read_text())["files_sha256"],
           "confirmatory_seeds": cfg["seeds"]["confirmatory"], "implementation_vil": cfg["_impl"]["vil"],
           "statement": "Recorded before any confirmatory BCCD-VIL-1.0 seed was run or any confirmatory outcome inspected."}
    write_json(FREEZE_RECORD, rec)
    print(json.dumps(rec, indent=1))
    return 0


def _check_freeze(cfg: dict) -> dict:
    if not FREEZE_RECORD.exists():
        raise SystemExit("no preregistration freeze record: run `vil-freeze` (and commit it) first")
    rec = json.loads(FREEZE_RECORD.read_text())
    if rec["hashes"] != cfg["_hashes"]:
        raise SystemExit(f"protocol/config/lock hashes changed since freeze: {rec['hashes']} vs {cfg['_hashes']}")
    if sha256_file(VIL_FROZEN / "vil_frozen_manifest.json") != rec["vil_frozen_manifest_sha256"]:
        raise SystemExit("VIL frozen manifest changed since freeze")
    return rec


# ----------------------------------------------------------------------------- per-seed execution
def _write_status(name: str, **kw) -> None:
    p = LOGS / f"{name}.status.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw)
    cur["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(p, cur)


def _strip(dec: dict) -> dict:
    return {k: v for k, v in dec.items() if not k.startswith("_")}


def evaluate_seed(seed: int, decisions: dict, eval_runner: HeadAblationRunner, eval_den_pp: np.ndarray, audit_runner: HeadAblationRunner,
                  p: int, ks: list[int]) -> tuple[list[dict], dict, dict]:
    """Audit sealed circuits with real interventions: eval prompts (blind or dev) + discovery-prompt audit."""
    rows, todo = [], []
    for method, dec in decisions.items():
        for role in ("final", "surrogate_at_104", "refit_128"):
            for k, rec in (dec.get(role) or {}).items():
                todo.append((method, role, int(k), rec))
        for i, rec in enumerate(dec.get("verified", [])):
            todo.append((method, f"verified_{i:02d}", int(rec["k"]), rec))
    uniq: dict[tuple, int] = {}
    for _, _, _, rec in todo:
        uniq.setdefault(tuple(rec["circuit"]), len(uniq))
    circuits = list(uniq)
    M = np.stack([retain_mask(p, C) for C in circuits])
    t0 = time.perf_counter()
    d_eval = np.abs(eval_runner.effects(M))  # (n, n_eval)
    d_disc = np.abs(audit_runner.effects(M))
    den_disc_pp = np.abs(audit_runner.effects(np.ones((1, p), np.uint8))[0])
    audit_seconds = time.perf_counter() - t0
    den_eval, den_disc = float(eval_den_pp.mean()), float(den_disc_pp.mean())
    pp = {"__denominator__": eval_den_pp.astype(np.float32)}
    for method, role, k, rec in todo:
        j = uniq[tuple(rec["circuit"])]
        num = float(d_eval[j].mean())
        row = {"seed": seed, "method": method, "role": role if not role.startswith("verified_") else "verified", "slot": role, "k": k,
               "circuit": json.dumps(rec["circuit"]), "numerator": num, "denominator": den_eval, "R": num / den_eval,
               "R_disc_audit": float(d_disc[j].mean()) / den_disc, "u_disc_audit": float(d_disc[j].mean()), "denominator_disc": den_disc,
               "pred_R": rec.get("pred_R"), "pred_u": rec.get("pred_u"), "P_faithful": rec.get("P_faithful"),
               "pred_R_q05": rec.get("pred_R_q05"), "pred_R_q95": rec.get("pred_R_q95"),
               "pred_y_lo90": rec.get("pred_y_lo90"), "pred_y_hi90": rec.get("pred_y_hi90"),
               "measured_u_discovery": rec.get("measured_u"), "round": rec.get("round"), "source": rec.get("source")}
        rows.append(row)
        if role in ("final", "surrogate_at_104", "refit_128"):
            pp[f"{role}__{method}__{k}"] = d_eval[j].astype(np.float32)
    return rows, pp, {"audit_seconds": audit_seconds, "n_audit_circuits": len(circuits), "eval_forward": int(len(circuits) * d_eval.shape[1]),
                      "disc_audit_forward": int((len(circuits) + 1) * d_disc.shape[1]), "denominator_disc": den_disc, "denominator_eval": den_eval}


def run_set(args) -> int:
    from . import tracking
    from .gate_c import load_model, to_promptset

    set_name = args.set
    cfg = load_vil_config()
    prov = vil_provenance(cfg)
    confirmatory = set_name == "confirmatory"
    variant = args.variant
    if confirmatory:
        if variant:
            raise SystemExit("variants are development-only")
        if prov["git_dirty"] and not args.allow_dirty:
            raise SystemExit("refusing confirmatory run on a dirty git worktree")
        freeze = _check_freeze(cfg)
        prov["freeze_record"] = freeze
    if not torch.cuda.is_available():
        raise SystemExit("VIL runs require the GPU host")
    fz = VIL_FROZEN / "vil_frozen_manifest.json"
    if not fz.exists():
        raise SystemExit("run `vil-prepare` first")
    fman = json.loads(fz.read_text())
    if sha256_file(VIL_FROZEN / "vil_splits.json") != fman["vil_splits_sha256"]:
        raise SystemExit("vil_splits.json hash mismatch")
    rounds_override = budget_override = None
    exploratory = set_name == "exploratory"
    if exploratory and not (variant and variant.startswith("B")):
        raise SystemExit("exploratory runs need a budget variant such as B256V0 or B128V48")
    if variant:
        if variant.startswith("rounds"):
            rounds_override = int(variant[len("rounds"):])
        elif variant.startswith("B") and exploratory:  # post-verdict exploratory budget/allocation variants
            b, v = variant[1:].split("V")
            budget_override = (int(b), int(v))
        elif not variant.startswith("rerun-"):  # rerun-<tag>: development re-execution after a fix, no overrides
            raise SystemExit(f"unknown variant {variant}")
    spec = spec_from_cfg(cfg, rounds_override, budget_override)
    methods = args.methods.split(",") if args.methods else list(METHODS_VIL)
    if confirmatory and methods != list(METHODS_VIL):
        raise SystemExit("confirmatory runs execute every preregistered method")
    if variant and variant.startswith("rounds"):
        methods = [m for m in methods if m in ("lasso_vil", "bayes_vil")]
    seeds = {"dev": cfg["seeds"]["development"], "confirmatory": cfg["seeds"]["confirmatory"], "bridge": cfg["seeds"]["bridge"],
             "exploratory": cfg["seeds"]["confirmatory"]}[set_name]
    seeds = [s for i, s in enumerate(seeds) if i % args.n_shards == args.shard]
    out = VIL_ROOT / (set_name if not variant else f"{set_name}_{variant}")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"vil-{set_name}{'-' + variant if variant else ''}-shard{args.shard}of{args.n_shards}"
    _write_status(tag, state="running", pid=os.getpid(), seeds=seeds)
    exp = EXPERIMENTS[set_name]
    tracking.setup(exp)
    ptags = tracking.base_tags({"protocol_version": PROTOCOL_ID, "protocol_doc_sha256": cfg["_hashes"]["08_VERIFICATION_PROTOCOL.md"],
                                "config_sha256": cfg["_hashes"]["preregistration_yaml"], "impl_config_sha256": cfg["_hashes"]["implementation_yaml"],
                                "git_sha": prov["git_sha"], "git_dirty": prov["git_dirty"], "uv_lock_sha256": prov["uv_lock_sha256"],
                                "torch_version": prov.get("torch_version")}, "VIL", confirmatory, role="parent", set=set_name,
                               variant=variant or "none", shard=f"{args.shard}/{args.n_shards}")
    pid = tracking.create_parent(exp, f"{tag}-parent", ptags)
    _write_status(tag, parent_run_id=pid)
    client = tracking.setup(exp).tracking.MlflowClient()
    try:
        torch.manual_seed(0)
        model, minfo = load_model("cuda")
        sp_ids = json.loads((VIL_FROZEN / "vil_splits.json").read_text())
        from .gate_c import eligible, load_mib

        byidx = {}
        for split in ("train", "dev", "test"):
            el, _ = eligible(model, load_mib(split)[0], split)
            byidx[split] = {it["idx"]: it for it in el}
        get = lambda lst: [byidx[it["src"]][it["idx"]] for it in lst]  # noqa: E731
        cand = json.loads((PILOT_FROZEN / "candidate_heads.json").read_text())
        heads = cand["heads"]
        bm = np.load(PILOT_FROZEN / "blind_masks.npy")
        reserved = frozenset(mask_key(m) for m in bm)
        eval_name = "dev_eval" if set_name == "dev" else "blind"
        ev_items = get(sp_ids["eval"][eval_name])
        if items_sha(ev_items) != fman["eval_sha256"][eval_name]:
            raise SystemExit(f"{eval_name} prompt hash mismatch")
        ev_ps = to_promptset(eval_name, ev_items, "cuda")
        eval_runner = make_runner(model, heads, ev_ps, cfg)
        eval_den_pp = np.abs(eval_runner.effects(np.ones((1, spec.p), np.uint8))[0])
        bcfg, solver = bayes_cfg(cfg), solver_cfg(cfg)
        dens = cfg["budget"]["design_densities"]
        npm = cfg["budget"]["prompts_per_mask"]
        for seed in seeds:
            udir = out / f"seed-{seed}"
            stp = udir / "status.json"
            if stp.exists() and json.loads(stp.read_text()).get("status") == "complete":
                print(json.dumps({"seed": seed, "status": "already complete"}), flush=True)
                continue
            udir.mkdir(parents=True, exist_ok=True)
            items = get(sp_ids["batches"][str(seed)])
            if items_sha(items) != fman["batch_sha256"][str(seed)]:
                raise SystemExit(f"discovery batch hash mismatch for seed {seed}")
            dps = to_promptset(f"disc-{seed}", items, "cuda")
            runner = make_runner(model, heads, dps, cfg)
            designs = seed_designs(seed, set(reserved), spec, dens)
            if not exploratory:  # exploratory budgets use other prefixes of the same frozen seeded streams
                for k_, v in designs.items():
                    if masks_sha256(v) != fman["design_sha256"][str(seed)][k_]:
                        raise SystemExit(f"design {k_} hash mismatch for seed {seed}")
            elif not np.array_equal(designs["warm"], seed_designs(seed, set(reserved), spec_from_cfg(cfg), dens)["warm"]):
                raise SystemExit("exploratory warm start differs from the frozen design")
            cache: dict = {}
            decisions, queries = {}, []
            t_seed = time.perf_counter()
            for method in methods:
                torch.cuda.reset_peak_memory_stats()
                orc = DiscoveryOracle(fn=oracle_fn(runner), p=spec.p, budget=spec.budget, method=method, unit_id=f"VIL-{seed}",
                                      prompt_batch_id=dps.sha256()[:12], reserved_keys=reserved, shared_cache=cache, n_prompts_per_mask=npm)
                w0 = time.perf_counter()
                dec = run_vil_method(method, orc, designs, spec, seed, bcfg, solver)
                dec["wall_total_method"] = time.perf_counter() - w0
                dec.pop("_data")
                dec["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
                dec["logical_oracle_seconds"] = float(sum(r["time_ms"] for r in orc.log) / 1000.0)
                dec["physical_mask_evals"] = int(sum(not r["reused_cache"] for r in orc.log))
                dec["cache_hits"] = int(sum(bool(r["reused_cache"]) for r in orc.log))
                dec["n_blind_or_reserved_queries"] = int(sum(1 for r in orc.log if bytes.fromhex(r["mask_bits"]) + bytes([spec.p % 256]) in reserved))
                for r in orc.log:
                    r["kind"] = None
                kinds = ["discovery"] * dec["n_discovery_masks"] + ["verification"] * dec["n_verification_masks"]
                for r, kd in zip(orc.log, kinds):
                    r["kind"] = kd
                queries.extend(orc.log)
                decisions[method] = dec
                print(json.dumps({"seed": seed, "method": method, "n_masks": dec["n_masks"], "wall": round(dec["wall_total_method"], 2)}), flush=True)
            write_json(udir / "decisions_sealed.json", {m: _strip(d) for m, d in decisions.items()})
            sealed_sha = sha256_file(udir / "decisions_sealed.json")
            # ---------------- evaluation strictly after sealing
            audit_runner = make_runner(model, heads, dps, cfg)
            rows, pp, ainfo = evaluate_seed(seed, decisions, eval_runner, eval_den_pp, audit_runner, spec.p, list(spec.ks))
            fdf = pd.DataFrame(rows)
            fdf["eval_set"] = eval_name
            fdf.to_csv(udir / "faithfulness.csv", index=False)
            np.savez_compressed(udir / "eval_abs_discrepancies.npz", **pp)
            pd.DataFrame(queries).to_parquet(udir / "queries.parquet", index=False)
            cost_rows = []
            for m, d in decisions.items():
                c = {"seed": seed, "method": m, "n_masks": d["n_masks"], "n_discovery_masks": d["n_discovery_masks"],
                     "n_verification_masks": d["n_verification_masks"], "logical_prompt_evals": d["n_prompt_evals_logical"],
                     "physical_mask_evals": d["physical_mask_evals"], "cache_hits": d["cache_hits"],
                     "logical_oracle_seconds": d["logical_oracle_seconds"], "peak_vram_mb": d["peak_vram_mb"],
                     "wall_total_method": d["wall_total_method"], "wall_e2e_measured": d["costs"]["wall_e2e"],
                     "n_blind_or_reserved_queries": d["n_blind_or_reserved_queries"]}
                for nm in ("wall", "cpu"):
                    for kk, vv in d["costs"][nm].items():
                        c[f"{nm}_{kk}"] = vv
                # logical end-to-end: CPU work measured + logical oracle seconds (cache-independent)
                c["wall_e2e_logical"] = d["logical_oracle_seconds"] + sum(d["costs"]["wall"].get(k, 0.0) for k in ("fit", "search", "acquisition"))
                cost_rows.append(c)
            pd.DataFrame(cost_rows).to_csv(udir / "costs.csv", index=False)
            run_ids = {}
            for m, d in decisions.items():
                tags = tracking.base_tags({"protocol_version": PROTOCOL_ID, "protocol_doc_sha256": cfg["_hashes"]["08_VERIFICATION_PROTOCOL.md"],
                                           "config_sha256": cfg["_hashes"]["preregistration_yaml"], "impl_config_sha256": cfg["_hashes"]["implementation_yaml"],
                                           "git_sha": prov["git_sha"], "git_dirty": prov["git_dirty"], "uv_lock_sha256": prov["uv_lock_sha256"],
                                           "torch_version": prov.get("torch_version")}, "VIL", confirmatory,
                                          world_id=f"VIL-{seed}", seed=seed, method=m, method_code=d["code"], task="ioi", model_id="gpt2",
                                          model_revision=json.dumps(minfo["hf_revision"]), candidate_set_sha256=cand["candidate_set_sha256"],
                                          split_sha256=fman["eval_sha256"][eval_name], prompt_batch_id=dps.sha256()[:12], device="cuda",
                                          budget_masks=str(spec.budget), set=set_name, variant=variant or "none", eval_set=eval_name,
                                          decisions_sealed_sha256=sealed_sha, verifying=str(m in VERIFYING).lower())
                with tracking.run(exp, f"VIL:{set_name}:{seed}:{m}", tags, pid) as r:
                    g = fdf[(fdf.method == m) & (fdf.role == "final")]
                    mets = {f"R_k{int(k)}": v for k, v in zip(g.k, g.R)}
                    mets.update({f"R_disc_audit_k{int(k)}": v for k, v in zip(g.k, g.R_disc_audit)})
                    mets.update({f"pred_R_k{int(k)}": v for k, v in zip(g.k, g.pred_R)})
                    mets["S"] = float(g.R.mean())
                    mets["n_faithful_R_le_0.10"] = int((g.R <= 0.10).sum())
                    cr = [c for c in cost_rows if c["method"] == m][0]
                    mets.update({k: v for k, v in cr.items() if isinstance(v, int | float) and k not in ("seed",)})
                    tracking.log_metrics_step(mets)
                    tracking.log_json(_strip(d), "decision.json")
                    run_ids[m] = r.info.run_id
            st = {"status": "complete", "seed": seed, "set": set_name, "variant": variant, "dir": str(udir), "decisions_sealed_sha256": sealed_sha,
                  "mlflow_runs": run_ids, "discovery_prompts_sha256": dps.sha256(), "eval_set": eval_name, "audit": ainfo,
                  "seed_wall_seconds": time.perf_counter() - t_seed, "git_sha": prov["git_sha"], "git_dirty": prov["git_dirty"],
                  "hashes": cfg["_hashes"], "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            write_json(stp, st)
            S = fdf[fdf.role == "final"].groupby("method").R.mean().round(4).to_dict()
            print(json.dumps({"seed": seed, "status": "complete", "wall": round(st["seed_wall_seconds"], 1)}), flush=True)
            if not confirmatory:
                print(json.dumps({"seed": seed, "S_" + eval_name: S}), flush=True)
            del S
        for nm, obj in {"provenance.json": prov, "spec.json": spec.__dict__}.items():
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                pth = Path(td) / nm
                pth.write_text(json.dumps(obj, indent=2, default=str))
                client.log_artifact(pid, str(pth))
        client.set_terminated(pid, status="FINISHED")
        _write_status(tag, state="complete", exit_code=0)
        return 0
    except BaseException as e:
        _write_status(tag, state="crashed", error=repr(e), traceback=traceback.format_exc())
        try:
            client.set_terminated(pid, status="FAILED")
        except Exception:
            pass
        raise
