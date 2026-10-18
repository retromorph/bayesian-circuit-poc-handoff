"""Gate C criteria (C-1..C-5) recomputed from sealed per-seed artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .discovery import NON_BAYES

CAND = "bayes_decision"


def load_c(out: Path, seeds: list[int]) -> dict:
    f, c, p, pp = [], [], [], {}
    for s in seeds:
        d = out / f"seed-{s}"
        if not (d / "status.json").exists():
            continue
        f.append(pd.read_csv(d / "faithfulness.csv"))
        c.append(pd.read_csv(d / "costs.csv"))
        p.append(pd.read_parquet(d / "blind_predictions.parquet"))
        z = np.load(d / "blind_abs_discrepancies.npz")
        pp[s] = {k: z[k] for k in z.files}
    return {"faith": pd.concat(f) if f else pd.DataFrame(), "costs": pd.concat(c) if c else pd.DataFrame(),
            "pred": pd.concat(p) if p else pd.DataFrame(), "pp": pp}


def s_table(faith: pd.DataFrame) -> pd.DataFrame:
    return faith.groupby(["method", "seed", "budget"]).R.mean().rename("S").reset_index()


def hierarchical_bootstrap(pp: dict, seeds: list[int], families: np.ndarray, ref: str, B: int, ks: list[int], reps: int, rng_seed: int) -> dict:
    rng = np.random.default_rng(rng_seed)
    fam_ids = np.unique(families)
    members = {f: np.where(families == f)[0] for f in fam_ids}
    diffs, rel = np.empty(reps), np.empty(reps)
    for r in range(reps):
        ss = rng.choice(seeds, size=len(seeds), replace=True)
        d_seed, sb_all, sr_all = [], [], []
        for s in ss:
            fams = rng.choice(fam_ids, size=len(fam_ids), replace=True)
            idx = np.concatenate([rng.choice(members[f], size=len(members[f]), replace=True) for f in fams])
            den = pp[s]["__denominator__"][idx].mean()
            sb = np.mean([pp[s][f"{CAND}__{B}__{k}"][idx].mean() / den for k in ks])
            sr = np.mean([pp[s][f"{ref}__{B}__{k}"][idx].mean() / den for k in ks])
            d_seed.append(sr - sb)
            sb_all.append(sb)
            sr_all.append(sr)
        diffs[r] = float(np.mean(d_seed))
        rel[r] = float(1 - np.mean(sb_all) / np.mean(sr_all))
    return {"reps": reps, "estimator": "mean over resampled seeds of S_ref - S_bayes (prompt families then prompts resampled within each seed)",
            "diff_mean": float(diffs.mean()), "ci90": [float(np.quantile(diffs, 0.05)), float(np.quantile(diffs, 0.95))],
            "relative_improvement_ci90": [float(np.quantile(rel, 0.05)), float(np.quantile(rel, 0.95))],
            "n_families": int(len(fam_ids)), "n_seeds": len(seeds)}


def gate_c_criteria(out: Path, cfg: dict, manifest: dict, seeds: list[int]) -> dict:
    from .criteria import gate_verdict

    gc = cfg["gate_c"]
    cr = gc["criteria"]
    B = gc["criterion_checkpoint_masks"]
    lo, hi = cfg["shared"]["accepted_coverage_interval"]
    data = load_c(out, seeds)
    res: dict = {"criteria": {}}
    crit = res["criteria"]
    if not manifest["validity"]["valid"]:
        for c in ["C-1", "C-2", "C-3", "C-4", "C-5"]:
            crit[c] = {"status": "INCONCLUSIVE", "observed": f"candidate pool validity failed: {manifest['validity']}"}
        res["verdict"] = "INCONCLUSIVE"
        return res
    if data["faith"].empty or data["faith"].seed.nunique() < gc["discovery"]["n_seeds"]:
        for c in ["C-1", "C-2", "C-3", "C-4", "C-5"]:
            crit[c] = {"status": "NOT_RUN", "observed": "incomplete seeds"}
        res["verdict"] = "NOT_RUN"
        return res
    S = s_table(data["faith"])
    pooled = S.groupby(["method", "budget"]).S.mean().unstack()
    res["S_pooled"] = pooled.to_dict()
    res["S_by_seed"] = S.pivot_table(index=["method", "budget"], columns="seed", values="S").reset_index().to_dict(orient="records")
    sb = float(pooled.loc[CAND, B])
    nb128 = {m: float(pooled.loc[m, B]) for m in NON_BAYES}
    best = min(nb128, key=nb128.get)
    best_s = nb128[best]
    # C-1
    if best_s <= 1e-9:
        crit["C-1"] = {"status": "INCONCLUSIVE", "observed": "best non-Bayesian S ~ 0"}
    else:
        ratio = sb / best_s
        seedwise = []
        for s in sorted(S.seed.unique()):
            sbs = float(S[(S.method == CAND) & (S.seed == s) & (S.budget == B)].S.iloc[0])
            bns = min(float(S[(S.method == m) & (S.seed == s) & (S.budget == B)].S.iloc[0]) for m in NON_BAYES)
            seedwise.append({"seed": int(s), "S_bayes": sbs, "S_best_nonbayes_seed": bns, "relative_gain": 1 - sbs / bns})
        nwin = sum(x["relative_gain"] >= cr["C-1"]["min_seedwise_relative_gain_fraction"] for x in seedwise)
        ok = ratio <= cr["C-1"]["max_score_ratio_to_best_non_bayesian_at_128"] and nwin >= cr["C-1"]["min_successful_seedwise_comparisons"]
        crit["C-1"] = {"status": "PASS" if ok else "FAIL",
                       "observed": f"S_bayes(128)={sb:.4f}, best non-Bayes {best} S(128)={best_s:.4f}, ratio={ratio:.3f} (need <=0.85); seeds with >=15% gain: {nwin}/5",
                       "ratio": ratio, "seedwise": seedwise, "best_non_bayes": best, "non_bayes_S128": nb128}
    # C-2
    nb256 = {m: float(pooled.loc[m, max(gc["discovery"]["budget_mask_checkpoints"])]) for m in NON_BAYES}
    best256 = min(nb256, key=nb256.get)
    crit["C-2"] = {"status": "PASS" if sb <= nb256[best256] else "FAIL",
                   "observed": f"S_bayes(128)={sb:.4f} vs best non-Bayes S(256)={nb256[best256]:.4f} ({best256})",
                   "non_bayes_S256": nb256}
    # C-3
    pr = data["pred"]
    pb = pr[(pr.method == CAND) & (pr.budget == B)]
    cov = float(np.mean((pb.truth >= pb.lo) & (pb.truth <= pb.hi)))
    by_d = pb.assign(c=(pb.truth >= pb.lo) & (pb.truth <= pb.hi), w=pb.hi - pb.lo).groupby("n_ablated").agg(coverage=("c", "mean"), width=("w", "mean"))
    cov0 = float(np.mean((pb.truth >= pb.lo_noprompt) & (pb.truth <= pb.hi_noprompt)))
    crit["C-3"] = {"status": "PASS" if lo <= cov <= hi else "FAIL",
                   "observed": f"coverage {cov:.3f} (mean width {float((pb.hi - pb.lo).mean()):.3f}; CRPS {float(pb.crps.mean()):.4f}); "
                               f"by #ablated: " + ", ".join(f"{int(k)}:{v.coverage:.2f}/w={v.width:.2f}" for k, v in by_d.iterrows()),
                   "coverage": cov, "coverage_without_prompt_term": cov0, "by_density": by_d.reset_index().to_dict(orient="records"),
                   "truth_range": [float(pb.truth.min()), float(pb.truth.max())]}
    # C-4
    dev_ref = json.loads((Path(__file__).resolve().parents[2] / "runs" / "gate_c" / "frozen" / "dev_reference.json").read_text())
    ref = dev_ref["reference_baseline"]
    splits = json.loads((Path(__file__).resolve().parents[2] / "runs" / "gate_c" / "frozen" / "splits.json").read_text())
    fam = np.array([it["template"] for it in splits["blind"]])
    _, fam_idx = np.unique(fam, return_inverse=True)
    boot = hierarchical_bootstrap(data["pp"], sorted(data["pp"]), fam_idx, ref, B, gc["discovery"]["retained_circuit_sizes"],
                                  cfg["_impl"]["gate_c"]["bootstrap_reps"], 4242)
    crit["C-4"] = {"status": "PASS" if boot["ci90"][0] > 0 else "FAIL",
                   "observed": f"reference={ref} (dev-preselected); paired 90% CI of S_ref-S_bayes = [{boot['ci90'][0]:.4f}, {boot['ci90'][1]:.4f}]",
                   "bootstrap": boot, "reference": ref}
    # C-5
    cs = data["costs"]
    c128 = cs[cs.budget == B].groupby("method").mean(numeric_only=True)
    scr = float(manifest["screening_cost"]["wall_seconds"])
    e_b = float(c128.loc[CAND, "wall_e2e"]) + scr
    e_n = float(c128.loc[best, "wall_e2e"]) + scr
    crit["C-5"] = {"status": "PASS" if e_b <= cr["C-5"]["end_to_end_elapsed_ratio_max_vs_best_non_bayesian_at_128"] * e_n else "FAIL",
                   "observed": f"end-to-end wall at B=128 (incl. {scr:.1f}s screening): bayes_decision {e_b:.1f}s vs {best} {e_n:.1f}s (ratio {e_b / e_n:.2f})",
                   "ratio": e_b / e_n,
                   "components": {m: {"gpu_oracle_wall": float(c128.loc[m, "wall_oracle"]), "cpu_fit_wall": float(c128.loc[m, "wall_fit_cum"]),
                                      "cpu_acquisition_wall": float(c128.loc[m, "wall_acquisition_cum"]), "e2e_wall_excl_screening": float(c128.loc[m, "wall_e2e"]),
                                      "cpu_e2e": float(c128.loc[m, "cpu_e2e"]), "extra_calibration_model_wall": float(c128.loc[m, "wall_extra_models_cum"])}
                                  for m in c128.index}}
    res["verdict"] = gate_verdict(crit)
    res["absolute_faithful_attainment"] = data["faith"].assign(f=data["faith"].R <= 0.10).groupby(["method", "budget", "k"]).f.mean().reset_index().to_dict(orient="records")
    return res
