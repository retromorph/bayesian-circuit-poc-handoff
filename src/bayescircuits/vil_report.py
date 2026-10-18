"""BCCD-VIL-1.0 report: recompute every criterion from sealed per-seed artifacts (no re-running).

Writes reports/verification_v1/{decision.json, run_manifest.json, generated_results.md, tables/*.csv, figures/*.png}.
The narrative report (final_report.md) cites these generated files.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, sha256_file
from .provenance import write_json
from .vil import METHOD_CODE, METHODS_VIL, NON_VERIFYING, VERIFYING
from .vil_run import (
    FREEZE_RECORD,
    PROTOCOL_ID,
    REPORT_DIR,
    VIL_FROZEN,
    VIL_ROOT,
    load_vil_config,
    protocol_hashes,
)

TABLES = REPORT_DIR / "tables"
FIGS = REPORT_DIR / "figures"


# ----------------------------------------------------------------------------- loading
def load_set(name: str) -> dict:
    d = VIL_ROOT / name
    faith, costs, pp, status, dec = [], [], {}, {}, {}
    for sd in sorted(d.glob("seed-*")):
        st = sd / "status.json"
        if not st.exists() or json.loads(st.read_text()).get("status") != "complete":
            continue
        s = int(sd.name.split("-")[1])
        status[s] = json.loads(st.read_text())
        f = pd.read_csv(sd / "faithfulness.csv")
        faith.append(f)
        costs.append(pd.read_csv(sd / "costs.csv"))
        z = np.load(sd / "eval_abs_discrepancies.npz")
        pp[s] = {k: z[k] for k in z.files}
        dec[s] = json.loads((sd / "decisions_sealed.json").read_text())
        if sha256_file(sd / "decisions_sealed.json") != status[s]["decisions_sealed_sha256"]:
            raise SystemExit(f"sealed decisions hash mismatch: {sd}")
    return {"faith": pd.concat(faith, ignore_index=True) if faith else pd.DataFrame(),
            "costs": pd.concat(costs, ignore_index=True) if costs else pd.DataFrame(),
            "pp": pp, "status": status, "decisions": dec}


def s_table(faith: pd.DataFrame, role: str = "final") -> pd.DataFrame:
    f = faith[faith.role == role]
    return f.groupby(["method", "seed"]).R.mean().rename("S").reset_index()


def families(eval_name: str) -> np.ndarray:
    sp = json.loads((VIL_FROZEN / "vil_splits.json").read_text())
    fam = np.array([it["template"] for it in sp["eval"][eval_name]])
    return np.unique(fam, return_inverse=True)[1]


# ----------------------------------------------------------------------------- bootstrap
def paired_bootstrap(pp: dict, seeds: list[int], fam: np.ndarray, cand_key: str, ref_key: str, ks: list[int], reps: int, rng_seed: int) -> dict:
    """Mean over resampled seeds of S_ref - S_cand; within each seed, template families then prompts resampled."""
    rng = np.random.default_rng(rng_seed)
    fam_ids = np.unique(fam)
    members = {f: np.where(fam == f)[0] for f in fam_ids}
    diffs, rel = np.empty(reps), np.empty(reps)
    for r in range(reps):
        ss = rng.choice(seeds, size=len(seeds), replace=True)
        d, sc_all, sr_all = [], [], []
        for s in ss:
            fs = rng.choice(fam_ids, size=len(fam_ids), replace=True)
            idx = np.concatenate([rng.choice(members[f], size=len(members[f]), replace=True) for f in fs])
            den = pp[s]["__denominator__"][idx].mean()
            sc = np.mean([pp[s][f"{cand_key}__{k}"][idx].mean() / den for k in ks])
            sr = np.mean([pp[s][f"{ref_key}__{k}"][idx].mean() / den for k in ks])
            d.append(sr - sc)
            sc_all.append(sc)
            sr_all.append(sr)
        diffs[r] = float(np.mean(d))
        rel[r] = float(1 - np.mean(sc_all) / np.mean(sr_all))
    return {"reps": reps, "rng_seed": rng_seed, "diff_mean": float(diffs.mean()),
            "ci90": [float(np.quantile(diffs, 0.05)), float(np.quantile(diffs, 0.95))],
            "relative_improvement_ci90": [float(np.quantile(rel, 0.05)), float(np.quantile(rel, 0.95))],
            "n_seeds": len(seeds), "n_families": int(len(fam_ids)),
            "estimator": "mean over resampled seeds of S_ref - S_cand (families, then prompts within family, resampled within seed)"}


# ----------------------------------------------------------------------------- criteria
def integrity(data: dict, n_expected: int, budget: int, n_disc: int, n_ver: int) -> dict:
    c = data["costs"]
    probs = []
    if data["faith"].empty or c.empty:
        return {"status": "INCONCLUSIVE", "observed": "no data"}
    seeds = sorted(c.seed.unique())
    if len(seeds) != n_expected:
        probs.append(f"{len(seeds)}/{n_expected} seeds complete")
    for m in METHODS_VIL:
        g = c[c.method == m]
        if len(g) != len(seeds):
            probs.append(f"{m}: {len(g)} seed rows")
        if (g.n_masks != budget).any():
            probs.append(f"{m}: masks != {budget}")
        if m in VERIFYING and ((g.n_discovery_masks != n_disc).any() or (g.n_verification_masks != n_ver).any()):
            probs.append(f"{m}: discovery/verification split != {n_disc}+{n_ver}")
        if m in NON_VERIFYING and (g.n_verification_masks != 0).any():
            probs.append(f"{m}: unexpected verification masks")
    if (c.n_blind_or_reserved_queries > 0).any():
        probs.append("reserved/blind mask queried")
    for s, st in data["status"].items():
        if not st.get("decisions_sealed_sha256"):
            probs.append(f"seed {s}: no sealed decisions")
    return {"status": "PASS" if not probs else "FAIL", "observed": "all methods x seeds: 128 logical masks (verifying 104+24), 0 reserved-mask queries, decisions sealed before evaluation" if not probs else "; ".join(probs),
            "n_seeds": len(seeds), "problems": probs}


def h_criteria(data: dict, cfg: dict, eval_name: str, adjudicate: bool = True) -> dict:
    cr = cfg["criteria"]
    ks = cfg["circuits"]["sizes"]
    reps, rseed = cr["bootstrap"]["reps"], cr["bootstrap"]["rng_seed"]
    S = s_table(data["faith"])
    seeds = sorted(S.seed.unique())
    piv = S.pivot(index="seed", columns="method", values="S")
    pooled = piv.mean()
    fam = families(eval_name)
    out: dict = {"S_pooled": pooled.to_dict(), "S_by_seed": piv.reset_index().to_dict(orient="records"), "n_seeds": len(seeds)}
    n_exp = len(cfg["seeds"]["confirmatory"]) if eval_name == "blind" else len(seeds)
    v4 = integrity(data, n_exp, cfg["budget"]["masks_per_method"], cfg["budget"]["verifying_methods"]["discovery_masks"],
                   cfg["budget"]["verifying_methods"]["verification_masks"])
    den = float(np.mean([data["pp"][s]["__denominator__"].mean() for s in seeds]))
    out["denominator_eval_mean"] = den

    def h1_for(cand: str) -> dict:
        comps = cr["H1"]["comparators"]
        best = min(comps, key=lambda c: pooled[c])
        ratio = float(pooled[cand] / pooled[best])
        res = {"V-1": {"status": "PASS" if ratio <= cr["H1"]["V-1"]["max_ratio"] else "FAIL",
                       "observed": f"pooled S({cand})={pooled[cand]:.4f} vs best non-verified {best} {pooled[best]:.4f}; ratio {ratio:.3f} (need <= {cr['H1']['V-1']['max_ratio']})",
                       "ratio": ratio, "best_comparator": best, "comparator_S": {c: float(pooled[c]) for c in comps}}}
        boots = {c: paired_bootstrap(data["pp"], seeds, fam, f"final__{cand}", f"final__{c}", ks, reps, rseed) for c in comps}
        ok2 = all(b["ci90"][0] > 0 for b in boots.values())
        res["V-2"] = {"status": "PASS" if ok2 else "FAIL",
                      "observed": "; ".join(f"vs {c}: 90% CI of S_c - S_cand [{b['ci90'][0]:+.4f}, {b['ci90'][1]:+.4f}]" for c, b in boots.items()),
                      "bootstrap": boots}
        seedmin = piv[comps].min(axis=1)
        wins = int((piv[cand] < seedmin).sum())
        need = cr["H1"]["V-3"]["min_seeds"] if eval_name == "blind" else int(np.ceil(0.75 * len(seeds)))
        res["V-3"] = {"status": "PASS" if wins >= need else "FAIL", "observed": f"{cand} beats the seed-wise best non-verified method in {wins}/{len(seeds)} seeds (need >= {need})",
                      "wins": wins}
        res["V-4"] = v4
        res["verdict"] = "INCONCLUSIVE" if v4["status"] != "PASS" else ("PASS" if all(res[k]["status"] == "PASS" for k in ("V-1", "V-2", "V-3")) else "FAIL")
        return res

    out["H1"] = h1_for(cr["H1"]["candidate"])
    out["H1_secondary_bayes_vil"] = h1_for(cr["H2"]["candidate"])
    # H2
    c2, r2 = cr["H2"]["candidate"], cr["H2"]["reference"]
    ratio2 = float(pooled[c2] / pooled[r2])
    b2 = paired_bootstrap(data["pp"], seeds, fam, f"final__{c2}", f"final__{r2}", ks, reps, rseed)
    wins2 = int((piv[c2] < piv[r2]).sum())
    need2 = cr["H2"]["H2-3"]["min_seeds"] if eval_name == "blind" else int(np.ceil(0.65 * len(seeds)))
    h2 = {"H2-1": {"status": "PASS" if ratio2 <= cr["H2"]["H2-1"]["max_ratio"] else "FAIL",
                   "observed": f"pooled S(bayes_vil)={pooled[c2]:.4f} vs S(lasso_vil)={pooled[r2]:.4f}; ratio {ratio2:.3f} (need <= {cr['H2']['H2-1']['max_ratio']})", "ratio": ratio2},
          "H2-2": {"status": "PASS" if b2["ci90"][0] > 0 else "FAIL",
                   "observed": f"90% CI of S_lasso_vil - S_bayes_vil [{b2['ci90'][0]:+.4f}, {b2['ci90'][1]:+.4f}]", "bootstrap": b2},
          "H2-3": {"status": "PASS" if wins2 >= need2 else "FAIL", "observed": f"bayes_vil < lasso_vil in {wins2}/{len(seeds)} seeds (need >= {need2})", "wins": wins2}}
    h2["verdict"] = "INCONCLUSIVE" if v4["status"] != "PASS" else ("PASS" if all(h2[k]["status"] == "PASS" for k in ("H2-1", "H2-2", "H2-3")) else "FAIL")
    out["H2"] = h2
    # secondary: active vs random verification
    b32 = paired_bootstrap(data["pp"], seeds, fam, "final__lasso_vil", "final__lasso_random_verify", ks, reps, rseed)
    out["secondary_M3_vs_M2"] = {"ratio": float(pooled["lasso_vil"] / pooled["lasso_random_verify"]), "bootstrap": b32,
                                 "wins": int((piv["lasso_vil"] < piv["lasso_random_verify"]).sum()), "n_seeds": len(seeds)}
    b42 = paired_bootstrap(data["pp"], seeds, fam, "final__bayes_vil", "final__lasso_random_verify", ks, reps, rseed)
    out["secondary_M4_vs_M2"] = {"ratio": float(pooled["bayes_vil"] / pooled["lasso_random_verify"]), "bootstrap": b42,
                                 "wins": int((piv["bayes_vil"] < piv["lasso_random_verify"]).sum())}
    b20 = paired_bootstrap(data["pp"], seeds, fam, "final__lasso_random_verify", "final__lasso_static_128", ks, reps, rseed)
    out["secondary_M2_vs_M0"] = {"ratio": float(pooled["lasso_random_verify"] / pooled["lasso_static_128"]), "bootstrap": b20,
                                 "wins": int((piv["lasso_random_verify"] < piv["lasso_static_128"]).sum())}
    b10 = paired_bootstrap(data["pp"], seeds, fam, "final__lasso_density_matched", "final__lasso_static_128", ks, reps, rseed)
    out["secondary_M1_vs_M0"] = {"ratio": float(pooled["lasso_density_matched"] / pooled["lasso_static_128"]), "bootstrap": b10,
                                 "wins": int((piv["lasso_density_matched"] < piv["lasso_static_128"]).sum())}
    # diagnostics: surrogate-only selections of verifying methods
    diag = {}
    for role in ("surrogate_at_104", "refit_128"):
        Sd = s_table(data["faith"], role)
        if not Sd.empty:
            diag[role] = Sd.groupby("method").S.mean().to_dict()
    out["diagnostic_S_pooled"] = diag
    f = data["faith"]
    fin = f[f.role == "final"]
    out["absolute_faithful_R_le_0.10"] = {m: int((fin[fin.method == m].R <= 0.10).sum()) for m in METHODS_VIL}
    out["absolute_faithful_total_circuits_per_method"] = int(len(seeds) * len(ks))
    out["min_R_any_final"] = float(fin.R.min())
    out["R_by_k"] = fin.groupby(["method", "k"]).R.mean().unstack().to_dict()
    return out


def optimism_table(faith: pd.DataFrame) -> pd.DataFrame:
    f = faith.copy()
    f["R_measured_disc"] = f.measured_u_discovery / f.denominator_disc
    rows = []
    for (m, role), g in f.groupby(["method", "role"]):
        rows.append({"method": m, "role": role, "n": len(g), "pred_R_mean": g.pred_R.mean(), "R_disc_audit_mean": g.R_disc_audit.mean(),
                     "R_blind_mean": g.R.mean(), "optimism_blind_minus_pred": (g.R - g.pred_R).mean(),
                     "optimism_disc_minus_pred": (g.R_disc_audit - g.pred_R).mean(),
                     "blind_minus_disc": (g.R - g.R_disc_audit).mean(),
                     "corr_pred_blind": g[["pred_R", "R"]].corr().iloc[0, 1] if len(g) > 2 else np.nan,
                     "corr_disc_blind": g[["R_disc_audit", "R"]].corr().iloc[0, 1] if len(g) > 2 else np.nan})
    return pd.DataFrame(rows)


def bayes_calibration(faith: pd.DataFrame) -> dict:
    """Circuit-level calibration of Bayesian predictions (secondary)."""
    out = {}
    for m in ("bayes_vil", "bayes_static_128"):
        g = faith[(faith.method == m) & faith.pred_y_lo90.notna()]
        if g.empty:
            continue
        # measured discovery u: verification outcome where available, else the post-hoc discovery-prompt audit
        u = g.measured_u_discovery.where(g.measured_u_discovery.notna(), g.u_disc_audit)
        cov = ((u >= g.pred_y_lo90) & (u <= g.pred_y_hi90))
        claims = g[g.P_faithful >= 0.95]
        out[m] = {"n_circuits": int(len(g)), "coverage90_discovery_u": float(cov.mean()),
                  "by_role": {r: float(cov[g.role == r].mean()) for r in g.role.unique()},
                  "mean_P_faithful": float(g.P_faithful.mean()), "n_claims_P_ge_0.95": int(len(claims)),
                  "n_invalid_claims_blind": int((claims.R > 0.10).sum()),
                  "frac_measured_above_hi90": float((u > g.pred_y_hi90).mean())}
    return out


# ----------------------------------------------------------------------------- figures
def _figs(data: dict, crit: dict, tag: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    names = []
    S = s_table(data["faith"])
    order = METHODS_VIL
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    vals = [S[S.method == m].S.values for m in order]
    ax[0].boxplot(vals, tick_labels=[f"{METHOD_CODE[m]}\n{m}" for m in order], showmeans=True)
    for i, v in enumerate(vals):
        ax[0].scatter(np.full(len(v), i + 1) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), v, s=10, alpha=0.6)
    ax[0].set_ylabel("S = mean_k R(C_k) (blind; lower is better)" if tag == "confirmatory" else f"S ({tag})")
    ax[0].tick_params(axis="x", labelsize=7)
    ax[0].set_title(f"Per-seed S by method ({tag}, B=128)")
    fin = data["faith"][data["faith"].role == "final"]
    for m in order:
        g = fin[fin.method == m].groupby("k").R.mean()
        ax[1].plot(g.index, g.values, marker="o", label=f"{METHOD_CODE[m]} {m}")
    ax[1].axhline(0.10, ls="--", c="k", lw=0.8)
    ax[1].set_xlabel("retained circuit size k")
    ax[1].set_ylabel("mean R(C_k)")
    ax[1].legend(fontsize=7)
    ax[1].set_title("R by circuit size (dashed: R = 0.10)")
    fig.tight_layout()
    n = f"vil_{tag}_S_by_method.png"
    fig.savefig(FIGS / n, dpi=130)
    plt.close(fig)
    names.append(n)
    # optimism: predicted vs blind R
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    f = data["faith"]
    for m in order:
        g = f[(f.method == m) & (f.role == "final")]
        ax[0].scatter(g.pred_R, g.R, s=12, label=METHOD_CODE[m])
    lim = [0, max(1.0, float(f.R.max()))]
    for a in ax[:2]:
        a.plot(lim, lim, "k--", lw=0.8)
    ax[0].set_xlabel("surrogate-predicted R at selection")
    ax[0].set_ylabel("blind R")
    ax[0].set_title("Selected circuits: prediction vs blind")
    ax[0].legend(fontsize=7)
    v = f[f.role == "verified"]
    for m in VERIFYING:
        g = v[v.method == m]
        ax[1].scatter(g.measured_u_discovery / g.denominator_disc, g.R, s=8, alpha=0.5, label=METHOD_CODE[m])
    ax[1].set_xlabel("measured discovery-prompt R (verification)")
    ax[1].set_ylabel("blind R")
    ax[1].set_title("All verified circuits: discovery vs blind")
    ax[1].legend(fontsize=7)
    for m in ("lasso_vil", "bayes_vil"):
        g = v[v.method == m].copy()
        if g.empty:
            continue
        g["Rm"] = g.measured_u_discovery / g.denominator_disc
        traj = []
        for _, gs in g.groupby(["seed", "k"]):
            gs = gs.sort_values("round")
            traj.append(gs.groupby("round").Rm.min().cummin().values)
        L = min(len(t) for t in traj)
        T = np.mean([t[:L] for t in traj], 0)
        ax[2].plot(np.arange(1, L + 1), T, marker="o", label=f"{METHOD_CODE[m]} {m}")
    ax[2].set_xlabel("verification round")
    ax[2].set_ylabel("best measured discovery R so far (mean over seeds, k)")
    ax[2].set_title("Verification trajectory")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    n = f"vil_{tag}_optimism.png"
    fig.savefig(FIGS / n, dpi=130)
    plt.close(fig)
    names.append(n)
    return names


# ----------------------------------------------------------------------------- main
def _fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def build_vil_report(args) -> int:
    cfg = load_vil_config()
    TABLES.mkdir(parents=True, exist_ok=True)
    md: list[str] = [f"# BCCD-VIL-1.0 — generated results ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})", ""]
    result: dict = {"protocol": PROTOCOL_ID, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    # ---- development (exploratory)
    dev = load_set("dev")
    dev_r4 = load_set("dev_rounds4")
    if not dev["faith"].empty:
        Sd = s_table(dev["faith"]).pivot(index="seed", columns="method", values="S")
        Sd.to_csv(TABLES / "vil_dev_S_by_seed.csv")
        md += ["## Development (exploratory; MIB dev evaluation prompts; never blind)", "", "| method | mean dev S | " + " | ".join(str(s) for s in Sd.index) + " |",
               "|---|---:|" + "---:|" * len(Sd.index)]
        for m in METHODS_VIL:
            if m in Sd:
                md.append(f"| {METHOD_CODE[m]} {m} | {Sd[m].mean():.4f} | " + " | ".join(f"{v:.3f}" for v in Sd[m]) + " |")
        sel = {"rounds8_lasso_vil": float(Sd["lasso_vil"].mean()) if "lasso_vil" in Sd else None}
        if not dev_r4["faith"].empty:
            S4 = s_table(dev_r4["faith"]).pivot(index="seed", columns="method", values="S")
            S4.to_csv(TABLES / "vil_dev_rounds4_S_by_seed.csv")
            sel["rounds4_lasso_vil"] = float(S4["lasso_vil"].mean()) if "lasso_vil" in S4 else None
            sel["rounds4_bayes_vil"] = float(S4["bayes_vil"].mean()) if "bayes_vil" in S4 else None
            sel["rounds8_bayes_vil"] = float(Sd["bayes_vil"].mean()) if "bayes_vil" in Sd else None
            if sel.get("rounds4_lasso_vil") is not None and sel.get("rounds8_lasso_vil") is not None:
                diff = sel["rounds4_lasso_vil"] - sel["rounds8_lasso_vil"]
                sel["rule_choice"] = 4 if diff < -0.005 else 8
            md += ["", f"Development schedule rule (prereg `decision.development_dependent_setting`): {json.dumps(sel)}"]
        result["development"] = {"S_mean": Sd.mean().to_dict(), "schedule_selection": sel, "n_seeds": int(len(Sd))}
        optimism_table(dev["faith"]).to_csv(TABLES / "vil_dev_optimism.csv", index=False)
        md.append("")
        if getattr(args, "dev_only", False):
            _figs(dev, {}, "dev")
            (REPORT_DIR / "generated_dev_results.md").write_text("\n".join(md) + "\n")
            write_json(REPORT_DIR / "dev_summary.json", result)
            print("\n".join(md))
            return 0
    # ---- confirmatory
    conf = load_set("confirmatory")
    freeze = json.loads(FREEZE_RECORD.read_text()) if FREEZE_RECORD.exists() else None
    if conf["faith"].empty:
        result["H1"] = {"verdict": "NOT_RUN"}
        result["H2"] = {"verdict": "NOT_RUN"}
    else:
        crit = h_criteria(conf, cfg, "blind")
        result["confirmatory"] = crit
        f = conf["faith"]
        f.to_csv(TABLES / "vil_faithfulness_all.csv", index=False)
        s_table(f).pivot(index="seed", columns="method", values="S").to_csv(TABLES / "vil_S_by_seed.csv")
        opt = optimism_table(f)
        opt.to_csv(TABLES / "vil_optimism.csv", index=False)
        conf["costs"].to_csv(TABLES / "vil_costs.csv", index=False)
        cal = bayes_calibration(f)
        result["bayes_circuit_calibration"] = cal
        cs = conf["costs"].groupby("method").mean(numeric_only=True)
        cost_cols = [c for c in ("n_masks", "n_discovery_masks", "n_verification_masks", "logical_prompt_evals", "physical_mask_evals", "cache_hits",
                                 "logical_oracle_seconds", "wall_fit", "wall_search", "wall_acquisition", "cpu_fit", "cpu_search", "cpu_acquisition",
                                 "wall_e2e_logical", "wall_e2e_measured", "wall_total_method", "wall_diagnostic", "peak_vram_mb") if c in cs]
        cs[cost_cols].to_csv(TABLES / "vil_costs_by_method.csv")
        result["costs_by_method_mean"] = cs[cost_cols].to_dict(orient="index")
        rows = []
        for h, key in (("H1", "H1"), ("H2", "H2")):
            for cid, v in crit[key].items():
                if cid != "verdict":
                    rows.append({"hypothesis": h, "criterion": cid, "status": v["status"], "observed": v["observed"]})
        pd.DataFrame(rows).to_csv(TABLES / "vil_criteria.csv", index=False)
        result["H1"] = {"verdict": crit["H1"]["verdict"], **{k: {kk: vv for kk, vv in v.items() if kk not in ("bootstrap",)} for k, v in crit["H1"].items() if k != "verdict"}}
        result["H2"] = {"verdict": crit["H2"]["verdict"], **{k: {kk: vv for kk, vv in v.items() if kk != "bootstrap"} for k, v in crit["H2"].items() if k != "verdict"}}
        h1, h2 = crit["H1"]["verdict"], crit["H2"]["verdict"]
        if "INCONCLUSIVE" in (h1,) or crit["denominator_eval_mean"] < 0.5:
            rec = "INCONCLUSIVE"
        elif h1 == "PASS" and h2 == "PASS":
            rec = "GO_BAYESIAN_VIL"
        elif h1 == "PASS":
            rec = "GO_VIL_WITHOUT_BAYES"
        else:
            rec = "NO_GO"
        result["recommendation"] = rec
        figs = _figs(conf, crit, "confirmatory")
        P = crit["S_pooled"]
        md += ["## Confirmatory (20 seeds, blind 128 MIB test prompts)", "", f"**H1: {h1} · H2: {h2} · recommendation: {rec}**", "",
               "| code | method | pooled S | R k=8 | R k=16 | R k=24 | #R<=0.10 | surrogate@104 S | refit@128 S |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for m in METHODS_VIL:
            rk = {k: crit["R_by_k"][k][m] for k in crit["R_by_k"]}
            md.append(f"| {METHOD_CODE[m]} | {m} | {P[m]:.4f} | " + " | ".join(f"{rk[k]:.4f}" for k in sorted(rk)) +
                      f" | {crit['absolute_faithful_R_le_0.10'][m]}/{crit['absolute_faithful_total_circuits_per_method']} | "
                      f"{_fmt(crit['diagnostic_S_pooled'].get('surrogate_at_104', {}).get(m))} | {_fmt(crit['diagnostic_S_pooled'].get('refit_128', {}).get(m))} |")
        md += ["", "| hypothesis | criterion | status | observed |", "|---|---|---|---|"] + [f"| {r['hypothesis']} | {r['criterion']} | {r['status']} | {r['observed']} |" for r in rows]
        md += ["", "Secondary (never override primary):", ""]
        for key in ("secondary_M3_vs_M2", "secondary_M4_vs_M2", "secondary_M2_vs_M0", "secondary_M1_vs_M0"):
            b = crit[key]
            md.append(f"- {key}: ratio {b['ratio']:.3f}; 90% CI of S_ref - S_cand [{b['bootstrap']['ci90'][0]:+.4f}, {b['bootstrap']['ci90'][1]:+.4f}]; seed wins {b['wins']}/{crit['n_seeds']}")
        hb = crit["H1_secondary_bayes_vil"]
        md.append(f"- H1 criteria applied to bayes_vil (secondary, not adjudicated): {hb['verdict']} — {hb['V-1']['observed']}; {hb['V-2']['observed']}; {hb['V-3']['observed']}")
        md += ["", "Optimism (winner's curse) by method and role:", "", opt.round(4).to_markdown(index=False), "",
               f"Bayesian circuit-level calibration: {json.dumps(cal)}", "", "Mean costs per method and seed:", "", cs[cost_cols].round(3).to_markdown(), ""]
        md += [f"![{n}](figures/{n})" for n in figs]
    # ---- bridge (secondary)
    br = load_set("bridge")
    if not br["faith"].empty:
        Sb = s_table(br["faith"]).pivot(index="seed", columns="method", values="S")
        Sb.to_csv(TABLES / "vil_bridge_S_by_seed.csv")
        rep = {}
        for s in Sb.index:
            pd_ = PROJECT_ROOT / "runs" / "gate_c" / f"seed-{s}" / "decisions_sealed.json"
            if pd_.exists():
                pil = json.loads(pd_.read_text())["lasso_pair@128"]["decision"]["selected"]
                mine = br["decisions"][s]["lasso_static_128"]["final"]
                rep[int(s)] = all(sorted(pil[k]) == sorted(mine[k]["circuit"]) for k in pil)
        result["bridge"] = {"S_pooled": Sb.mean().to_dict(), "lasso_static_128_reproduces_pilot_lasso_pair_128": rep}
        md += ["", "## Bridge (pilot seeds 31001-31005; secondary)", "", "| method | pooled S |", "|---|---:|"] + [f"| {m} | {Sb[m].mean():.4f} |" for m in METHODS_VIL if m in Sb]
        md += ["", f"lasso_static_128 reproduces pilot lasso_pair@128 circuits: {rep}"]
    # ---- manifest
    manifest = {"protocol": PROTOCOL_ID, "freeze_record": freeze, "current_hashes": protocol_hashes(),
                "vil_frozen_manifest_sha256": sha256_file(VIL_FROZEN / "vil_frozen_manifest.json") if (VIL_FROZEN / "vil_frozen_manifest.json").exists() else None,
                "mlflow": {name: {str(s): st["mlflow_runs"] for s, st in d["status"].items()} for name, d in (("confirmatory", conf), ("bridge", br), ("dev", dev))},
                "seed_status": {name: {str(s): {k: st.get(k) for k in ("decisions_sealed_sha256", "git_sha", "git_dirty", "completed_utc", "seed_wall_seconds")}
                                       for s, st in d["status"].items()} for name, d in (("confirmatory", conf), ("bridge", br), ("dev", dev))}}
    write_json(REPORT_DIR / "run_manifest.json", manifest)
    write_json(REPORT_DIR / "decision.json", result)
    (REPORT_DIR / "generated_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:80]))
    return 0
