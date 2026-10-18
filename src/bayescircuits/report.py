"""Final report builder.

Recomputes every gate criterion from saved per-unit artifacts (no experiment is re-run),
cross-checks against the gate summaries written at run time, and writes
reports/{decision.json, run_manifest.json, final_report.md, tables/*.csv, figures/*.png}.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, sha256_file
from .criteria import gate_a_criteria, gate_verdict, overall_verdict
from .discovery import METHODS
from .provenance import provenance, write_json

REP = PROJECT_ROOT / "reports"
RUNS = PROJECT_ROOT / "runs"
THRESH = {
    "A-1": ">=16/20 A1 worlds: top-2 predicted nonadditivity pairs = planted pairs (B=128)",
    "A-2": ">=16/20 A1 worlds recover >=3/4 exact size-3 minimal circuits; 0 invalid P>=0.95 claims",
    "A-3": "A0 Bayes MAE <= 1.10 x first-order Lasso (zero-error fallback: normalised MAE <= 0.01)",
    "A-4": "pooled A0-A2 90% predictive coverage in [0.85, 0.95]",
    "A-5": "A3 coverage in [0.85,0.95] OR misfit flags >=16/20 with <=12/60 A0-A2 false positives",
    "B-1": ">=8/10 valid models: >=3/4 planted size-2 circuits + a blind-faithful proposal by B=128",
    "B-2": "median masks-to-recovery <= 0.75 x best non-Bayesian median (censored at 160)",
    "B-3": "median masks-to-recovery <= 0.80 x bayes_static median",
    "B-4": "0 invalid high-confidence (P>=0.95) circuit claims",
    "C-1": "S_Bayes(128) <= 0.85 x best non-Bayes S(128) pooled AND >=15% gain in >=4/5 seeds",
    "C-2": "S_Bayes(128) <= best non-Bayes S(256)",
    "C-3": "90% predictive coverage of blind mean intervention effects in [0.85, 0.95]",
    "C-4": "paired hierarchical bootstrap 90% CI lower bound > 0 vs dev-preselected baseline",
    "C-5": "end-to-end elapsed at B=128 <= 2 x strongest non-Bayesian baseline",
}


def _read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _mlflow_ids(gate_dir: Path, pattern: str) -> dict:
    out = {}
    for f in sorted(gate_dir.glob(pattern)):
        try:
            d = json.loads(f.read_text())
            out[f.parent.name] = d.get("mlflow_runs", d) if isinstance(d, dict) else d
        except Exception:
            pass
    return out


def _fig_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


COLORS = {"lasso_1st": "#9e9e9e", "lasso_pair": "#1f77b4", "gbdt": "#2ca02c", "bayes_static": "#ff7f0e",
          "bayes_uncertainty": "#9467bd", "bayes_decision": "#d62728"}


def build_gate_a(cfg: dict, tables: Path, figs: Path) -> dict:
    p = RUNS / "gate_a" / "rows_all.csv"
    if not p.exists():
        return {"verdict": "NOT_RUN", "criteria": {c: {"status": "NOT_RUN", "observed": "no confirmatory artifacts"} for c in ["A-1", "A-2", "A-3", "A-4", "A-5"]}}
    rows = pd.read_csv(p).assign(split="confirmatory")
    th = json.loads((PROJECT_ROOT / "configs" / "gate_a_misfit_threshold.json").read_text())
    crit = gate_a_criteria(rows, cfg, th["threshold"])
    rows["misfit_flag"] = np.where(rows.misfit_stat.notna(), rows.misfit_stat > th["threshold"], np.nan)
    cols = ["regime", "world_seed", "method", "budget", "top2_pairs_correct", "true_pair_ranks", "circuits_recovered", "n_proposals", "proposal_size",
            "n_high_conf_claims", "invalid_high_conf_claims", "test_mae", "test_rmse", "test_mae_normalized", "coverage_90", "interval_width_90", "crps",
            "misfit_stat", "misfit_flag", "discovery_masks", "discovery_context_evals", "cost_wall_e2e", "cost_cpu_e2e"]
    rows[[c for c in cols if c in rows]].to_csv(tables / "gate_a_worlds.csv", index=False)
    b = rows[rows.budget == 128]
    agg = b.groupby(["regime", "method"]).agg(
        n=("world_seed", "nunique"), mae=("test_mae", "mean"), nmae=("test_mae_normalized", "mean"), coverage=("coverage_90", "mean"),
        width=("interval_width_90", "mean"), top2=("top2_pairs_correct", "mean"), rec3=("circuits_recovered", lambda x: (x >= 3).sum()),
        invalid=("invalid_high_conf_claims", lambda x: x.sum(min_count=1)), misfit_flags=("misfit_flag", lambda x: x.astype(float).sum(min_count=1)),
        cpu=("cost_cpu_e2e", "mean")).reset_index()
    agg.to_csv(tables / "gate_a_summary_B128.csv", index=False)
    # recovery curves by budget (A1)
    a1 = rows[rows.regime == "A1"].groupby(["method", "budget"]).agg(top2=("top2_pairs_correct", "sum"),
                                                                       rec3=("circuits_recovered", lambda x: int((x >= 3).sum()))).reset_index()
    a1.to_csv(tables / "gate_a_A1_by_budget.csv", index=False)
    plt = _fig_setup()
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for m in METHODS:
        g = a1[a1.method == m]
        ax[0].plot(g.budget, g.rec3, "-o", color=COLORS[m], label=m, ms=3)
        ax[1].plot(g.budget, g.top2, "-o", color=COLORS[m], ms=3)
    ax[0].axhline(16, ls="--", c="k", lw=0.7)
    ax[1].axhline(16, ls="--", c="k", lw=0.7)
    ax[0].set(title="A1: worlds recovering >=3/4 circuits", xlabel="masks", ylabel="# of 20 worlds")
    ax[1].set(title="A1: top-2 pairs correct", xlabel="masks")
    bb = agg[agg.method.str.startswith("bayes")]
    for i, m in enumerate(["bayes_static", "bayes_uncertainty", "bayes_decision"]):
        g = bb[bb.method == m]
        ax[2].bar(np.arange(len(g)) + 0.27 * (i - 1), g.coverage, 0.27, color=COLORS[m], label=m)
    ax[2].set_xticks(range(4), ["A0", "A1", "A2", "A3"])
    ax[2].axhspan(0.85, 0.95, color="k", alpha=0.08)
    ax[2].set(title="90% interval coverage (B=128)", ylim=(0.5, 1.02))
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figs / "gate_a_overview.png")
    plt.close(fig)
    saved = _read_json(RUNS / "gate_a" / "summary.json") or {}
    consistent = {k: (saved.get("criteria", {}).get(k, {}).get("status") == v["status"]) for k, v in crit.items()}
    return {"criteria": crit, "verdict": gate_verdict(crit), "misfit_threshold": th, "summary_table": agg.to_dict(orient="records"),
            "consistent_with_runtime_summary": consistent, "parent_run_id": saved.get("parent_run_id"),
            "n_worlds": int(rows.groupby(["regime", "world_seed"]).ngroups)}


def build_gate_b(cfg: dict, tables: Path, figs: Path) -> dict:
    from .gate_b import gate_b_criteria

    p = RUNS / "gate_b" / "rows_all.csv"
    log = _read_json(RUNS / "gate_b" / "construction_log.json")
    if not p.exists() or log is None:
        return {"verdict": "NOT_RUN", "criteria": {c: {"status": "NOT_RUN", "observed": "no confirmatory artifacts"} for c in ["B-1", "B-2", "B-3", "B-4"]}}
    rows = pd.read_csv(p)
    crit = gate_b_criteria(rows, cfg, log["n_valid"])
    rows.to_csv(tables / "gate_b_instances.csv", index=False)
    val = pd.DataFrame([{"seed": a["seed"], "valid": a["valid"], "clean_acc": a["clean_accuracy"],
                         "min_singleton_acc": min(s["accuracy"] for s in a["singleton"].values()),
                         "pairA_acc": a["pair"]["A"]["accuracy"], "pairB_acc": a["pair"]["B"]["accuracy"],
                         "max_singleton_pair_ratio": a["max_singleton_to_pair_ratio"], "true_circuit_R_max": max(a["true_circuit_R"]),
                         "min_other_size2_R": a.get("max_R_other_size2"), "n_faithful_size_le2": len(a["faithful_size_le2"]),
                         "model_sha256": a["model_sha256"]} for a in log["attempts"]])
    val.to_csv(tables / "gate_b_construction.csv", index=False)
    ev = rows.groupby(["method", "model_seed"]).agg(event_time=("event_time", "first"), censored=("censored", "first")).reset_index()
    px = {}
    for d in sorted((RUNS / "gate_b").glob("model-*/proxyspex.json")):
        px[d.parent.name] = json.loads(d.read_text())
    plt = _fig_setup()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for i, m in enumerate(METHODS):
        g = ev[ev.method == m]
        ax[0].scatter(np.full(len(g), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(g)), g.event_time, color=COLORS[m], s=14)
        ax[0].plot([i - 0.3, i + 0.3], [g.event_time.median()] * 2, c="k")
    ax[0].set_xticks(range(len(METHODS)), METHODS, rotation=30, fontsize=7)
    ax[0].set(ylabel="masks to recovery (160 = censored)", title="Gate B event times")
    rb = rows.groupby(["method", "budget"]).success.sum().reset_index()
    for m in METHODS:
        g = rb[rb.method == m]
        ax[1].plot(g.budget, g.success, "-o", color=COLORS[m], label=m, ms=3)
    ax[1].axhline(8, ls="--", c="k", lw=0.7)
    ax[1].set(xlabel="masks", ylabel="# models recovered (of 10)", title="Gate B recovery by budget")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figs / "gate_b_overview.png")
    plt.close(fig)
    saved = _read_json(RUNS / "gate_b" / "summary.json") or {}
    consistent = {k: (saved.get("criteria", {}).get(k, {}).get("status") == v["status"]) for k, v in crit.items()}
    return {"criteria": crit, "verdict": gate_verdict(crit), "construction": {k: log[k] for k in ("n_attempts", "n_valid", "valid_seeds", "status")},
            "event_times": ev.to_dict(orient="records"), "proxyspex": {k: {"top2_pairs_correct": v.get("top2_pairs_correct"), "oracle_calls": v.get("oracle_calls")} for k, v in px.items()},
            "consistent_with_runtime_summary": consistent, "parent_run_id": saved.get("parent_run_id")}


def build_gate_c(cfg: dict, tables: Path, figs: Path) -> dict:
    from .criteria_c import gate_c_criteria, load_c, s_table

    out = RUNS / "gate_c"
    manifest = _read_json(out / "frozen" / "frozen_manifest.json")
    seeds = cfg["seeds"]["gate_c_discovery"]
    if manifest is None or not any((out / f"seed-{s}" / "status.json").exists() for s in seeds):
        return {"verdict": "NOT_RUN", "criteria": {c: {"status": "NOT_RUN", "observed": "no confirmatory artifacts"} for c in ["C-1", "C-2", "C-3", "C-4", "C-5"]},
                "manifest": manifest}
    res = gate_c_criteria(out, cfg, manifest, seeds)
    data = load_c(out, seeds)
    f = data["faith"].assign(candidate_set_sha256=manifest["candidate_set_sha256"])
    S = s_table(data["faith"])
    f = f.merge(S, on=["method", "seed", "budget"])
    f.to_csv(tables / "gate_c_faithfulness.csv", index=False)
    pr = data["pred"]
    pr = pr.assign(cov=(pr.truth >= pr.lo) & (pr.truth <= pr.hi), width=pr.hi - pr.lo, abserr=(pr.pred - pr.truth).abs())
    pt = pr.groupby(["seed", "method", "budget", "n_ablated"]).agg(mae=("abserr", "mean"), coverage_90=("cov", "mean"), width_90=("width", "mean"),
                                                                  crps=("crps", "mean")).reset_index()
    pt.loc[~pt.method.str.startswith("bayes"), ["coverage_90", "width_90", "crps"]] = np.nan
    pt.to_csv(tables / "gate_c_prediction.csv", index=False)
    data["costs"].to_csv(tables / "costs_gate_c.csv", index=False)
    plt = _fig_setup()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    pooled = S.groupby(["method", "budget"]).S.agg(["mean", "std"]).reset_index()
    for m in METHODS:
        g = pooled[pooled.method == m]
        ax[0].errorbar(g.budget, g["mean"], yerr=g["std"], fmt="-o", color=COLORS[m], label=m, ms=3, capsize=2)
    ax[0].set(xscale="log", xlabel="mask budget B", ylabel="S (mean blind R over k)", title="Gate C: S vs budget")
    ax[0].set_xticks([64, 128, 256], ["64", "128", "256"])
    ax[0].legend(fontsize=6)
    fk = data["faith"][data["faith"].budget == 128].groupby(["method", "k"]).R.mean().reset_index()
    for m in METHODS:
        g = fk[fk.method == m]
        ax[1].plot(g.k, g.R, "-o", color=COLORS[m], ms=3)
    ax[1].axhline(0.1, ls="--", c="k", lw=0.7)
    ax[1].set(xlabel="retained heads k", ylabel="blind R(C)", title="Faithfulness-size (B=128)")
    pb = pr[(pr.method == "bayes_decision") & (pr.budget == 128)]
    ax[2].errorbar(pb.truth, pb.pred, yerr=[pb.pred - pb.lo, pb.hi - pb.pred], fmt=".", ms=2, alpha=0.4, color=COLORS["bayes_decision"])
    lim = [min(pb.truth.min(), pb.pred.min()), max(pb.truth.max(), pb.pred.max())]
    ax[2].plot(lim, lim, "k--", lw=0.7)
    ax[2].set(xlabel="blind mean effect (truth)", ylabel="predicted (90% PI)", title="bayes_decision B=128 predictions")
    fig.tight_layout()
    fig.savefig(figs / "gate_c_overview.png")
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    cd = pt[(pt.method.str.startswith("bayes")) & (pt.budget == 128)].groupby(["method", "n_ablated"]).agg(c=("coverage_90", "mean"), w=("width_90", "mean")).reset_index()
    for m in ["bayes_static", "bayes_uncertainty", "bayes_decision"]:
        g = cd[cd.method == m]
        ax[0].plot(g.n_ablated, g.c, "-o", color=COLORS[m], label=m, ms=3)
        ax[1].plot(g.n_ablated, g.w, "-o", color=COLORS[m], ms=3)
    ax[0].axhspan(0.85, 0.95, color="k", alpha=0.08)
    ax[0].set(xlabel="# ablated heads", ylabel="coverage", title="C: coverage by density (B=128)")
    ax[1].set(xlabel="# ablated heads", ylabel="mean 90% width", title="C: interval width by density")
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figs / "gate_c_calibration.png")
    plt.close(fig)
    c = data["costs"][data["costs"].budget == 128].groupby("method").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    bottom = np.zeros(len(c))
    for col, lab in (("wall_oracle", "GPU oracle"), ("wall_fit_cum", "CPU fitting"), ("wall_acquisition_cum", "CPU acquisition")):
        ax.bar(c.index, c[col], bottom=bottom, label=lab)
        bottom += c[col].to_numpy()
    ax.set_xticks(range(len(c)), c.index, rotation=30, fontsize=7)
    ax.set(ylabel="seconds (mean over seeds)", title="Gate C cost breakdown to B=128 (excl. shared screening)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "gate_c_costs.png")
    plt.close(fig)
    saved = _read_json(out / "summary.json") or {}
    consistent = {k: (saved.get("criteria", {}).get(k, {}).get("status") == v["status"]) for k, v in res["criteria"].items()}
    res.update({"manifest": manifest, "consistent_with_runtime_summary": consistent, "parent_run_id": saved.get("parent_run_id"),
                "mlflow_runs": {s: (_read_json(out / f"seed-{s}" / "status.json") or {}).get("mlflow_runs") for s in seeds}})
    return res


def build_report(cfg: dict) -> int:
    tables, figs = REP / "tables", REP / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    A = build_gate_a(cfg, tables, figs)
    Bg = build_gate_b(cfg, tables, figs)
    C = build_gate_c(cfg, tables, figs)
    gates = {"A": A["verdict"], "B": Bg["verdict"], "C": C["verdict"]}
    overall = overall_verdict(gates)
    status_files = {p.name: _read_json(p) for p in sorted((PROJECT_ROOT / "logs").glob("*.status.json"))}
    rid = {"gate_a_parent": A.get("parent_run_id"), "gate_b_parent": Bg.get("parent_run_id"), "gate_c_parent": C.get("parent_run_id"),
           "gate_c_prepare_parent": (status_files.get("prepare-c.status.json") or {}).get("parent_run_id"),
           "gate_a_dev_parent": (status_files.get("prepare-a.status.json") or {}).get("parent_run_id")}
    gate_rows = []
    for gname, G in (("A", A), ("B", Bg), ("C", C)):
        for cid, v in G["criteria"].items():
            gate_rows.append({"criterion": cid, "threshold": THRESH[cid], "observed": v.get("observed"), "status": v["status"],
                              "run_ids": rid.get(f"gate_{gname.lower()}_parent")})
    pd.DataFrame(gate_rows).to_csv(tables / "gates.csv", index=False)
    decision = {
        "protocol": cfg["protocol"]["id"], "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_verdict": overall, "overall_rule": "GO iff Gate A, B and C all PASS",
        "gates": {g: {"verdict": v, "criteria": {r["criterion"]: {k: r[k] for k in ("status", "threshold", "observed")} for r in gate_rows if r["criterion"].startswith(g)}}
                  for g, v in gates.items()},
        "candidate_method": "bayes_decision", "mlflow_parent_runs": rid,
        "recomputed_from_artifacts": True,
        "consistency_with_runtime_summaries": {"A": A.get("consistent_with_runtime_summary"), "B": Bg.get("consistent_with_runtime_summary"),
                                               "C": C.get("consistent_with_runtime_summary")},
    }
    write_json(REP / "decision.json", decision)
    prov = provenance(cfg)
    manifest = {
        "provenance": prov, "mlflow_tracking": "http://127.0.0.1:5001 (remote loopback; ssh -N -L 127.0.0.1:5001:127.0.0.1:5001 <host>)",
        "mlflow_parent_runs": rid, "status_files": status_files,
        "gate_a": {"misfit_threshold": A.get("misfit_threshold"), "world_mlflow_runs": _mlflow_ids(RUNS / "gate_a" / "confirmatory", "*/mlflow_runs.json"),
                   "threshold_file_sha256": sha256_file(PROJECT_ROOT / "configs" / "gate_a_misfit_threshold.json") if (PROJECT_ROOT / "configs" / "gate_a_misfit_threshold.json").exists() else None},
        "gate_b": {"construction": Bg.get("construction"), "model_status": {p.parent.name: _read_json(p) for p in sorted((RUNS / "gate_b").glob("model-*/status.json"))}},
        "gate_c": {"frozen": C.get("manifest"), "seed_status": {p.parent.name: _read_json(p) for p in sorted((RUNS / "gate_c").glob("seed-*/status.json"))}},
        "external": {"mib_ioi_dataset": "mib-bench/ioi@e5f3468f3af4c0883be35cd3bced8c711c95d286", "shapiq": prov.get("shapiq_version"),
                     "transformer_lens": prov.get("transformer_lens_version"), "third_party_git_repos": "none vendored or cloned"},
        "artifact_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in sorted(tables.glob("*.csv"))},
    }
    write_json(REP / "run_manifest.json", manifest)
    full = {"A": A, "B": Bg, "C": {k: v for k, v in C.items() if k != "manifest"}}
    write_json(REP / "gate_details.json", full)
    from .report_md import render_report

    (REP / "final_report.md").write_text(render_report(cfg, decision, A, Bg, C, manifest))
    print(json.dumps({"overall": overall, "gates": gates}, indent=1))
    return 0
