"""Render reports/final_report.md from recomputed gate results (+ hand-written narrative)."""

from __future__ import annotations

import json

import pandas as pd

from .config import PROJECT_ROOT
from .discovery import METHODS

NARRATIVE = PROJECT_ROOT / "reports" / "narrative.md"


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return "—" if x != x else f"{x:.{nd}f}"
    return str(x)


def _fmt_int(x):
    try:
        if x is None or x != x:
            return "—"
        return str(int(x))
    except (TypeError, ValueError):
        return "—"


def _crit_table(decision: dict, rid: dict) -> str:
    from .report import THRESH

    lines = ["| Criterion | Frozen threshold | Observed | Status | Supporting MLflow run IDs |", "|---|---|---|---|---|"]
    for g, gv in decision["gates"].items():
        for c, v in gv["criteria"].items():
            lines.append(f"| {c} | {THRESH[c]} | {v['observed']} | **{v['status']}** | parent `{rid.get(f'gate_{g.lower()}_parent')}` |")
    return "\n".join(lines)


def render_report(cfg: dict, decision: dict, A: dict, B: dict, C: dict, manifest: dict) -> str:
    prov = manifest["provenance"]
    rid = decision["mlflow_parent_runs"]
    out = []
    out.append("# Bayesian Causal Circuit Discovery — Go/No-Go Pilot Report\n")
    out.append(f"**Date (UTC):** {decision['generated_utc']}  ")
    out.append(f"**Protocol:** {cfg['protocol']['id']} (protocol doc SHA-256 `{prov.get('protocol_doc_sha256')}`; config SHA-256 `{prov.get('config_sha256')}`)  ")
    out.append(f"**Git revision / uv lock:** `{prov.get('git_sha')}` (dirty={prov.get('git_dirty')}) / `{prov.get('uv_lock_sha256')}`  ")
    out.append(f"**Host / GPU / driver / torch:** {prov.get('platform')} / {prov.get('nvidia_smi')} / torch {prov.get('torch_version')} (CUDA build {prov.get('torch_cuda_build')})  ")
    out.append(f"**Overall verdict:** **{decision['overall_verdict']}** (Gate A {decision['gates']['A']['verdict']}, Gate B {decision['gates']['B']['verdict']}, Gate C {decision['gates']['C']['verdict']})  ")
    out.append(f"**MLflow tracking:** remote `http://127.0.0.1:5001` via SSH tunnel; parent runs {json.dumps(rid)}\n")
    if NARRATIVE.exists():
        out.append(NARRATIVE.read_text())
    # ---- Gate A tables
    out.append("\n## Appendix A — Gate A numerical results (B=128, 20 confirmatory worlds per regime)\n")
    if "summary_table" in A:
        df = pd.DataFrame(A["summary_table"])
        out.append("| Regime | Method | MAE | norm. MAE | 90% coverage | mean width | top-2 pairs (frac) | worlds >=3/4 circuits | invalid claims | misfit flags | CPU s (e2e) |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in df.iterrows():
            out.append(f"| {r.regime} | {r.method} | {_fmt(r.mae)} | {_fmt(r.nmae)} | {_fmt(r.coverage, 3)} | {_fmt(r.width, 3)} | {_fmt(r.top2, 2)} | "
                       f"{int(r.rec3)} | {_fmt_int(r.invalid)} | {_fmt_int(r.misfit_flags)} | {_fmt(r.cpu, 2)} |")
        out.append(f"\nMisfit threshold (frozen from development worlds): `{json.dumps({k: A['misfit_threshold'][k] for k in ('threshold', 'rule', 'dev_control_max', 'dev_A3_min')})}`\n")
        out.append("![Gate A](figures/gate_a_overview.png)\n")
    # ---- Gate B
    out.append("\n## Appendix B — Gate B numerical results\n")
    if "event_times" in B:
        ev = pd.DataFrame(B["event_times"])
        piv = ev.pivot(index="model_seed", columns="method", values="event_time")[METHODS]
        out.append("Event time = first checkpoint with >=3/4 planted size-2 circuits among the (<=4) proposals and >=1 blind-faithful proposal; 160 = censored.\n")
        out.append("| Model seed | " + " | ".join(METHODS) + " |")
        out.append("|---|" + "---:|" * len(METHODS))
        for s, r in piv.iterrows():
            out.append(f"| {s} | " + " | ".join(str(int(v)) for v in r.values) + " |")
        med = ev.groupby("method").event_time.median()
        nc = ev.groupby("method").censored.sum()
        out.append("| **median** | " + " | ".join(f"**{med[m]:g}**" for m in METHODS) + " |")
        out.append("| censored | " + " | ".join(str(int(nc[m])) for m in METHODS) + " |")
        out.append(f"\nConstruction: {json.dumps(B['construction'])}\n")
        out.append(f"ProxySPEX (official shapiq, FBII, own 128 coalitions) top-2 pair retrieval per model: {json.dumps(B.get('proxyspex'))}\n")
        out.append("![Gate B](figures/gate_b_overview.png)\n")
    # ---- Gate C
    out.append("\n## Appendix C — Gate C numerical results\n")
    if "S_pooled" in C:
        sp = pd.DataFrame(C["S_pooled"])
        comp = C["criteria"].get("C-5", {}).get("components", {})
        out.append("| Method | S(64) | S(128) | S(256) | GPU oracle s @128 | CPU fit s @128 | CPU acq s @128 | End-to-end s @128 (excl. screening) |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for m in METHODS:
            row = sp.loc[m] if m in sp.index else None
            cm = comp.get(m, {})
            out.append(f"| {m} | {_fmt(row.get(64) if row is not None else None)} | {_fmt(row.get(128) if row is not None else None)} | "
                       f"{_fmt(row.get(256) if row is not None else None)} | {_fmt(cm.get('gpu_oracle_wall'), 2)} | {_fmt(cm.get('cpu_fit_wall'), 2)} | "
                       f"{_fmt(cm.get('cpu_acquisition_wall'), 2)} | {_fmt(cm.get('e2e_wall_excl_screening'), 2)} |")
        out.append("\nPer-seed S (rows: method, budget):\n")
        bys = pd.DataFrame(C["S_by_seed"])
        cols = [c for c in bys.columns if c not in ("method", "budget")]
        out.append("| method | B | " + " | ".join(str(c) for c in cols) + " |")
        out.append("|---|---:|" + "---:|" * len(cols))
        for _, r in bys.iterrows():
            out.append(f"| {r.method} | {int(r.budget)} | " + " | ".join(_fmt(r[c]) for c in cols) + " |")
        out.append("\n![Gate C](figures/gate_c_overview.png)\n\n![Calibration](figures/gate_c_calibration.png)\n\n![Costs](figures/gate_c_costs.png)\n")
    # ---- matrix
    out.append("\n## Mandatory Go/No-Go matrix (recomputed from saved artifacts)\n")
    out.append(_crit_table(decision, rid))
    out.append(f"\n\n**Formal overall verdict: {decision['overall_verdict']}** — rule: {decision['overall_rule']}.\n")
    out.append(f"\nRuntime-summary consistency check (recomputed status == status written at run time): `{json.dumps(decision['consistency_with_runtime_summaries'])}`\n")
    return "\n".join(out)
