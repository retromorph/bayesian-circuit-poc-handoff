"""Command-line entry point: ``python -m bayescircuits.cli <command> --config ...``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import PROJECT_ROOT, load_config, run_root, sha256_file  # noqa: E402
from .provenance import provenance, write_json  # noqa: E402

THRESH_FILE = PROJECT_ROOT / "configs" / "gate_a_misfit_threshold.json"


def _status_path(gate: str) -> Path:
    return PROJECT_ROOT / "logs" / f"{gate}.status.json"


def _write_status(gate: str, **kw) -> None:
    p = _status_path(gate)
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw)
    cur["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(p, cur)


def _require_clean(prov: dict, confirmatory: bool, allow_dirty: bool) -> None:
    if confirmatory and prov.get("git_dirty") and not allow_dirty:
        raise SystemExit("refusing confirmatory run on a dirty git worktree (commit first or pass --allow-dirty)")


# ----------------------------------------------------------------------------------- doctor
def cmd_doctor(args) -> int:
    cfg = load_config(args.config)
    prov = provenance(cfg)
    rep = {"provenance": prov, "checks": {}}
    try:
        from .tracking import roundtrip_check, tracking_uri

        rep["checks"]["mlflow_uri"] = tracking_uri()
        rep["checks"]["mlflow_roundtrip"] = roundtrip_check()
    except Exception as e:
        rep["checks"]["mlflow_roundtrip"] = {"ok": False, "error": repr(e)}
    try:
        import torch

        if torch.cuda.is_available():
            x = torch.randn(256, 256, device="cuda")
            rep["checks"]["cuda_matmul"] = float((x @ x).sum().item()) is not None
            rep["checks"]["cuda_free_mib"] = int(torch.cuda.mem_get_info()[0] // 2**20)
    except Exception as e:
        rep["checks"]["cuda"] = repr(e)
    print(json.dumps(rep, indent=2, default=str))
    write_json(PROJECT_ROOT / "reports" / "doctor.json", rep)
    return 0


# ----------------------------------------------------------------------------------- gate A
def _log_gate_a_world(res: dict, parent_id: str | None, prov: dict, confirmatory: bool, experiment: str) -> None:
    if res.get("status") != "complete":
        return
    from . import tracking

    d = Path(res["dir"])
    rows = pd.read_csv(d / "rows.csv")
    dec = json.loads((d / "decisions.json").read_text())
    q = pd.read_parquet(d / "queries.parquet")
    run_ids = {}
    for method, g in rows.groupby("method"):
        tags = tracking.base_tags(
            prov, "A", confirmatory, world_id=f"{res['regime']}-{res['seed']}", seed=res["seed"], method=method,
            task=f"synthetic-{res['regime']}", model_id="synthetic", split_sha256=res["blind_masks_sha256"],
            budget_masks="32,64,96,128", device="cpu", split=res["split"],
        )
        with tracking.run(experiment, f"A:{res['regime']}-{res['seed']}:{method}", tags, parent_id) as r:
            for _, row in g.iterrows():
                tracking.log_metrics_step(
                    {k: row.get(k) for k in row.index if k not in ("regime", "world_seed", "split", "method", "proposal_true_R", "true_pair_ranks")
                     and not isinstance(row.get(k), str)},
                    step=int(row["budget"]),
                )
            tracking.log_json({k: v for k, v in dec.items() if k.startswith(method + "@")}, "decisions.json")
            qm = q[q.method == method]
            tmp = d / f"_queries_{method}.csv"
            qm.to_csv(tmp, index=False)
            import mlflow

            mlflow.log_artifact(str(tmp), artifact_path="queries")
            tmp.unlink()
            run_ids[method] = r.info.run_id
    write_json(d / "mlflow_runs.json", run_ids)


def _gate_parent(experiment: str, gate: str, prov: dict, confirmatory: bool, status_key: str):
    from . import tracking

    st = json.loads(_status_path(gate).read_text()) if _status_path(gate).exists() else {}
    pid = st.get(status_key)
    mlflow = tracking.setup(experiment)
    if pid:
        try:
            mlflow.tracking.MlflowClient().get_run(pid)
            return pid
        except Exception:
            pid = None
    tags = tracking.base_tags(prov, gate.split("-")[1].upper() if "-" in gate else gate, confirmatory, role="parent")
    pid = tracking.create_parent(experiment, f"{gate}-parent", tags)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        pp = Path(td) / "provenance.json"
        pp.write_text(json.dumps(prov, indent=2, default=str))
        mlflow.tracking.MlflowClient().log_artifact(pid, str(pp))
    _write_status(gate, **{status_key: pid})
    return pid


def _finish_parent(pid: str, status: str, artifacts: dict, metrics: dict | None = None) -> None:
    from . import tracking

    mlflow = tracking.setup("bccd/aggregate")
    client = mlflow.tracking.MlflowClient()
    for name, obj in artifacts.items():
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / name
            p.write_text(json.dumps(obj, indent=2, default=str))
            client.log_artifact(pid, str(p))
    for k, v in (metrics or {}).items():
        if v is not None:
            client.log_metric(pid, k, float(v))
    client.set_terminated(pid, status=status)


def gate_a_rows(confirmatory: bool, split: str) -> pd.DataFrame:
    base = run_root(confirmatory) / "gate_a" / split
    frames = [pd.read_csv(p) for p in sorted(base.glob("*/rows.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_misfit_threshold(dev_rows: pd.DataFrame) -> dict:
    r = dev_rows[(dev_rows.method == "bayes_decision") & (dev_rows.budget == 128)]
    ctrl = r[r.regime.isin(["A0", "A1", "A2"])].misfit_stat.to_numpy()
    a3 = r[r.regime == "A3"].misfit_stat.to_numpy()
    c, a = float(np.max(ctrl)), float(np.min(a3))
    if c < a:
        t = float(np.sqrt(max(c, 1e-12) * a))
        rule = "geometric midpoint"
    else:
        cands = np.unique(np.concatenate([ctrl, a3]))
        errs = [((ctrl > t).sum() + (a3 <= t).sum(), -t) for t in cands]
        best = min(range(len(cands)), key=lambda i: errs[i])
        t = float(cands[best])
        rule = "min dev misclassification, ties to larger t"
    return {"threshold": t, "rule": rule, "dev_control_max": c, "dev_A3_min": a,
            "dev_control_values": ctrl.tolist(), "dev_A3_values": a3.tolist(), "n_dev_control": len(ctrl), "n_dev_A3": len(a3)}


def cmd_prepare(args) -> int:
    cfg = load_config(args.config)
    prov = provenance(cfg)
    stages = ["a", "b", "c"] if args.stage == "all" else [args.stage]
    if "a" in stages:
        from .gate_a import run_gate_a

        if THRESH_FILE.exists():
            print(f"misfit threshold already frozen: {THRESH_FILE.read_text()}")
        else:
            exp = "bccd/gate-a-dev"
            pid = _gate_parent(exp, "prepare-a", prov, False, "parent_run_id")
            res = run_gate_a(cfg, "development", confirmatory=False,
                             on_done=lambda r: _log_gate_a_world(r, pid, prov, False, exp))
            if any(r.get("status") != "complete" for r in res):
                _finish_parent(pid, "FAILED", {"results.json": res})
                raise SystemExit("development worlds failed")
            dev = gate_a_rows(False, "development")
            th = compute_misfit_threshold(dev)
            th.update({"protocol": cfg["protocol"]["id"], "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "git_sha": prov["git_sha"], "impl_config_sha256": cfg["_impl_sha256"]})
            write_json(THRESH_FILE, th)
            dev.to_csv(run_root(False) / "gate_a" / "development_rows.csv", index=False)
            _finish_parent(pid, "FINISHED", {"misfit_threshold.json": th})
            print(json.dumps(th, indent=2))
    if "c" in stages:
        from .gate_c import prepare_gate_c

        prepare_gate_c(cfg, prov)
    return 0


def cmd_gate_a(args) -> int:
    from .criteria import gate_a_criteria, gate_verdict
    from .gate_a import run_gate_a

    cfg = load_config(args.config)
    prov = provenance(cfg)
    confirmatory = args.max_worlds is None and not args.dry_run
    _require_clean(prov, confirmatory, args.allow_dirty)
    if not THRESH_FILE.exists():
        raise SystemExit("misfit threshold not frozen: run `prepare --stage a` first")
    th = json.loads(THRESH_FILE.read_text())
    exp = "bccd/gate-a" if confirmatory else "bccd/dev-smoke"
    gate = "gate-a" if confirmatory else "gate-a-smoke"
    _write_status(gate, state="running", started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), pid=os.getpid())
    pid = _gate_parent(exp, gate, prov, confirmatory, "parent_run_id")
    try:
        res = run_gate_a(cfg, "confirmatory" if confirmatory else "smoke", confirmatory=confirmatory,
                         max_worlds=args.max_worlds if not confirmatory else None,
                         on_done=lambda r: _log_gate_a_world(r, pid, prov, confirmatory, exp))
        failed = [r for r in res if r.get("status") != "complete"]
        rows = gate_a_rows(confirmatory, "confirmatory" if confirmatory else "smoke")
        out = run_root(confirmatory) / "gate_a"
        rows.to_csv(out / "rows_all.csv", index=False)
        crit = gate_a_criteria(rows.assign(split="confirmatory"), cfg, th["threshold"]) if confirmatory else {}
        summary = {"confirmatory": confirmatory, "n_worlds": len(res), "n_failed": len(failed), "criteria": crit,
                   "verdict": gate_verdict(crit) if crit else "EXPLORATORY", "misfit_threshold": th,
                   "threshold_file_sha256": sha256_file(THRESH_FILE), "provenance": prov, "parent_run_id": pid}
        write_json(out / "summary.json", summary)
        _finish_parent(pid, "FAILED" if failed else "FINISHED", {"gate_a_summary.json": summary})
        _write_status(gate, state="failed" if failed else "complete", exit_code=1 if failed else 0,
                      completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), verdict=summary["verdict"])
        print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2, default=str))
        return 1 if failed else 0
    except BaseException as e:
        _write_status(gate, state="crashed", error=repr(e), traceback=traceback.format_exc())
        try:
            _finish_parent(pid, "FAILED", {"error.json": {"error": repr(e)}})
        except Exception:
            pass
        raise


def cmd_gate_b(args) -> int:
    from .gate_b import cmd_gate_b as run

    return run(args, load_config(args.config))


def cmd_gate_c(args) -> int:
    from .gate_c import cmd_gate_c as run

    return run(args, load_config(args.config))


def cmd_diagnostics(args) -> int:
    from .diagnostics import run_diagnostics

    return run_diagnostics(load_config(args.config), args.which)


def cmd_report(args) -> int:
    from .report import build_report

    return build_report(load_config(args.config))


def cmd_vil_prepare(args) -> int:
    from .vil_run import cmd_vil_prepare as run

    return run(args)


def cmd_vil_freeze(args) -> int:
    from .vil_run import cmd_vil_freeze as run

    return run(args)


def cmd_vil_run(args) -> int:
    from .vil_run import run_set

    return run_set(args)


def cmd_vil_report(args) -> int:
    from .vil_report import build_vil_report

    return build_vil_report(args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bayescircuits")
    sub = ap.add_subparsers(dest="cmd", required=True)
    # BCCD-VIL-1.0 (verification-in-the-loop follow-up; configs/preregistration_verification_v1.yaml)
    vp = sub.add_parser("vil-prepare", help="freeze VIL prompt batches, evaluation sets and mask designs")
    vp.add_argument("--config", default=None)
    vf = sub.add_parser("vil-freeze", help="record protocol/config/source hashes before confirmatory runs")
    vf.add_argument("--force", action="store_true")
    vr = sub.add_parser("vil-run", help="run one VIL set (dev | confirmatory | bridge)")
    vr.add_argument("--set", required=True, choices=["dev", "confirmatory", "bridge", "exploratory"])
    vr.add_argument("--shard", type=int, default=0)
    vr.add_argument("--n-shards", type=int, default=1)
    vr.add_argument("--methods", default=None, help="development only: comma-separated subset")
    vr.add_argument("--variant", default=None, help="development only: e.g. rounds4")
    vr.add_argument("--allow-dirty", action="store_true")
    vrep = sub.add_parser("vil-report", help="recompute VIL criteria from sealed artifacts; write reports/verification_v1")
    vrep.add_argument("--dev-only", action="store_true")
    for name in ["doctor", "prepare", "gate-a", "gate-b", "gate-c", "diagnostics", "report"]:
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "preregistration.yaml"))
        sp.add_argument("--dry-run", action="store_true", help="exploratory smoke run (never confirmatory)")
        sp.add_argument("--allow-dirty", action="store_true")
        if name == "prepare":
            sp.add_argument("--stage", default="all", choices=["all", "a", "b", "c"])
        if name == "gate-a":
            sp.add_argument("--max-worlds", type=int, default=None, help="exploratory: worlds per regime")
        if name == "diagnostics":
            sp.add_argument("--which", default="all", choices=["all", "proxyspex", "transfer", "replicate"])
        if name in ("gate-b", "gate-c"):
            sp.add_argument("--max-seeds", type=int, default=None, help="exploratory: number of units")
    args = ap.parse_args(argv)
    fn = {"doctor": cmd_doctor, "prepare": cmd_prepare, "gate-a": cmd_gate_a, "gate-b": cmd_gate_b,
          "gate-c": cmd_gate_c, "diagnostics": cmd_diagnostics, "report": cmd_report,
          "vil-prepare": cmd_vil_prepare, "vil-freeze": cmd_vil_freeze, "vil-run": cmd_vil_run, "vil-report": cmd_vil_report}[args.cmd]
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
