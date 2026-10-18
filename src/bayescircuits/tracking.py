"""MLflow logging helpers (local server on the remote host, loopback only)."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

DEFAULT_URI = "http://127.0.0.1:5001"

TAG_KEYS = [
    "protocol_version",
    "protocol_sha256",
    "config_sha256",
    "git_sha",
    "uv_lock_sha256",
    "model_id",
    "model_revision",
    "task",
    "gate",
    "world_id",
    "seed",
    "method",
    "budget_masks",
    "prompt_batch_id",
    "intervention_semantics",
    "candidate_set_sha256",
    "split_sha256",
    "confirmatory",
    "device",
    "torch_version",
]


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI)


def setup(experiment: str):
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment)
    return mlflow


def base_tags(prov: dict, gate: str, confirmatory: bool, **extra) -> dict:
    t = {
        "protocol_version": prov.get("protocol_version"),
        "protocol_sha256": prov.get("protocol_doc_sha256"),
        "config_sha256": prov.get("config_sha256"),
        "impl_config_sha256": prov.get("impl_config_sha256"),
        "git_sha": prov.get("git_sha"),
        "git_dirty": prov.get("git_dirty"),
        "uv_lock_sha256": prov.get("uv_lock_sha256"),
        "gate": gate,
        "confirmatory": str(bool(confirmatory)).lower(),
        "torch_version": prov.get("torch_version"),
        "intervention_semantics": "mask1=ablate; zero head output (hook_z) at all positions; retain(C)=ablate complement in pool",
    }
    t.update({k: v for k, v in extra.items() if v is not None})
    return {k: str(v) for k, v in t.items() if v is not None}


@contextmanager
def run(experiment: str, run_name: str, tags: dict, parent_run_id: str | None = None):
    mlflow = setup(experiment)
    tags = dict(tags)
    if parent_run_id:
        tags["mlflow.parentRunId"] = parent_run_id
    active = mlflow.start_run(run_name=run_name, tags=tags)
    try:
        yield active
    except BaseException:
        mlflow.end_run(status="FAILED")
        raise
    else:
        mlflow.end_run(status="FINISHED")


def log_json(obj, name: str, artifact_path: str | None = None) -> None:
    import mlflow

    from .provenance import _default

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / name
        p.write_text(json.dumps(obj, indent=2, default=_default))
        mlflow.log_artifact(str(p), artifact_path=artifact_path)


def log_metrics_step(metrics: dict, step: int | None = None) -> None:
    import math

    import mlflow

    clean = {}
    for k, v in metrics.items():
        if v is None:
            continue  # never log missing values as zero
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            clean[k] = fv
    if clean:
        mlflow.log_metrics(clean, step=step)


def roundtrip_check(experiment: str = "bccd/dev-smoke") -> dict:
    """Create run, log param/metric/artifact, download artifact, verify, end run."""
    import mlflow

    setup(experiment)
    payload = {"roundtrip": "ok", "value": 3.14159}
    with mlflow.start_run(run_name="artifact-roundtrip", tags={"confirmatory": "false", "purpose": "roundtrip"}) as r:
        mlflow.log_param("p", 1)
        mlflow.log_metric("m", 2.5)
        log_json(payload, "roundtrip.json")
        rid = r.info.run_id
    with tempfile.TemporaryDirectory() as d:
        local = mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="roundtrip.json", dst_path=d)
        got = json.loads(Path(local).read_text())
    client = mlflow.tracking.MlflowClient()
    info = client.get_run(rid)
    ok = got == payload and info.data.metrics.get("m") == 2.5 and info.info.status == "FINISHED"
    return {"run_id": rid, "ok": bool(ok), "artifact_uri": info.info.artifact_uri, "status": info.info.status}


def create_parent(experiment: str, run_name: str, tags: dict) -> str:
    mlflow = setup(experiment)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    r = client.create_run(exp.experiment_id, run_name=run_name, tags=tags)
    return r.info.run_id
