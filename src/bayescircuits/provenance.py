"""Provenance capture: source revision, lockfile, environment, hardware."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from .config import PROJECT_ROOT, sha256_file


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return None


def git_info() -> dict:
    sha = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return {"git_sha": sha, "git_dirty": bool(status), "git_status": status}


def env_info() -> dict:
    info: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname_hash": __import__("hashlib").sha256(socket.gethostname().encode()).hexdigest()[:12],
        "cpu_count": os.cpu_count(),
        "uv_version": _run(["uv", "--version"]) or _run([str(Path.home() / ".local/bin/uv"), "--version"]),
    }
    for mod in ["numpy", "scipy", "sklearn", "lightgbm", "shapiq", "mlflow", "torch", "transformer_lens", "transformers", "pandas"]:
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", None)
        except Exception as e:  # pragma: no cover - environment specific
            info[f"{mod}_version"] = f"unavailable: {e.__class__.__name__}"
    try:
        import torch

        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_total_mib"] = int(torch.cuda.get_device_properties(0).total_memory // 2**20)
    except Exception:
        pass
    smi = shutil.which("nvidia-smi")
    if smi:
        info["nvidia_smi"] = _run([smi, "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"])
    try:
        import psutil

        info["ram_total_gib"] = round(psutil.virtual_memory().total / 2**30, 1)
        info["disk_free_gib"] = round(shutil.disk_usage(PROJECT_ROOT).free / 2**30, 1)
    except Exception:
        pass
    return info


def lock_info() -> dict:
    lock = PROJECT_ROOT / "uv.lock"
    return {"uv_lock_sha256": sha256_file(lock) if lock.exists() else None}


def provenance(cfg: dict) -> dict:
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_version": cfg["protocol"]["id"],
        "config_sha256": cfg["_sha256"],
        "impl_config_sha256": cfg["_impl_sha256"],
        "protocol_doc_sha256": cfg["_protocol_sha256"],
        **git_info(),
        **lock_info(),
        **env_info(),
    }


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
    os.replace(tmp, path)


def _default(o):
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set | frozenset):
        return sorted(o)
    if isinstance(o, tuple):
        return list(o)
    return str(o)
