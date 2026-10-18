"""Configuration loading and hashing (pre-registration is the source of truth)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DOC = PROJECT_ROOT / "docs" / "01_PROTOCOL.md"
IMPL_CONFIG = PROJECT_ROOT / "configs" / "implementation.yaml"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(path: str | Path) -> dict:
    """Load the pre-registration plus the frozen implementation settings."""
    path = Path(path)
    cfg = load_yaml(path)
    cfg["_path"] = str(path)
    cfg["_sha256"] = sha256_file(path)
    impl = load_yaml(IMPL_CONFIG)
    cfg["_impl"] = impl
    cfg["_impl_sha256"] = sha256_file(IMPL_CONFIG)
    cfg["_protocol_sha256"] = sha256_file(PROTOCOL_DOC) if PROTOCOL_DOC.exists() else None
    return cfg


def run_root(confirmatory: bool) -> Path:
    return PROJECT_ROOT / ("runs" if confirmatory else "runs_exploratory")
