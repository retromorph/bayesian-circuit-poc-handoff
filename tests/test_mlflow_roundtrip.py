import os
import urllib.request

import pytest

from bayescircuits.tracking import roundtrip_check, tracking_uri


def _up() -> bool:
    try:
        urllib.request.urlopen(tracking_uri() + "/health", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _up() or os.environ.get("BCCD_SKIP_MLFLOW") == "1", reason="MLflow server not reachable")
def test_mlflow_artifact_roundtrip():
    res = roundtrip_check("bccd/dev-smoke")
    assert res["ok"], res
