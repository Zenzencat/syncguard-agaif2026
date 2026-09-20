"""Shared test fixtures.

The DB path override has to be set before api.db is imported anywhere, because
DEFAULT_DB_PATH is resolved at module import time -- pytest loads conftest.py before test
modules, which is what makes this work. Without it the suite would write into the real
data/syncguard.db and pollute whatever the dashboard is showing.
"""
import os
import tempfile
from pathlib import Path

import pytest

_TMP_DB_DIR = tempfile.mkdtemp(prefix="syncguard-tests-")
os.environ["SYNCGUARD_DB_PATH"] = str(Path(_TMP_DB_DIR) / "test_syncguard.db")

from fastapi.testclient import TestClient  # noqa: E402  (must follow the env var above)

from api.main import app  # noqa: E402
from api.model_service import DEFAULT_MODEL_PATH, FALLBACK_MODEL_PATH  # noqa: E402

requires_model = pytest.mark.skipif(
    not (DEFAULT_MODEL_PATH.exists() or FALLBACK_MODEL_PATH.exists()),
    reason="No trained model artifact -- run `make train` first.",
)


@pytest.fixture(scope="session")
def client():
    """One TestClient for the whole session: the lifespan loads the model and the SHAP
    explainer, which is slow enough that per-test startup would dominate the run."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def tower_ids(client) -> list[str]:
    """Real tower_keys straight from GET /towers -- the tests never hardcode a site id."""
    towers = client.get("/towers").json()
    assert towers, "expected the real 136-tower table to be loaded"
    return [t["tower_key"] for t in towers]
