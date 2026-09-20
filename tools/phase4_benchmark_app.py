"""Phase 4 benchmark-only ASGI wrapper.

This module reuses the shipped ``api.main:app`` routes and lifespan, then swaps the
runtime tower registry for an explicitly synthetic registry after startup.  It exists
only so the Phase 4 benchmark can exercise the real HTTP /ingest path at 1,000 and
10,000 synthetic tower IDs without changing the production API or the real 136-row
tower data.  Feature values remain resampled rows from the real parquet table.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from fastapi import Query

from api.ingest import IngestService
from api.main import app, lifespan as shipped_lifespan
from api.spatial import LiveCorrelationEngine


@app.get("/_phase4/batch")
async def phase4_batch(
    start: int = Query(ge=0),
    count: int = Query(ge=1, le=500),
    tower_count: int = Query(ge=1),
    seed: int = Query(),
    day: int = Query(default=20, ge=1, le=28),
    batch_id: str = Query(),
):
    """Benchmark-only payload source: resamples real parquet rows server-side.

    The client then POSTs the returned payload to the shipped /ingest route.  Keeping
    parquet/pyarrow inside the long-lived server avoids the Windows bundled-Python
    client runtime's native teardown issue; it does not change what /ingest measures.
    """

    if tower_count != int(os.environ.get("SYNTHETIC_TOWER_COUNT", "136")):
        raise ValueError("tower_count must match SYNTHETIC_TOWER_COUNT")
    df = pd.read_parquet(
        os.path.join(os.path.dirname(__file__), "..", "processed", "syncguard_features.parquet")
    )
    sampled = df.sample(n=count, replace=True, random_state=seed + start)
    base = datetime(2026, 9, day, tzinfo=timezone.utc)
    observations = []
    for offset, (_, row) in enumerate(sampled.iterrows()):
        idx = start + offset
        observation = {
            feature: (None if pd.isna(row[feature]) else float(row[feature]))
            for feature in app.state.model_service.feature_cols
        }
        observation.update({
            "tower_id": f"SYNTH_TOWER_{(idx % tower_count) + 1:05d}",
            "timestamp": (base + timedelta(seconds=idx)).isoformat().replace("+00:00", "Z"),
        })
        observations.append(observation)
    return {
        "batch_id": batch_id,
        "observations": observations,
        "feature_provenance": "RESAMPLED REAL FEATURES, SYNTHETIC TOWERS",
    }


def synthetic_towers(n: int) -> pd.DataFrame:
    """Build N synthetic IDs and deterministic synthetic coordinates.

    The coordinates are deliberately a small regular grid rather than copies of the
    real registry.  This makes the benchmark's tower layer unambiguous: all tower IDs
    and locations in this wrapper are synthetic, while all model feature rows are real
    observations sampled from the committed parquet dataset.
    """

    side = max(1, int(np.ceil(np.sqrt(n))))
    i = np.arange(n, dtype=float)
    return pd.DataFrame({
        "site_id": [f"SYNTH_TOWER_{k + 1:05d}" for k in range(n)],
        "tower_key": [f"SYNTH_TOWER_{k + 1:05d}" for k in range(n)],
        "site_name": [f"Synthetic tower {k + 1:05d}" for k in range(n)],
        "lat": (0.0001 * (i % side)).tolist(),
        "lon": (0.0001 * np.floor(i / side)).tolist(),
    })


@asynccontextmanager
async def benchmark_lifespan(app_):
    async with shipped_lifespan(app_):
        n = int(os.environ.get("SYNTHETIC_TOWER_COUNT", "136"))
        if n < 1:
            raise RuntimeError("SYNTHETIC_TOWER_COUNT must be positive")
        towers = synthetic_towers(n)
        app_.state.towers = towers
        app_.state.correlation_engine = LiveCorrelationEngine(
            towers, app_.state.event_store
        )
        app_.state.ingest_service = IngestService(
            app_.state.model_service,
            app_.state.event_store,
            towers,
            app_.state.correlation_engine,
            app_.state.event_bus,
            baseline=app_.state.baseline,
        )
        yield


# Uvicorn uses the app router's lifespan context.  Keeping the original FastAPI app
# preserves every shipped route, middleware, auth behavior, and request instrumentation.
app.router.lifespan_context = benchmark_lifespan
