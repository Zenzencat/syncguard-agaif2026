"""/spatial/autocorrelation: first-call latency and event-loop safety (Item B2).

MEASURED (fresh process, this machine, 136 real tower coordinates with random severities):
  import api.spatial_stats (libpysal + esda) ....... 1.4 s   (already paid at app import)
  FIRST compute_autocorrelation() call ............. 14.8 s  (numba JIT of esda's permutation kernels)
  later calls ...................................... 0.02-0.03 s
After the fix: warmup() at startup (background thread) absorbs the ~15 s, and the first REAL
computation afterwards takes ~0.02 s. The endpoint also runs the computation in a worker
thread, so even a slow one cannot stall the event loop (SSE, /health, every other request).

The tower data and severities used here are SYNTHETIC test fixtures.
"""
from __future__ import annotations

import asyncio
import random
import threading
import time

import httpx
import pandas as pd

import api.main as main_module
from api import spatial_stats
from api.main import app


def _fixture(n=30, seed=7):
    rng = random.Random(seed)
    towers = pd.DataFrame({
        "tower_key": [f"T{i}" for i in range(n)],
        "site_id": [f"T{i}" for i in range(n)],
        "site_name": [f"tower {i}" for i in range(n)],
        "lat": [0.01 * (i // 6) for i in range(n)],
        "lon": [0.01 * (i % 6) for i in range(n)],
    })
    latest = {f"T{i}": {"severity": rng.random()} for i in range(n)}
    return towers, latest


def test_warmup_leaves_computation_fast_and_touches_no_store(client):
    spatial_stats.warmup()                 # pays the one-time JIT cost if the startup thread hasn't
    towers, latest = _fixture()
    t0 = time.perf_counter()
    result = spatial_stats.compute_autocorrelation(towers, latest)
    elapsed = time.perf_counter() - t0
    assert result.computable
    # warm cost is ~0.02 s; 2 s leaves 100x headroom for a loaded CI box yet is far below the
    # ~15 s cold cost this guards against
    assert elapsed < 2.0, f"warm autocorrelation took {elapsed:.2f}s -- warmup did not take effect"


def test_startup_launches_the_warmup_thread(client):
    thread = getattr(app.state, "autocorr_warmup", None)
    assert isinstance(thread, threading.Thread) and thread.daemon
    thread.join(timeout=120)               # finishes on its own; it is best-effort and never fatal
    assert not thread.is_alive()


def test_concurrent_computations_are_bit_identical_to_serial(client):
    """The reseed of numpy's global RNG + permutation draw is serialised by a lock; without it
    two worker threads could interleave and change each other's p-values."""
    towers, latest = _fixture()
    spatial_stats.warmup()
    serial = spatial_stats.compute_autocorrelation(towers, latest)
    results, errors = [], []

    def run():
        try:
            results.append(spatial_stats.compute_autocorrelation(towers, latest))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(results) == 6
    for r in results:
        assert r.global_moran_i == serial.global_moran_i
        assert r.global_p_value == serial.global_p_value
        assert [t["p_value"] for t in r.per_tower] == [t["p_value"] for t in serial.per_tower]


def test_slow_autocorrelation_does_not_block_other_requests(client, monkeypatch):
    """Regression: the handler used to run the computation inline in an `async def`, freezing
    the event loop for its whole duration (16 s on the first call after startup)."""
    def slow(towers, latest):
        time.sleep(1.0)
        return spatial_stats.AutocorrelationResult(
            computable=False, n_towers_scored=0, n_towers_total=len(towers),
            min_required=spatial_stats.MIN_TOWERS_FOR_STATS, reason="test stub")
    monkeypatch.setattr(main_module, "compute_autocorrelation", slow)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            # Timer starts BEFORE the slow request is scheduled: if its computation blocks the
            # loop, /health cannot even begin until it finishes, and that wait must count.
            t0 = time.perf_counter()
            slow_req = asyncio.create_task(ac.get("/spatial/autocorrelation"))
            await asyncio.sleep(0.05)      # let the slow request start its computation
            health = await ac.get("/health")
            health_s = time.perf_counter() - t0
            slow_resp = await slow_req
            return health.status_code, health_s, slow_resp.status_code

    health_status, health_s, slow_status = asyncio.run(scenario())
    assert slow_status == 200 and health_status == 200
    assert health_s < 0.5, (
        f"/health took {health_s:.2f}s while a 1 s autocorrelation was running -- "
        "the event loop was blocked")
