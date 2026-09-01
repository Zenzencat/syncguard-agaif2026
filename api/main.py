"""SyncGuard serving layer: wraps the trained detector in a FastAPI service, adds a
simulated live-feed replay mode, and layers real distance-weighted spatial correlation and
SQLite persistence on top -- see api/model_service.py, api/replay.py, api/spatial.py, api/db.py
for the pieces this wires together, and ROBUSTNESS_NOTES.md / spatial_layer_notes.md for the
REAL-vs-SIMULATED framing this project holds itself to.

Run: uvicorn api.main:app --reload   (from the repo root, venv activated)
"""
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse

from api.schemas import TelemetryInput, ScoreResponse, HealthResponse
from api.model_service import ModelService, ModelNotFoundError
from api.db import EventStore
from api.spatial import load_towers, TowerAttributor, LiveCorrelationEngine
from api.replay import ReplayManager, EventBus, list_run_ids

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_PATH = REPO_ROOT / "syncguard_interactive_summary.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model_service = ModelService()
    except ModelNotFoundError as e:
        # Let the app start (so /health reports the real problem instead of a crash-loop),
        # but every scoring endpoint will 503 until `make train` has been run.
        print(f"[startup] {e}")
        app.state.model_service = None

    app.state.event_store = EventStore()
    towers = load_towers()
    app.state.towers = towers
    app.state.tower_attributor = TowerAttributor(towers)
    app.state.correlation_engine = LiveCorrelationEngine(towers, app.state.event_store)
    app.state.event_bus = EventBus()
    app.state.replay_manager = ReplayManager(
        app.state.model_service, app.state.event_store, app.state.tower_attributor,
        app.state.correlation_engine, app.state.event_bus,
    ) if app.state.model_service else None

    yield

    if app.state.replay_manager:
        app.state.replay_manager.stop()
    app.state.event_store.close()


app = FastAPI(
    title="SyncGuard API",
    description="GNSS spoofing/jamming detection for telecom timing infrastructure -- "
                 "scoring, live replay, and real distance-weighted spatial correlation.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _require_model(app_: FastAPI) -> ModelService:
    if app_.state.model_service is None:
        raise HTTPException(503, "No trained model loaded -- run `make train` first, then restart the service.")
    return app_.state.model_service


@app.get("/health", response_model=HealthResponse)
async def health():
    ms = app.state.model_service
    return HealthResponse(
        status="ok" if ms else "degraded",
        model_loaded=ms is not None,
        model_version=ms.model_version if ms else None,
        towers_loaded=len(app.state.towers),
        replay_running=bool(app.state.replay_manager and app.state.replay_manager.status["status"] == "running"),
    )


@app.post("/score", response_model=ScoreResponse)
async def score(telemetry: TelemetryInput):
    ms = _require_model(app)
    features = telemetry.model_dump(exclude={"receiver_id", "tower_site_id"})
    result = ms.score(features)

    tower, corr = None, None
    if telemetry.tower_site_id is not None:
        towers = app.state.towers
        # Match against tower_key first (the unique internal id, e.g. what /towers and
        # replay-attributed events use), falling back to site_id -- 16 of the 136 real
        # towers share the literal site_id "tbg" (see api/spatial.py::load_towers), so a
        # caller matching on raw site_id there gets the first such row deterministically.
        match = towers[towers["tower_key"] == telemetry.tower_site_id]
        if match.empty:
            match = towers[towers["site_id"] == telemetry.tower_site_id]
        if match.empty:
            raise HTTPException(422, f"Unknown tower_site_id: {telemetry.tower_site_id!r}")
        t = match.iloc[0]
        tower_key = t["tower_key"]
        tower = {"site_id": tower_key, "site_name": t["site_name"], "lat": float(t["lat"]), "lon": float(t["lon"])}
        result_corr = app.state.correlation_engine.correlate(tower_key)
        corr = {
            "correlation_score": result_corr.correlation_score,
            "n_neighbors_considered": result_corr.n_neighbors_considered,
            "window_seconds": result_corr.window_seconds,
        }

    event_id = app.state.event_store.insert_event({
        "source": "api",
        "probability": result["probability"],
        "severity": result["severity"],
        "predicted_label": result["predicted_label"],
        "model_version": result["model_version"],
        "tower_site_id": tower["site_id"] if tower else None,
        "tower_site_name": tower["site_name"] if tower else None,
        "tower_lat": tower["lat"] if tower else None,
        "tower_lon": tower["lon"] if tower else None,
        "correlation_score": corr["correlation_score"] if corr else None,
        "features": features,
    })

    return ScoreResponse(**result, event_id=event_id, tower=tower, correlation=corr)


@app.get("/towers")
async def towers():
    return app.state.towers.to_dict(orient="records")


@app.get("/events")
async def events(limit: int = Query(default=200, le=2000)):
    return app.state.event_store.recent_events(limit=limit)


@app.get("/events/map")
async def events_map():
    """Latest scored event per tower -- what the dashboard renders as the current map state."""
    return list(app.state.event_store.latest_severity_per_tower().values())


@app.get("/replay/runs")
async def replay_runs():
    return list_run_ids()


@app.post("/replay/start")
async def replay_start(run_id: str | None = None, speed: float = 10.0):
    if app.state.replay_manager is None:
        raise HTTPException(503, "No trained model loaded -- run `make train` first, then restart the service.")
    try:
        return app.state.replay_manager.start(run_id, speed)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(409, str(e))


@app.post("/replay/stop")
async def replay_stop():
    if app.state.replay_manager is None:
        raise HTTPException(503, "No trained model loaded.")
    return app.state.replay_manager.stop()


@app.get("/replay/status")
async def replay_status():
    if app.state.replay_manager is None:
        return {"status": "unavailable", "reason": "no trained model loaded"}
    return app.state.replay_manager.status


@app.get("/stream/events")
async def stream_events():
    if app.state.replay_manager is None:
        raise HTTPException(503, "No trained model loaded.")
    bus: EventBus = app.state.event_bus
    queue = bus.subscribe()

    async def gen():
        try:
            yield "event: connected\ndata: {}\n\n"
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            bus.unsubscribe(queue)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard")
async def dashboard():
    if not DASHBOARD_PATH.exists():
        raise HTTPException(404, "syncguard_interactive_summary.html not found")
    return FileResponse(DASHBOARD_PATH)
