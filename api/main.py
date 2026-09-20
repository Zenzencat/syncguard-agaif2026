"""SyncGuard serving layer: wraps the trained detector in a FastAPI service, adds a
simulated live-feed replay mode, and layers real distance-weighted spatial correlation and
SQLite persistence on top -- see api/model_service.py, api/replay.py, api/spatial.py, api/db.py
for the pieces this wires together, and ROBUSTNESS_NOTES.md / spatial_layer_notes.md for the
REAL-vs-SIMULATED framing this project holds itself to.

Run: uvicorn api.main:app --reload   (from the repo root, venv activated)
"""
import asyncio
import csv
import io
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse

from api.schemas import (TelemetryInput, ScoreResponse, HealthResponse, AutocorrelationResponse,
                         ExplainResponse, IngestBatch, IngestResponse, FeedbackRequest,
                         FeedbackRecord, FeedbackSummary)
from api.model_service import ModelService, ModelNotFoundError
from api.db import EventStore
from api.spatial import load_towers, TowerAttributor, LiveCorrelationEngine
from api.spatial_stats import compute_autocorrelation
from api.replay import ReplayManager, EventBus, list_run_ids
from api.ingest import IngestService, UnknownTowerError

# Attached verbatim to every /feedback/summary response. The numbers are real, but the
# population they describe is chosen by analysts, not sampled -- so they are not an estimate
# of the detector's precision in the field, and must not be quoted as one. See
# FEEDBACK_LOOP.md.
PRECISION_CAVEAT = (
    "MEASURED over labeled events only. Analysts choose which events to label, so this is a "
    "self-selected, non-random subset of all scored events -- not a random sample. These "
    "figures describe that subset and nothing wider. They are NOT the model's field "
    "precision, NOT a validation result, and NOT comparable to the held-out test metrics in "
    "the model reports. No retraining uses these labels -- see FEEDBACK_LOOP.md."
)

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
    app.state.ingest_service = IngestService(
        app.state.model_service, app.state.event_store, towers,
        app.state.correlation_engine, app.state.event_bus,
    ) if app.state.model_service else None
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
    top_features = ms.explain(features)  # always computed for /score -- ad-hoc, single-row, latency is imperceptible (see SHAP_EXPLAINABILITY.md)

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
        "top_features": top_features,
    })

    return ScoreResponse(**result, event_id=event_id, tower=tower, correlation=corr, top_features=top_features)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(batch: IngestBatch):
    """Batch ingestion of per-tower observation windows from an external collector.

    Routes through the same ModelService.score() call as POST /score -- same artifact, same
    23 features, same 0.52 threshold, same single-threaded predict_proba -- then adds
    per-tower hysteresis, (tower_id, timestamp) deduplication, out-of-order detection, the
    same SQLite persistence, and the same SSE publish replay uses, so ingested events appear
    on the dashboard and in the live spatial statistics exactly like replayed ones.

    The 23 features must be computed caller-side; INGESTION_CONTRACT.md documents which raw
    receiver observables produce each one. HTTP only -- MQTT is noted as future work there,
    not implemented. See api/ingest.py for the full REAL/CALLER-SUPPLIED framing: the
    tower each observation claims to come from is trusted, not verified, and no real receiver
    has ever fed this endpoint.
    """
    _require_model(app)
    if app.state.ingest_service is None:
        raise HTTPException(503, "No trained model loaded -- run `make train` first, then restart the service.")
    try:
        return IngestResponse(**await app.state.ingest_service.ingest(batch))
    except UnknownTowerError as e:
        raise HTTPException(422, str(e))


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


@app.get("/events/{event_id}/explain", response_model=ExplainResponse)
async def explain_event(event_id: int):
    """On-demand SHAP explanation for an already-scored event. If it already has one
    (computed live for /score or a slow-enough replay -- see api/replay.py's
    LIVE_EXPLAIN_MAX_SPEED), returns it as-is (cached=True), no recomputation. Otherwise
    recomputes it from the event's own stored feature values and persists the result back
    onto the row, so asking again doesn't recompute. This is what makes fast-replay events
    lazily, not permanently, unexplainable -- see SHAP_EXPLAINABILITY.md."""
    ms = _require_model(app)
    event = app.state.event_store.get_event(event_id)
    if event is None:
        raise HTTPException(404, f"No scored event with id={event_id}")

    if event.get("top_features") is not None:
        return ExplainResponse(event_id=event_id, top_features=event["top_features"], cached=True)

    if not event.get("features_json"):
        raise HTTPException(422, f"Event {event_id} has no stored feature values to explain "
                                  f"(scored before this endpoint existed).")
    features = json.loads(event["features_json"])
    top_features = ms.explain(features)
    app.state.event_store.update_event_top_features(event_id, top_features)
    return ExplainResponse(event_id=event_id, top_features=top_features, cached=False)


@app.post("/events/{event_id}/feedback", response_model=FeedbackRecord)
async def submit_feedback(event_id: int, feedback: FeedbackRequest):
    """Record an analyst's verdict on a scored event: confirmed (real) or dismissed (false
    alarm).

    Re-submitting replaces the current label and bumps `revision`; the response's
    `previous_label` says what it replaced. Every submission, including superseded ones, is
    appended to an audit log (`event_feedback_log`) -- a change of mind is information, not
    noise.

    These labels are STORED ONLY. Nothing in this repository retrains on them, no model is
    updated, and no automated action follows from a label. See FEEDBACK_LOOP.md.
    """
    store = app.state.event_store
    if store.get_event(event_id) is None:
        raise HTTPException(404, f"No scored event with id={event_id}")
    record = store.upsert_feedback(
        event_id, feedback.label, note=feedback.note, analyst=feedback.analyst
    )
    return FeedbackRecord(**record)


@app.get("/events/{event_id}/feedback", response_model=FeedbackRecord)
async def get_event_feedback(event_id: int):
    """Current label for one event. 404 if the event does not exist; a record with
    `label: null, revision: 0` if the event exists but has never been labeled -- the
    dashboard uses that to render its buttons in the unlabeled state."""
    store = app.state.event_store
    if store.get_event(event_id) is None:
        raise HTTPException(404, f"No scored event with id={event_id}")
    record = store.get_feedback(event_id)
    if record is None:
        return FeedbackRecord(event_id=event_id, label=None, revision=0)
    return FeedbackRecord(**record)


@app.get("/events/{event_id}/feedback/history")
async def get_event_feedback_history(event_id: int):
    """Every label ever submitted for this event, oldest first, including superseded ones."""
    store = app.state.event_store
    if store.get_event(event_id) is None:
        raise HTTPException(404, f"No scored event with id={event_id}")
    return store.feedback_history(event_id)


@app.get("/feedback/export")
async def feedback_export():
    """Every labeled event as CSV: the label, who set it and when, the full scored-event row,
    and each of the 23 model features flattened into its own column.

    This is the artifact a future retraining effort would start from. Producing it is as far
    as this repo goes -- nothing consumes it. See FEEDBACK_LOOP.md.

    Feature columns are emitted in the model's own `feature_cols` order when a model is
    loaded, so the CSV's column order matches the artifact's. Without a model (the degraded
    /health state) the order falls back to whatever the stored rows contain, which is stable
    but not guaranteed to match.
    """
    store = app.state.event_store
    rows = store.labeled_events()

    ms = app.state.model_service
    if ms is not None:
        feature_cols = list(ms.feature_cols)
    else:
        seen: list[str] = []
        for r in rows:
            if r.get("features_json"):
                for k in json.loads(r["features_json"]):
                    if k not in seen:
                        seen.append(k)
        feature_cols = seen

    meta_cols = [
        "event_id", "feedback_label", "feedback_analyst", "feedback_note",
        "labeled_at", "first_labeled_at", "feedback_revision",
        "created_at", "source", "run_id", "scenario_id", "obs_timestamp", "ingest_batch_id",
        "attack_type", "true_attack",
        "probability", "severity", "predicted_label", "alert_state", "model_version",
        "tower_site_id", "tower_site_name", "tower_lat", "tower_lon", "correlation_score",
    ]

    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(meta_cols + [f"feature_{c}" for c in feature_cols])
    for r in rows:
        features = json.loads(r["features_json"]) if r.get("features_json") else {}
        writer.writerow(
            [r.get(c) if r.get(c) is not None else "" for c in meta_cols]
            + [features.get(c) if features.get(c) is not None else "" for c in feature_cols]
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition":
                 f'attachment; filename="syncguard_feedback_{stamp}.csv"'},
    )


@app.get("/feedback/summary", response_model=FeedbackSummary)
async def feedback_summary():
    """Label counts, plus alert precision over LABELED EVENTS ONLY.

    Two precision figures, because "alert" has two defensible meanings here:
      - `predicted_attack_precision` -- over labeled events the model called 'attack'
        (threshold 0.52, per-reading, no debouncing);
      - `hysteresis_alert_precision` -- over labeled events whose tower/session was in
        hysteresis state 'alerting' (3 consecutive above-threshold readings).

    Both come back with the `caveat` field attached, and both are null while their
    denominator is 0. Read PRECISION_CAVEAT above before quoting either number anywhere.
    """
    return FeedbackSummary(**app.state.event_store.feedback_summary(),
                           caveat=PRECISION_CAVEAT)


@app.get("/spatial/autocorrelation", response_model=AutocorrelationResponse)
async def spatial_autocorrelation():
    """Global Moran's I + per-tower Local Moran's I (LISA), computed fresh from the current
    scored_events state -- see api/spatial_stats.py for the full REAL/SIMULATED framing and
    SPATIAL_STATISTICS.md for the method writeup. Distinct from, and additive to, the
    hand-rolled live correlation in api/spatial.py and the offline SIMULATED-epicenter layer
    in build_spatial_simulation.py -- neither of those is touched by this endpoint."""
    latest = app.state.event_store.latest_severity_per_tower()
    result = compute_autocorrelation(app.state.towers, latest)
    return AutocorrelationResponse(**result.__dict__)


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
