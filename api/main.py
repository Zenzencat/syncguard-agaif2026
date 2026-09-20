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
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, JSONResponse, PlainTextResponse,
                               RedirectResponse, Response, StreamingResponse)

from api.schemas import (TelemetryInput, ScoreResponse, HealthResponse, AutocorrelationResponse,
                         ExplainResponse, IngestBatch, IngestResponse, FeedbackRequest,
                         FeedbackRecord, FeedbackSummary, IncidentsResponse)
from api.model_service import ModelService, ModelNotFoundError
from api.db import EventStore
from api.spatial import load_towers, TowerAttributor, EpicenterWeightedAttributor, LiveCorrelationEngine
from api.exposure import attach_exposure, rank_priority, records_with_nulls
from api.incidents import build_incidents, INCIDENT_WINDOW_SECONDS, INCIDENT_DISTANCE_KM
from api.spatial_stats import compute_autocorrelation
from api.replay import ReplayManager, EventBus, list_run_ids
from api.ingest import IngestService, UnknownTowerError
from api.auth import ApiKeyAuth, SESSION_COOKIE, API_KEY_HEADER
from api.observability import (METRICS, REQUEST_ID_HEADER, configure_logging,
                               new_request_id, request_id_var)
from api.plausibility import FeatureBaseline, BaselineUnavailable, PSI_MINOR, PSI_MAJOR
from api.evaluation import compute_evaluation

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
ASSET_DIR = REPO_ROOT / "assets"
PLOTLY_PATH = ASSET_DIR / "plotly-2.35.2.min.js"
BASEMAP_PATH = ASSET_DIR / "offline_basemap.geojson"
BASEMAP_FALLBACK_PATH = ASSET_DIR / "offline_basemap_natural_earth_fallback.geojson"
EVALUATION_DATASET_PATH = REPO_ROOT / "processed" / "syncguard_features.parquet"

log = configure_logging()

# Auth is OPT-IN: this is disabled unless SYNCGUARD_API_KEY is set. With it unset, every
# route behaves exactly as it did before Phase 3 and `docker compose up` needs no config.
# See api/auth.py, including why /stream/events uses a cookie rather than a query token.
AUTH = ApiKeyAuth()

# Number of most-recent scored events GET /drift samples by default.
DRIFT_DEFAULT_SAMPLE = 1000
DRIFT_MAX_SAMPLE = 20000


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model_service = ModelService()
    except ModelNotFoundError as e:
        # Let the app start (so /health reports the real problem instead of a crash-loop),
        # but every scoring endpoint will 503 until `make train` has been run.
        print(f"[startup] {e}")
        app.state.model_service = None

    try:
        app.state.baseline = FeatureBaseline()
    except BaselineUnavailable as e:
        # Not fatal: input warnings and /drift degrade to "unavailable", everything else
        # works. Regenerate with `python build_feature_baseline.py`.
        log.warning("feature baseline unavailable", extra={"error": str(e)})
        app.state.baseline = None

    app.state.event_store = EventStore()
    # Real tower records plus ESTIMATED nearby population (pop_1km / pop_2km, joined by
    # (lat, lon) -- never site_id). See api/exposure.py and PRIORITIZE_NOTES.md.
    towers = attach_exposure(load_towers())
    app.state.towers = towers
    app.state.tower_attributor = TowerAttributor(towers)
    # Named replay attribution modes -- see api/spatial.py::EpicenterWeightedAttributor and
    # api/replay.py::ReplayManager. round_robin is the API default; the dashboard's NOC tab
    # requests epicenter so its incident queue groups into a small number of localized
    # incidents instead of one spanning all 136 towers.
    app.state.replay_attributors = {
        "round_robin": app.state.tower_attributor,
        "epicenter": EpicenterWeightedAttributor(towers),
    }
    app.state.correlation_engine = LiveCorrelationEngine(towers, app.state.event_store)
    app.state.event_bus = EventBus()
    app.state.ingest_service = IngestService(
        app.state.model_service, app.state.event_store, towers,
        app.state.correlation_engine, app.state.event_bus,
        baseline=app.state.baseline,
    ) if app.state.model_service else None
    app.state.replay_manager = ReplayManager(
        app.state.model_service, app.state.event_store, app.state.replay_attributors,
        app.state.correlation_engine, app.state.event_bus,
    ) if app.state.model_service else None

    ms = app.state.model_service
    if ms is not None:
        app.state.evaluation = compute_evaluation(ms, EVALUATION_DATASET_PATH)
        METRICS.set_gauge("syncguard_model_info", 1, {
            "model_tag": ms.model_tag,
            "model_version": ms.model_version,
            "model_sha256_12": ms.model_sha256[:12],
            "decision_threshold": f"{ms.decision_threshold:.4f}",
            "n_features": str(len(ms.feature_cols)),
        })
    METRICS.set_gauge("syncguard_build_info", 1, {
        "service": "syncguard",
        "auth_enabled": str(AUTH.enabled).lower(),
        "baseline_loaded": str(app.state.baseline is not None).lower(),
    })
    log.info("startup complete", extra={
        "model_loaded": ms is not None,
        "model_tag": ms.model_tag if ms else None,
        "towers": len(app.state.towers),
        "auth_enabled": AUTH.enabled,          # never the key itself
        "auth_key_is_weak": AUTH.weak_key or None,
        "baseline_loaded": app.state.baseline is not None,
    })
    if AUTH.weak_key:
        log.warning("SYNCGUARD_API_KEY is shorter than the recommended minimum "
                    "(16 characters). Auth is active but the key is weak.")
    if not AUTH.enabled:
        log.info("SYNCGUARD_API_KEY is not set -- authentication is DISABLED and every "
                 "route is open. This is the default so the demo path needs no config.")

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
# CORS posture depends on whether auth is on, because the two cannot be configured the same
# way. A browser refuses to send credentials (the session cookie /stream/events needs) to an
# origin whose Access-Control-Allow-Origin is "*", so wildcard CORS and cookie auth are
# mutually exclusive -- see api/auth.py's SSE section.
#
#   auth off  -> wildcard, no credentials. Exactly the pre-Phase-3 behaviour.
#   auth on   -> only the origins named in SYNCGUARD_CORS_ORIGINS (comma-separated), with
#                credentials allowed. The default is an empty list, which is correct for the
#                normal case: the dashboard is served by this same service, so its requests
#                are same-origin and never hit CORS at all.
_CORS_ORIGINS = [o.strip() for o in os.environ.get("SYNCGUARD_CORS_ORIGINS", "").split(",")
                 if o.strip()]
if AUTH.enabled:
    app.add_middleware(
        CORSMiddleware, allow_origins=_CORS_ORIGINS, allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )


def _route_template(request: Request) -> str:
    """The matched route's path template ("/events/{event_id}/explain"), not the concrete
    path. Metrics labelled with concrete paths would grow one time series per event id and
    blow up the registry's cardinality."""
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


@app.middleware("http")
async def observability_and_auth(request: Request, call_next):
    """Assigns a request id, enforces auth (when enabled), records metrics, and emits one
    structured log line per request.

    Nothing here logs headers, cookies or bodies, so a credential cannot reach a log line.
    """
    incoming = request.headers.get(REQUEST_ID_HEADER)
    request_id = (incoming or new_request_id())[:64]
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    path = request.url.path

    try:
        if not AUTH.authorize(path, request.headers, request.cookies):
            presented = (AUTH.extract_key(request.headers) is not None
                         or SESSION_COOKIE in request.cookies)
            reason = "invalid_credential" if presented else "missing_credential"
            METRICS.inc("syncguard_auth_failures_total", {"reason": reason})
            # Logged without any part of the credential -- only whether one was presented.
            log.warning("auth rejected", extra={"path": path, "method": request.method,
                                                "reason": reason})
            response = JSONResponse(
                status_code=401,
                content={"detail": f"API key required. Send it as the {API_KEY_HEADER} "
                                   f"header, or as 'Authorization: Bearer <key>'. Browsers "
                                   f"using the live event stream should POST the key to "
                                   f"/auth/session first -- see api/auth.py."},
                headers={"WWW-Authenticate": f'ApiKey realm="syncguard", header="{API_KEY_HEADER}"'},
            )
        else:
            response = await call_next(request)

        duration = time.perf_counter() - started
        template = _route_template(request)
        METRICS.inc("syncguard_requests_total", {
            "method": request.method, "path": template, "status": str(response.status_code),
        })
        METRICS.observe("syncguard_request_duration_seconds", duration, {
            "method": request.method, "path": template,
        })
        # /metrics scrapes and the SSE stream would otherwise dominate the log at INFO.
        if path not in ("/metrics", "/stream/events"):
            log.info("request", extra={
                "method": request.method, "path": path, "route": template,
                "status": response.status_code, "duration_ms": round(duration * 1000, 2),
            })
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    except Exception:
        duration = time.perf_counter() - started
        METRICS.inc("syncguard_requests_total", {
            "method": request.method, "path": _route_template(request), "status": "500",
        })
        log.exception("unhandled error", extra={
            "method": request.method, "path": path,
            "duration_ms": round(duration * 1000, 2),
        })
        raise
    finally:
        request_id_var.reset(token)


def _input_warnings(app_: FastAPI, features: dict) -> list[dict]:
    """Plausibility check against the training baseline. Advisory: the caller scores the
    features regardless of what comes back. Returns [] when no baseline is loaded, which is
    indistinguishable in the response from "nothing was out of range" -- GET /health's
    baseline_loaded is how a caller tells those apart."""
    baseline = getattr(app_.state, "baseline", None)
    if baseline is None:
        return []
    warnings = baseline.check(features)
    for w in warnings:
        METRICS.inc("syncguard_input_warnings_total",
                    {"feature": w["feature"], "direction": w["direction"]})
    return warnings


def _require_model(app_: FastAPI) -> ModelService:
    if app_.state.model_service is None:
        raise HTTPException(503, "No trained model loaded -- run `make train` first, then restart the service.")
    return app_.state.model_service


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness plus the model version tag. Always exempt from authentication so a container
    healthcheck, load balancer or uptime probe needs no secret distributed to it -- it
    reports no event data, no scores and no telemetry."""
    ms = app.state.model_service
    baseline = getattr(app.state, "baseline", None)
    return HealthResponse(
        status="ok" if ms else "degraded",
        model_loaded=ms is not None,
        model_version=ms.model_version if ms else None,
        towers_loaded=len(app.state.towers),
        replay_running=bool(app.state.replay_manager and app.state.replay_manager.status["status"] == "running"),
        model_tag=ms.model_tag if ms else None,
        model_info=ms.version_info if ms else None,
        auth_required=AUTH.enabled,
        auth_key_is_weak=AUTH.weak_key if AUTH.enabled else None,
        baseline_loaded=baseline is not None,
        baseline_generated_at=baseline.generated_at if baseline else None,
    )


@app.post("/auth/session")
async def auth_session(request: Request):
    """Exchange an API key for a short-lived HttpOnly session cookie.

    This exists for one reason: the dashboard's live feed uses EventSource, which cannot send
    custom headers, and putting the key in the stream URL would leak it into access logs,
    proxy logs, browser history and Referer headers. The key is sent here once, in a header or
    a JSON body, and comes back as an opaque random token in a cookie the browser attaches
    automatically. api/auth.py documents the full tradeoff, including that sessions are
    in-process and do not survive a restart.

    With auth disabled this is a no-op that reports as much, so the dashboard can call it
    unconditionally.
    """
    if not AUTH.enabled:
        return {"auth_required": False, "authenticated": True,
                "detail": "Authentication is disabled (SYNCGUARD_API_KEY is not set)."}

    presented = AUTH.extract_key(request.headers)
    if not presented:
        try:
            body = await request.json()
            if isinstance(body, dict):
                presented = body.get("api_key")
        except Exception:
            presented = None

    if not AUTH.check_key(presented):
        METRICS.inc("syncguard_auth_failures_total", {"reason": "session_bad_key"})
        log.warning("auth session rejected", extra={"path": "/auth/session"})
        raise HTTPException(401, "Invalid API key.")

    token, ttl = AUTH.sessions.issue()
    response = JSONResponse({"auth_required": True, "authenticated": True,
                             "expires_in_seconds": ttl})
    response.set_cookie(
        SESSION_COOKIE, token,
        max_age=ttl,
        httponly=True,       # unreadable from JavaScript
        samesite="strict",   # a third-party page cannot ride the session
        secure=AUTH.cookie_secure,  # off by default for the plain-HTTP localhost demo;
                                    # set SYNCGUARD_COOKIE_SECURE=1 behind TLS
        path="/",
    )
    log.info("auth session issued", extra={"ttl_seconds": ttl})
    return response


@app.post("/auth/logout")
async def auth_logout(request: Request):
    """Revoke this browser's session cookie."""
    AUTH.sessions.revoke(request.cookies.get(SESSION_COOKIE))
    response = JSONResponse({"authenticated": False})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@app.get("/metrics")
async def metrics():
    """Prometheus text exposition format.

    Requires the API key when auth is on: it carries no secrets, but request counts, latency
    and alert volumes are operational information. Unlike EventSource, a Prometheus scraper
    can send a header, so there is no reason to exempt it.
    """
    return PlainTextResponse(METRICS.render(),
                             media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/drift")
async def drift(limit: int = Query(default=DRIFT_DEFAULT_SAMPLE, ge=1, le=DRIFT_MAX_SAMPLE),
                source: str | None = Query(default=None,
                                           description="Restrict to one event source: 'ingest', 'replay' or 'api'")):
    """Per-feature PSI of recently scored inputs against the training baseline.

    **The baseline is Jammertest 2024: a Norwegian test range, one receiver, September 2024.
    Real ASEAN telecom input is EXPECTED to drift from it.** A high PSI here is therefore not
    by itself evidence of a fault -- it is the expected reading for genuinely different
    infrastructure, and it is exactly as consistent with "the model is being asked about data
    it was never trained on" as with "something changed". Treat it as a prompt to look, never
    as a verdict. See ASSUMPTIONS_PRODUCTION.md.

    PSI is computed over the baseline's own deciles. Features whose baseline is near-constant
    (fixType) have no usable binning and are reported as skipped rather than silently
    dropped. Fewer than 100 live samples for a feature returns no PSI for it, because PSI over
    10 bins is too noisy to mean anything at that size.
    """
    baseline = getattr(app.state, "baseline", None)
    if baseline is None:
        raise HTTPException(503, "No feature baseline loaded. Run "
                                 "`python build_feature_baseline.py` and restart.")

    rows = app.state.event_store.recent_feature_rows(limit=limit, source=source)
    per_feature, skipped = [], []
    for name in baseline.features:
        values = [r.get(name) for r in rows if r.get(name) is not None]
        result = baseline.psi(name, values)
        if result is None:
            skipped.append({
                "feature": name,
                "reason": ("baseline near-constant, no usable binning"
                           if baseline.features[name].get("psi_degenerate")
                           else f"fewer than 100 live samples ({len(values)})"),
            })
        else:
            per_feature.append(result)

    per_feature.sort(key=lambda r: -r["psi"])
    n_major = sum(1 for r in per_feature if r["severity"] == "major")
    n_minor = sum(1 for r in per_feature if r["severity"] == "minor")

    return {
        "computable": bool(per_feature),
        "n_events_sampled": len(rows),
        "source_filter": source,
        "method": "PSI over the training baseline's deciles; see api/plausibility.py",
        "thresholds": {"minor": PSI_MINOR, "major": PSI_MAJOR,
                       "note": "Conventional PSI cut points, not values derived from this "
                               "project's data."},
        "baseline": {
            "generated_at": baseline.generated_at,
            "n_rows": baseline.n_rows,
            "n_runs": baseline.n_runs,
            "provenance": baseline.provenance,
        },
        "flagged": n_major > 0 or n_minor > 0,
        "n_major": n_major,
        "n_minor": n_minor,
        "caveat": (
            "EXPECTED TO DRIFT. The baseline is Norwegian test-range data (Jammertest 2024, "
            "one receiver). Real ASEAN telecom input is expected to fall outside it, so a "
            "high PSI here is not evidence of a fault. See ASSUMPTIONS_PRODUCTION.md."
        ),
        "per_feature": per_feature,
        "skipped": skipped,
    }


@app.post("/score", response_model=ScoreResponse)
async def score(telemetry: TelemetryInput):
    ms = _require_model(app)
    features = telemetry.model_dump(exclude={"receiver_id", "tower_site_id"})
    # Advisory only, and computed before scoring purely so the two travel together in the
    # response -- it does not gate the call below. See api/plausibility.py.
    input_warnings = _input_warnings(app, features)
    result = ms.score(features)
    METRICS.inc("syncguard_scores_total",
                {"path": "/score", "predicted_label": result["predicted_label"]})
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

    METRICS.inc("syncguard_events_persisted_total", {"source": "api"})
    return ScoreResponse(**result, event_id=event_id, tower=tower, correlation=corr,
                         top_features=top_features, input_warnings=input_warnings)


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
        result = await app.state.ingest_service.ingest(batch)
    except UnknownTowerError as e:
        raise HTTPException(422, str(e))

    for outcome, count in (("scored", result["scored"]),
                           ("duplicate", result["duplicates"]),
                           ("out_of_order", result["out_of_order"])):
        if count:
            METRICS.inc("syncguard_ingest_observations_total", {"outcome": outcome}, count)
    log.info("ingest batch", extra={
        "batch_id": batch.batch_id, "received": result["received"],
        "scored": result["scored"], "duplicates": result["duplicates"],
        "out_of_order": result["out_of_order"], "alerts": result["alerts"],
        "input_warnings": result["input_warning_count"],
    })
    return IngestResponse(**result)


@app.get("/towers")
async def towers():
    """The real tower records. `pop_1km` / `pop_2km` are ESTIMATED people within 1 km / 2 km
    (Meta Data for Good / CIESIN HRSL, ~2020, CC BY 4.0) -- an exposure proxy, not people
    served -- and are null where no estimate is available. They overlap between neighbouring
    towers and must not be summed."""
    return records_with_nulls(app.state.towers)


@app.get("/events")
async def events(limit: int = Query(default=200, le=2000)):
    return app.state.event_store.recent_events(limit=limit)


@app.get("/events/map")
async def events_map():
    """Latest scored event per tower -- what the dashboard renders as the current map state."""
    return list(app.state.event_store.latest_severity_per_tower().values())


@app.get("/incidents", response_model=IncidentsResponse)
async def incidents(limit: int = Query(default=2000, le=2000)):
    """The NOC tab's incident queue: flagged events grouped into incidents. See
    api/incidents.py for the grouping rule -- time AND distance chaining, not a real
    spatial-clustering algorithm."""
    events_rows = app.state.event_store.recent_events(limit=limit)
    event_ids = [e["id"] for e in events_rows]
    feedback_by_event = app.state.event_store.feedback_for_events(event_ids)
    pop_2km_by_tower = app.state.towers.set_index("tower_key")["pop_2km"].dropna().to_dict()
    grouped = build_incidents(events_rows, feedback_by_event, pop_2km_by_tower)
    return IncidentsResponse(
        incidents=grouped,
        window_seconds=INCIDENT_WINDOW_SECONDS,
        distance_km=INCIDENT_DISTANCE_KM,
        method_note=(
            "Consecutive alerting events (hysteresis-confirmed, or threshold-flagged for "
            "stateless /score calls) are chained into one incident when no more than "
            f"{INCIDENT_WINDOW_SECONDS}s apart by created_at (server insert time, 'recording "
            f"time' during replay) AND no more than {INCIDENT_DISTANCE_KM:g}km apart by real "
            "tower distance, never across two different replay runs. A demo heuristic, not a "
            "real spatial-clustering algorithm. Status comes from existing analyst "
            "confirm/dismiss labels on the incident's peak-severity event; nothing here is "
            "learned. See api/incidents.py."
        ),
    )


@app.get("/priority")
async def priority():
    """Which flagged towers to look at first: gate, then rank by estimated nearby population.

    Gate: severity decides which towers are flagged -- a tower is in the list only if its
    latest event is in hysteresis state 'alerting' (or, for events that carry no hysteresis
    state such as POST /score, its per-reading verdict is 'attack', i.e. probability >= 0.52).
    Rank: flagged towers are ordered by ESTIMATED people within 2 km (`pop_2km`), descending;
    severity is a secondary field. This is deliberately NOT severity x population, which would
    just be a population ranking -- see PRIORITIZE_NOTES.md.

    Populations are an exposure proxy (~2020 estimates), not people served, and they overlap
    between towers -- the response carries no total, and none should be computed from it.
    """
    latest = app.state.event_store.latest_severity_per_tower()
    ms = app.state.model_service
    threshold = ms.decision_threshold if ms is not None else 0.52
    return rank_priority(latest, app.state.towers, threshold=threshold)


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
    METRICS.inc("syncguard_feedback_total", {
        "label": record["label"],
        "relabel": str(record["previous_label"] is not None).lower(),
    })
    log.info("analyst feedback", extra={
        "event_id": event_id, "label": record["label"],
        "previous_label": record["previous_label"], "revision": record["revision"],
    })
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
async def replay_start(run_id: str | None = None, speed: float = 10.0, attribution: str | None = None):
    """`attribution`: 'round_robin' (default) | 'epicenter' -- which SIMULATED tower
    attribution mechanism replay uses for this run. See api/spatial.py."""
    if app.state.replay_manager is None:
        raise HTTPException(503, "No trained model loaded -- run `make train` first, then restart the service.")
    try:
        return app.state.replay_manager.start(run_id, speed, attribution)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


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


@app.get("/evaluation")
async def evaluation():
    """Held-out evidence recomputed from the exact shipped artifact; never retrains."""
    if app.state.model_service is None:
        raise HTTPException(503, "No trained model loaded.")
    return app.state.evaluation


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


@app.get("/assets/plotly-2.35.2.min.js", include_in_schema=False)
async def dashboard_plotly():
    """Vendored Plotly build: the dashboard must render with no internet connection."""
    if not PLOTLY_PATH.exists():
        raise HTTPException(404, "vendored Plotly asset not found")
    return FileResponse(PLOTLY_PATH, media_type="text/javascript")


@app.get("/assets/offline_basemap.geojson", include_in_schema=False)
async def dashboard_basemap():
    """OSM vector basemap (coastline, water, named rivers, major roads, built-up areas, place
    labels) for the local tower-map bounding box -- built once, offline, by
    tools/build_basemap.py. See assets/OFFLINE_BASEMAP_ATTRIBUTION.md."""
    if not BASEMAP_PATH.exists():
        raise HTTPException(404, "offline basemap asset not found")
    return FileResponse(BASEMAP_PATH, media_type="application/geo+json")


@app.get("/assets/offline_basemap_natural_earth_fallback.geojson", include_in_schema=False)
async def dashboard_basemap_fallback():
    """Fallback only: the original small Natural Earth land/coastline/rivers layer, used by
    the dashboard only if the OSM basemap above fails to load."""
    if not BASEMAP_FALLBACK_PATH.exists():
        raise HTTPException(404, "fallback basemap asset not found")
    return FileResponse(BASEMAP_FALLBACK_PATH, media_type="application/geo+json")
