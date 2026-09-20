"""POST /ingest -- batch ingestion of per-tower observation windows.

This is the adapter that lets something other than the bundled replay feed the detector: an
edge agent, a collector, a test harness. It deliberately does NOT introduce a second scoring
path. Every ingested observation goes through the same ModelService.score() call that POST
/score uses, against the same loaded artifact, the same 23 features, the same 0.52 decision
threshold, and the same single-threaded predict_proba. Nothing about the model changed to
make ingestion possible.

What ingestion adds on top of /score:
  - per-tower alert hysteresis (api/hysteresis.py's TowerHysteresisRegistry), which /score
    deliberately does not have -- see that module for why the two paths differ;
  - deduplication on (tower_id, observation timestamp);
  - out-of-order detection against what has already been accepted for that tower;
  - the same SQLite persistence and the same SSE publish replay uses, so ingested events land
    on the dashboard map, in the event log, and in the live spatial-statistics state
    (api/spatial_stats.py reads the same scored_events rows) exactly like replayed ones.

REAL / SIMULATED framing, unchanged by this module:
  - REAL: the model, its scoring, the 136 tower coordinates, the haversine distances, and the
    distance-weighted correlation over whatever severities are actually in the store.
  - CALLER-SUPPLIED: the tower each observation claims to come from. For ingestion this is
    not simulated the way replay's round-robin attribution is -- the caller states it. It is
    also not verified: this service cannot check that the submitted features were really
    observed at that tower. It is trusted input, not established fact.
  - NOT a claim of field validation. No real GNSS receiver at a real Telkomsel tower has ever
    fed this endpoint. See ASSUMPTIONS_PRODUCTION.md (Phase 3) and INGESTION_CONTRACT.md.

Transport: HTTP only. MQTT / Kafka style transports are plausible future work for a real edge
fleet and are explicitly NOT built here -- see INGESTION_CONTRACT.md's "Future work" section.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pandas as pd

from api.hysteresis import TowerHysteresisRegistry
from api.observability import METRICS

# SHAP TreeExplainer costs ~45-55ms/row (exact Tree SHAP over 300 trees -- SHAP_EXPLAINABILITY.md).
# Same tradeoff replay makes with LIVE_EXPLAIN_MAX_SPEED, applied to batch size instead of
# replay speed: small batches get explanations inline (a 20-row batch pays ~1s of SHAP, which
# is acceptable for an interactive submission), larger ones skip it so a 500-row batch isn't
# dominated by explanation cost. Skipped rows are not permanently unexplainable --
# GET /events/{id}/explain computes them lazily from the row's own stored feature values.
INGEST_LIVE_EXPLAIN_MAX_BATCH = 20

# How many observations to score before yielding control back to the asyncio event loop.
# Scoring is synchronous CPU work on the single-threaded path (a hard project constraint), so
# a large batch would otherwise stall the SSE stream and every other request for its whole
# duration. This does not make scoring concurrent -- it just stops one batch from starving
# the loop.
YIELD_EVERY = 25


def iso_utc(ts: datetime) -> str:
    """Single canonical string form for observation timestamps.

    Everything downstream (the dedup lookup, the out-of-order comparison in
    EventStore.max_ingested_obs_timestamp) compares these as plain strings, which is only
    correct if every stored value has identical formatting and the same timezone. Forcing
    UTC + a fixed-width, offset-less format here is what makes lexicographic order equal
    chronological order.
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


class UnknownTowerError(ValueError):
    """Raised for a tower_id that is not in the real tower table. Surfaces as a 422."""

    def __init__(self, tower_id: str):
        self.tower_id = tower_id
        super().__init__(
            f"Unknown tower_id: {tower_id!r}. Must be a tower_key or site_id from GET /towers."
        )


class IngestService:
    def __init__(self, model_service, event_store, towers: pd.DataFrame, correlation_engine,
                 event_bus=None, hysteresis: TowerHysteresisRegistry | None = None,
                 baseline=None):
        self._model = model_service
        # Training-distribution baseline for input plausibility warnings. Optional: when it
        # is None, warnings are simply absent and scoring is unaffected -- the check never
        # gates anything. See api/plausibility.py.
        self._baseline = baseline
        self._store = event_store
        self._towers = towers
        self._correlation = correlation_engine
        self._bus = event_bus
        self.hysteresis = hysteresis or TowerHysteresisRegistry()
        # tower_key -> resolved dict, and site_id -> the same. tower_key wins on a collision,
        # matching the resolution order POST /score already uses (see main.py) -- 16 of the 136
        # real towers share the literal site_id "tbg" (api/spatial.py::load_towers), so a
        # caller matching on raw site_id there deterministically gets the first such row.
        # Materialized as plain Python values at startup, not kept as pandas rows: this is on
        # the per-observation hot path and a DataFrame lookup per observation would be pure
        # overhead.
        self._by_key: dict[str, dict] = {}
        self._by_site_id: dict[str, dict] = {}
        for r in towers.to_dict(orient="records"):
            resolved = {
                "site_id": r["tower_key"],  # the unique internal key -- see load_towers()
                "site_name": r["site_name"],
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
            }
            self._by_key[r["tower_key"]] = resolved
            self._by_site_id.setdefault(r["site_id"], resolved)

    def resolve_tower(self, tower_id: str) -> dict:
        resolved = self._by_key.get(tower_id)
        if resolved is None:
            resolved = self._by_site_id.get(tower_id)
        if resolved is None:
            raise UnknownTowerError(tower_id)
        return resolved

    async def ingest(self, batch) -> dict:
        """Score, persist and publish one batch. `batch` is a schemas.IngestBatch.

        Every tower_id is resolved up front, so a batch containing an unknown tower is
        rejected whole (422) rather than half-applied. After that point the batch is processed
        in ascending timestamp order regardless of the order it was submitted in, so per-tower
        hysteresis always sees a monotonic sequence.
        """
        observations = list(batch.observations)
        resolved = [self.resolve_tower(o.tower_id) for o in observations]  # raises -> 422

        # Count submission-order inversions before sorting, so the response can state plainly
        # that the batch was reordered rather than silently fixing it.
        reordered = 0
        seen_max: dict[str, datetime] = {}
        for obs in observations:
            prev = seen_max.get(obs.tower_id)
            if prev is not None and obs.timestamp < prev:
                reordered += 1
            else:
                seen_max[obs.tower_id] = obs.timestamp

        pairs = sorted(zip(observations, resolved), key=lambda p: p[0].timestamp)

        # Per-tower high-water mark as of the START of this batch. Out-of-order is judged
        # against already-accepted data, not against the batch's own internal ordering (which
        # the sort above has already normalized).
        baseline: dict[str, str | None] = {}
        for _, tower in pairs:
            key = tower["site_id"]
            if key not in baseline:
                baseline[key] = self._store.max_ingested_obs_timestamp(key)

        live_explain = len(observations) <= INGEST_LIVE_EXPLAIN_MAX_BATCH
        results, n_scored, n_dup, n_ooo, n_alerts = [], 0, 0, 0, 0
        n_warnings = 0

        for i, (obs, tower) in enumerate(pairs):
            if i and i % YIELD_EVERY == 0:
                await asyncio.sleep(0)  # don't starve the event loop -- see YIELD_EVERY

            tower_key = tower["site_id"]
            obs_iso = iso_utc(obs.timestamp)

            existing = self._store.find_ingested_event(tower_key, obs_iso)
            if existing is not None:
                n_dup += 1
                results.append({
                    "tower_id": tower_key,
                    "timestamp": obs.timestamp,
                    "event_id": existing["id"],
                    "probability": existing["probability"],
                    "severity": existing["severity"],
                    "predicted_label": existing["predicted_label"],
                    "alert_state": None,  # hysteresis is not re-run for a duplicate
                    "correlation_score": existing["correlation_score"],
                    "duplicate": True,
                    "out_of_order": False,
                    "explained": existing.get("top_features") is not None,
                    "input_warnings": [],  # not re-checked; the original was checked once
                })
                continue

            features = obs.model_dump(exclude={"tower_id", "timestamp"})
            # Advisory only -- never gates the scoring call below.
            warnings = self._baseline.check(features) if self._baseline is not None else []
            for w in warnings:
                METRICS.inc("syncguard_input_warnings_total",
                            {"feature": w["feature"], "direction": w["direction"]})
            n_warnings += len(warnings)

            result = self._model.score(features)  # same path as POST /score
            METRICS.inc("syncguard_scores_total",
                        {"path": "/ingest", "predicted_label": result["predicted_label"]})
            top_features = self._model.explain(features) if live_explain else None

            base = baseline.get(tower_key)
            out_of_order = base is not None and obs_iso < base
            if out_of_order:
                # A reading older than data already accepted for this tower is not the "next"
                # reading in that tower's sequence, so feeding it to the streak counter would
                # corrupt the state of a sensor that has since moved on. It is still scored
                # and persisted -- nothing is dropped -- it just carries the tower's current
                # alert_state rather than advancing it. See INGESTION_CONTRACT.md.
                n_ooo += 1
                alert_state = self.hysteresis.state(tower_key)
            else:
                alert_state = self.hysteresis.update(
                    tower_key, result["predicted_label"] == "attack"
                )
                baseline[tower_key] = obs_iso

            corr = self._correlation.correlate(tower_key)
            created_at = datetime.now(timezone.utc).isoformat()

            event_id = self._store.insert_event({
                "created_at": created_at,
                "source": "ingest",
                "probability": result["probability"],
                "severity": result["severity"],
                "predicted_label": result["predicted_label"],
                "model_version": result["model_version"],
                "tower_site_id": tower_key,
                "tower_site_name": tower["site_name"],
                "tower_lat": tower["lat"],
                "tower_lon": tower["lon"],
                "correlation_score": corr.correlation_score,
                "features": features,
                "top_features": top_features,
                "alert_state": alert_state,
                "obs_timestamp": obs_iso,
                "ingest_batch_id": batch.batch_id,
            })

            if self._bus is not None:
                # Same envelope shape replay publishes, so the dashboard's existing SSE
                # handler renders ingested events with no client-side special-casing.
                # attack_type/true_attack are absent on purpose: ingested data carries no
                # ground truth, unlike a replayed row (see api/replay.py).
                self._bus.publish({
                    "event_id": event_id,
                    "created_at": created_at,
                    "source": "ingest",
                    "obs_timestamp": obs_iso,
                    "batch_id": batch.batch_id,
                    "attack_type": None,
                    "true_attack": None,
                    **result,
                    "tower": tower,
                    "correlation_score": corr.correlation_score,
                    "top_features": top_features,
                    "alert_state": alert_state,
                    "out_of_order": out_of_order,
                })

            n_scored += 1
            METRICS.inc("syncguard_events_persisted_total", {"source": "ingest"})
            if alert_state == "alerting":
                n_alerts += 1
                METRICS.inc("syncguard_alerts_total", {"source": "ingest"})

            results.append({
                "tower_id": tower_key,
                "timestamp": obs.timestamp,
                "event_id": event_id,
                "probability": result["probability"],
                "severity": result["severity"],
                "predicted_label": result["predicted_label"],
                "alert_state": alert_state,
                "correlation_score": corr.correlation_score,
                "duplicate": False,
                "out_of_order": out_of_order,
                "explained": top_features is not None,
                "input_warnings": warnings,
            })

        return {
            "batch_id": batch.batch_id,
            "received": len(observations),
            "scored": n_scored,
            "duplicates": n_dup,
            "out_of_order": n_ooo,
            "reordered_in_batch": reordered,
            "alerts": n_alerts,
            "live_explain": live_explain,
            "input_warning_count": n_warnings,
            "model_tag": getattr(self._model, "model_tag", None),
            "results": results,
        }
