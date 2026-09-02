"""Simulated live-feed mode: replays syncguard_features.parquet row-by-row, in real
recording-time order, against the scoring pipeline -- so the demo shows "live" detection
instead of static batch scoring. Runs as a single asyncio background task inside the FastAPI
process (see main.py) -- no separate service, no message broker, no polling loop outside of
the asyncio event loop itself.

Ground truth (attack_type, true attack label) travels with each replayed row because it's in
the dataset -- it's stored alongside the model's prediction (source='replay') purely so the
dashboard can show "did the detector get this one right" during the demo. A row scored via
POST /score has no such ground truth, which is why db.py's true_attack/attack_type columns
are nullable.
"""
from __future__ import annotations
import asyncio
import functools
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "processed" / "syncguard_features.parquet"
DEFAULT_SCENARIO = {"attack_type": "Spoofing", "scenario_id": "2.1.1"}  # same scenario demo_prediction_visualization.py uses
MIN_SLEEP_S = 0.02
MAX_ROW_GAP_S = 5.0  # clamp real inter-row gaps (e.g. across a data dropout) so replay doesn't stall

# SHAP TreeExplainer costs ~45-55ms/row (exact Tree SHAP over 300 trees, confirmed -- no
# faster mode without sacrificing exactness, see SHAP_EXPLAINABILITY.md) -- computing it
# inline on every replayed row would throttle replay to ~20 rows/sec regardless of the
# requested speed multiplier. LIVE_EXPLAIN_MAX_SPEED is the speed threshold below which
# that cost is absorbed into the replay's own natural per-row cadence anyway (at speed=20,
# the ~1Hz dataset's per-row gap is already ~50ms -- almost exactly the SHAP cost -- so
# inline explanation is nearly free there) and above which it's skipped so replay stays at
# full requested speed. Skipped rows are not permanently unexplainable: GET
# /events/{id}/explain computes SHAP on-demand from the event's own stored feature values,
# lazily, the first time anyone actually asks -- see SHAP_EXPLAINABILITY.md's full writeup
# of this tradeoff.
LIVE_EXPLAIN_MAX_SPEED = 20.0


@functools.lru_cache(maxsize=1)
def _load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["real_time"] = pd.to_datetime(df["real_time"])
    return df


def list_run_ids() -> list[dict]:
    df = _load_dataset()
    g = df.groupby("run_id").agg(
        attack_type=("attack_type", "first"),
        scenario_id=("scenario_id", "first"),
        rover_state=("rover_state", "first"),
        n_rows=("run_id", "size"),
    ).reset_index()
    return g.to_dict(orient="records")


def _resolve_run_id(run_id: str | None) -> str:
    df = _load_dataset()
    if run_id:
        if run_id not in df["run_id"].unique():
            raise ValueError(f"Unknown run_id: {run_id}")
        return run_id
    match = df.loc[
        (df["attack_type"] == DEFAULT_SCENARIO["attack_type"]) &
        (df["scenario_id"] == DEFAULT_SCENARIO["scenario_id"]),
        "run_id",
    ].unique()
    return match[0] if len(match) else df["run_id"].iloc[0]


class EventBus:
    """Minimal async pub/sub for the SSE endpoint -- one Queue per connected client."""

    def __init__(self):
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers.discard(q)

    def publish(self, event: dict):
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # slow consumer -- drop rather than block replay


class ReplayManager:
    def __init__(self, model_service, event_store, tower_attributor, correlation_engine, bus: EventBus):
        self._model = model_service
        self._store = event_store
        self._attributor = tower_attributor
        self._correlation = correlation_engine
        self._bus = bus
        self._task: asyncio.Task | None = None
        self._status = "idle"
        self._run_id: str | None = None
        self._speed = 1.0
        self._rows_replayed = 0
        self._total_rows = 0
        self._error: str | None = None
        self._live_explain = True

    @property
    def status(self) -> dict:
        return {
            "status": self._status,
            "run_id": self._run_id,
            "speed": self._speed,
            "rows_replayed": self._rows_replayed,
            "total_rows": self._total_rows,
            "error": self._error,
            "live_explain": self._live_explain,
        }

    def start(self, run_id: str | None, speed: float):
        if self._task and not self._task.done():
            raise RuntimeError("Replay already running -- call /replay/stop first.")
        resolved_run_id = _resolve_run_id(run_id)
        self._run_id, self._speed = resolved_run_id, max(speed, 0.1)
        self._rows_replayed = 0
        self._error = None
        self._live_explain = self._speed <= LIVE_EXPLAIN_MAX_SPEED
        self._status = "running"
        self._task = asyncio.create_task(self._run(resolved_run_id, self._speed))
        return self.status

    def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
        self._status = "stopped"
        return self.status

    async def _run(self, run_id: str, speed: float):
        df = _load_dataset()
        rows = df[df["run_id"] == run_id].sort_values("real_time").reset_index(drop=True)
        self._total_rows = len(rows)
        prev_time = None
        try:
            for _, row in rows.iterrows():
                features = {c: (None if pd.isna(row[c]) else float(row[c]))
                            for c in self._model.feature_cols}
                result = self._model.score(features)
                top_features = self._model.explain(features) if self._live_explain else None

                tower = self._attributor.next_tower()
                corr = self._correlation.correlate(tower["site_id"])
                created_at = datetime.now(timezone.utc).isoformat()

                event_id = self._store.insert_event({
                    "created_at": created_at,
                    "source": "replay",
                    "run_id": run_id,
                    "scenario_id": row.get("scenario_id"),
                    "attack_type": row.get("attack_type"),
                    "true_attack": int(row["attack"]),
                    "probability": result["probability"],
                    "severity": result["severity"],
                    "predicted_label": result["predicted_label"],
                    "model_version": result["model_version"],
                    "tower_site_id": tower["site_id"],
                    "tower_site_name": tower["site_name"],
                    "tower_lat": tower["lat"],
                    "tower_lon": tower["lon"],
                    "correlation_score": corr.correlation_score,
                    "features": features,
                    "top_features": top_features,
                })

                self._bus.publish({
                    "event_id": event_id,
                    "created_at": created_at,
                    "run_id": run_id,
                    "scenario_id": row.get("scenario_id"),
                    "attack_type": row.get("attack_type"),
                    "true_attack": int(row["attack"]),
                    **result,
                    "tower": tower,
                    "correlation_score": corr.correlation_score,
                    "top_features": top_features,
                })

                self._rows_replayed += 1
                real_time = row["real_time"]
                dt = MIN_SLEEP_S
                if prev_time is not None:
                    gap = (real_time - prev_time).total_seconds()
                    dt = min(max(gap, 0.0), MAX_ROW_GAP_S) / speed
                    dt = max(dt, MIN_SLEEP_S)
                prev_time = real_time
                await asyncio.sleep(dt)
            self._status = "finished"
        except asyncio.CancelledError:
            self._status = "stopped"
            raise
        except Exception as e:
            self._status = "error"
            self._error = f"{type(e).__name__}: {e}"
            raise
