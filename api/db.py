"""SQLite persistence for scored events. Deliberately plain stdlib sqlite3 (no ORM, no async
driver) -- dependency-light per the project's constraints, and the demo's write volume (one
row per scored event, at most tens per second during a fast replay) is nowhere near where
that would matter. A single connection is shared across the app and guarded by a lock, since
sqlite3 connections aren't safe for concurrent use across threads/tasks without one.
"""
import json
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timezone

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "syncguard.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scored_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,              -- 'api' | 'replay'
    run_id TEXT,
    scenario_id TEXT,
    attack_type TEXT,                  -- ground truth, only populated by replay (from the dataset)
    true_attack INTEGER,               -- ground truth label, only populated by replay
    probability REAL NOT NULL,
    severity REAL NOT NULL,
    predicted_label TEXT NOT NULL,
    model_version TEXT,
    tower_site_id TEXT,                -- SIMULATED attribution for replay (round-robin); caller-supplied for direct /score calls
    tower_site_name TEXT,
    tower_lat REAL,
    tower_lon REAL,
    correlation_score REAL,
    features_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_scored_events_created_at ON scored_events(created_at);
CREATE INDEX IF NOT EXISTS idx_scored_events_tower ON scored_events(tower_site_id, created_at);
"""


class EventStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def insert_event(self, event: dict) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO scored_events
                   (created_at, source, run_id, scenario_id, attack_type, true_attack,
                    probability, severity, predicted_label, model_version,
                    tower_site_id, tower_site_name, tower_lat, tower_lon,
                    correlation_score, features_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    event.get("created_at") or datetime.now(timezone.utc).isoformat(),
                    event["source"],
                    event.get("run_id"),
                    event.get("scenario_id"),
                    event.get("attack_type"),
                    event.get("true_attack"),
                    event["probability"],
                    event["severity"],
                    event["predicted_label"],
                    event.get("model_version"),
                    event.get("tower_site_id"),
                    event.get("tower_site_name"),
                    event.get("tower_lat"),
                    event.get("tower_lon"),
                    event.get("correlation_score"),
                    json.dumps(event.get("features")) if event.get("features") is not None else None,
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    def recent_events(self, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM scored_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def events_for_tower_since(self, tower_site_id: str, since_iso: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM scored_events WHERE tower_site_id = ? AND created_at >= ? "
                "ORDER BY id DESC",
                (tower_site_id, since_iso),
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_events_all_towers_since(self, since_iso: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM scored_events WHERE tower_site_id IS NOT NULL AND created_at >= ? "
                "ORDER BY id DESC",
                (since_iso,),
            ).fetchall()
        return [dict(r) for r in rows]

    def latest_severity_per_tower(self) -> dict[str, dict]:
        """Most recent scored event for each tower that has ever been attributed one --
        what the dashboard map renders when idle (no time-window filtering)."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT s.* FROM scored_events s
                   JOIN (SELECT tower_site_id, MAX(id) AS max_id FROM scored_events
                         WHERE tower_site_id IS NOT NULL GROUP BY tower_site_id) latest
                   ON s.tower_site_id = latest.tower_site_id AND s.id = latest.max_id"""
            ).fetchall()
        return {r["tower_site_id"]: dict(r) for r in rows}

    def event_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM scored_events").fetchone()[0]

    def close(self):
        with self._lock:
            self._conn.close()
