"""SQLite persistence for scored events. Deliberately plain stdlib sqlite3 (no ORM, no async
driver) -- dependency-light per the project's constraints, and the demo's write volume (one
row per scored event, at most tens per second during a fast replay) is nowhere near where
that would matter. A single connection is shared across the app and guarded by a lock, since
sqlite3 connections aren't safe for concurrent use across threads/tasks without one.
"""
import json
import os
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timezone

# SYNCGUARD_DB_PATH overrides the default location. Added so the test suite can point the
# whole app at a throwaway DB instead of the real one -- the container and normal local runs
# don't set it and get the committed default.
DEFAULT_DB_PATH = Path(os.environ.get("SYNCGUARD_DB_PATH")
                       or Path(__file__).resolve().parent.parent / "data" / "syncguard.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scored_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,              -- 'api' (POST /score) | 'replay' | 'ingest' (POST /ingest)
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

# Added after the initial schema above (SHAP explainability -- see SHAP_EXPLAINABILITY.md).
# NULL means "not explained yet" (e.g. scored during a fast replay, above the speed
# threshold where live SHAP is skipped -- see api/replay.py) -- distinct from an empty list,
# which json.dumps([]) would produce and this code never actually stores, since explain()
# always returns at least one feature. Applied via idempotent ALTER TABLE in
# EventStore.__init__ so existing local DBs from before this change don't break.
_MIGRATIONS = [
    "ALTER TABLE scored_events ADD COLUMN top_features_json TEXT",
    # Alert hysteresis (api/hysteresis.py, OPERATIONAL_METRICS.md) -- 'normal' or 'alerting',
    # computed per replay session over the recording's real temporal order, never per
    # simulated tower. NULL for events scored before this migration or via /score (hysteresis
    # doesn't apply to stateless ad-hoc calls -- see api/hysteresis.py's module docstring).
    "ALTER TABLE scored_events ADD COLUMN alert_state TEXT",
    # Ingestion adapter (POST /ingest, api/ingest.py, INGESTION_CONTRACT.md).
    # obs_timestamp is the caller-stated UTC observation time of the window -- distinct from
    # created_at, which is when this service scored it. NULL for /score and replay rows
    # (neither carries a caller-stated observation time), which is also why deduplication is
    # scoped to rows with source='ingest': a NULL obs_timestamp can never collide.
    "ALTER TABLE scored_events ADD COLUMN obs_timestamp TEXT",
    "ALTER TABLE scored_events ADD COLUMN ingest_batch_id TEXT",
]

_POST_MIGRATION_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_scored_events_ingest_dedup
    ON scored_events(tower_site_id, obs_timestamp) WHERE obs_timestamp IS NOT NULL;
"""

# Analyst confirm/dismiss (POST /events/{id}/feedback -- see FEEDBACK_LOOP.md).
#
# Two tables on purpose:
#   event_feedback      -- exactly one row per event: the CURRENT label. A relabel replaces
#                          it (upsert on the primary key), so "what does the analyst say
#                          about event N" has a single unambiguous answer, and the precision
#                          numbers in /feedback/summary can't double-count a re-labeled event.
#   event_feedback_log  -- append-only, every submission ever made, including superseded
#                          ones. A relabel is information, not noise: it records that an
#                          analyst changed their mind, which is exactly the kind of signal a
#                          future retraining effort would want to weigh (or exclude)
#                          deliberately rather than never know about.
#
# No FOREIGN KEY constraint: sqlite3 does not enforce them unless PRAGMA foreign_keys=ON is
# set per connection, so declaring one here would be decorative. Event existence is checked
# in the API layer instead, which is also where a useful 404 can be produced.
_FEEDBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_feedback (
    event_id   INTEGER PRIMARY KEY,
    label      TEXT NOT NULL,          -- 'confirmed' | 'dismissed'
    note       TEXT,
    analyst    TEXT,
    created_at TEXT NOT NULL,          -- when this event was FIRST labeled
    updated_at TEXT NOT NULL,          -- when the CURRENT label was set
    revision   INTEGER NOT NULL        -- 1 on first label, +1 per relabel
);
CREATE INDEX IF NOT EXISTS idx_event_feedback_label ON event_feedback(label);

CREATE TABLE IF NOT EXISTS event_feedback_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id       INTEGER NOT NULL,
    label          TEXT NOT NULL,
    previous_label TEXT,               -- NULL on the first label for an event
    note           TEXT,
    analyst        TEXT,
    created_at     TEXT NOT NULL,
    revision       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_event_feedback_log_event ON event_feedback_log(event_id, id);
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
            for stmt in _MIGRATIONS:
                try:
                    self._conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e):
                        raise  # a real migration failure, not just "already applied"
            # Runs after the ALTERs above, since it indexes a column they add.
            self._conn.executescript(_POST_MIGRATION_INDEXES)
            self._conn.executescript(_FEEDBACK_SCHEMA)
            self._conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        raw = d.pop("top_features_json", None)
        d["top_features"] = json.loads(raw) if raw is not None else None
        return d

    def insert_event(self, event: dict) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO scored_events
                   (created_at, source, run_id, scenario_id, attack_type, true_attack,
                    probability, severity, predicted_label, model_version,
                    tower_site_id, tower_site_name, tower_lat, tower_lon,
                    correlation_score, features_json, top_features_json, alert_state,
                    obs_timestamp, ingest_batch_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    json.dumps(event.get("top_features")) if event.get("top_features") is not None else None,
                    event.get("alert_state"),
                    event.get("obs_timestamp"),
                    event.get("ingest_batch_id"),
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    def get_event(self, event_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM scored_events WHERE id = ?", (event_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def update_event_top_features(self, event_id: int, top_features: list[dict]) -> None:
        """Persists a lazily-computed (on-demand) explanation back onto its event, so
        requesting it again doesn't recompute -- see GET /events/{id}/explain."""
        with self._lock:
            self._conn.execute(
                "UPDATE scored_events SET top_features_json = ? WHERE id = ?",
                (json.dumps(top_features), event_id),
            )
            self._conn.commit()

    def recent_events(self, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM scored_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

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
        return {r["tower_site_id"]: self._row_to_dict(r) for r in rows}

    def find_ingested_event(self, tower_site_id: str, obs_timestamp_iso: str) -> dict | None:
        """Duplicate lookup for POST /ingest: has this exact (tower, observation timestamp)
        already been ingested? Scoped to source='ingest' rows -- a /score or replay row has a
        NULL obs_timestamp and cannot collide. Returns the ORIGINAL row, so a duplicate
        submission can be answered with the first event_id instead of scoring again."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM scored_events WHERE source = 'ingest' AND tower_site_id = ? "
                "AND obs_timestamp = ? ORDER BY id ASC LIMIT 1",
                (tower_site_id, obs_timestamp_iso),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def max_ingested_obs_timestamp(self, tower_site_id: str) -> str | None:
        """Newest observation timestamp already ingested for this tower, or None. Used to
        flag out-of-order observations -- compared as an ISO-8601 string, which orders
        correctly because api/ingest.py normalizes every timestamp to UTC with a fixed
        format before it is stored (see _iso_utc there)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(obs_timestamp) FROM scored_events "
                "WHERE source = 'ingest' AND tower_site_id = ?",
                (tower_site_id,),
            ).fetchone()
        return row[0] if row and row[0] is not None else None

    def recent_feature_rows(self, limit: int = 1000, source: str | None = None) -> list[dict]:
        """Stored feature vectors from the most recent scored events -- the live sample
        GET /drift compares against the training baseline.

        Rows with no stored features (scored before features_json existed) are skipped rather
        than counted as empty, so they cannot dilute a distribution.
        """
        sql = ("SELECT features_json FROM scored_events WHERE features_json IS NOT NULL")
        params: list = []
        if source:
            sql += " AND source = ?"
            params.append(source)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            try:
                out.append(json.loads(r["features_json"]))
            except (TypeError, ValueError):
                continue
        return out

    # ---------------------------------------------------------------- feedback
    # Analyst confirm/dismiss. These store labels; nothing reads them back into the model.
    # See FEEDBACK_LOOP.md -- there is no closed loop and no retraining anywhere in this repo.

    def get_feedback(self, event_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM event_feedback WHERE event_id = ?", (event_id,)
            ).fetchone()
        return dict(row) if row else None

    def feedback_for_events(self, event_ids: list[int]) -> dict[int, dict]:
        """Bulk lookup for grouping events into incidents (api/incidents.py) -- one query
        instead of one get_feedback() call per event."""
        if not event_ids:
            return {}
        placeholders = ",".join("?" * len(event_ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM event_feedback WHERE event_id IN ({placeholders})", event_ids
            ).fetchall()
        return {r["event_id"]: dict(r) for r in rows}

    def upsert_feedback(self, event_id: int, label: str, note: str | None = None,
                        analyst: str | None = None) -> dict:
        """Set (or replace) the current label for an event, and append to the audit log.

        Both writes happen inside one transaction under the shared lock, so the log can never
        record a submission the current-label table did not accept, or vice versa.

        A relabel keeps the ORIGINAL created_at (when the event was first labeled) and bumps
        revision. The returned record carries previous_label so the caller can tell a first
        label from a change of mind.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                existing = self._conn.execute(
                    "SELECT label, created_at, revision FROM event_feedback WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
                previous_label = existing["label"] if existing else None
                created_at = existing["created_at"] if existing else now
                revision = (existing["revision"] + 1) if existing else 1

                self._conn.execute(
                    """INSERT INTO event_feedback
                           (event_id, label, note, analyst, created_at, updated_at, revision)
                       VALUES (?,?,?,?,?,?,?)
                       ON CONFLICT(event_id) DO UPDATE SET
                           label=excluded.label, note=excluded.note, analyst=excluded.analyst,
                           updated_at=excluded.updated_at, revision=excluded.revision""",
                    (event_id, label, note, analyst, created_at, now, revision),
                )
                self._conn.execute(
                    """INSERT INTO event_feedback_log
                           (event_id, label, previous_label, note, analyst, created_at, revision)
                       VALUES (?,?,?,?,?,?,?)""",
                    (event_id, label, previous_label, note, analyst, now, revision),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        return {
            "event_id": event_id, "label": label, "note": note, "analyst": analyst,
            "created_at": created_at, "updated_at": now, "revision": revision,
            "previous_label": previous_label,
        }

    def feedback_history(self, event_id: int) -> list[dict]:
        """Every label ever submitted for this event, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM event_feedback_log WHERE event_id = ? ORDER BY id ASC",
                (event_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def labeled_events(self) -> list[dict]:
        """Every labeled event, joined to its scored_events row -- the export dataset.

        INNER JOIN on purpose: a label whose event row no longer exists (a wiped DB reused
        alongside a stale label table) is excluded rather than exported with empty features.
        """
        with self._lock:
            rows = self._conn.execute(
                """SELECT f.event_id   AS event_id,
                          f.label      AS feedback_label,
                          f.note       AS feedback_note,
                          f.analyst    AS feedback_analyst,
                          f.created_at AS first_labeled_at,
                          f.updated_at AS labeled_at,
                          f.revision   AS feedback_revision,
                          e.*
                   FROM event_feedback f
                   JOIN scored_events e ON e.id = f.event_id
                   ORDER BY f.event_id ASC"""
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d.pop("id", None)  # duplicate of event_id, introduced by the join
            out.append(d)
        return out

    def feedback_summary(self) -> dict:
        """Counts, plus alert precision computed over LABELED EVENTS ONLY.

        The precision denominators are deliberately narrow and these numbers are deliberately
        NOT called model accuracy. See FEEDBACK_LOOP.md: analysts choose which events to
        label, so the labeled set is a self-selected, non-random sample of the scored set. A
        precision figure over it describes that sample and nothing wider.
        """
        with self._lock:
            total_events = self._conn.execute(
                "SELECT COUNT(*) FROM scored_events").fetchone()[0]
            by_label = dict(self._conn.execute(
                "SELECT label, COUNT(*) FROM event_feedback GROUP BY label").fetchall())
            n_relabeled = self._conn.execute(
                "SELECT COUNT(*) FROM event_feedback WHERE revision > 1").fetchone()[0]
            n_submissions = self._conn.execute(
                "SELECT COUNT(*) FROM event_feedback_log").fetchone()[0]
            by_source = dict(self._conn.execute(
                """SELECT e.source, COUNT(*) FROM event_feedback f
                   JOIN scored_events e ON e.id = f.event_id GROUP BY e.source""").fetchall())
            n_analysts = self._conn.execute(
                "SELECT COUNT(DISTINCT analyst) FROM event_feedback "
                "WHERE analyst IS NOT NULL AND analyst != ''").fetchone()[0]

            def precision(where: str) -> tuple[int, int]:
                row = self._conn.execute(
                    "SELECT SUM(CASE WHEN f.label='confirmed' THEN 1 ELSE 0 END), COUNT(*) "
                    "FROM event_feedback f JOIN scored_events e ON e.id = f.event_id "
                    "WHERE " + where
                ).fetchone()
                return (row[0] or 0), (row[1] or 0)

            pred_conf, pred_total = precision("e.predicted_label = 'attack'")
            hyst_conf, hyst_total = precision("e.alert_state = 'alerting'")

        confirmed = by_label.get("confirmed", 0)
        dismissed = by_label.get("dismissed", 0)
        return {
            "total_events": total_events,
            "total_labeled": confirmed + dismissed,
            "unlabeled": total_events - (confirmed + dismissed),
            "confirmed": confirmed,
            "dismissed": dismissed,
            "relabeled_events": n_relabeled,
            "total_submissions": n_submissions,
            "distinct_analysts": n_analysts,
            "labeled_by_source": by_source,
            "predicted_attack_labeled": pred_total,
            "predicted_attack_confirmed": pred_conf,
            "predicted_attack_precision": (pred_conf / pred_total) if pred_total else None,
            "hysteresis_alerting_labeled": hyst_total,
            "hysteresis_alerting_confirmed": hyst_conf,
            "hysteresis_alert_precision": (hyst_conf / hyst_total) if hyst_total else None,
        }

    def event_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM scored_events").fetchone()[0]

    def clear_all(self) -> None:
        """Demo-mode-only reset (see POST /demo/reset): deletes every scored event, analyst
        label and label-history row. Does not touch tower data or the model -- those aren't
        stored here. Irreversible; callers are responsible for gating who can call this."""
        with self._lock:
            try:
                self._conn.execute("DELETE FROM scored_events")
                self._conn.execute("DELETE FROM event_feedback")
                self._conn.execute("DELETE FROM event_feedback_log")
                self._conn.execute(
                    "DELETE FROM sqlite_sequence WHERE name IN "
                    "('scored_events', 'event_feedback_log')"
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self):
        with self._lock:
            self._conn.close()
