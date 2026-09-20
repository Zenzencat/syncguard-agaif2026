"""The incident queue must be stable across page reloads and across a full replay (Item D).

Diagnosed (MEASURED, throwaway server, temp DB):
  * GET /incidents recomputed incidents from only the newest 2,000 events of any kind. One
    replay is 2,503 rows, so the earliest incidents slid out of the response mid-replay, and a
    long-running incident was RENAMED (id = INC-<first event id in the window>): storage held 6
    incident groups while the API returned 1, under a different id.
  * Re-running the same scenario right after a first run did not open a new incident: run_id is
    the recording, identical on a re-run, so the new alerts were chained into the previous
    incident -- which stayed 'Dismissed' after an analyst dismissed it.
Nothing needed new storage: the events were already persisted. Events and towers in these tests
are SYNTHETIC fixtures.
"""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from api.db import EventStore, INCIDENT_EVENT_CAP
from api.incidents import build_incidents
from api.main import app
from api.replay import ReplayManager

BASE = datetime(2099, 1, 1, tzinfo=timezone.utc)


def _ev(store: EventStore, second: float, *, state="alerting", label="attack",
        tower="TWR-A", session=None, run_id="run-x") -> int:
    """`second` is seconds after BASE. Callers keep it non-decreasing, like real rows: in the
    real store created_at is monotonic with id (0 violations across 1,226 measured rows)."""
    return store.insert_event({
        "created_at": (BASE + timedelta(seconds=second)).isoformat(),
        "source": "replay", "run_id": run_id, "replay_session": session,
        "probability": 0.9, "severity": 0.9, "predicted_label": label,
        "tower_site_id": tower, "tower_site_name": "Fixture Tower",
        "tower_lat": 0.0, "tower_lon": 100.0, "alert_state": state,
    })


@pytest.fixture
def store(tmp_path, client, monkeypatch):
    """A fresh store swapped in for the app's, so these tests never touch shared test data."""
    s = EventStore(tmp_path / "stability.db")
    monkeypatch.setattr(app.state, "event_store", s)
    yield s
    s.close()


def test_flagged_events_returns_only_flagged_rows_with_slim_columns(store):
    a = _ev(store, 0, state="alerting")
    _ev(store, 1, state="normal", label="clean")
    c = _ev(store, 2, state=None, label="attack")     # stateless /score-style row, attack verdict
    _ev(store, 3, state=None, label="clean")
    rows = store.flagged_events()
    assert [r["id"] for r in rows] == [c, a]           # newest first
    assert "features_json" not in rows[0] and "top_features_json" not in rows[0]
    assert store.flagged_events(limit=1) == rows[:1]


def test_incident_survives_more_than_2000_later_events_with_the_same_id(store, client):
    first = _ev(store, 0)
    _ev(store, 1)
    _ev(store, 2)
    for i in range(2500):                               # a full replay's worth of normal readings
        _ev(store, 3 + i * 0.001, state="normal", label="clean")
    # what /incidents used to do: newest 2,000 events of any kind -> the incident is gone
    assert build_incidents(store.recent_events(limit=2000), {}) == []
    body = client.get("/incidents").json()
    assert [i["incident_id"] for i in body["incidents"]] == [f"INC-{first}"]
    assert body["incidents"][0]["event_count"] == 3


def test_long_running_incident_keeps_its_id_as_it_grows_past_the_old_window(store, client):
    first = _ev(store, 0)
    before = client.get("/incidents").json()["incidents"][0]
    for i in range(2600):                               # sustained alert: same tower, same run
        _ev(store, 1 + i * 0.01)
    after = client.get("/incidents").json()["incidents"]
    assert len(after) == 1
    assert after[0]["incident_id"] == before["incident_id"] == f"INC-{first}"
    assert after[0]["event_count"] == 2601


def test_reload_returns_identical_incident_list(store, client):
    """Two reads of the API (what a page reload does) return byte-identical incident lists."""
    for i in range(5):
        _ev(store, i)
    first = client.get("/incidents").json()["incidents"]
    second = client.get("/incidents").json()["incidents"]
    assert first == second and len(first) == 1


def test_incidents_limit_parameter_is_bounded(client):
    assert client.get("/incidents", params={"limit": 0}).status_code == 422
    assert client.get("/incidents", params={"limit": INCIDENT_EVENT_CAP + 1}).status_code == 422
    assert client.get("/incidents", params={"limit": 10}).status_code == 200


def test_feedback_status_still_reaches_incident_when_many_events(store, client):
    """feedback_for_events is chunked; a label on an early event must still set the status."""
    ids = [_ev(store, i * 0.01) for i in range(1200)]   # > 2 chunks of 500
    store.upsert_feedback(ids[3], "dismissed", analyst="t")
    inc = client.get("/incidents").json()["incidents"][0]
    assert inc["status"] == "Dismissed" and inc["labeled_event_count"] == 1


# ---- replay sessions -------------------------------------------------------------------

def test_two_replays_of_the_same_scenario_do_not_merge_into_one_incident():
    def ev(i, session):
        return {"id": i, "tower_site_id": "A", "tower_site_name": "A", "tower_lat": 0.0,
                "tower_lon": 100.0, "severity": 0.9, "alert_state": "alerting",
                "predicted_label": "attack", "created_at": f"2099-01-01T00:00:{i:02d}+00:00",
                "run_id": "same-recording", "replay_session": session, "source": "replay"}
    two_runs = [ev(1, "s1"), ev(2, "s1"), ev(3, "s2"), ev(4, "s2")]   # 1 s apart: inside the window
    incidents = build_incidents(two_runs, {})
    assert sorted(i["incident_id"] for i in incidents) == ["INC-1", "INC-3"]
    one_run = [ev(1, "s1"), ev(2, "s1"), ev(3, "s1")]
    assert len(build_incidents(one_run, {})) == 1


def test_rows_without_a_session_keep_the_old_run_id_behaviour():
    """Rows written before the replay_session column existed have NULL there."""
    def ev(i):
        return {"id": i, "tower_site_id": "A", "tower_site_name": "A", "tower_lat": 0.0,
                "tower_lon": 100.0, "severity": 0.9, "alert_state": "alerting",
                "predicted_label": "attack", "created_at": f"2099-01-01T00:00:{i:02d}+00:00",
                "run_id": "same-recording", "replay_session": None, "source": "replay"}
    assert len(build_incidents([ev(1), ev(2), ev(3)], {})) == 1


def test_a_rerun_opens_a_new_incident_and_does_not_inherit_a_dismissal(store, client):
    first_run = [_ev(store, i, session="s1") for i in range(3)]
    store.upsert_feedback(first_run[0], "dismissed", analyst="t")
    second_run = [_ev(store, 10 + i, session="s2") for i in range(3)]   # 7 s later, same tower
    incidents = {i["incident_id"]: i for i in client.get("/incidents").json()["incidents"]}
    assert set(incidents) == {f"INC-{first_run[0]}", f"INC-{second_run[0]}"}
    assert incidents[f"INC-{first_run[0]}"]["status"] == "Dismissed"
    assert incidents[f"INC-{second_run[0]}"]["status"] == "New"


def test_each_replay_start_gets_a_fresh_session_id():
    async def scenario():
        mgr = ReplayManager(None, None, {"round_robin": object()}, None, None)
        sessions = []
        for _ in range(2):
            mgr.start(None, 10.0)
            sessions.append(mgr._session_id)
            mgr.stop()
            await asyncio.sleep(0)                      # let the cancelled task finish
        return sessions
    s1, s2 = asyncio.run(scenario())
    assert s1 and s2 and s1 != s2


# ---- migration -------------------------------------------------------------------------

def test_replay_session_migration_is_additive_on_a_legacy_db(tmp_path):
    """A DB created before the column existed keeps its rows and gains a NULL replay_session."""
    path = tmp_path / "legacy.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE scored_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, source TEXT NOT NULL,
            run_id TEXT, scenario_id TEXT, attack_type TEXT, true_attack INTEGER,
            probability REAL NOT NULL, severity REAL NOT NULL, predicted_label TEXT NOT NULL,
            model_version TEXT, tower_site_id TEXT, tower_site_name TEXT, tower_lat REAL,
            tower_lon REAL, correlation_score REAL, features_json TEXT);
        INSERT INTO scored_events (created_at, source, probability, severity, predicted_label)
            VALUES ('2099-01-01T00:00:00+00:00', 'api', 0.9, 0.9, 'attack');
    """)
    con.commit(); con.close()
    store = EventStore(path)
    try:
        cols = {r[1] for r in store._conn.execute("PRAGMA table_info(scored_events)")}
        assert "replay_session" in cols
        old = store.recent_events(limit=5)
        assert len(old) == 1 and old[0]["replay_session"] is None and old[0]["id"] == 1
        new_id = _ev(store, 1, session="abc")
        assert new_id == 2                              # ids continue; nothing reassigned
        assert [r["id"] for r in store.flagged_events()] == [2, 1]
    finally:
        store.close()
