"""POST /demo/reset -- the "New session" button's endpoint. Off unless SYNCGUARD_DEMO_MODE=1
(read per-request, not cached, so it's monkeypatch-able here); irreversibly stops any running
replay and clears scored_events/event_feedback/event_feedback_log. Also gated by API-key auth
like any other non-exempt route (see tests/test_ops.py::test_protected_post_routes_require_a_key).

The `client` fixture's event_store is session-scoped and shared with every other test file,
so the clearing test below swaps in a throwaway EventStore for its own duration (monkeypatch
reverts it automatically) instead of touching real shared state.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

from api.db import EventStore


def test_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("SYNCGUARD_DEMO_MODE", raising=False)
    r = client.post("/demo/reset")
    assert r.status_code == 403
    assert "SYNCGUARD_DEMO_MODE" in r.json()["detail"]


def test_disabled_when_env_var_is_not_exactly_1(client, monkeypatch):
    monkeypatch.setenv("SYNCGUARD_DEMO_MODE", "true")
    r = client.post("/demo/reset")
    assert r.status_code == 403


def test_health_reports_demo_mode(client, monkeypatch):
    monkeypatch.delenv("SYNCGUARD_DEMO_MODE", raising=False)
    assert client.get("/health").json()["demo_mode"] is False
    monkeypatch.setenv("SYNCGUARD_DEMO_MODE", "1")
    assert client.get("/health").json()["demo_mode"] is True


def test_enabled_clears_events_and_feedback_and_stops_replay(client, monkeypatch, tower_ids):
    monkeypatch.setenv("SYNCGUARD_DEMO_MODE", "1")
    tmp_db = Path(tempfile.mkdtemp(prefix="syncguard-demo-reset-test-")) / "isolated.db"
    store = EventStore(db_path=tmp_db)
    monkeypatch.setattr(client.app.state, "event_store", store)

    event_id = store.insert_event({
        "created_at": "2099-01-01T00:00:00+00:00", "source": "api", "run_id": None,
        "scenario_id": None, "attack_type": None, "true_attack": None,
        "probability": 0.9, "severity": 0.9, "predicted_label": "attack",
        "model_version": "test", "tower_site_id": tower_ids[0], "tower_site_name": "Fixture",
        "tower_lat": 0.0, "tower_lon": 100.0, "correlation_score": 0.0,
        "features": {}, "top_features": None, "alert_state": "alerting",
    })
    store.upsert_feedback(event_id, "confirmed", analyst="tester")
    assert store.event_count() == 1
    assert store.get_feedback(event_id) is not None

    r = client.post("/demo/reset")
    assert r.status_code == 200
    assert r.json() == {"status": "reset", "demo_mode": True}

    assert store.event_count() == 0
    assert store.get_feedback(event_id) is None
    assert store.recent_events() == []


def test_dashboard_new_session_button_is_hidden_by_default_and_wired_to_the_endpoint(client):
    html = client.get("/dashboard").text
    assert 'id="btn-new-session"' in html
    m = re.search(r'<button[^>]*id="btn-new-session"[^>]*>', html)
    assert m and "hidden" in m.group(0), "the button must start hidden -- only demo_mode reveals it"
    assert "api('/demo/reset', {method:'POST'})" in html
    assert "health.demo_mode" in html


def test_enabled_stops_a_running_replay(client, monkeypatch):
    monkeypatch.setenv("SYNCGUARD_DEMO_MODE", "1")
    if client.app.state.replay_manager is None:
        return  # no model loaded in this environment -- nothing to stop
    started = client.post("/replay/start", params={"speed": 50.0})
    if started.status_code != 200:
        return  # e.g. a replay from another test is already running -- not this test's concern
    assert client.get("/replay/status").json()["status"] == "running"
    client.post("/demo/reset")
    assert client.get("/replay/status").json()["status"] != "running"
