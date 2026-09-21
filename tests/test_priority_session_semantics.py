"""/priority flags the towers that ALERTED IN THE SESSION, marks which are alerting now, and agrees with
the dashboard's solid + hollow diamonds.

Before: the gate was each tower's LATEST event only. Replay hysteresis is per recording, so once a recording
returned to normal, towers that simply received no later event kept an old 'alerting' event: /priority
still reported them flagged while the map (correctly) showed nothing alerting, and towers that had cleared
dropped out of the count entirely. Tower locations are REAL; population is an ESTIMATE; every event in this
module is a SYNTHETIC fixture, not a field observation.
"""
from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from api.db import EventStore
from api.exposure import is_alerting_now, is_flagged, latest_replay_state, rank_priority


def _towers(keys) -> pd.DataFrame:
    return pd.DataFrame([
        {"tower_key": k, "site_id": k, "site_name": k, "lat": 0.0 + i * 0.01, "lon": 100.0,
         "pop_1km": 10.0 * (i + 1), "pop_2km": 100.0 * (i + 1)}   # later towers have MORE people
        for i, k in enumerate(keys)
    ])


def _ev(eid, state, source="replay", label=None, sev=0.9):
    return {"id": eid, "alert_state": state, "source": source,
            "predicted_label": label or ("attack" if state == "alerting" else "clean"),
            "severity": sev, "probability": sev, "created_at": "2099-01-01T00:00:00+00:00"}


def _by_key(body):
    return {r["tower_key"]: r for r in body["towers"]}


# ---------------------------------------------------------------- the rule

def test_towers_that_cleared_stay_in_the_flagged_set_marked_cleared():
    towers = _towers(["a", "b", "c"])
    latest = {"a": _ev(1, "alerting"), "b": _ev(2, "normal"), "c": _ev(3, "normal")}
    body = rank_priority(latest, towers, session_flagged={"a", "b"})     # b alerted earlier, c never did
    rows = _by_key(body)
    assert set(rows) == {"a", "b"}
    # recording's latest event (id 3) is 'normal' -> nothing is alerting now, both are cleared
    assert rows["a"]["state"] == "cleared" and rows["a"]["alerting_now"] is False
    assert rows["b"]["state"] == "cleared"
    assert (body["n_flagged"], body["n_alerting_now"], body["n_cleared"]) == (2, 0, 2)


def test_replay_towers_are_alerting_now_only_while_the_recording_is_alerting():
    towers = _towers(["a", "b", "c"])
    still = {"a": _ev(1, "alerting"), "b": _ev(2, "alerting"), "c": _ev(3, "alerting")}
    body = rank_priority(still, towers, session_flagged={"a", "b", "c"})
    assert (body["n_flagged"], body["n_alerting_now"], body["n_cleared"]) == (3, 3, 0)
    # the recording returns to normal on tower c's later event; a and b never got another event
    ended = {"a": _ev(1, "alerting"), "b": _ev(2, "alerting"), "c": _ev(9, "normal")}
    body = rank_priority(ended, towers, session_flagged={"a", "b", "c"})
    assert (body["n_flagged"], body["n_alerting_now"], body["n_cleared"]) == (3, 0, 3), \
        "no tower may be 'alerting now' once the recording is normal"


def test_ingest_towers_keep_per_tower_semantics():
    towers = _towers(["i1", "i2", "r"])
    latest = {"i1": _ev(1, "alerting", "ingest"), "i2": _ev(2, "normal", "ingest"),
              "r": _ev(9, "normal", "replay")}     # an unrelated replay recording is currently normal
    body = rank_priority(latest, towers, session_flagged={"i1", "i2"})
    rows = _by_key(body)
    assert rows["i1"]["alerting_now"] is True, "an ingest tower's own latest state is its current state"
    assert rows["i2"]["alerting_now"] is False and rows["i2"]["state"] == "cleared"


def test_an_event_with_no_hysteresis_state_is_never_alerting_now_but_is_still_flagged():
    towers = _towers(["s"])
    latest = {"s": _ev(1, None, "api", label="attack")}
    assert is_flagged(latest["s"])[0] is True
    assert is_alerting_now(latest["s"], None) is False
    body = rank_priority(latest, towers)      # no history passed: latest-flagged still gets in
    assert body["n_flagged"] == 1 and body["towers"][0]["state"] == "cleared"


def test_ranking_is_still_by_population_and_a_cleared_tower_is_not_demoted():
    towers = _towers(["low", "mid", "high"])          # pop_2km 100 / 200 / 300
    latest = {"low": _ev(1, "alerting"), "mid": _ev(2, "alerting"), "high": _ev(9, "normal")}
    body = rank_priority(latest, towers, session_flagged={"low", "mid", "high"})
    assert [r["tower_key"] for r in body["towers"]] == ["high", "mid", "low"]
    assert [r["rank"] for r in body["towers"]] == [1, 2, 3]
    assert body["towers"][0]["state"] == "cleared"


def test_without_history_the_gate_is_the_latest_event_as_before():
    towers = _towers(["a", "b"])
    latest = {"a": _ev(1, "alerting"), "b": _ev(2, "normal")}
    body = rank_priority(latest, towers)
    assert [r["tower_key"] for r in body["towers"]] == ["a"]


def test_latest_replay_state_is_read_from_the_most_recent_replay_event():
    latest = {"a": _ev(4, "alerting"), "b": _ev(7, "normal"), "i": _ev(99, "alerting", "ingest")}
    assert latest_replay_state(latest) == "normal"        # the ingest event is newer but is not replay
    assert latest_replay_state({"i": _ev(1, "alerting", "ingest")}) is None


# ---------------------------------------------------------------- the store and the endpoint

def _row(state, tower, label=None):
    return {"created_at": "2099-01-01T00:00:00+00:00", "source": "replay", "run_id": "r", "scenario_id": "s",
            "attack_type": "Spoofing", "true_attack": 1, "probability": 0.9, "severity": 0.9,
            "predicted_label": label or ("attack" if state == "alerting" else "clean"), "model_version": "t",
            "tower_site_id": tower, "tower_site_name": tower, "tower_lat": 0.0, "tower_lon": 100.0,
            "correlation_score": 0.0, "features": {}, "top_features": None, "alert_state": state}


def test_store_reports_every_tower_with_a_flagged_event_in_the_session():
    store = EventStore(db_path=Path(tempfile.mkdtemp(prefix="syncguard-prio-")) / "p.db")
    for state, tower in (("alerting", "a"), ("normal", "a"), ("normal", "b"), ("alerting", "c"),
                         (None, "d")):
        store.insert_event(_row(state, tower, label="attack" if tower == "d" else None))
    store.insert_event(_row(None, "e", label="clean"))
    assert store.flagged_tower_ids() == {"a", "c", "d"}       # a cleared, c still alerting, d threshold verdict
    store.clear_all()
    assert store.flagged_tower_ids() == set(), "a new session starts with an empty flagged set"


def test_endpoint_returns_the_session_set_with_state_fields(client, monkeypatch, tower_ids):
    store = client.app.state.event_store
    a, b, c = tower_ids[:3]
    monkeypatch.setattr(store, "latest_severity_per_tower",
                        lambda: {a: _ev(1, "alerting"), b: _ev(2, "normal"), c: _ev(3, "normal")})
    monkeypatch.setattr(store, "flagged_tower_ids", lambda: {a, b})
    body = client.get("/priority").json()
    assert {r["tower_key"] for r in body["towers"]} == {a, b}
    assert all(r["state"] == "cleared" and r["alerting_now"] is False for r in body["towers"])
    assert (body["n_flagged"], body["n_alerting_now"], body["n_cleared"]) == (2, 0, 2)
    assert "alerted at any point in the current session" in body["method"]


# ---------------------------------------------------------------- server rule == dashboard rule (differential)

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@needs_node
def test_api_and_dashboard_agree_on_alerting_now_and_cleared_for_random_sessions(client):
    """Runs the dashboard's own classifyAlerting() (from the served page) and the API's rank_priority() on
    the same random sessions. The header counter is n_flagged and 'active alerts' is the solid diamonds, so
    n_flagged == solid + hollow and n_alerting_now == solid must hold for every session."""
    html = client.get("/dashboard").text
    block = re.search(r"// ---- alert state BEGIN.*?// ---- alert state END ----", html, re.S).group(0)
    rng = random.Random(20260921)
    sessions = []
    for _ in range(60):
        keys = [f"t{i}" for i in range(rng.randint(1, 9))]
        latest, session = {}, set()
        for eid, k in enumerate(keys, start=1):
            src = rng.choice(["replay", "replay", "ingest", "api"])
            state = None if src == "api" else rng.choice(["alerting", "normal"])
            latest[k] = _ev(eid, state, src, label=rng.choice(["attack", "clean"]) if state is None else None)
            if is_flagged(latest[k])[0] or rng.random() < 0.4:     # alerted now, or at some earlier point
                session.add(k)
        sessions.append({"latest": latest, "session": sorted(session)})
    js = block + f"""
    const sessions = {json.dumps(sessions)};
    console.log(JSON.stringify(sessions.map(s => {{
      const rs = (() => {{ const r = Object.values(s.latest).filter(e => e.source === 'replay');
                          return r.length ? r.reduce((a, b) => a.id > b.id ? a : b).alert_state : null; }})();
      const towers = Object.entries(s.latest).map(([k, e]) => ({{site_id: k, alert_state: e.alert_state, source: e.source}}));
      const r = classifyAlerting(towers, rs, new Set(s.session));
      return {{now: r.alertingNow.map(t => t.site_id).sort(), cleared: r.cleared.map(t => t.site_id).sort()}};
    }})));"""
    # a script file, not `node -e`: sixty sessions overflow the Windows command-line length limit
    with tempfile.TemporaryDirectory(prefix="syncguard-diff-") as tmp:
        script = Path(tmp) / "classify.js"
        script.write_text(js, encoding="utf-8")
        out = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    dash = json.loads(out.stdout)
    for s, d in zip(sessions, dash):
        towers = _towers(sorted(s["latest"]))
        body = rank_priority(s["latest"], towers, session_flagged=set(s["session"]))
        api_now = sorted(r["tower_key"] for r in body["towers"] if r["alerting_now"])
        api_cleared = sorted(r["tower_key"] for r in body["towers"] if not r["alerting_now"])
        assert api_now == d["now"], s
        assert api_cleared == d["cleared"], s
        assert body["n_flagged"] == len(d["now"]) + len(d["cleared"])       # header == solid + hollow
        assert body["n_alerting_now"] == len(d["now"])                       # active alerts == solid
