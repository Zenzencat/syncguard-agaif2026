"""NOC tab incident queue: grouping flagged events into incidents (api/incidents.py).

Test events and their tower attribution here are SYNTHETIC fixtures, not field observations
-- same convention as tests/test_priority.py.
"""
from __future__ import annotations

from api.incidents import build_incidents

# Tower A and B are ~1km apart (within the 5km default distance_km); C is ~50km from A.
TOWER_COORDS = {
    "A": (0.0, 100.0),
    "B": (0.009, 100.0),
    "C": (0.45, 100.0),
}


def _event(id, *, tower="A", severity=0.8, alert_state="alerting",
           predicted_label="attack", created_at="2099-01-01T00:00:00+00:00",
           attack_type=None, run_id=None, source=None, coords=True):
    lat, lon = TOWER_COORDS.get(tower, (None, None)) if coords else (None, None)
    return {
        "id": id, "tower_site_id": tower, "tower_site_name": f"Tower {tower}",
        "tower_lat": lat, "tower_lon": lon,
        "severity": severity, "alert_state": alert_state,
        "predicted_label": predicted_label, "created_at": created_at,
        "attack_type": attack_type, "run_id": run_id, "source": source,
    }


def test_non_flagged_events_are_excluded():
    events = [_event(1, alert_state="normal", predicted_label="clean")]
    assert build_incidents(events, {}) == []


def test_events_within_window_on_one_tower_form_one_incident():
    events = [
        _event(1, created_at="2099-01-01T00:00:00+00:00"),
        _event(2, created_at="2099-01-01T00:01:30+00:00"),  # 90s later, within 120s window
    ]
    incidents = build_incidents(events, {})
    assert len(incidents) == 1
    assert incidents[0]["event_ids"] == [1, 2]
    assert incidents[0]["event_count"] == 2


def test_events_beyond_window_split_into_separate_incidents():
    events = [
        _event(1, created_at="2099-01-01T00:00:00+00:00"),
        _event(2, created_at="2099-01-01T00:05:00+00:00"),  # 300s later, beyond 120s window
    ]
    incidents = build_incidents(events, {})
    assert len(incidents) == 2


def test_nearby_towers_within_window_merge_into_one_incident():
    events = [
        _event(1, tower="A", created_at="2099-01-01T00:00:00+00:00"),
        _event(2, tower="B", created_at="2099-01-01T00:00:30+00:00"),  # ~1km from A
    ]
    incidents = build_incidents(events, {})
    assert len(incidents) == 1
    towers = {t["site_id"] for t in incidents[0]["towers"]}
    assert towers == {"A", "B"}


def test_distant_towers_within_window_do_not_merge():
    events = [
        _event(1, tower="A", created_at="2099-01-01T00:00:00+00:00"),
        _event(2, tower="C", created_at="2099-01-01T00:00:30+00:00"),  # ~50km from A, same window
    ]
    incidents = build_incidents(events, {})
    assert len(incidents) == 2, "towers 50km apart must not merge even within the time window"


def test_unknown_tower_coordinates_fail_closed_and_do_not_merge():
    events = [
        _event(1, tower="A", created_at="2099-01-01T00:00:00+00:00"),
        _event(2, tower="D", created_at="2099-01-01T00:00:30+00:00", coords=False),
    ]
    incidents = build_incidents(events, {})
    assert len(incidents) == 2


def test_incident_id_and_top_event_use_most_severe_member():
    events = [
        _event(1, severity=0.6, created_at="2099-01-01T00:00:00+00:00"),
        _event(2, severity=0.9, created_at="2099-01-01T00:00:10+00:00"),
    ]
    incidents = build_incidents(events, {})
    assert incidents[0]["incident_id"] == "INC-1"
    assert incidents[0]["top_event_id"] == 2
    assert incidents[0]["severity"] == 0.9


def test_status_confirmed_wins_over_dismissed():
    events = [_event(1), _event(2, created_at="2099-01-01T00:00:10+00:00")]
    feedback = {1: {"label": "dismissed"}, 2: {"label": "confirmed"}}
    incidents = build_incidents(events, feedback)
    assert incidents[0]["status"] == "Confirmed"


def test_status_dismissed_when_no_confirm():
    events = [_event(1)]
    incidents = build_incidents(events, {1: {"label": "dismissed"}})
    assert incidents[0]["status"] == "Dismissed"


def test_status_new_when_unlabeled():
    events = [_event(1)]
    incidents = build_incidents(events, {})
    assert incidents[0]["status"] == "New"


def test_sorted_most_severe_first_then_newest_first_on_tie():
    events = [
        _event(1, severity=0.5, created_at="2099-01-01T00:00:00+00:00"),
        _event(2, severity=0.9, created_at="2099-01-01T01:00:00+00:00"),
        _event(3, severity=0.9, created_at="2099-01-01T02:00:00+00:00"),
    ]
    # each pair > 120s apart -> 3 separate incidents
    incidents = build_incidents(events, {})
    assert [inc["top_event_id"] for inc in incidents] == [3, 2, 1]


def test_stateless_score_events_use_threshold_gate():
    events = [_event(1, alert_state=None, predicted_label="attack")]
    incidents = build_incidents(events, {})
    assert len(incidents) == 1


def test_two_separate_replay_runs_never_merge_even_within_window():
    events = [
        _event(1, created_at="2099-01-01T00:00:00+00:00", run_id="run-A", source="replay"),
        _event(2, created_at="2099-01-01T00:00:30+00:00", run_id="run-B", source="replay"),
    ]
    incidents = build_incidents(events, {})
    assert len(incidents) == 2
    assert incidents[0]["event_ids"] != incidents[1]["event_ids"]


def test_same_run_id_within_window_still_merges():
    events = [
        _event(1, created_at="2099-01-01T00:00:00+00:00", run_id="run-A", source="replay"),
        _event(2, created_at="2099-01-01T00:00:30+00:00", run_id="run-A", source="replay"),
    ]
    incidents = build_incidents(events, {})
    assert len(incidents) == 1


def test_labeled_event_count_counts_only_labeled_members():
    events = [_event(1), _event(2, created_at="2099-01-01T00:00:10+00:00"),
              _event(3, created_at="2099-01-01T00:00:20+00:00")]
    feedback = {2: {"label": "confirmed"}}
    incidents = build_incidents(events, feedback)
    assert incidents[0]["event_count"] == 3
    assert incidents[0]["labeled_event_count"] == 1


def test_tower_attribution_simulated_flag_follows_replay_source():
    replay_events = [_event(1, source="replay")]
    assert build_incidents(replay_events, {})[0]["tower_attribution_simulated"] is True

    ingest_events = [_event(2, source="ingest")]
    assert build_incidents(ingest_events, {})[0]["tower_attribution_simulated"] is False


def test_max_pop_2km_is_the_highest_single_tower_not_a_sum():
    events = [
        _event(1, tower="A", created_at="2099-01-01T00:00:00+00:00"),
        _event(2, tower="B", created_at="2099-01-01T00:00:30+00:00"),
    ]
    pop_by_tower = {"A": 1000.0, "B": 5000.0}
    incidents = build_incidents(events, {}, pop_by_tower)
    assert incidents[0]["max_pop_2km"] == 5000.0  # not 6000 (the sum)


def test_max_pop_2km_is_null_when_no_population_data_given():
    events = [_event(1)]
    incidents = build_incidents(events, {})
    assert incidents[0]["max_pop_2km"] is None


def test_incidents_endpoint_empty_state(client, monkeypatch):
    monkeypatch.setattr(client.app.state.event_store, "recent_events", lambda limit=2000: [])
    monkeypatch.setattr(client.app.state.event_store, "feedback_for_events", lambda ids: {})
    response = client.get("/incidents")
    assert response.status_code == 200
    body = response.json()
    assert body["incidents"] == []
    assert body["window_seconds"] == 120
    assert body["distance_km"] == 5.0


def test_incidents_endpoint_groups_real_scored_events(client, monkeypatch, tower_ids):
    tower_key = tower_ids[0]
    fake_events = [
        {"id": 101, "tower_site_id": tower_key, "tower_site_name": "Fixture Tower",
         "tower_lat": 0.0, "tower_lon": 100.0,
         "severity": 0.95, "alert_state": "alerting", "predicted_label": "attack",
         "created_at": "2099-01-01T00:00:00+00:00", "attack_type": None},
    ]
    monkeypatch.setattr(client.app.state.event_store, "recent_events", lambda limit=2000: fake_events)
    monkeypatch.setattr(client.app.state.event_store, "feedback_for_events", lambda ids: {})
    response = client.get("/incidents")
    assert response.status_code == 200
    body = response.json()
    assert len(body["incidents"]) == 1
    assert body["incidents"][0]["status"] == "New"
    assert body["incidents"][0]["towers"][0]["site_id"] == tower_key
