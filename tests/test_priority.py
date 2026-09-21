"""Phase 6: severity gates which towers are flagged; ESTIMATED nearby population ranks them.

Tower locations are REAL. Population is an ESTIMATE (~2020 exposure proxy). Test events and
their tower attribution in this module are SYNTHETIC fixtures, not field observations.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from api.exposure import attach_exposure, rank_priority
from api.spatial import load_towers


def _tower_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"tower_key": "low-pop", "site_id": "A", "site_name": "A", "lat": 0.0,
         "lon": 100.0, "pop_1km": 100.0, "pop_2km": 500.0},
        {"tower_key": "high-pop", "site_id": "B", "site_name": "B", "lat": 0.1,
         "lon": 100.1, "pop_1km": 500.0, "pop_2km": 5000.0},
        {"tower_key": "tie-high-severity", "site_id": "C", "site_name": "C", "lat": 0.2,
         "lon": 100.2, "pop_1km": 500.0, "pop_2km": 5000.0},
        {"tower_key": "missing-pop", "site_id": "D", "site_name": "D", "lat": 0.3,
         "lon": 100.3, "pop_1km": np.nan, "pop_2km": np.nan},
        {"tower_key": "not-flagged", "site_id": "E", "site_name": "E", "lat": 0.4,
         "lon": 100.4, "pop_1km": 9000.0, "pop_2km": 90000.0},
    ])


def _event(*, severity: float, alert_state="alerting", predicted_label="attack") -> dict:
    return {
        "id": 1, "severity": severity, "probability": severity,
        "alert_state": alert_state, "predicted_label": predicted_label,
        "source": "synthetic-test", "created_at": "2099-01-01T00:00:00+00:00",
    }


def _has_key_fragment(value, fragment: str) -> bool:
    if isinstance(value, dict):
        return any(fragment in str(key).lower() or _has_key_fragment(item, fragment)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_has_key_fragment(item, fragment) for item in value)
    return False


def test_gate_excludes_unflagged_towers_and_uses_threshold_fallback():
    events = {
        "low-pop": _event(severity=0.9),
        "not-flagged": _event(severity=0.99, alert_state="normal"),
        "high-pop": _event(severity=0.8, alert_state=None, predicted_label="attack"),
        "tie-high-severity": _event(severity=0.95, alert_state=None,
                                    predicted_label="clean"),
    }
    body = rank_priority(events, _tower_table())
    assert [row["tower_key"] for row in body["towers"]] == ["high-pop", "low-pop"]
    assert body["gate"]["flagged_by_basis"] == {"hysteresis": 1, "threshold": 1}


def test_population_orders_first_and_severity_only_breaks_ties():
    events = {
        "low-pop": _event(severity=0.99),
        "high-pop": _event(severity=0.6),
        "tie-high-severity": _event(severity=0.9),
    }
    rows = rank_priority(events, _tower_table())["towers"]
    assert [row["tower_key"] for row in rows] == [
        "tie-high-severity", "high-pop", "low-pop"]
    assert [row["rank"] for row in rows] == [1, 2, 3]


def test_null_population_sorts_last_and_is_marked_unavailable():
    events = {
        "missing-pop": _event(severity=1.0),
        "low-pop": _event(severity=0.6),
    }
    rows = rank_priority(events, _tower_table())["towers"]
    assert rows[-1]["tower_key"] == "missing-pop"
    assert rows[-1]["pop_1km"] is None
    assert rows[-1]["pop_2km"] is None
    assert rows[-1]["population_status"] == "population unavailable"


def test_real_coordinate_join_covers_all_towers_and_keeps_tbg_rows_distinct():
    towers = attach_exposure(load_towers())
    assert len(towers) == 136
    assert towers["tower_key"].is_unique
    assert towers[["pop_1km", "pop_2km"]].notna().all().all()
    tbg = towers[towers["site_id"] == "tbg"]
    assert len(tbg) == 16
    assert tbg["tower_key"].nunique() == 16
    assert tbg[["lat", "lon"]].drop_duplicates().shape[0] == 16


def test_priority_response_contains_no_total_field_anywhere():
    body = rank_priority({"low-pop": _event(severity=0.8)}, _tower_table())
    assert not _has_key_fragment(body, "total")


def test_priority_endpoint_empty_state(client, monkeypatch):
    monkeypatch.setattr(client.app.state.event_store, "latest_severity_per_tower", lambda: {})
    response = client.get("/priority")
    assert response.status_code == 200
    body = response.json()
    assert body["towers"] == []
    assert body["n_flagged"] == 0
    assert "No towers have alerted this session" in body["empty_state"]
    assert not _has_key_fragment(body, "total")


def test_towers_endpoint_serializes_missing_population_as_json_null(client, monkeypatch):
    towers = client.app.state.towers.copy()
    towers.loc[towers.index[0], "pop_2km"] = np.nan
    monkeypatch.setattr(client.app.state, "towers", towers)
    response = client.get("/towers")
    assert response.status_code == 200
    assert response.json()[0]["pop_2km"] is None
    assert "NaN" not in response.text


def test_priority_numbers_are_json_safe():
    body = rank_priority({"missing-pop": _event(severity=0.8)}, _tower_table())
    for row in body["towers"]:
        for value in row.values():
            assert not (isinstance(value, float) and math.isnan(value))
