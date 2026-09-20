"""A tower table with no exposure columns (no pop_1km / pop_2km) must degrade to "no
population data available", never a 500.

Found by running the Phase 4 benchmark server standalone: its synthetic tower registry never
goes through attach_exposure(), so GET /incidents raised KeyError: 'pop_2km'. The production
path always attaches the columns (all-NaN when the exposure CSV is missing), so this is a
robustness fix, not a change to any number. Tower rows here are SYNTHETIC test fixtures.
"""
from __future__ import annotations

import pandas as pd

from api.exposure import pop_2km_by_tower, rank_priority
from api.main import app


def _towers_without_exposure() -> pd.DataFrame:
    towers = app.state.towers
    return towers.drop(columns=[c for c in ("pop_1km", "pop_2km") if c in towers.columns])


def test_pop_2km_by_tower_returns_empty_without_the_column(client):
    towers = _towers_without_exposure()
    assert "pop_2km" not in towers.columns
    assert pop_2km_by_tower(towers) == {}


def test_pop_2km_by_tower_drops_missing_estimates_and_keeps_real_ones():
    towers = pd.DataFrame({
        "tower_key": ["A", "B", "C"],
        "pop_2km": [1200.0, float("nan"), 0.0],
    })
    assert pop_2km_by_tower(towers) == {"A": 1200.0, "C": 0.0}


def test_get_incidents_200_when_tower_table_has_no_exposure_columns(client, monkeypatch):
    monkeypatch.setattr(app.state, "towers", _towers_without_exposure())
    r = client.get("/incidents")
    assert r.status_code == 200
    body = r.json()
    assert "incidents" in body
    # No exposure data -> no incident may claim a population figure.
    assert all(inc["max_pop_2km"] is None for inc in body["incidents"])


def test_rank_priority_lists_flagged_tower_as_population_unavailable_without_columns(client):
    towers = _towers_without_exposure()
    key = towers["tower_key"].iloc[0]
    latest = {key: {"id": 1, "severity": 0.9, "probability": 0.9,
                    "alert_state": "alerting", "source": "ingest",
                    "created_at": "2099-01-01T00:00:00+00:00"}}
    out = rank_priority(latest, towers)
    assert out["n_flagged"] == 1
    row = out["towers"][0]
    assert row["pop_2km"] is None and row["pop_1km"] is None
    assert row["population_status"] == "population unavailable"
