"""SIMULATED tower attribution is reproducible: seeded, and reset at every replay start.

Before: the epicenter random walk was unseeded and never reset, so it carried on from wherever the
previous replay left it -- two replays of the same scenario got different tower sequences (and the
round-robin index likewise kept counting). The tower each event is attributed to is SIMULATED either
way; this only makes the simulation repeatable, which a demo needs.
"""
from __future__ import annotations

import time

import pandas as pd
import pytest

from api.spatial import (DEFAULT_ATTRIBUTION_SEED, EpicenterRandomWalk, TowerAttributor,
                         attribution_seed_from_env)
from conftest import requires_model


def _towers(n: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({"tower_key": f"t{i:02d}", "site_id": f"s{i:02d}", "site_name": f"T{i}",
                     "lat": 0.0 + 0.02 * (i % 6), "lon": 100.0 + 0.02 * (i // 6)})
    return pd.DataFrame(rows)


def _walk(attributor, n: int) -> list[str]:
    return [attributor.next_tower()["site_id"] for _ in range(n)]


# ---------------------------------------------------------------- the walk

def test_reset_returns_the_walk_to_the_epicenter_and_replays_the_same_sequence():
    a = EpicenterRandomWalk(_towers(), seed=7)
    first = _walk(a, 60)
    assert first[0] == a.epicenter_tower_key
    a.reset()
    second = _walk(a, 60)
    assert second == first, "one seed, one sequence: a reset walk must repeat itself exactly"
    assert second[0] == a.epicenter_tower_key


def test_without_a_reset_the_walk_carries_on_from_where_it_stopped():
    """The old behaviour, kept as documentation of what reset() is for."""
    a = EpicenterRandomWalk(_towers(), seed=7)
    first = _walk(a, 60)
    assert _walk(a, 60) != first


def test_different_seeds_give_different_walks():
    assert _walk(EpicenterRandomWalk(_towers(), seed=1), 80) != _walk(EpicenterRandomWalk(_towers(), seed=2), 80)


def test_an_unseeded_walk_still_resets_to_the_epicenter():
    a = EpicenterRandomWalk(_towers(), seed=None)
    _walk(a, 20)
    a.reset()
    assert a.next_tower()["site_id"] == a.epicenter_tower_key


def test_round_robin_reset_restarts_at_the_first_tower():
    a = TowerAttributor(_towers(10))
    assert _walk(a, 4) == ["t00", "t01", "t02", "t03"]
    a.reset()
    assert _walk(a, 3) == ["t00", "t01", "t02"]


# ---------------------------------------------------------------- default seed and override

def test_default_seed_is_fixed_and_overridable_from_the_environment(monkeypatch):
    monkeypatch.delenv("SYNCGUARD_ATTRIBUTION_SEED", raising=False)
    assert attribution_seed_from_env() == DEFAULT_ATTRIBUTION_SEED
    monkeypatch.setenv("SYNCGUARD_ATTRIBUTION_SEED", "12345")
    assert attribution_seed_from_env() == 12345
    monkeypatch.setenv("SYNCGUARD_ATTRIBUTION_SEED", "  ")
    assert attribution_seed_from_env() == DEFAULT_ATTRIBUTION_SEED
    monkeypatch.setenv("SYNCGUARD_ATTRIBUTION_SEED", "not-a-number")
    with pytest.raises(ValueError, match="SYNCGUARD_ATTRIBUTION_SEED"):
        attribution_seed_from_env()


def test_app_wires_the_epicenter_walk_with_the_default_seed(client):
    walk = client.app.state.replay_attributors["epicenter"]
    assert isinstance(walk, EpicenterRandomWalk)
    assert walk._seed == attribution_seed_from_env()


def test_the_old_misleading_name_is_gone_from_the_codebase():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    hits = []
    for path in list(root.glob("api/*.py")) + [root / "syncguard_interactive_summary.html"]:
        if "EpicenterWeightedAttributor" in path.read_text(encoding="utf-8"):
            # the class docstring mentions its former name once, on purpose
            if path.name != "spatial.py":
                hits.append(path.name)
    assert hits == [], f"stale references to the old class name: {hits}"
    assert "epicenter-weighted" not in (root / "syncguard_interactive_summary.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------- two real back-to-back replays

def _replay_towers(client, n_rows: int) -> list[str]:
    store = client.app.state.event_store
    store.clear_all()
    started = client.post("/replay/start", params={"speed": 1000.0, "attribution": "epicenter"})
    assert started.status_code == 200, started.text
    deadline = time.time() + 60
    while time.time() < deadline and client.get("/replay/status").json()["rows_replayed"] < n_rows:
        time.sleep(0.05)
    client.post("/replay/stop")
    while time.time() < deadline and client.get("/replay/status").json()["status"] == "running":
        time.sleep(0.05)
    events = sorted(store.recent_events(limit=2000), key=lambda e: e["id"])[:n_rows]
    assert len(events) == n_rows
    return [e["tower_site_id"] for e in events]


@requires_model
def test_two_back_to_back_replays_of_the_same_scenario_produce_identical_tower_sequences(client):
    if client.app.state.replay_manager is None:
        pytest.skip("no trained model loaded")
    a = _replay_towers(client, 120)
    b = _replay_towers(client, 120)
    client.app.state.event_store.clear_all()
    assert a == b
    assert a[0] == client.app.state.replay_attributors["epicenter"].epicenter_tower_key
    assert len(set(a)) > 1, "the walk should actually move (a constant sequence would prove nothing)"
