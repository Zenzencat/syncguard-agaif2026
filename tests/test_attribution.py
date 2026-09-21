"""SIMULATED replay tower-attribution mechanisms (api/spatial.py). Tower coordinates here are
synthetic fixtures, not real data -- same convention as tests/test_priority.py.
"""
from __future__ import annotations

import pandas as pd

from api.spatial import EpicenterRandomWalk, SpatiallyPersistentAttributor


def _towers() -> pd.DataFrame:
    # A tight cluster centered exactly on "center", plus two outliers placed symmetrically
    # opposite each other so they cancel out in the centroid instead of dragging it off to one
    # side -- keeps "center" unambiguously nearest the centroid.
    return pd.DataFrame([
        {"tower_key": "center", "site_id": "center", "site_name": "Center", "lat": 0.000, "lon": 100.000},
        {"tower_key": "near1", "site_id": "near1", "site_name": "Near1", "lat": 0.005, "lon": 100.000},
        {"tower_key": "near2", "site_id": "near2", "site_name": "Near2", "lat": -0.005, "lon": 100.000},
        {"tower_key": "near3", "site_id": "near3", "site_name": "Near3", "lat": 0.000, "lon": 100.005},
        {"tower_key": "far", "site_id": "far", "site_name": "Far", "lat": 5.000, "lon": 105.000},
        {"tower_key": "far2", "site_id": "far2", "site_name": "Far2", "lat": -5.000, "lon": 95.000},
    ])


def test_epicenter_is_the_real_tower_nearest_the_centroid():
    attributor = EpicenterRandomWalk(_towers(), seed=1)
    assert attributor.epicenter_tower_key == "center"


def test_first_event_lands_on_the_epicenter():
    attributor = EpicenterRandomWalk(_towers(), seed=1)
    first = attributor.next_tower()
    assert first["site_id"] == "center"


def test_stay_probability_one_never_moves():
    attributor = EpicenterRandomWalk(_towers(), stay_prob=1.0, seed=1)
    towers = [attributor.next_tower()["site_id"] for _ in range(10)]
    assert towers == ["center"] * 10


def test_stay_probability_zero_always_moves_to_a_real_neighbor():
    attributor = EpicenterRandomWalk(_towers(), stay_prob=0.0, k=3, seed=1)
    towers = [attributor.next_tower()["site_id"] for _ in range(10)]
    assert towers[0] == "center"
    assert all(t not in ("far", "far2") for t in towers), "the distant outliers should never be a nearest neighbor of the cluster"
    assert len(set(towers[1:])) >= 1  # moves among the tight cluster's neighbors


def test_deterministic_with_a_fixed_seed():
    seq_a = [EpicenterRandomWalk(_towers(), seed=42).next_tower()["site_id"] for _ in range(8)]
    seq_b = [EpicenterRandomWalk(_towers(), seed=42).next_tower()["site_id"] for _ in range(8)]
    assert seq_a == seq_b


def test_next_tower_shape_matches_other_attributors():
    attributor = EpicenterRandomWalk(_towers(), seed=1)
    tower = attributor.next_tower()
    assert set(tower.keys()) == {"site_id", "site_name", "lat", "lon"}


def test_spatially_persistent_attributor_start_tower_key_anchors_first_event():
    attributor = SpatiallyPersistentAttributor(_towers(), start_tower_key="far", seed=1)
    assert attributor.next_tower()["site_id"] == "far"


def test_spatially_persistent_attributor_default_start_is_still_random():
    # No start_tower_key -- behaviour is unchanged from before this option existed: some
    # uniformly random tower, not necessarily "far".
    seen = {SpatiallyPersistentAttributor(_towers(), seed=s).next_tower()["site_id"]
            for s in range(20)}
    assert len(seen) > 1, "20 different seeds should not all pick the same first tower"
