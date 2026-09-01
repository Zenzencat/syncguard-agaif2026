"""Live spatial correlation over REAL neighboring towers -- distinct from, and not a
replacement for, build_spatial_simulation.py's offline SIMULATED-epicenter/distance-decay
layer (that script, its outputs in spatial_processed/, and spatial_layer_notes.md are
untouched by this module).

What is REAL here:
  - The 136 Telkomsel tower coordinates/site metadata (same source as the offline layer:
    spatial_raw/Module 6_AD1002_Dataset (Tower)/menaratelepon_ar_50k.csv).
  - The haversine distance computation between towers.
  - The correlation itself: when a tower's live-scored severity crosses a threshold, its
    correlation score is a genuine distance-weighted aggregate of *other real towers'
    actual recent scored severities* within a real time window -- not a fixed, hand-picked
    epicenter with an assumed spread. If three real neighboring towers all happen to score
    high in the same window, that's what drives this tower's correlation score up; if none
    do, it doesn't, regardless of distance.

What is SIMULATED here (see TowerAttributor below):
  - Which physical tower a given scored event "comes from". The Jammertest 2024 dataset is a
    single receiver's log, not a multi-tower deployment, so there is no real per-event tower
    attribution to use. Events are assigned to real towers by deterministic round-robin
    cycling (not hand-picked, not random-per-run) -- clearly SIMULATED and documented as such
    everywhere this module's output surfaces (API response, DB column comments, dashboard).
  - The exponential distance-decay weighting function and its decay constant, same as the
    offline layer: a documented simplifying assumption standing in for "nearby infrastructure
    sharing correlated GNSS timing anomalies," not a measured RF propagation model.

Net effect vs. the offline layer: the offline layer asks "if a spoofing event started at one
fixed, simulated epicenter, how would simulated severity plausibly spread over real
geography?" This module asks "given what has actually been scored recently at each real
tower (via the live replay/API), how correlated does this tower's neighborhood currently
look?" -- computed fresh from live data every time, not from a fixed assumed origin.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd

TOWER_CSV = (Path(__file__).resolve().parent.parent / "spatial_raw" /
             "Module 6_AD1002_Dataset (Tower)" / "menaratelepon_ar_50k.csv")

# Same decay constant and formula as build_spatial_simulation.py's DECAY_KM_SIMULATED, reused
# here for consistency -- it is exactly as much a modeling assumption in this live setting as
# it is in the offline one; nothing about applying it to live data makes it "more real".
DECAY_KM = 2.0
CORRELATION_WINDOW_SECONDS = 120


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_towers() -> pd.DataFrame:
    """REAL tower data, with one real-world wrinkle handled: 16 of the 136 rows in
    menaratelepon_ar_50k.csv share the literal site_id "tbg" (a placeholder used by whoever
    compiled the source spreadsheet for these particular sites, not a data error we're
    introducing) -- they're still 136 distinct real towers with distinct real coordinates and
    (mostly) distinct real site_name values, just not unique on site_id. Everything in this
    module needs a unique per-tower key for lookups/joins, so a `tower_key` column
    disambiguates duplicates deterministically (site_id, or site_id__N for the Nth
    duplicate) -- this is the key used internally (attribution, correlation, the DB's
    tower_site_id column); site_id/site_name themselves are left exactly as sourced."""
    df = pd.read_csv(TOWER_CSV)
    towers = df[["site_id", "site_name", "desa", "kec", "lat", "long"]].copy()
    towers = towers.rename(columns={"long": "lon"}).reset_index(drop=True)
    dup = towers["site_id"].duplicated(keep=False)
    towers["tower_key"] = towers["site_id"]
    towers.loc[dup, "tower_key"] = (
        towers.loc[dup, "site_id"] + "__" + towers.loc[dup].groupby("site_id").cumcount().astype(str)
    )
    assert towers["tower_key"].is_unique, "tower_key must be unique after disambiguation"
    return towers


class TowerAttributor:
    """SIMULATED: assigns each live-scored event to a real tower by deterministic
    round-robin. See module docstring -- this is the one part of this module that is not
    grounded in real per-event data, because no such data exists for this dataset."""

    def __init__(self, towers: pd.DataFrame):
        self._towers = towers.reset_index(drop=True)
        self._next_idx = 0

    def next_tower(self) -> dict:
        row = self._towers.iloc[self._next_idx % len(self._towers)]
        self._next_idx += 1
        return {
            "site_id": row["tower_key"],  # disambiguated, unique -- see load_towers()
            "site_name": row["site_name"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
        }

    @property
    def towers(self) -> pd.DataFrame:
        return self._towers


@dataclass
class CorrelationResult:
    correlation_score: float
    n_neighbors_considered: int
    window_seconds: int
    decay_km: float


class LiveCorrelationEngine:
    """Real distance-weighted correlation against real neighboring towers' real recent
    scored severities, within a real time window. Queries the shared EventStore rather than
    keeping its own state, so correlation always reflects exactly what's persisted."""

    def __init__(self, towers: pd.DataFrame, event_store, window_seconds: int = CORRELATION_WINDOW_SECONDS,
                 decay_km: float = DECAY_KM):
        self._towers = towers.set_index("tower_key")  # unique key -- see load_towers()
        self._store = event_store
        self.window_seconds = window_seconds
        self.decay_km = decay_km

    def correlate(self, tower_site_id: str, exclude_event_id: int | None = None) -> CorrelationResult:
        if tower_site_id not in self._towers.index:
            return CorrelationResult(0.0, 0, self.window_seconds, self.decay_km)

        this_tower = self._towers.loc[tower_site_id]
        since = (datetime.now(timezone.utc) - timedelta(seconds=self.window_seconds)).isoformat()
        recent = self._store.recent_events_all_towers_since(since)
        recent = [e for e in recent
                  if e["tower_site_id"] != tower_site_id and e["id"] != exclude_event_id]

        if not recent:
            return CorrelationResult(0.0, 0, self.window_seconds, self.decay_km)

        neighbor_ids = [e["tower_site_id"] for e in recent]
        known = [nid in self._towers.index for nid in neighbor_ids]
        recent = [e for e, ok in zip(recent, known) if ok]
        if not recent:
            return CorrelationResult(0.0, 0, self.window_seconds, self.decay_km)

        neighbor_lats = np.array([self._towers.loc[e["tower_site_id"], "lat"] for e in recent])
        neighbor_lons = np.array([self._towers.loc[e["tower_site_id"], "lon"] for e in recent])
        severities = np.array([e["severity"] for e in recent])

        dist_km = haversine_km(this_tower["lat"], this_tower["lon"], neighbor_lats, neighbor_lons)
        weights = np.exp(-dist_km / self.decay_km)

        # Weighted mean of real recent neighbor severities, weighted by real distance decay.
        # Falls back to 0 if all weights vanish (everything in-window is far away).
        total_weight = weights.sum()
        score = float((weights * severities).sum() / total_weight) if total_weight > 1e-9 else 0.0

        return CorrelationResult(
            correlation_score=round(score, 4),
            n_neighbors_considered=len(recent),
            window_seconds=self.window_seconds,
            decay_km=self.decay_km,
        )
