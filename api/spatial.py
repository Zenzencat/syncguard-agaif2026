"""Live spatial correlation over REAL neighboring towers -- distinct from, and not a
replacement for, build_spatial_simulation.py's offline SIMULATED-epicenter/distance-decay
layer (that script, its outputs in spatial_processed/, and spatial_layer_notes.md are
untouched by this module).

What is REAL here:
  - The 136 real telecom tower coordinates/site metadata, operator labels mixed (same source
    as the offline layer: spatial_raw/Module 6_AD1002_Dataset (Tower)/menaratelepon_ar_50k.csv).
  - The haversine distance computation between towers.
  - The correlation itself: when a tower's live-scored severity crosses a threshold, its
    correlation score is a genuine distance-weighted aggregate of *other real towers'
    actual recent scored severities* within a real time window -- not a fixed, hand-picked
    epicenter with an assumed spread. If three real neighboring towers all happen to score
    high in the same window, that's what drives this tower's correlation score up; if none
    do, it doesn't, regardless of distance.

What is SIMULATED here (see TowerAttributor, SpatiallyPersistentAttributor and
EpicenterRandomWalk below):
  - Which physical tower a given scored event "comes from". The Jammertest 2024 dataset is a
    single receiver's log, not a multi-tower deployment, so there is no real per-event tower
    attribution to use, and no ground-truth fix is possible for any mechanism below -- the
    events happened at a test range in Norway, the towers are real infrastructure in
    Indonesia. Three SIMULATED attribution mechanisms are available: `TowerAttributor`
    (deterministic round-robin cycling, memoryless, no relationship to geography -- the API
    default), `SpatiallyPersistentAttributor` (a biased random walk over real tower geometry
    from a random starting tower, encoding the one real, disclosed assumption that sustained
    attacks persist and drift locally rather than teleporting), and
    `EpicenterRandomWalk` (the same persistent walk, anchored to start at a fixed
    simulated epicenter -- what the dashboard's NOC tab requests, so its incident queue has
    something spatially localized to group). All three are clearly SIMULATED and documented
    as such everywhere this module's output surfaces (API response, DB column comments,
    dashboard); see SPATIAL_STATISTICS.md for the honest Moran's I result the first two
    produce.
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
import os
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

    def reset(self) -> None:
        """Back to tower 0. Called at every replay start so a replay's tower sequence does not
        depend on how many events an earlier replay happened to consume."""
        self._next_idx = 0

    @property
    def towers(self) -> pd.DataFrame:
        return self._towers


# Default seed for the SIMULATED epicenter random walk, so a replay's tower attribution is the
# same every run. Override with SYNCGUARD_ATTRIBUTION_SEED (an integer).
DEFAULT_ATTRIBUTION_SEED = 20260921


def attribution_seed_from_env() -> int:
    raw = os.environ.get("SYNCGUARD_ATTRIBUTION_SEED", "").strip()
    if not raw:
        return DEFAULT_ATTRIBUTION_SEED
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"SYNCGUARD_ATTRIBUTION_SEED must be an integer, got {raw!r}") from exc


ATTRIBUTION_K_NEIGHBORS = 5   # deliberately named distinctly from api/spatial_stats.py's
                               # K_NEIGHBORS -- both happen to be 5 (SPATIAL_STATISTICS.md's
                               # "conventional middle value, 4-8" reasoning applies to both),
                               # but they are unrelated parameters of two different mechanisms;
                               # changing one does not imply changing the other.
STAY_PROBABILITY = 0.7        # P(next event stays at the current tower) -- see class docstring


class SpatiallyPersistentAttributor:
    """SIMULATED, alternative to TowerAttributor. Still no ground truth to recover -- the
    spoofing events happened at a test range in Norway, the 136 towers are real infrastructure
    in Indonesia, there was never a real link between "this detected event" and "this specific
    tower." This does not change that. What it changes is which *placeholder* mechanism
    generates the SIMULATED attribution, replacing round-robin's memoryless cycling (each
    event's tower is independent of the last, fixed order, no relationship to anything real --
    which structurally guarantees near-zero spatial autocorrelation almost by construction,
    the same way the old offline epicenter-decay CSV structurally guaranteed a strong positive
    one) with a mechanism that encodes one real, statable, and honestly-disclosed assumption:
    sustained attacks tend to persist and drift locally, not teleport to a random distant
    tower on every single event.

    Mechanism -- a biased random walk over the real tower graph:
      1. First event of a session: pick a uniformly random real tower.
      2. Each subsequent event: stay at the current tower with probability STAY_PROBABILITY
         (default 0.7); otherwise move to one of its ATTRIBUTION_K_NEIGHBORS nearest real
         towers (real haversine distance, same constant as haversine_km below), chosen with
         probability inversely proportional to distance (closer neighbors more likely).

    This produces a spatially-coherent path across real geography -- still SIMULATED, still
    not a claim about where any real attack was, but a more structurally defensible
    placeholder than a mechanism with no relationship to geography at all. See
    SPATIAL_STATISTICS.md's "Attribution methodology" section for the honest result this
    produces and how it compares to round-robin's.
    """

    def __init__(self, towers: pd.DataFrame, stay_prob: float = STAY_PROBABILITY,
                 k: int = ATTRIBUTION_K_NEIGHBORS, seed: int | None = None,
                 start_tower_key: str | None = None):
        """`start_tower_key`: if given, the walk's first tower is fixed to this tower_key
        instead of a uniformly random one -- used by EpicenterRandomWalk below to
        anchor the walk's start at a fixed simulated epicenter while keeping this class's
        real k-NN persistence for every step after that."""
        self._towers = towers.reset_index(drop=True)
        self._stay_prob = stay_prob
        self._k = k
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._current_idx: int | None = None
        self._start_idx: int | None = None
        if start_tower_key is not None:
            matches = self._towers.index[self._towers["tower_key"] == start_tower_key]
            if len(matches):
                self._start_idx = int(matches[0])
        self._neighbor_idx, self._neighbor_weights = self._build_neighbor_table()

    def _build_neighbor_table(self) -> tuple[np.ndarray, np.ndarray]:
        """Precompute, for every tower, its k nearest real neighbors (by real haversine
        distance) and inverse-distance move probabilities -- built once at construction, not
        recomputed per event."""
        n = len(self._towers)
        lats = self._towers["lat"].to_numpy()
        lons = self._towers["lon"].to_numpy()
        neighbor_idx = np.zeros((n, self._k), dtype=int)
        neighbor_weights = np.zeros((n, self._k), dtype=float)
        for i in range(n):
            dist = haversine_km(lats[i], lons[i], lats, lons)
            dist[i] = np.inf  # exclude self as a neighbor of itself
            nearest = np.argsort(dist)[: self._k]
            inv = 1.0 / np.maximum(dist[nearest], 1e-6)
            neighbor_idx[i] = nearest
            neighbor_weights[i] = inv / inv.sum()
        return neighbor_idx, neighbor_weights

    def reset(self) -> None:
        """Back to the start of the walk: forget the current tower and re-seed the RNG, so one
        seed always gives one tower sequence. Called at every replay start. Before this, the walk
        carried on from wherever the previous replay left it, on a random stream nothing had
        seeded, so two replays of the same scenario got different tower sequences. With
        seed=None the RNG is fresh OS entropy again, i.e. still non-deterministic by choice."""
        self._rng = np.random.default_rng(self._seed)
        self._current_idx = None

    def next_tower(self) -> dict:
        if self._current_idx is None:
            self._current_idx = (self._start_idx if self._start_idx is not None
                                 else int(self._rng.integers(0, len(self._towers))))
        elif self._rng.random() >= self._stay_prob:
            candidates = self._neighbor_idx[self._current_idx]
            weights = self._neighbor_weights[self._current_idx]
            self._current_idx = int(self._rng.choice(candidates, p=weights))
        row = self._towers.iloc[self._current_idx]
        return {
            "site_id": row["tower_key"],  # disambiguated, unique -- see load_towers()
            "site_name": row["site_name"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
        }

    @property
    def towers(self) -> pd.DataFrame:
        return self._towers


class EpicenterRandomWalk(SpatiallyPersistentAttributor):
    """SIMULATED, alternative to TowerAttributor: a real k-NN persistent walk (see
    SpatiallyPersistentAttributor above -- same stay/move mechanism, same real haversine
    neighbor table) anchored to start at a fixed simulated epicenter instead of a uniformly
    random tower, so a live replay reads as "an attack starting near <epicenter> and
    drifting locally" instead of round-robin's structurally-random spread across all 136
    towers. This exists so the NOC incident queue (api/incidents.py, which requires nearby
    towers AND nearby time to join one incident) has something spatially coherent to group
    under live replay -- round-robin stays the API default and is what tests / the
    spatial-statistics results still exercise.

    Named for what it does now: a seeded random walk that starts at the epicenter. (It used to
    be called EpicenterRandomWalk, but nothing is weighted by distance from the epicenter
    any more -- the v1 mechanism below was replaced.) Deterministic by default: it is seeded
    (DEFAULT_ATTRIBUTION_SEED, override with SYNCGUARD_ATTRIBUTION_SEED) and reset() at every
    replay start, so two replays of one scenario produce the same tower sequence.

    v1 of this class independently redrew a tower every event, weighted by
    exp(-distance_from_epicenter_km / decay_km) (the same decay narrative
    build_spatial_simulation.py uses for the static Severity Map). MEASURED on a fresh
    database, default Spoofing 2.1.1 replay: that gave 742 incidents (mean 2.56 events each,
    54% single-event) -- an improvement on round-robin's 1,695, but nowhere near a usably
    small, localized incident count, because independent per-draw sampling does not make
    *consecutive* draws land near each other, even when each draw individually favors the
    epicenter's neighborhood. Anchoring SpatiallyPersistentAttributor's already-real,
    already-measured (SPATIAL_STATISTICS.md) persistence mechanism at the epicenter fixes
    that: consecutive events now share a real tower or one of its real nearest neighbors far
    more often, by construction.

    The epicenter is picked the same way build_spatial_simulation.py picks it -- the real
    tower nearest the geometric centroid of all 136 real tower coordinates, deterministic,
    not cherry-picked -- computed independently here rather than importing that script
    (a standalone plotting/report tool, not a library, and left untouched).

    Still SIMULATED tower attribution: there is no real per-event link between a Norway
    test-range GNSS recording and any Indonesian tower. This changes only which placeholder
    mechanism generates that attribution.
    """

    def __init__(self, towers: pd.DataFrame, stay_prob: float = STAY_PROBABILITY,
                 k: int = ATTRIBUTION_K_NEIGHBORS, seed: int | None = None):
        towers = towers.reset_index(drop=True)
        lats = towers["lat"].to_numpy()
        lons = towers["lon"].to_numpy()
        centroid_lat, centroid_lon = float(lats.mean()), float(lons.mean())
        d_to_centroid = haversine_km(centroid_lat, centroid_lon, lats, lons)
        epicenter_idx = int(np.argmin(d_to_centroid))
        self.epicenter_tower_key = towers.iloc[epicenter_idx]["tower_key"]
        self.epicenter_site_name = towers.iloc[epicenter_idx]["site_name"]
        super().__init__(towers, stay_prob=stay_prob, k=k, seed=seed,
                         start_tower_key=self.epicenter_tower_key)


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
