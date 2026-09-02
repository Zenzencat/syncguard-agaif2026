"""Established spatial statistics -- global Moran's I and Local Moran's I (LISA) -- over the
live tower network. A second, distinct GeoAI layer alongside api/spatial.py's hand-rolled
distance-weighted correlation: that module answers "how correlated does this one tower's
neighborhood currently look" per-event; this one answers "right now, across the whole
network, are anomalies spatially clustered or randomly scattered" (global Moran's I), and
"which specific towers are part of a statistically significant cluster, and which are
isolated outliers" (Local Moran's I / LISA). Neither replaces the other; both stay available.
Uses esda/libpysal -- the standard, citable PySAL implementations (Moran, 1950; Anselin, 1995
for LISA) -- not a custom reimplementation.

REAL vs SIMULATED, same discipline as the rest of this project (see api/spatial.py and
spatial_layer_notes.md for the original statement this follows):
  REAL: the 136 tower coordinates; the k-nearest-neighbors spatial weights built from real
    great-circle distance between them (same EARTH_RADIUS_KM constant as api/spatial.py's
    haversine_km); the severity values fed in (real predict_proba output from the trained
    detector, via api/model_service.py); and the Moran's I / Local Moran's I computation
    itself -- a standard, established statistical method applied exactly as the literature
    defines it.
  SIMULATED (inherited, unchanged, not re-introduced here): which physical tower a given
    scored event is attributed to -- see api/spatial.py's TowerAttributor docstring. The
    statistics in this module are 100% real math; the one input that carries a caveat is the
    same one already disclosed everywhere else a scored event surfaces (API responses, DB
    columns, the dashboard).

Missing towers: not every one of the 136 real towers has a scored event at a given moment,
especially early in a replay. This module never imputes a placeholder severity for a
tower with no data -- that would invent a third, unlabeled category of data alongside the
REAL/SIMULATED split. Every computation here runs over whichever subset of towers currently
has at least one scored event (via EventStore.latest_severity_per_tower()), and refuses to
compute anything below MIN_TOWERS_FOR_STATS, returning an explicit "not enough data yet"
result instead of a misleading early number.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local

K_NEIGHBORS = 5          # see SPATIAL_STATISTICS.md for why KNN over distance-band, and why 5
MIN_TOWERS_FOR_STATS = 15  # mathematical floor is K_NEIGHBORS + 1 = 6; 15 gives more
                            # meaningful permutation-test power -- see SPATIAL_STATISTICS.md
N_PERMUTATIONS = 999
RNG_SEED = 42             # same convention as ROBUSTNESS_NOTES.md / train_improved_model.py
SIGNIFICANCE_ALPHA = 0.05
EARTH_RADIUS_KM = 6371.0088  # identical constant to api/spatial.py's haversine_km

# esda's Moran_Local quadrant convention (verified empirically, not just from docs -- see
# SPATIAL_STATISTICS.md's worked example): 1=HH, 2=LH, 3=LL, 4=HL. 0 is this module's own
# convention for "not statistically significant", not an esda value.
QUADRANT_LABELS = {
    0: "Not significant",
    1: "High-High (hotspot)",
    2: "Low-High (spatial outlier)",
    3: "Low-Low (coldspot)",
    4: "High-Low (spatial outlier)",
}


@dataclass
class AutocorrelationResult:
    computable: bool
    n_towers_scored: int
    n_towers_total: int
    min_required: int
    k_neighbors: int | None = None
    global_moran_i: float | None = None
    global_p_value: float | None = None
    global_z_score: float | None = None
    global_expected_i: float | None = None
    reason: str | None = None
    per_tower: list[dict] = field(default_factory=list)


def build_knn_weights(towers_subset: pd.DataFrame, k: int) -> KNN:
    """towers_subset must have 'lon'/'lat' columns, in the SAME row order the caller will use
    for the severity array -- KNN.from_array's neighbor indices are positional."""
    coords = towers_subset[["lon", "lat"]].values
    w = KNN.from_array(coords, k=k, radius=EARTH_RADIUS_KM)
    w.transform = "r"
    return w


def compute_autocorrelation(towers: pd.DataFrame, latest_by_tower: dict[str, dict]) -> AutocorrelationResult:
    """towers: the full 136-row REAL tower table (must have tower_key, site_id, site_name,
    lat, lon -- see api/spatial.py::load_towers()). latest_by_tower: the dict returned by
    EventStore.latest_severity_per_tower() (tower_key -> most recent scored event). Only
    towers present in latest_by_tower are used; nothing is imputed for the rest."""
    towers_idx = towers.set_index("tower_key")
    n_total = len(towers)
    present_keys = [k for k in latest_by_tower if k in towers_idx.index]
    n_scored = len(present_keys)

    if n_scored < MIN_TOWERS_FOR_STATS:
        return AutocorrelationResult(
            computable=False, n_towers_scored=n_scored, n_towers_total=n_total,
            min_required=MIN_TOWERS_FOR_STATS,
            reason=f"Only {n_scored}/{MIN_TOWERS_FOR_STATS} required towers have a scored "
                   f"event yet -- statistics not meaningful this early in a replay.",
        )

    sub = towers_idx.loc[present_keys].reset_index()  # row order == present_keys order
    severities = np.array([latest_by_tower[k]["severity"] for k in present_keys], dtype=float)

    if np.isclose(severities.std(), 0.0):
        return AutocorrelationResult(
            computable=False, n_towers_scored=n_scored, n_towers_total=n_total,
            min_required=MIN_TOWERS_FOR_STATS,
            reason="All scored towers currently report identical severity -- Moran's I is "
                   "undefined with zero variance.",
        )

    k = min(K_NEIGHBORS, n_scored - 1)
    w = build_knn_weights(sub, k=k)

    # esda.moran.Moran (this version, 2.10.0) has no seed= parameter -- it draws permutations
    # from numpy's global RNG, so reproducibility requires reseeding immediately before the
    # call. Verified empirically (see SPATIAL_STATISTICS.md): two calls with the same reseed
    # give bit-identical I and p_sim. Moran_Local *does* take seed= directly (and n_jobs=1,
    # passed explicitly here even though it's already the default, for the same determinism
    # discipline as api/model_service.py).
    np.random.seed(RNG_SEED)
    global_mi = Moran(severities, w, permutations=N_PERMUTATIONS)
    local_mi = Moran_Local(severities, w, permutations=N_PERMUTATIONS, seed=RNG_SEED, n_jobs=1)

    per_tower = []
    for i, tower_key in enumerate(present_keys):
        row = sub.iloc[i]
        significant = bool(local_mi.p_sim[i] < SIGNIFICANCE_ALPHA)
        quadrant = int(local_mi.q[i]) if significant else 0
        per_tower.append({
            "tower_key": tower_key,
            "site_id": row["site_id"],
            "site_name": row["site_name"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "severity": float(severities[i]),
            "local_moran_i": float(local_mi.Is[i]),
            "p_value": float(local_mi.p_sim[i]),
            "significant": significant,
            "lisa_quadrant": quadrant,
            "lisa_label": QUADRANT_LABELS[quadrant],
        })

    return AutocorrelationResult(
        computable=True, n_towers_scored=n_scored, n_towers_total=n_total,
        min_required=MIN_TOWERS_FOR_STATS, k_neighbors=k,
        global_moran_i=float(global_mi.I), global_p_value=float(global_mi.p_sim),
        global_z_score=float(global_mi.z_sim), global_expected_i=float(global_mi.EI),
        per_tower=per_tower,
    )
