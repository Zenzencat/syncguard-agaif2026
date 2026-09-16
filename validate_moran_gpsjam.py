"""Proof-of-method validation: does this project's Moran's I / LISA code path detect real
spatial clustering on real, independently-corroborated GPS-interference data -- with no
synthetic attribution involved at all?

Why this exists: every attribution mechanism tried on the Indonesian tower network (see
SPATIAL_STATISTICS.md's "three-mechanism synthesis") was undone by a structural artifact of
the mechanism itself. None of that ever tested whether Moran's I/LISA *as implemented in this
codebase* actually works on real clustered data -- it only ever tested placeholder attribution,
which was never going to resolve either way. This script tests the method itself, fully
separate from the SyncGuard tower narrative: different domain (aircraft-altitude GPS
interference, not ground-based telecom spoofing), different geometry (global H3 hex grid, not
136 real towers), different purpose (validate the method, not narrate SyncGuard's problem).

Data source: GPSJam.org, daily global GPS-interference H3-resolution-4 hex CSVs derived from
real ADS-B Exchange aircraft navigation-accuracy reports (John Wiseman, CC-BY), back to
2022-02-14. Confirmed directly by reading the site's actual client-side source
(github.com/wiseman/gpsjam.org, mirrored at github.com/guofengji/gpsjam.org since
github.com/wiseman/gpsjam.org itself returned 404 for direct raw fetches -- see
GPSJAM_VALIDATION.md for the full trail), specifically views/index.ejs's loadH3file() and
loadManifest() functions:

  - Per-day data: GET https://gpsjam.org/data/{YYYY-MM-DD}-h3_4.csv
    Columns (confirmed from the site's own d3.csv parsing code): hex, count_good_aircraft,
    count_bad_aircraft. `hex` is an H3 resolution-4 cell index (hex string).
  - Manifest of available dates: GET https://gpsjam.org/data/manifest.csv
    Columns: date, suspect (true/false), num_bad_aircraft_hexes.
  - GPSJam's own interference-level formula (verbatim from their client JS):
        bad_frac = (count_bad_aircraft - 1) / (count_bad_aircraft + count_good_aircraft)
        low  if bad_frac < 0.02
        med  if bad_frac < 0.10
        high otherwise
    The "-1" is GPSJam's own choice (reads as a small-sample continuity correction), not
    something introduced here -- reused verbatim for fidelity to their own definition, as the
    continuous severity value fed into Moran's I below.

IMPORTANT -- this script has NOT been run end-to-end. gpsjam.org is blocked by this
environment's network egress policy (confirmed: CONNECT tunnel fails with HTTP 403, consistent
with an organization-level policy decision, not a transient failure -- per this environment's
own operating rules, that is reported, not routed around). Every fact above about the data
schema was confirmed by reading the site's actual deployed source code, not assumed. The
extraction and Moran's I logic below follows this project's own established conventions
exactly (RNG_SEED=42, 999 permutations, KNN vs. contiguity reasoning) but should be smoke-
tested against one real manifest.csv + one real day's CSV before its output is trusted -- see
GPSJAM_VALIDATION.md for what that would involve and why it wasn't possible from here.

Spatial weights -- hex contiguity, not KNN: unlike the 136 irregularly-spaced Telkomsel towers
(where KNN k=5 was the defensible choice -- see SPATIAL_STATISTICS.md), H3 hex cells form a
regular grid where every interior cell has exactly 6 edge-adjacent neighbors. There is no
rook-vs-queen distinction on a hex grid (each hex shares a full edge with each neighbor, unlike
a square grid's ambiguous corner-touching case) -- so first-ring contiguity (`h3.grid_disk(hex,
1)` minus the cell itself) is the single natural, standard choice for hex-grid spatial weights,
used here instead of KNN.

Usage:
    python validate_moran_gpsjam.py proof-of-method --start 2022-05-01 --end 2022-05-07
    python validate_moran_gpsjam.py region-check --start 2022-05-01 --end 2022-05-07

Requires (not part of requirements.txt/requirements-api.txt -- this is a standalone,
one-off validation utility, not part of the shipped SyncGuard system): requests, h3>=4,
pandas, numpy, libpysal, esda.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import requests
from libpysal.weights import W
from esda.moran import Moran, Moran_Local

BASE_URL = "https://gpsjam.org/data"
H3_RESOLUTION = 4
RNG_SEED = 42  # same convention as api/spatial_stats.py / ROBUSTNESS_NOTES.md
N_PERMUTATIONS = 999
SIGNIFICANCE_ALPHA = 0.05
MIN_HEXES_FOR_STATS = 15  # same convention/value as api/spatial_stats.py's
                           # MIN_TOWERS_FOR_STATS -- see that module for why 15,
                           # not just the mathematical floor of k_neighbors+1

OUT_DIR = Path(__file__).resolve().parent / "spatial_processed"

# Bounding boxes (lat_min, lat_max, lon_min, lon_max) -- real geography, not simulated.
REGIONS = {
    # Black Sea / Crimea / Sea of Azov -- independently corroborated real jamming
    # (news reporting, Bellingcat, EASA advisories) on specific dates since 2022.
    "black_sea_crimea": (43.0, 46.5, 30.0, 37.0),
    # Kubu Raya + Pontianak, West Kalimantan -- the actual SyncGuard target region.
    # Same bounding box already verified clean elsewhere in this project
    # (spatial_layer_notes.md): lat -0.86..+0.10, lon 109.15..110.07.
    "kubu_raya_pontianak": (-0.86, 0.10, 109.15, 110.07),
}


def fetch_manifest() -> pd.DataFrame:
    resp = requests.get(f"{BASE_URL}/manifest.csv", timeout=30)
    resp.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(resp.text))


def fetch_day(date_str: str) -> pd.DataFrame:
    """date_str: 'YYYY-MM-DD'. Returns the raw per-hex CSV for that day, globally."""
    url = f"{BASE_URL}/{date_str}-h3_4.csv"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    df["date"] = date_str
    return df


def add_severity_and_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """Adds GPSJam's own bad_frac formula (verbatim) and real H3-decoded lat/lon centroids."""
    df = df.copy()
    df["bad_frac"] = (df["count_bad_aircraft"] - 1) / (
        df["count_bad_aircraft"] + df["count_good_aircraft"]
    )
    latlon = df["hex"].apply(lambda h: h3.cell_to_latlng(h))
    df["lat"] = latlon.apply(lambda t: t[0])
    df["lon"] = latlon.apply(lambda t: t[1])
    return df


def filter_region(df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    lat_min, lat_max, lon_min, lon_max = REGIONS[region_name]
    mask = (
        (df["lat"] >= lat_min) & (df["lat"] <= lat_max) &
        (df["lon"] >= lon_min) & (df["lon"] <= lon_max)
    )
    return df[mask].reset_index(drop=True)


def aggregate_date_range(dates: list[str], region_name: str) -> pd.DataFrame:
    """Fetches each day, filters to the region, and aggregates count_good/count_bad per hex
    across the whole date range (sum of real per-day aircraft counts) -- gives one severity
    value per hex over the range, same shape as this project's other single-snapshot inputs to
    compute_autocorrelation."""
    frames = []
    for d in dates:
        try:
            day_df = fetch_day(d)
        except requests.HTTPError as e:
            print(f"  [skip] {d}: {e}")
            continue
        day_df = add_severity_and_latlon(day_df)
        region_df = filter_region(day_df, region_name)
        frames.append(region_df[["hex", "lat", "lon", "count_good_aircraft", "count_bad_aircraft"]])
    if not frames:
        raise RuntimeError(f"No data fetched for region={region_name}, dates={dates}")
    all_df = pd.concat(frames, ignore_index=True)
    agg = all_df.groupby(["hex", "lat", "lon"], as_index=False).agg(
        count_good_aircraft=("count_good_aircraft", "sum"),
        count_bad_aircraft=("count_bad_aircraft", "sum"),
    )
    agg["bad_frac"] = (agg["count_bad_aircraft"] - 1) / (
        agg["count_bad_aircraft"] + agg["count_good_aircraft"]
    )
    return agg


def build_hex_contiguity_weights(hexes: list[str]) -> W:
    """First-ring H3 grid contiguity -- the natural spatial-weights choice for a regular hex
    grid (see module docstring for why this replaces KNN here). Neighbors not present in
    `hexes` (outside the region / no data) are dropped, same discipline as
    api/spatial_stats.py never imputing a placeholder for missing towers."""
    hex_set = set(hexes)
    neighbors = {}
    for h in hexes:
        ring = h3.grid_disk(h, 1)
        ring.discard(h)
        neighbors[h] = [n for n in ring if n in hex_set]
    w = W(neighbors)
    w.transform = "r"
    return w


def run_moran(agg: pd.DataFrame, label: str):
    n = len(agg)
    print(f"\n=== {label}: {n} hexes with data ===")
    if n < MIN_HEXES_FOR_STATS:
        print(f"  NOT COMPUTABLE: only {n}/{MIN_HEXES_FOR_STATS} required hexes have data. "
              f"This is an honest result, not a failure -- report it as such, the same way "
              f"api/spatial_stats.py reports 'not enough data yet' rather than a misleading "
              f"early number.")
        return None

    hexes = agg["hex"].tolist()
    w = build_hex_contiguity_weights(hexes)
    isolated = [h for h, ns in w.neighbors.items() if len(ns) == 0]
    if isolated:
        print(f"  {len(isolated)} hex(es) have zero in-region neighbors and will contribute "
              f"nothing to the statistic (same known limitation as any contiguity-weights "
              f"graph at a region boundary): {isolated[:5]}{'...' if len(isolated) > 5 else ''}")

    severities = agg["bad_frac"].to_numpy(dtype=float)
    if np.isclose(severities.std(), 0.0):
        print("  NOT COMPUTABLE: all hexes report identical severity -- Moran's I undefined "
              "with zero variance.")
        return None

    np.random.seed(RNG_SEED)
    global_mi = Moran(severities, w, permutations=N_PERMUTATIONS)
    local_mi = Moran_Local(severities, w, permutations=N_PERMUTATIONS, seed=RNG_SEED, n_jobs=1)

    n_sig = int((local_mi.p_sim < SIGNIFICANCE_ALPHA).sum())
    sig = "SIGNIFICANT" if global_mi.p_sim < SIGNIFICANCE_ALPHA else "not significant"
    print(f"  global_moran_i={global_mi.I:+.4f}  p={global_mi.p_sim:.4f}  ({sig})  "
          f"z={global_mi.z_sim:+.3f}  E[I]={global_mi.EI:+.4f}")
    print(f"  significant LISA hexes: {n_sig}/{n}")
    return {
        "n_hexes": n, "global_moran_i": float(global_mi.I), "p_value": float(global_mi.p_sim),
        "z_score": float(global_mi.z_sim), "expected_i": float(global_mi.EI),
        "n_significant_lisa": n_sig,
    }


def daterange(start: str, end: str) -> list[str]:
    dates = pd.date_range(start, end, freq="D")
    return [d.strftime("%Y-%m-%d") for d in dates]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["proof-of-method", "region-check", "manifest"])
    parser.add_argument("--start", help="YYYY-MM-DD")
    parser.add_argument("--end", help="YYYY-MM-DD")
    args = parser.parse_args()

    if args.mode == "manifest":
        m = fetch_manifest()
        print(m.head(20))
        print(f"... {len(m)} rows total. Earliest: {m['date'].min()}  Latest: {m['date'].max()}")
        return

    dates = daterange(args.start, args.end)
    region = "black_sea_crimea" if args.mode == "proof-of-method" else "kubu_raya_pontianak"
    label = ("Proof-of-method (Black Sea / Crimea, known real jamming)"
             if args.mode == "proof-of-method" else
             "Honest regional check (Kubu Raya / Pontianak)")

    print(f"Fetching {len(dates)} day(s) of real GPSJam data for region={region}...")
    agg = aggregate_date_range(dates, region)
    result = run_moran(agg, label)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"gpsjam_{region}_{args.start}_{args.end}.csv"
    agg.to_csv(out_path, index=False)
    print(f"\nWrote per-hex data to {out_path}")
    if result:
        print(f"Result: {result}")


if __name__ == "__main__":
    sys.exit(main())
