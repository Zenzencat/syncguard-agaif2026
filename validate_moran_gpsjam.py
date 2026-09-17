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

Status: RUN, against real data, both modes -- see GPSJAM_VALIDATION.md for full results.
gpsjam.org itself is blocked by this environment's network egress policy (confirmed: CONNECT
tunnel fails with HTTP 403, an organization-level policy decision, reported rather than routed
around), so the manifest and the seven daily CSVs used here (2024-04-01 through 2024-04-07)
were fetched from a machine that could reach the site and supplied via `--data-dir` instead of
over HTTP. Every fact in this docstring about the data schema was confirmed by reading the
site's actual deployed source code, not assumed.

Spatial weights -- hex contiguity, not KNN: unlike the 136 irregularly-spaced Telkomsel towers
(where KNN k=5 was the defensible choice -- see SPATIAL_STATISTICS.md), H3 hex cells form a
regular grid where every interior cell has exactly 6 edge-adjacent neighbors. There is no
rook-vs-queen distinction on a hex grid (each hex shares a full edge with each neighbor, unlike
a square grid's ambiguous corner-touching case) -- so first-ring contiguity (`h3.grid_disk(hex,
1)` minus the cell itself) is the single natural, standard choice for hex-grid spatial weights,
used here instead of KNN.

Usage:
    python validate_moran_gpsjam.py proof-of-method --start 2024-04-01 --end 2024-04-07 --data-dir gpsjam_raw
    python validate_moran_gpsjam.py region-check --start 2024-04-01 --end 2024-04-07 --data-dir gpsjam_raw
    # Omit --data-dir to fetch from gpsjam.org directly, from an environment that can reach it.

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
LOCAL_DATA_DIR: Path | None = None  # set by main() from --data-dir; when set, reads local
                                     # files instead of hitting the network at all -- used
                                     # when gpsjam.org itself isn't reachable but the specific
                                     # files have been fetched some other way (see
                                     # GPSJAM_VALIDATION.md)
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
    if LOCAL_DATA_DIR is not None:
        path = LOCAL_DATA_DIR / "manifest.csv"
        print(f"  [local] reading {path}")
        return pd.read_csv(path)
    resp = requests.get(f"{BASE_URL}/manifest.csv", timeout=30)
    resp.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(resp.text))


def fetch_day(date_str: str) -> pd.DataFrame:
    """date_str: 'YYYY-MM-DD'. Returns the raw per-hex CSV for that day, globally. Reads from
    LOCAL_DATA_DIR if set (files fetched some other way -- see GPSJAM_VALIDATION.md), else
    hits gpsjam.org directly."""
    if LOCAL_DATA_DIR is not None:
        path = LOCAL_DATA_DIR / f"{date_str}-h3_4.csv"
        if not path.exists():
            raise FileNotFoundError(f"No local file for {date_str}: {path}")
        print(f"  [local] reading {path}")
        df = pd.read_csv(path)
        df["date"] = date_str
        return df
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
        except (requests.HTTPError, FileNotFoundError) as e:
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
        ring = set(h3.grid_disk(h, 1))  # h3-py v4 returns a list, not a set
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
        "_agg": agg, "_w": w, "_severities": severities,
        "_local_q": local_mi.q, "_local_p": local_mi.p_sim,
    }


QUADRANT_COLORS = {0: "#c9ccd1", 1: "#d1263b", 2: "#7fb3e8", 3: "#1f5fa8", 4: "#f0a35c"}
QUADRANT_LABELS = {
    0: "Not significant", 1: "High-High (hotspot)", 2: "Low-High (spatial outlier)",
    3: "Low-Low (coldspot)", 4: "High-Low (spatial outlier)",
}


def plot_result(result: dict, label: str, out_prefix: Path):
    """LISA map + Moran scatter for a computable result -- same visual convention as
    build_spatial_autocorrelation_live_demo.py / build_spatial_autocorrelation_persistent_demo.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = result["_agg"]
    w = result["_w"]
    sev = result["_severities"]
    quads = np.array([
        int(q) if p < SIGNIFICANCE_ALPHA else 0
        for q, p in zip(result["_local_q"], result["_local_p"])
    ])
    lats = agg["lat"].to_numpy()
    lons = agg["lon"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.2, 8.2), dpi=200)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for q in [0, 2, 4, 1, 3]:
        mask = quads == q
        if not mask.any():
            continue
        ax.scatter(lons[mask], lats[mask], s=90 if q != 0 else 45,
                   c=QUADRANT_COLORS[q], edgecolors="#2a2a2a" if q != 0 else "none",
                   linewidths=0.6, alpha=0.95 if q != 0 else 0.6,
                   label=f"{QUADRANT_LABELS[q]} ({mask.sum()})", zorder=3 if q != 0 else 2)
    ax.set_xlabel("Longitude", fontsize=11); ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title(f"Local Moran's I (LISA) -- real GPSJam H3 hexes\n{label}",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.15)
    for spine in ax.spines.values():
        spine.set_color("#888888")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_lisa_map.png", facecolor="white")
    plt.close(fig)

    z = (sev - sev.mean()) / sev.std()
    lag = w.sparse @ z
    colors = [QUADRANT_COLORS[q] for q in quads]
    fig2, ax2 = plt.subplots(figsize=(8.2, 8.2), dpi=200)
    fig2.patch.set_facecolor("white"); ax2.set_facecolor("white")
    ax2.scatter(z, lag, c=colors, s=60, edgecolors="#2a2a2a", linewidths=0.5, alpha=0.9, zorder=3)
    ax2.axhline(0, color="#999999", linewidth=0.8); ax2.axvline(0, color="#999999", linewidth=0.8)
    m, b = np.polyfit(z, lag, 1)
    xs = np.linspace(z.min(), z.max(), 100)
    sig_str = "significant" if result["p_value"] < SIGNIFICANCE_ALPHA else "not significant"
    ax2.plot(xs, m * xs + b, color="#d1263b", linewidth=2.2, zorder=4,
             label=f"Moran's I = {result['global_moran_i']:.3f}  (p = {result['p_value']:.3f}, {sig_str})")
    ax2.set_xlabel("Standardized bad_frac (z)", fontsize=11)
    ax2.set_ylabel("Spatial lag (H3 first-ring contiguity)", fontsize=11)
    ax2.set_title(f"Moran Scatter Plot -- real GPSJam H3 hexes\n{label}", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax2.grid(True, alpha=0.15)
    for spine in ax2.spines.values():
        spine.set_color("#888888")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_moran_scatter.png", facecolor="white")
    plt.close(fig2)
    print(f"  Wrote {out_prefix}_lisa_map.png")
    print(f"  Wrote {out_prefix}_moran_scatter.png")


def daterange(start: str, end: str) -> list[str]:
    dates = pd.date_range(start, end, freq="D")
    return [d.strftime("%Y-%m-%d") for d in dates]


def main():
    global LOCAL_DATA_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["proof-of-method", "region-check", "manifest"])
    parser.add_argument("--start", help="YYYY-MM-DD")
    parser.add_argument("--end", help="YYYY-MM-DD")
    parser.add_argument("--data-dir", help="Read manifest.csv/{date}-h3_4.csv from this local "
                         "directory instead of gpsjam.org (used when the host itself isn't "
                         "reachable but the specific files were fetched some other way -- see "
                         "GPSJAM_VALIDATION.md).")
    args = parser.parse_args()
    if args.data_dir:
        LOCAL_DATA_DIR = Path(args.data_dir)

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
        printable = {k: v for k, v in result.items() if not k.startswith("_")}
        print(f"Result: {printable}")
        prefix = OUT_DIR / f"gpsjam_{region}_{args.start}_{args.end}"
        plot_result(result, label, prefix)


if __name__ == "__main__":
    sys.exit(main())
