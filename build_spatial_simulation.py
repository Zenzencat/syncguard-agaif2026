"""
SyncGuard spatial layer builder.

Produces a NEW, clearly-labeled SIMULATED spatial anomaly-spread layer laid on top of
REAL ASEAN telecom infrastructure locations (136 real Telkomsel tower sites, Kubu Raya /
Pontianak, West Kalimantan, Indonesia -- source: menaratelepon_ar_50k.csv, bootcamp
Module 6 materials).

What is REAL here: the 136 tower coordinates/site metadata, and the RandomForest
detector's predict_proba() output distribution (reproduced in-memory from the exact
validated Pipeline in train_baseline_model.py, self-checked against
baseline_model_report.md before use -- see reproduce_validated_model() below).

What is SIMULATED here: which site is the spoofing epicenter, and how "detected anomaly
severity" spreads to geographically nearby sites. No public dataset of real ASEAN
base-station GNSS timing under spoofing exists (see dataset_notes.md), so this spread is a
documented simplifying assumption (distance decay), NOT measured data. Every derived
column/variable/file that comes out of the simulated step is suffixed `_SIMULATED` (or
lives under a `SIMULATED` file name) so the real/simulated line can never be lost by
truncation, copy-paste, or a stray rename.

Does NOT modify baseline_model_report.md, extract_features.py, or train_baseline_model.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless -- no display server, no blocking GUI backend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

sys.stdout.reconfigure(encoding="utf-8")

DATASET_DIR = Path(__file__).resolve().parent
NORWAY_PROCESSED_DIR = DATASET_DIR / "processed"
TOWER_CSV = DATASET_DIR / "spatial_raw" / "Module 6_AD1002_Dataset (Tower)" / "menaratelepon_ar_50k.csv"
OUT_DIR = DATASET_DIR / "spatial_processed"
OUT_DIR.mkdir(exist_ok=True)

RNG_SEED = 42  # same seed used throughout this repo's model code, for reproducibility


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Step 1: reproduce the ALREADY-VALIDATED detector in-memory (verbatim logic
# from demo_prediction_visualization.py / train_baseline_model.py) and pull
# its REAL predict_proba() output distribution -- this grounds the simulated
# severity scores in the real model's actual confidence behavior instead of
# made-up numbers.
# ---------------------------------------------------------------------------
def reproduce_validated_model():
    log("[1/4] Reproducing validated RandomForest detector in-memory (Norway/Jammertest data)...")
    df = pd.read_parquet(NORWAY_PROCESSED_DIR / "syncguard_features.parquet")
    df["real_time"] = pd.to_datetime(df["real_time"])

    TEST_SELECTION = [
        ("Jamming", "1.10.6", "dynamic"),
        ("Jamming", "1.6.4", "stationary"),
        ("Meaconing", "3.2.8", "dynamic"),
        ("Meaconing", "3.2.7", "stationary"),
        ("Spoofing", "2.3.2", "dynamic"),
        ("Spoofing", "2.1.1", "stationary"),
        ("Spoofing + Jamming", "2.6.4", "dynamic"),
        ("Spoofing + Jamming", "2.6.3", "stationary"),
    ]
    test_run_ids = set()
    for atype, sid, rstate in TEST_SELECTION:
        match = df.loc[(df["attack_type"] == atype) & (df["scenario_id"] == sid) &
                        (df["rover_state"] == rstate), "run_id"].unique()
        assert len(match) == 1, f"Expected exactly 1 run for {(atype, sid, rstate)}"
        test_run_ids.add(match[0])

    EXCLUDE_COLS = {
        "iTOW", "real_time", "lat", "lon", "height", "clock_drift_proxy_s",
        "attack", "scenario_id", "attack_type", "rover_state", "power_w_max", "bands",
        "n_attack_windows", "window_start_in_range", "run_id",
    }
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    is_test = df["run_id"].isin(test_run_ids)
    train_df, test_df = df[~is_test], df[is_test]
    X_train, y_train = train_df[feature_cols], train_df["attack"]
    X_test, y_test = test_df[feature_cols], test_df["attack"]

    clf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RNG_SEED, n_jobs=-1)),
    ])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    reproduced_report = classification_report(
        y_test, y_pred, target_names=["clean(0)", "attack(1)"], digits=3)
    published_report = (DATASET_DIR / "baseline_model_report.md").read_text(encoding="utf-8")
    if reproduced_report.strip() not in published_report:
        log("Self-check FAILED: reproduced report does not match baseline_model_report.md. Stopping.")
        log(reproduced_report)
        sys.exit(1)
    log("  Self-check PASSED -- this is the same validated model as baseline_model_report.md.")

    proba_test = clf.predict_proba(X_test)[:, 1]
    attack_proba = proba_test[y_test.values == 1]
    clean_proba = proba_test[y_test.values == 0]
    log(f"  Real model proba on held-out attack rows: median={np.median(attack_proba):.3f}, "
        f"n={len(attack_proba)}")
    log(f"  Real model proba on held-out clean rows:  median={np.median(clean_proba):.3f}, "
        f"n={len(clean_proba)}")
    return attack_proba, clean_proba


# ---------------------------------------------------------------------------
# Step 2: real tower locations (136 real Telkomsel sites, Kubu Raya/Pontianak)
# ---------------------------------------------------------------------------
def load_real_towers():
    log("[2/4] Loading real tower locations (menaratelepon_ar_50k.csv)...")
    df = pd.read_csv(TOWER_CSV)
    towers = df[["site_id", "site_name", "desa", "kec", "lat", "long"]].copy()
    towers = towers.rename(columns={"long": "lon"})
    log(f"  Loaded {len(towers)} real tower sites.")
    return towers


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Step 3: SIMULATED spoofing-spread layer, anchored to the real model's own
# proba distribution (not arbitrary numbers).
# ---------------------------------------------------------------------------
DECAY_KM_SIMULATED = 2.0  # simplifying assumption: severity decays e-fold every ~2km
                           # (documented in spatial_layer_notes.md -- not a measured RF
                           # propagation constant, calibrated so the flagged set grows from
                           # ~3 sites at t=0 to ~15-20 sites at the final snapshot, out of
                           # 136 total -- a plausible localized-neighborhood footprint, not
                           # "everything nearby lights up")


def build_simulated_spread(towers, attack_proba, clean_proba, n_snapshots=4):
    log("[3/4] Building SIMULATED distance-decay anomaly spread over the real tower network...")

    centroid_lat, centroid_lon = towers["lat"].mean(), towers["lon"].mean()
    d_to_centroid = haversine_km(towers["lat"], towers["lon"], centroid_lat, centroid_lon)
    epicenter_idx = d_to_centroid.idxmin()
    epicenter = towers.loc[epicenter_idx]
    log(f"  SIMULATED epicenter (deterministic pick: real site nearest the cluster's "
        f"geometric centroid, not cherry-picked): {epicenter['site_id']} / {epicenter['site_name']}")

    dist_km = haversine_km(epicenter["lat"], epicenter["lon"], towers["lat"], towers["lon"])

    floor = float(np.median(clean_proba))       # real model's own "quiet" confidence level
    ceiling = float(np.percentile(attack_proba, 90))  # real model's own strong-detection level

    snapshots = []
    max_radius_km = dist_km.max()
    for t in range(n_snapshots):
        frac = (t + 1) / n_snapshots
        decay_km_t = DECAY_KM_SIMULATED * (0.5 + 1.5 * frac)  # spread widens over time steps
        severity = floor + (ceiling - floor) * np.exp(-dist_km / decay_km_t)
        snapshots.append(severity)

    final_severity = snapshots[-1]
    out = towers.copy()
    out["distance_from_epicenter_km"] = dist_km
    out["simulated_anomaly_severity_SIMULATED"] = final_severity
    out["flagged_anomalous_SIMULATED"] = out["simulated_anomaly_severity_SIMULATED"] >= 0.5
    out["priority_rank_SIMULATED"] = out["simulated_anomaly_severity_SIMULATED"].rank(
        ascending=False, method="min").astype(int)
    out = out.sort_values("priority_rank_SIMULATED")

    n_flagged = int(out["flagged_anomalous_SIMULATED"].sum())
    log(f"  SIMULATED result: {n_flagged}/{len(out)} sites flagged anomalous "
        f"(severity >= 0.5), floor={floor:.3f}, ceiling={ceiling:.3f}")

    csv_path = OUT_DIR / "simulated_spatial_anomaly_SIMULATED.csv"
    out.to_csv(csv_path, index=False)
    log(f"  Saved {csv_path}")

    return out, epicenter, snapshots, floor, ceiling


# ---------------------------------------------------------------------------
# Step 4: static map (no basemap tiles / no network calls -- plain lat/lon
# scatter with latitude-corrected aspect ratio, safe for a headless run)
# ---------------------------------------------------------------------------
def plot_map(out, epicenter, floor, ceiling):
    log("[4/4] Rendering SIMULATED spatial anomaly map (static, offline, no basemap tiles)...")
    mean_lat_rad = np.radians(out["lat"].mean())

    fig, ax = plt.subplots(figsize=(11, 9))
    sizes = 30 + 220 * out["simulated_anomaly_severity_SIMULATED"].clip(0, 1)
    sc = ax.scatter(out["lon"], out["lat"], c=out["simulated_anomaly_severity_SIMULATED"],
                     s=sizes, cmap="YlOrRd", vmin=floor, vmax=ceiling,
                     edgecolor="black", linewidth=0.4, zorder=3)
    ax.scatter([epicenter["lon"]], [epicenter["lat"]], marker="*", s=500,
               facecolor="none", edgecolor="blue", linewidth=2, zorder=4)

    top5 = out[out["priority_rank_SIMULATED"] <= 5]
    offsets = [(10, 10), (10, -14), (-70, 10), (10, 24), (-70, -14)]
    for (_, row), (dx, dy) in zip(top5.iterrows(), offsets):
        ax.annotate(f"#{row['priority_rank_SIMULATED']} {row['site_id']}",
                     (row["lon"], row["lat"]), fontsize=7.5, xytext=(dx, dy),
                     textcoords="offset points",
                     arrowprops=dict(arrowstyle="-", lw=0.6, color="gray"))

    ax.set_aspect(1 / np.cos(mean_lat_rad))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "SIMULATED spoofing-spread severity over REAL Telkomsel tower locations\n"
        "(136 real sites, Kubu Raya/Pontianak, West Kalimantan, Indonesia -- "
        "severity/spread is a simulation, coordinates are real)",
        fontsize=10)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("SIMULATED anomaly severity\n(anchored to real detector's proba range)")
    legend_handles = [
        Line2D([0], [0], marker="*", color="blue", linestyle="None", markersize=15,
               markerfacecolor="none", markeredgewidth=2, label="SIMULATED epicenter"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="darkred", markersize=10,
               label="high SIMULATED severity"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightyellow",
               markeredgecolor="black", markersize=10, label="low SIMULATED severity"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 1))
    out_path = OUT_DIR / "spatial_anomaly_map_SIMULATED.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    log(f"  Saved {out_path}")
    return out_path


def plot_spread_over_time(towers, epicenter, snapshots, floor, ceiling):
    log("  Rendering SIMULATED spread-over-time small multiples...")
    mean_lat_rad = np.radians(towers["lat"].mean())
    n = len(snapshots)

    # Zoom to the epicenter's neighborhood -- at full-cluster (130km) extent the spread
    # between snapshots is a few pixels and invisible; this crop is what actually shows it.
    zoom_deg = 0.25
    xlim = (epicenter["lon"] - zoom_deg, epicenter["lon"] + zoom_deg)
    ylim = (epicenter["lat"] - zoom_deg, epicenter["lat"] + zoom_deg)

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.8), sharex=True, sharey=True)
    for t, (ax, sev) in enumerate(zip(axes, snapshots)):
        flagged = sev >= 0.5
        sizes = 25 + 300 * sev.clip(0, 1) ** 3  # cubed so only genuinely high severity grows visibly
        sc = ax.scatter(towers["lon"], towers["lat"], c=sev, s=sizes, cmap="YlOrRd",
                         vmin=floor, vmax=ceiling, edgecolor="black", linewidth=0.4)
        ax.scatter([epicenter["lon"]], [epicenter["lat"]], marker="*", s=280,
                   facecolor="none", edgecolor="blue", linewidth=1.8, zorder=4)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect(1 / np.cos(mean_lat_rad))
        ax.set_title(f"t={t}  ({int(flagged.sum())} flagged)", fontsize=10)
        ax.set_xlabel("Longitude")
        if t == 0:
            ax.set_ylabel("Latitude")
    fig.suptitle(
        "SIMULATED anomaly spread over time (illustrative time steps, real tower "
        "coordinates, simulated propagation) -- widening distance-decay radius per step",
        fontsize=10)
    fig.colorbar(sc, ax=axes, shrink=0.7, label="SIMULATED severity")
    out_path = OUT_DIR / "spatial_anomaly_spread_over_time_SIMULATED.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log(f"  Saved {out_path}")
    return out_path


def main():
    attack_proba, clean_proba = reproduce_validated_model()
    towers = load_real_towers()
    out, epicenter, snapshots, floor, ceiling = build_simulated_spread(towers, attack_proba, clean_proba)
    plot_map(out, epicenter, floor, ceiling)
    plot_spread_over_time(towers, epicenter, snapshots, floor, ceiling)
    log("\nDone. All outputs in " + str(OUT_DIR))


if __name__ == "__main__":
    main()
