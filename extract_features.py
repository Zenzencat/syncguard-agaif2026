"""
SyncGuard dataset pipeline: Jammertest 2024 (Zenodo 15911589) -> time-series features.

For each scenario folder (scenario.json + rinex.csv + nav_pvt.csv + mon_rf.csv):
  - nav_pvt.csv (~1 Hz receiver PVT solution): position, velocity, DOP, accuracy estimates,
    plus a clock-drift proxy derived from iTOW (receiver's internal GPS time-of-week) vs the
    logging computer's real_time wall-clock timestamps.
  - rinex.csv (per-satellite, per-epoch observables): pseudorange, carrier phase, Doppler,
    SNR (C/N0 proxy) for L1/L2. Aggregated per epoch into a code-Doppler consistency residual
    (a standard spoofing indicator: actual pseudorange rate vs. Doppler-predicted rate) plus
    SNR/Doppler summary stats.
  - mon_rf.csv (~1 Hz RF monitor): jamming indicator and AGC counts, averaged across the two
    antenna paths.

Rows are labeled attack=1/0 using scenario.json's attack_log windows, shifted -2h to convert
from the dataset's local CEST timestamps (mislabeled with a "Z"/UTC suffix) to true UTC,
matching the real_time column. See dataset_notes.md for how this offset was determined.
"""
import json
import re
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_ROOT = SCRIPT_DIR / "raw"
OUT_DIR = SCRIPT_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C = 299792458.0
F_L1 = 1575.42e6
LAMBDA_L1 = C / F_L1

LOCAL_TZ_OFFSET_HOURS = 2  # Bleik, Norway was on CEST (UTC+2) during Jammertest 2024 (Sept 2024)


def find_scenario_dirs(root: Path):
    return sorted(p.parent for p in root.rglob("scenario.json"))


def parse_attack_windows(scenario_json: dict):
    """Pair sequential started/end events into (start_utc, end_utc) windows, applying the
    local->UTC offset. Returns a list of (pd.Timestamp, pd.Timestamp)."""
    windows = []
    pending_start = None
    for ev in scenario_json.get("attack_log", []):
        ts = pd.Timestamp(ev["timestamp_utc"]) - timedelta(hours=LOCAL_TZ_OFFSET_HOURS)
        ts = ts.tz_localize(None)
        if "start" in ev["event"].lower():
            pending_start = ts
        elif "end" in ev["event"].lower():
            if pending_start is not None:
                windows.append((pending_start, ts))
                pending_start = None
    return windows


def label_attack(times: pd.Series, windows):
    label = pd.Series(0, index=times.index, dtype="int8")
    for start, end in windows:
        label |= ((times >= start) & (times <= end)).astype("int8")
    return label


def load_nav_pvt(path: Path) -> pd.DataFrame:
    cols = ["real_time", "iTOW", "fixType", "gSpeed", "hAcc", "vAcc", "sAcc", "headAcc",
            "pDOP", "numSV", "lat", "lon", "height", "velN", "velE", "velD"]
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    df["real_time"] = pd.to_datetime(df["real_time"], errors="coerce")
    df = df.dropna(subset=["real_time"]).sort_values("real_time").reset_index(drop=True)
    for c in cols:
        if c != "real_time" and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clock drift proxy: receiver-internal time progression (iTOW, ms) vs logging wall clock.
    itow_dt = df["iTOW"].diff() / 1000.0
    realtime_dt = df["real_time"].diff().dt.total_seconds()
    df["clock_drift_proxy_s"] = itow_dt - realtime_dt

    # Position deviation from the scenario's own first-60s median fix (stationary scenarios
    # only -- for dynamic scenarios this column is left as NaN; see dataset_notes.md).
    ref_window = df[df["real_time"] < df["real_time"].iloc[0] + timedelta(seconds=60)]
    ref_lat, ref_lon, ref_h = ref_window[["lat", "lon", "height"]].median()
    dlat_m = (df["lat"] - ref_lat) * 111320.0
    dlon_m = (df["lon"] - ref_lon) * 111320.0 * np.cos(np.radians(ref_lat))
    dh_m = df["height"] - ref_h
    df["pos_dev_m"] = np.sqrt(dlat_m**2 + dlon_m**2 + dh_m**2)

    return df


def load_rinex_features(path: Path) -> pd.DataFrame:
    cols = ["time", "satellite", "pseudorange_L1", "doppler_L1", "snr_L1"]
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ["pseudorange_L1", "doppler_L1", "snr_L1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values(["satellite", "time"])

    # A small fraction of snr_L1 samples are corrupted logging artifacts (values up to ~4e7,
    # far outside the physically valid C/N0 range). Drop them rather than let them dominate
    # the mean/std aggregates.
    df.loc[(df["snr_L1"] < 0) | (df["snr_L1"] > 60), "snr_L1"] = np.nan

    # Code-Doppler consistency residual per satellite: actual pseudorange rate vs the rate
    # predicted from the reported Doppler shift. Common-mode receiver clock drift largely
    # cancels between the two terms, leaving multipath/spoofing-induced disagreement.
    # Native per-satellite cadence is ~0.2s; gaps beyond 1s indicate a reacquisition
    # (satellite dropped and relocked on a discontinuous pseudorange), which produces a rate
    # artifact rather than a physically meaningful residual, so those are excluded.
    dt = df.groupby("satellite")["time"].diff().dt.total_seconds()
    dpr = df.groupby("satellite")["pseudorange_L1"].diff()
    actual_rate = dpr / dt
    predicted_rate = -df["doppler_L1"] * LAMBDA_L1
    residual = actual_rate - predicted_rate
    residual[(dt <= 0) | (dt > 1.0) | (residual.abs() > 5000)] = np.nan
    df["pr_doppler_residual"] = residual

    agg = df.groupby("time").agg(
        n_sats_l1=("satellite", "count"),
        snr_l1_mean=("snr_L1", "mean"),
        snr_l1_std=("snr_L1", "std"),
        snr_l1_min=("snr_L1", "min"),
        doppler_l1_mean=("doppler_L1", "mean"),
        doppler_l1_std=("doppler_L1", "std"),
        pr_doppler_residual_mean=("pr_doppler_residual", "mean"),
        pr_doppler_residual_std=("pr_doppler_residual", "std"),
    ).reset_index()
    return agg


def load_mon_rf_features(path: Path) -> pd.DataFrame:
    cols = ["real_time", "jamInd_01", "jamInd_02", "agcCnt_01", "agcCnt_02",
            "noisePerMS_01", "noisePerMS_02"]
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    df["real_time"] = pd.to_datetime(df["real_time"], errors="coerce")
    for c in cols:
        if c != "real_time":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["real_time"]).sort_values("real_time")
    agg = df.groupby("real_time").agg(
        jam_ind_mean=("jamInd_01", "mean"),
        agc_cnt_mean=("agcCnt_01", "mean"),
        noise_per_ms_mean=("noisePerMS_01", "mean"),
    ).reset_index()
    return agg


def process_scenario(sdir: Path) -> pd.DataFrame | None:
    sjson_path = sdir / "scenario.json"
    if not sjson_path.exists():
        return None
    sjson = json.loads(sjson_path.read_text())

    nav_path, rinex_path, monrf_path = sdir / "nav_pvt.csv", sdir / "rinex.csv", sdir / "mon_rf.csv"
    if not (nav_path.exists() and rinex_path.exists() and monrf_path.exists()):
        print(f"  SKIP {sdir} (missing csv)")
        return None

    nav = load_nav_pvt(nav_path)
    rinex = load_rinex_features(rinex_path)
    monrf = load_mon_rf_features(monrf_path)

    merged = pd.merge_asof(nav, rinex, left_on="real_time", right_on="time",
                            direction="nearest", tolerance=pd.Timedelta("1s"))
    merged = pd.merge_asof(merged, monrf, on="real_time",
                            direction="nearest", tolerance=pd.Timedelta("1s"))
    merged = merged.drop(columns=["time"], errors="ignore")

    windows = parse_attack_windows(sjson)
    merged["attack"] = label_attack(merged["real_time"], windows)

    merged["run_id"] = str(sdir.relative_to(RAW_ROOT))
    merged["scenario_id"] = sjson.get("scenario_id")
    merged["attack_type"] = sjson.get("attack_type")
    attack_params = sjson.get("attack_parameters", {})
    merged["rover_state"] = attack_params.get("Rover states") or attack_params.get("rover_states")
    merged["power_w_max"] = sjson.get("attack_parameters", {}).get("power_max")
    merged["bands"] = sjson.get("attack_parameters", {}).get("frequency_band")
    merged["n_attack_windows"] = len(windows)
    merged["window_start_in_range"] = bool(windows) and any(
        merged["real_time"].min() <= w[0] <= merged["real_time"].max() for w in windows
    )

    return merged


def main():
    scenario_dirs = find_scenario_dirs(RAW_ROOT)
    print(f"Found {len(scenario_dirs)} scenario folders")

    frames = []
    for sdir in scenario_dirs:
        rel = sdir.relative_to(RAW_ROOT)
        print(f"Processing {rel} ...")
        try:
            feat = process_scenario(sdir)
        except Exception as e:
            print(f"  ERROR {rel}: {e}")
            continue
        if feat is None:
            continue
        n_attack = int(feat["attack"].sum())
        print(f"  rows={len(feat)} attack_rows={n_attack} window_in_range={feat['window_start_in_range'].iloc[0]}")
        frames.append(feat)

    full = pd.concat(frames, ignore_index=True)
    full.to_csv(OUT_DIR / "syncguard_features.csv", index=False)
    full.to_parquet(OUT_DIR / "syncguard_features.parquet", index=False)
    print(f"\nWrote {len(full)} rows to {OUT_DIR}")
    print(full["attack_type"].value_counts())
    print(full["attack"].value_counts())


if __name__ == "__main__":
    main()
