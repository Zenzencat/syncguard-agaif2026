"""
SyncGuard temporal-coherence feature probe: SUPPLEMENTARY feature extraction.

Distinct axis from extract_spoofing_features.py (SPOOFING_FEATURES.md, rejected). That
attempt reduced rinex.csv to PER-EPOCH SPATIAL aggregates (dispersion across satellites at
one instant) and misread degraded sky visibility as attack on dynamic recordings. This
extracts TEMPORAL COHERENCE per satellite over time:

  - CMC drift (highest priority): per-SV trailing-window SLOPE of code-minus-carrier
    (PR_L1 - lambda*carrier_phase_L1). Authentic signals keep code/carrier locked (only slow
    ionospheric drift); a spoofer walking a receiver introduces a coherent CMC ramp. Jamming
    raises noise but does not create a coherent code/carrier divergence -> this should not
    cost jamming recall, and should not fire on a receiver tracking few but authentic
    signals.
  - Doppler smoothness: per-SV trailing-window std of the 2nd difference of Doppler (a
    deployable discrete-curvature proxy for "not a smooth orbital-motion trajectory" -- the
    Step-0 probe used a quadratic-fit residual; the 2nd-difference std is the causal,
    vectorizable form and the feature is marginal either way, see TEMPORAL_COHERENCE.md).

Two candidates from the Step-0 plan are DELIBERATELY EXCLUDED (TEMPORAL_COHERENCE.md Step 0):
  - reacquisition rate -- strong spoofing signal but fires catastrophically on 1.11.7's
    obstructed-sky clean segment (trailing-10s rate 0.347, 30x any attack), the exact
    degraded-reception failure mode that sank SPOOFING_FEATURES.md.
  - instantaneous tracked-set Jaccard -- dead (1.000/1.000 on every recording at 5 Hz).

WHAT THIS DOES NOT TOUCH: extract_features.py, processed/syncguard_features.parquet,
models/*.joblib, api/, or any deployed threshold. Writes a SEPARATE artifact,
processed/temporal_features_supplement.parquet, keyed by (run_id, real_time) on the same
nav_pvt spine as the main parquet (identical merge_asof and run_id strings).

Leakage discipline: every trailing window is strictly causal -- slope/curvature over the
trailing WIN epochs WITHIN the current cycle-slip-aware arc, current epoch inclusive, no
future rows. Cold start (fewer than WIN epochs into an arc) is NaN -> existing
SimpleImputer(median), same as pos_dev_m.

Carrier-phase sign convention (d(phase)/dt ~= -doppler) is checked per recording; a
recording that violates it has its cmc_* columns NaN'd (the doppler-only columns are kept).
"""
import json
import os
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "temporal_features_supplement.parquet"


def resolve_raw_root() -> Path:
    env = os.environ.get("SYNCGUARD_RAW_ROOT")
    cands = ([Path(env)] if env else []) + [SCRIPT_DIR / "raw",
             SCRIPT_DIR.parent / "agaif-materials" / "dataset" / "raw"]
    for c in cands:
        if c.exists() and any(c.rglob("scenario.json")):
            return c
    raise SystemExit("Raw scenario tree not found. Set SYNCGUARD_RAW_ROOT.")


RAW_ROOT = resolve_raw_root()
C = 299792458.0
LAMBDA_L1 = C / 1575.42e6
LOCAL_TZ_OFFSET_HOURS = 2

GAP_S = 0.35            # dt above this = tracking break
SLIP_CYC = 5.0          # |phase-rate + doppler| * dt above this many cycles = cycle slip -> new arc
WIN = 20               # trailing-window epochs (~4 s at 5 Hz) for slope / curvature
CP_SIGN_SANITY_HZ = 5.0
SNR_VALID = (0.0, 60.0)
CMC_SLOPE_HI_MPS = 3.0  # threshold for sp_cmc_slope_frac_hi

# closed-form trailing linear-slope constants for a fixed window of length WIN, positions 0..WIN-1
_SK = WIN * (WIN - 1) / 2.0                       # sum(k)
_SKK = (WIN - 1) * WIN * (2 * WIN - 1) / 6.0      # sum(k^2)
_DENOM = WIN * _SKK - _SK * _SK


def find_scenario_dirs(root: Path):
    return sorted(p.parent for p in root.rglob("scenario.json"))


def parse_attack_windows(sj: dict):
    out, ps = [], None
    for ev in sj.get("attack_log", []):
        ts = (pd.Timestamp(ev["timestamp_utc"]) - timedelta(hours=LOCAL_TZ_OFFSET_HOURS)).tz_localize(None)
        if "start" in ev["event"].lower():
            ps = ts
        elif "end" in ev["event"].lower() and ps is not None:
            out.append((ps, ts)); ps = None
    return out


def load_nav_timeline(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in ("real_time",), low_memory=False)
    df["real_time"] = pd.to_datetime(df["real_time"], errors="coerce")
    df = df.dropna(subset=["real_time"]).sort_values("real_time").reset_index(drop=True)
    df["real_time"] = df["real_time"].astype("datetime64[us]")
    return df


def epoch_features(path: Path) -> pd.DataFrame:
    cols = ["time", "satellite", "pseudorange_L1", "carrier_phase_L1", "doppler_L1", "snr_L1"]
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in cols:
        if c not in ("time", "satellite"):
            df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["time", "satellite", "pseudorange_L1", "carrier_phase_L1",
                           "doppler_L1"]).sort_values(["satellite", "time"]).reset_index(drop=True)

    dt = df.groupby("satellite")["time"].diff().dt.total_seconds()
    cp_rate = df.groupby("satellite")["carrier_phase_L1"].diff() / dt
    cp_resid = cp_rate + df["doppler_L1"]

    med_resid = cp_resid.abs().median()
    cp_sign_ok = bool(np.isfinite(med_resid) and med_resid <= CP_SIGN_SANITY_HZ)

    slip = dt.isna() | (dt > GAP_S) | (cp_resid.abs() * dt.fillna(0.0) > SLIP_CYC)
    df["arc"] = slip.groupby(df["satellite"]).cumsum()
    # order rows by (satellite, arc, time) so each cycle-slip-aware arc is a contiguous block;
    # then a plain global rolling(WIN) window is entirely within one arc iff the within-arc
    # position p >= WIN-1 -- no per-group rolling needed (that path is ~100x slower here).
    df = df.sort_values(["satellite", "arc", "time"]).reset_index(drop=True)
    df["p"] = df.groupby(["satellite", "arc"]).cumcount().astype(float)
    df["cmc"] = df["pseudorange_L1"] - LAMBDA_L1 * df["carrier_phase_L1"]

    # --- trailing-window |CMC slope| (closed form over a contiguous WIN block; causal) ---
    df["_py"] = df["p"] * df["cmc"]
    sum_y = df["cmc"].rolling(WIN).sum()
    sum_py = df["_py"].rolling(WIN).sum()
    sum_ky = sum_py - (df["p"] - (WIN - 1)) * sum_y                 # sum of local_k * y over window
    slope = ((WIN * sum_ky - _SK * sum_y) / _DENOM).abs()
    df["cmc_slope_abs"] = slope.where(df["p"] >= WIN - 1)          # NaN until the window fits in one arc

    # --- trailing-window Doppler curvature: std of 2nd difference of Doppler ---
    d2 = df["doppler_L1"].diff().diff().where(df["p"] >= 2)
    df["dop_curv"] = d2.rolling(WIN).std().where(df["p"] >= WIN + 1)

    if not cp_sign_ok:
        df["cmc_slope_abs"] = np.nan

    df.loc[(df["snr_L1"] < SNR_VALID[0]) | (df["snr_L1"] > SNR_VALID[1]), "snr_L1"] = np.nan

    agg = df.groupby("time").agg(
        sp_cmc_slope_p50=("cmc_slope_abs", "median"),
        sp_cmc_slope_max=("cmc_slope_abs", "max"),
        sp_cmc_slope_frac_hi=("cmc_slope_abs", lambda s: (s > CMC_SLOPE_HI_MPS).mean() if s.notna().any() else np.nan),
        sp_dop_smooth_p50=("dop_curv", "median"),
        sp_dop_smooth_max=("dop_curv", "max"),
    ).reset_index()
    agg.attrs["cp_sign_ok"] = cp_sign_ok
    agg.attrs["cp_median_resid"] = float(med_resid) if np.isfinite(med_resid) else float("nan")
    return agg


def process_scenario(sdir: Path):
    sj_path, nav_path, rx_path = sdir / "scenario.json", sdir / "nav_pvt.csv", sdir / "rinex.csv"
    if not (sj_path.exists() and nav_path.exists() and rx_path.exists()):
        print(f"  SKIP {sdir}"); return None
    run_id = str(sdir.relative_to(RAW_ROOT))
    nav = load_nav_timeline(nav_path)
    ep = epoch_features(rx_path)
    cp_ok, cp_med = ep.attrs["cp_sign_ok"], ep.attrs["cp_median_resid"]
    ep = ep.sort_values("time")
    ep["time"] = ep["time"].astype("datetime64[us]")
    merged = pd.merge_asof(nav, ep, left_on="real_time", right_on="time",
                           direction="nearest", tolerance=pd.Timedelta("1s")).drop(columns=["time"], errors="ignore")
    merged.insert(0, "run_id", run_id)
    cov = ", ".join(f"{c.replace('sp_',''):s}:{merged[c].notna().mean():.0%}"
                    for c in ("sp_cmc_slope_p50", "sp_dop_smooth_p50"))
    print(f"  rows={len(merged):>5}  cp_sign_ok={cp_ok} (|resid|={cp_med:.2f}Hz)  coverage {cov}", flush=True)
    return merged


def main():
    print(f"RAW_ROOT = {RAW_ROOT}")
    frames = []
    for sdir in find_scenario_dirs(RAW_ROOT):
        print(sdir.relative_to(RAW_ROOT))
        out = process_scenario(sdir)
        if out is not None:
            frames.append(out)
    full = pd.concat(frames, ignore_index=True)
    full.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(full)} rows x {full.shape[1]} cols -> {OUT_PATH}")
    for c in [c for c in full.columns if c.startswith("sp_")]:
        print(f"  {c:<24} coverage {full[c].notna().mean():.1%}")


if __name__ == "__main__":
    main()
