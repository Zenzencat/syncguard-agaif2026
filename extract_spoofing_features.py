"""
SyncGuard spoofing-specific feature probe: SUPPLEMENTARY feature extraction.

Motivation (see SPOOFING_FEATURES.md for the full writeup): baseline_model_report.md and
SHAP_EXPLAINABILITY.md both note the shipped detector leans on u-blox's own RF-monitor
fields (AGC count, noise floor, jamming indicator) rather than novel spoofing-specific
signal. This script mines structure that `extract_features.py` collapses away at its
per-epoch aggregation step and never exposes to the model:

  - carrier_phase_L1  -- fully populated in rinex.csv, currently unused. Enables a
    carrier-phase-rate vs Doppler consistency residual (distinct from the existing *code*
    pseudorange-rate residual: carrier phase is ~mm precision vs ~1 m code noise, so this
    isolates true phase-lock breaks / cycle slips) and code-minus-carrier (CMC) jump
    detection (a meaconing/spoofing splice produces CMC discontinuities).
  - the *_L2 columns -- present for ~40-46% of observations, currently unused. A
    single-frequency L1 spoofer/meaconer cannot reproduce L2, so the fraction of
    satellites still tracked on L2 collapses under spoofing/meaconing (but not under pure
    jamming -- which is the point; see SPOOFING_FEATURES.md).
  - the constellation prefix in `satellite` (G/E/C/R/S) -- currently only used as a count.
    Spoofers/meaconers hit some constellations harder than others, so the spread of mean
    C/N0 *across constellations* carries signal the single pooled snr_l1_std does not.

WHAT THIS DOES NOT TOUCH: extract_features.py, processed/syncguard_features.parquet,
models/*.joblib, api/, or any deployed threshold. It writes a SEPARATE artifact,
processed/spoofing_features_supplement.parquet, keyed by (run_id, real_time) on the exact
same nav_pvt timeline the main parquet uses (identical merge_asof(nearest, 1s) logic and
identical run_id strings -- str(scenario_dir.relative_to(RAW_ROOT))). spoofing_features_
experiment.py left-joins the two and evaluates BASE vs BASE+NEW under the established
GroupKFold harness. Nothing here is promoted to a model feature unless that experiment
clears the honest bar and the user approves.

Leakage discipline (same as session_normalization_experiment.py):
  - every per-epoch statistic is computed strictly within one epoch -- no temporal lookahead.
  - the carrier-phase and CMC residuals use per-satellite *consecutive-epoch* differences
    (current and immediately-preceding epoch only), the same causal construction as
    extract_features.py's pr_doppler_residual, with the same dt<=1s reacquisition guard.
  - the one trailing-window feature in the proposal (nsat_l1_roll_std_30) is NOT built here
    -- it is derived causally from the existing n_sats_l1 column inside the experiment
    script, where the 1 Hz timeline is already clean.

Assumption flagged (verified empirically on 6 scenarios, and re-checked per recording here):
  u-blox's carrier_phase_L1 sign convention is such that d(carrier_phase_L1)/dt ~= -doppler_L1
  (residual median ~0.15 cyc/s). Any recording whose median |residual| exceeds
  CP_SIGN_SANITY_HZ is treated as violating this assumption and has its cp_* columns set to
  NaN for that recording (handled downstream by the existing SimpleImputer(median)).
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
OUT_PATH = OUT_DIR / "spoofing_features_supplement.parquet"


def resolve_raw_root() -> Path:
    """Raw scenario tree. Priority: $SYNCGUARD_RAW_ROOT, then repo-local raw/, then the
    sibling agaif-materials copy this project was developed against."""
    env = os.environ.get("SYNCGUARD_RAW_ROOT")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(SCRIPT_DIR / "raw")
    candidates.append(SCRIPT_DIR.parent / "agaif-materials" / "dataset" / "raw")
    for c in candidates:
        if c.exists() and any(c.rglob("scenario.json")):
            return c
    raise SystemExit(
        "Could not locate the raw scenario tree. Set SYNCGUARD_RAW_ROOT to the directory "
        "containing Jamming/ Spoofing/ Meaconing/ Jamming+Spoofing/."
    )


RAW_ROOT = resolve_raw_root()

C = 299792458.0
F_L1 = 1575.42e6
LAMBDA_L1 = C / F_L1

LOCAL_TZ_OFFSET_HOURS = 2  # same -2h CEST->UTC correction as extract_features.py
SNR_VALID = (0.0, 60.0)     # physically valid C/N0 window; same filter as extract_features.py
MAX_DT_S = 1.0              # per-satellite gap beyond this = reacquisition, not a real rate
CP_RESID_CLIP_HZ = 1.0e4    # |carrier-phase residual| above this = logging artifact, drop
CMC_STEP_CLIP_M = 1.0e5     # |CMC step| above this = logging artifact, drop
CP_SIGN_SANITY_HZ = 5.0     # per-recording median |cp residual| above this => sign assumption violated
LOW_CN0_DBHZ = 30.0         # "weak" satellite threshold for frac_low_cn0


def find_scenario_dirs(root: Path):
    return sorted(p.parent for p in root.rglob("scenario.json"))


def parse_attack_windows(scenario_json: dict):
    windows, pending_start = [], None
    for ev in scenario_json.get("attack_log", []):
        ts = pd.Timestamp(ev["timestamp_utc"]) - timedelta(hours=LOCAL_TZ_OFFSET_HOURS)
        ts = ts.tz_localize(None)
        if "start" in ev["event"].lower():
            pending_start = ts
        elif "end" in ev["event"].lower() and pending_start is not None:
            windows.append((pending_start, ts))
            pending_start = None
    return windows


def load_nav_timeline(path: Path) -> pd.DataFrame:
    """Just the real_time column -- the 1 Hz spine the main parquet is keyed on."""
    df = pd.read_csv(path, usecols=lambda c: c in ("real_time",), low_memory=False)
    df["real_time"] = pd.to_datetime(df["real_time"], errors="coerce")
    df = df.dropna(subset=["real_time"]).sort_values("real_time").reset_index(drop=True)
    df["real_time"] = df["real_time"].astype("datetime64[us]")
    return df


def epoch_features_from_rinex(path: Path, run_id: str) -> pd.DataFrame:
    cols = ["time", "satellite", "pseudorange_L1", "carrier_phase_L1", "doppler_L1",
            "snr_L1", "pseudorange_L2", "snr_L2"]
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in cols:
        if c not in ("time", "satellite"):
            df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["time", "satellite"]).sort_values(["satellite", "time"])

    for c in ("snr_L1", "snr_L2"):
        df.loc[(df[c] < SNR_VALID[0]) | (df[c] > SNR_VALID[1]), c] = np.nan

    df["constellation"] = df["satellite"].str[0]

    # --- per-satellite consecutive-epoch quantities (causal) ---
    dt = df.groupby("satellite")["time"].diff().dt.total_seconds()
    ok = (dt > 0) & (dt <= MAX_DT_S)

    # carrier-phase-rate vs Doppler consistency. Convention: d(phase)/dt ~= -doppler,
    # so (rate + doppler) ~= 0 in steady phase lock; a cycle slip spikes it.
    cp_rate = df.groupby("satellite")["carrier_phase_L1"].diff() / dt
    cp_resid = cp_rate + df["doppler_L1"]
    cp_resid = cp_resid.where(ok)
    cp_resid[cp_resid.abs() > CP_RESID_CLIP_HZ] = np.nan
    df["cp_doppler_resid"] = cp_resid

    # code-minus-carrier: PR_L1 - lambda*phase_L1. Arbitrary per-arc ambiguity offset, so
    # only the epoch-to-epoch STEP is meaningful; a splice/handoff produces a jump.
    cmc = df["pseudorange_L1"] - LAMBDA_L1 * df["carrier_phase_L1"]
    cmc_step = cmc.groupby(df["satellite"]).diff().abs()
    cmc_step = cmc_step.where(ok)
    cmc_step[cmc_step > CMC_STEP_CLIP_M] = np.nan
    df["cmc_l1_step"] = cmc_step

    # per-recording carrier-phase sign sanity check
    med_resid = df["cp_doppler_resid"].abs().median()
    cp_sign_ok = bool(np.isfinite(med_resid) and med_resid <= CP_SIGN_SANITY_HZ)
    if not cp_sign_ok:
        df["cp_doppler_resid"] = np.nan
        df["cmc_l1_step"] = np.nan  # CMC step shares the carrier-phase assumption

    df["_l2"] = df["pseudorange_L2"].notna().astype(float)
    df["_cn0_l1_l2"] = df["snr_L1"] - df["snr_L2"]
    df["_cp_abs"] = df["cp_doppler_resid"].abs()
    df["_low_cn0"] = np.where(df["snr_L1"].notna(), (df["snr_L1"] < LOW_CN0_DBHZ).astype(float), np.nan)
    df["_dopp_ad"] = (df["doppler_L1"] - df.groupby("time")["doppler_L1"].transform("median")).abs()

    # main per-epoch aggregation -- all vectorized
    agg = df.groupby("time").agg(
        sp_cp_doppler_resid_std=("cp_doppler_resid", "std"),
        sp_cp_doppler_resid_maxabs=("_cp_abs", "max"),
        sp_cmc_l1_step_std=("cmc_l1_step", "std"),
        sp_cmc_l1_step_maxabs=("cmc_l1_step", "max"),   # cmc_l1_step is already |.|
        sp_frac_l2_tracked=("_l2", "mean"),
        sp_cn0_l1_minus_l2_mean=("_cn0_l1_l2", "mean"),
        sp_n_const=("constellation", "nunique"),
        sp_frac_low_cn0=("_low_cn0", "mean"),
        sp_doppler_l1_mad=("_dopp_ad", "median"),
    )

    # cross-constellation C/N0 spread: max-min of per-constellation mean C/N0, needs >=2
    # constellations with a valid mean this epoch
    cm = (df.dropna(subset=["snr_L1"])
            .groupby(["time", "constellation"])["snr_L1"].mean()
            .groupby("time").agg(lambda s: s.max() - s.min() if s.count() >= 2 else np.nan))
    agg["sp_xconst_cn0_spread"] = cm
    agg = agg.reset_index()
    agg.attrs["cp_sign_ok"] = cp_sign_ok
    agg.attrs["cp_median_resid"] = float(med_resid) if np.isfinite(med_resid) else float("nan")
    return agg


def process_scenario(sdir: Path):
    sjson_path = sdir / "scenario.json"
    nav_path, rinex_path = sdir / "nav_pvt.csv", sdir / "rinex.csv"
    if not (sjson_path.exists() and nav_path.exists() and rinex_path.exists()):
        print(f"  SKIP {sdir} (missing file)")
        return None
    run_id = str(sdir.relative_to(RAW_ROOT))

    nav = load_nav_timeline(nav_path)
    epoch = epoch_features_from_rinex(rinex_path, run_id)
    cp_sign_ok, cp_median_resid = epoch.attrs["cp_sign_ok"], epoch.attrs["cp_median_resid"]
    epoch = epoch.sort_values("time")
    epoch["time"] = epoch["time"].astype("datetime64[us]")

    merged = pd.merge_asof(nav, epoch, left_on="real_time", right_on="time",
                           direction="nearest", tolerance=pd.Timedelta("1s"))
    merged = merged.drop(columns=["time"], errors="ignore")
    merged.insert(0, "run_id", run_id)

    print(f"  rows={len(merged):>5}  cp_sign_ok={cp_sign_ok} "
          f"(median|resid|={cp_median_resid:.3f} Hz)  "
          f"coverage=" + ", ".join(f"{c.replace('sp_',''):s}:{merged[c].notna().mean():.0%}"
                                    for c in ("sp_xconst_cn0_spread", "sp_frac_l2_tracked",
                                              "sp_cp_doppler_resid_std")), flush=True)
    return merged


def main():
    sdirs = find_scenario_dirs(RAW_ROOT)
    print(f"RAW_ROOT = {RAW_ROOT}")
    print(f"Found {len(sdirs)} scenario folders\n")
    frames = []
    for sdir in sdirs:
        print(f"{sdir.relative_to(RAW_ROOT)}")
        out = process_scenario(sdir)
        if out is not None:
            frames.append(out)
    full = pd.concat(frames, ignore_index=True)
    full.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(full)} rows x {full.shape[1]} cols -> {OUT_PATH}")
    print("\nOverall non-null coverage:")
    for c in [c for c in full.columns if c.startswith("sp_")]:
        print(f"  {c:<28} {full[c].notna().mean():.1%}")


if __name__ == "__main__":
    main()
