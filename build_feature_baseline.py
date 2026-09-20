"""Generate api/feature_baseline.json -- the training-distribution reference the serving
layer uses for input plausibility warnings (/score, /ingest) and drift detection (/drift).

Reads processed/syncguard_features.parquet, which is the feature table the shipped model was
trained on, and writes per-feature summary statistics. It does NOT read, modify or re-run any
training script, and it does not touch the model: this is a read-only summary of the data
the model already saw.

The output is committed so the serving layer has a deterministic baseline that does not
depend on the parquet being present at runtime, and so the exact numbers behind every
published warning are reviewable in git rather than recomputed differently on each machine.

Regenerate with:  python build_feature_baseline.py

--------------------------------------------------------------------------------
METHOD (stated because every published number has to be traceable to one)

Bounds -- what "implausible" means
  low  = p0.1 - MARGIN * (p99.9 - p0.1)
  high = p99.9 + MARGIN * (p99.9 - p0.1)
  with MARGIN = 0.25, i.e. the p0.1-p99.9 span widened by 25% on each side.

  p0.1/p99.9 rather than min/max so a handful of corrupted samples cannot stretch a bound to
  uselessness; the 25% margin so that ordinary values slightly outside the recorded range do
  not produce warning spam. The margin is a judgement call, not a derived quantity -- it is
  stated here and in every API response so nobody has to guess at it.

  A widened low bound below zero is clamped to zero for features the request schema already
  declares as ge=0 (api/schemas.py), since a negative value there is rejected before scoring
  and an unreachable bound would be noise.

PSI bins -- for /drift
  10 bins from the baseline's own deciles (p0, p10, ..., p100), with the outer edges set to
  -inf/+inf so live values outside the training range still fall in the first/last bin
  instead of being dropped. Bins are deduplicated: a feature whose deciles collapse (a
  near-constant column such as fixType) gets fewer than 10 bins, and that is recorded so
  /drift can report it rather than silently divide by zero.
--------------------------------------------------------------------------------
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
PARQUET = REPO_ROOT / "processed" / "syncguard_features.parquet"
MODEL = REPO_ROOT / "models" / "model.joblib"
OUT = REPO_ROOT / "api" / "feature_baseline.json"

MARGIN = 0.25          # fraction of the p0.1-p99.9 span added to each side
LOW_Q, HIGH_Q = 0.001, 0.999
N_PSI_BINS = 10

# Features whose request schema declares ge=0 (api/schemas.py). A widened low bound below
# zero is clamped for these -- the schema rejects negatives before scoring, so an unreachable
# bound would only produce noise.
NON_NEGATIVE = {
    "hAcc", "vAcc", "sAcc", "headAcc", "pDOP", "numSV", "pos_dev_m",
    "n_sats_l1", "snr_l1_mean", "snr_l1_std", "snr_l1_min",
    "doppler_l1_std", "pr_doppler_residual_std",
    "jam_ind_mean", "agc_cnt_mean", "noise_per_ms_mean",
}


def main() -> int:
    if not PARQUET.exists():
        print(f"Missing {PARQUET}. This file ships with the repo; see README.md.")
        return 1
    if not MODEL.exists():
        print(f"Missing {MODEL}. Run `make train` first -- the feature order comes from the "
              f"trained artifact, not from this script.")
        return 1

    feature_cols = list(joblib.load(MODEL)["feature_cols"])
    df = pd.read_parquet(PARQUET)

    features: dict[str, dict] = {}
    for col in feature_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        lo, hi = float(s.quantile(LOW_Q)), float(s.quantile(HIGH_Q))
        span = hi - lo
        bound_low = lo - MARGIN * span
        bound_high = hi + MARGIN * span
        clamped = False
        if col in NON_NEGATIVE and bound_low < 0.0:
            bound_low, clamped = 0.0, True

        # PSI bin edges from the baseline's own deciles, open at both ends.
        edges = [float(s.quantile(q)) for q in np.linspace(0, 1, N_PSI_BINS + 1)]
        edges = sorted(set(edges))
        degenerate = len(edges) < 3  # fewer than 2 usable bins
        if not degenerate:
            edges[0], edges[-1] = float("-inf"), float("inf")
            counts, _ = np.histogram(s.to_numpy(), bins=edges)
            proportions = (counts / counts.sum()).tolist()
        else:
            proportions = []

        features[col] = {
            "n": int(s.size),
            "n_missing": int(len(df) - s.size),
            "p001": lo,
            "p999": hi,
            "median": float(s.median()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "bound_low": bound_low,
            "bound_high": bound_high,
            "bound_low_clamped_to_zero": clamped,
            "psi_bin_edges": [None if np.isinf(e) else e for e in edges] if not degenerate else [],
            "psi_baseline_proportions": proportions,
            "psi_degenerate": degenerate,
        }

    payload = {
        "_comment": (
            "Training-distribution baseline for input plausibility warnings and drift "
            "detection. Generated by build_feature_baseline.py from the REAL Jammertest 2024 "
            "feature table the shipped model was trained on. Do not hand-edit -- regenerate."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_parquet": str(PARQUET.relative_to(REPO_ROOT)).replace("\\", "/"),
        "n_rows": int(len(df)),
        "n_runs": int(df["run_id"].nunique()) if "run_id" in df.columns else None,
        "method": {
            "bounds": (f"[p{LOW_Q*100:g} - {MARGIN}*(p{HIGH_Q*100:g} - p{LOW_Q*100:g}), "
                       f"p{HIGH_Q*100:g} + {MARGIN}*(p{HIGH_Q*100:g} - p{LOW_Q*100:g})]"),
            "margin": MARGIN,
            "low_quantile": LOW_Q,
            "high_quantile": HIGH_Q,
            "psi_bins": N_PSI_BINS,
            "psi_bin_source": "baseline deciles, outer edges open (-inf / +inf)",
        },
        "provenance": (
            "REAL: Jammertest 2024 (Zenodo 15911589), Bleik test range, Norway, September "
            "2024. One receiver. This is a test-range baseline, NOT ASEAN telecom "
            "infrastructure -- real deployment input is EXPECTED to drift from it. See "
            "ASSUMPTIONS_PRODUCTION.md."
        ),
        "features": features,
    }

    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}: {len(features)} features from "
          f"{len(df)} rows.")

    n_clamped = sum(1 for f in features.values() if f["bound_low_clamped_to_zero"])
    n_degenerate = sum(1 for f in features.values() if f["psi_degenerate"])
    print(f"  low bound clamped to zero: {n_clamped}")
    print(f"  degenerate PSI binning (near-constant feature): {n_degenerate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
