# SyncGuard spatial layer notes

## Why this exists

Mentor feedback from Dr. Asmidar (2026-08-10): the SyncGuard prototype had no spatial
component at all — it didn't show *where* anomalies occur, whether *neighbouring sites*
see correlated anomalies, or *which sites should be prioritized*. This document and the
outputs in `spatial_processed/` are the response: a new spatial layer added on top of the
existing, already-validated Norway/Jammertest-2024 detector. **The detector itself was not
touched or retrained** — `baseline_model_report.md`, `extract_features.py`, and
`train_baseline_model.py` are unchanged. This is additive.

## The hard rule this document follows

Real and simulated data are never allowed to blur together — not in code, not in file
names, not in plot titles, not in prose. Concretely:

- Every column, filename, and plot title produced by the spatial-simulation step carries an
  explicit `SIMULATED` marker (e.g. `simulated_anomaly_severity_SIMULATED`,
  `spatial_anomaly_map_SIMULATED.png`). This is deliberately loud/redundant so the label
  can't be lost by truncation, a copy-paste, or a later rename.
- The tower **coordinates and site metadata** are real and are never labeled "simulated."
- The **anomaly severity, spread, and epicenter** are simulated and are never presented as
  measured or observed.

## What is REAL

**Source**: `menaratelepon_ar_50k.csv`, from the bootcamp's Module 6 (AD1002) materials —
`agaif-materials/downloads/Module 6_AD1002_Dataset (Tower).zip`.

- **Country/region**: Indonesia — Kubu Raya and Pontianak, West Kalimantan (Kalimantan
  Barat).
- **Coverage**: 136 real PT. Telkomsel cellular tower sites (`site_id`, `site_name`,
  `desa`/village, `kec`/district, `lat`/`long`, tower type/height, construction metadata).
  Verified clean on load: no missing lat/long, no duplicate coordinates, single coherent
  region (lat range roughly -0.86 to +0.10, lon range roughly 109.15 to 110.07).
- This is real ASEAN telecom infrastructure. It was chosen over a messier ~3,547-row
  Bandung/West Java tower file (needed cleanup work not affordable on this timeline) and
  over registering for OpenCellID or Malaysia tower data (registration/access lead time
  didn't fit the ~2-day window remaining before the deadline).
- **Also real**: the RandomForest detector's `predict_proba()` output distribution, pulled
  by re-running the exact validated `Pipeline` from `train_baseline_model.py` in-memory
  (same features, same held-out recordings, same hyperparameters, same `random_state=42`)
  and self-checking the reproduced classification report against
  `baseline_model_report.md` before trusting any of its numbers (see
  `build_spatial_simulation.py::reproduce_validated_model()`). On the held-out Norway test
  set this run produced: median proba on true attack rows = **0.953** (n=10,940), median
  proba on true clean rows = **0.430** (n=3,137). These two numbers anchor the simulated
  severity scale below — they are the real model's own confidence floor and ceiling, not
  invented numbers.

## What is SIMULATED, and why

No public dataset of real ASEAN base-station GNSS timing under spoofing exists (confirmed
during the original dataset search — see `dataset_notes.md`). There is therefore no way to
show a *measured* spoofing spread across a real telecom network on this timeline. Instead,
`build_spatial_simulation.py` builds a documented, simplifying simulation:

1. **Epicenter** (SIMULATED): the real tower site nearest the geometric centroid of the
   136-site cluster is picked as the deterministic epicenter — `16MPW043` / "Rd To Rasau
   Jaya" (Rasau Jaya, Kubu Raya). Deterministic and centroid-based rather than hand-picked,
   so the choice isn't tuned for a more dramatic-looking map.
2. **Distance decay** (SIMULATED): severity at every other real site is computed as
   `floor + (ceiling - floor) * exp(-distance_km / decay_km)`, where `distance_km` is the
   real haversine distance from the epicenter, and `floor`/`ceiling` are the real detector's
   own median clean/attack proba values above (0.430 / 0.993, using the 90th percentile of
   attack-row proba as the ceiling — 0.993 — as the "confidently detected" reference point).
   This is a simplifying assumption, not a measured RF propagation model — it stands in for
   "nearby infrastructure sharing correlated GNSS timing anomalies during a spoofing event,"
   which is directionally plausible (shared regional GNSS visibility/geometry) but not
   something this dataset can verify quantitatively.
3. **Decay constant**: `DECAY_KM_SIMULATED = 2.0` km, calibrated (not measured) so the
   flagged-anomalous set (severity ≥ 0.5) grows from **3 sites at t=0 to 21 sites at the
   final time step**, out of 136 total. This was deliberately tuned away from an earlier,
   looser constant that flagged 116/136 sites (85%) — too diffuse to demonstrate
   prioritization or a "neighbourhood" effect. 21/136 (~15%) reads as a localized,
   plausible footprint instead of "everything nearby lights up."
4. **Spread over time** (SIMULATED): 4 illustrative time steps (`t=0..3`) with a widening
   decay radius, to show the flagged footprint growing outward from the epicenter — not a
   modeled time-series, just a widening-radius illustration of what "spread" would look
   like.

## How this connects to the validated Norway-based detector

The spatial layer does not replace or re-score the detector's evidence — it reuses the
detector's own output distribution as the severity scale, so a SIMULATED "high severity"
site on the map corresponds to the same proba range the real, validated model assigns to
real attack rows in the Norway/Jammertest-2024 recordings (~0.95 median), and a
SIMULATED "low severity" site corresponds to the same range as real clean rows (~0.43
median). In other words: the *shape and range* of the severity scale is evidence-grounded;
the *spatial placement and propagation* on top of it is simulated. The abstract/deck should
state both halves of that sentence together — never just the first half.

## Outputs

All in `agaif-materials/dataset/spatial_processed/`:

- `simulated_spatial_anomaly_SIMULATED.csv` — per-site table: real `site_id`/`site_name`/
  `desa`/`kec`/`lat`/`lon`, real `distance_from_epicenter_km`, and the three SIMULATED
  columns (`simulated_anomaly_severity_SIMULATED`, `flagged_anomalous_SIMULATED`,
  `priority_rank_SIMULATED`).
- `spatial_anomaly_map_SIMULATED.png` — static map (real coordinates, no basemap tiles /
  no network calls), colored/sized by simulated severity, epicenter marked, top-5 priority
  sites labeled.
- `spatial_anomaly_spread_over_time_SIMULATED.png` — 4-panel small-multiple, zoomed to the
  epicenter's neighborhood, showing the flagged count growing 3 → 6 → 12 → 21 across the
  illustrative time steps.

Regenerate with:
```
cd agaif-materials/dataset
.venv/Scripts/python.exe build_spatial_simulation.py
```
Runs in well under a minute (headless `Agg` backend, no network calls, no live geocoding —
the tower CSV already has lat/long).
