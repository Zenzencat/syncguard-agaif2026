# SyncGuard dataset notes

## What was actually used, and why (read this first for the Abstract)

The original plan was FGI-JSDR (FGI OSNMA dataset) processed through FGI-GSRx (MATLAB) into
pseudorange/C-N0/clock-bias/Doppler features. That plan was **abandoned before any download**,
for two compounding reasons, both confirmed directly in this environment:

1. **No MATLAB.** Not installed, not on PATH, no `Program Files\MATLAB`. FGI-GSRx's own docs
   only reference MATLAB's Parallel Computing Toolbox as an optional extra — there is no
   documented Octave, standalone, or Docker path that avoids a full MATLAB install.
2. **The "smallest" FGI option wasn't actually small.** The FGI OSNMA dataset is raw I/Q only
   (26 MHz sampling, 8-bit real samples, ~740s/scenario) — **~19 GB per scenario, ~38 GB
   total** for two files. FGI-SpoofRepo (the fallback) is the same story, and larger. Every
   FGI dataset in this family requires FGI-GSRx to extract *any* usable observable — there is
   no pre-processed shortcut available by picking a different FGI dataset.

Given the deadline (Project Abstract + Dataset Usage Rights due 12 Aug 9PM GMT+8), pulling a
MATLAB license, moving 38 GB, and debugging an unfamiliar SDR toolchain was judged not
tractable in the time available, per explicit instruction to stop and flag rather than sink the
budget into it.

**What replaced it:** the user pointed to an arXiv paper (Liu & Papadimitratos, KTH,
*"Self-Supervised Federated GNSS Spoofing Detection with Opportunistic Data,"*
arXiv:2505.06171) that used a Jammertest 2024 dataset (6 Android phones, Bleik, Norway). That
*specific* dataset has **no independent public release** — it's referenced only via
`jammertest.no/about-2/`, the event organizer's general page, and was privately collected by
the KTH authors. However, searching further surfaced a **separate, independently downloadable
dataset from the same Jammertest 2024 event**, released by a different research group on
Zenodo — see below. That is the dataset this pipeline actually uses.

## Dataset used

**GNSS Dataset Under Jamming, Spoofing, and Meaconing Conditions (JammerTest 2024)**
Sayyaf, M. I., Ortiz, M., & Renaudin, V. (2025). Zenodo. https://doi.org/10.5281/zenodo.15911589
(v3, published 2025-07-15; versions v1/v2 also exist at 10.5281/zenodo.15910564 and
10.5281/zenodo.15911359 — v3 was used here).

- **License**: GNU General Public License v3.0 or later, as stated on the Zenodo record's
  "Rights" field. GPL is an unusual choice for a *dataset* (it's a software license) — flag
  this explicitly to whoever reviews the Dataset Usage Rights declaration; it may be a
  Zenodo-default artifact rather than a deliberate choice by the authors, but it's what's
  actually listed.
- **Requested citation** (from the dataset page): V. Renaudin, M. I. Sayyaf, F. L. Bourhis and
  M. Ortiz, "GNSS Positioning Under Threat: The Rising Risk to Existing Systems and the Role of
  Alternative Indoor and Seamless Navigation Technologies," IEEE Journal of Indoor and Seamless
  Positioning and Navigation, doi: 10.1109/JISPIN.2025.3629705.
- **Size**: 375.3 MB compressed (`GNSS_DATASET_JAMMING_SPOOFING.tar.gz`), ~1.4 GB extracted.
  Downloaded to `agaif-materials/dataset/raw/`, md5 verified against the checksum published on
  the Zenodo page (`c7b00c63bb1ee5db80c7692a2b06e169`) — matched exactly.
- **Content**: real u-blox GNSS receiver logs recorded during Jammertest 2024 (Andøya Space
  Defense test range, Bleik, Norway), across 24 scenarios covering Jamming (7), Spoofing (8),
  Meaconing (5), and combined Jamming+Spoofing (4) attacks, at multiple power levels and
  frequency bands, both stationary and dynamic (vehicle-mounted) receiver states. Each scenario
  folder contains `rinex.csv` (per-satellite, per-epoch pseudorange/carrier phase/Doppler/SNR
  for L1 and L2), `nav_pvt.csv` (~1 Hz receiver position/velocity/time solution), `mon_rf.csv`
  (~1 Hz RF monitor: jamming indicator, AGC, noise floor), plus the raw `.ubx` binary log,
  `scenario.json` (attack metadata + timestamped attack log), and `info.txt`.

**Crucially, this ships pre-processed RINEX-equivalent observables directly — no SDR/MATLAB
processing step was needed at all.** This is real GNSS-receiver data under real jamming/
spoofing/meaconing attacks at a controlled test range, not synthetic data.

## Framing gap (must go in the Project Abstract, not hidden)

This dataset comes from **u-blox GNSS receivers/rovers** (some vehicle-mounted, some
stationary) at a public GNSS interference test range — **not from 4G/5G base station timing
infrastructure or a GNSS-disciplined oscillator (GNSSDO)**. SyncGuard's premise — that spoofing
detection built on receiver-level GNSS observable anomalies generalizes to base-station GNSSDO
timing — remains a stated assumption, not something this data demonstrates directly, because no
public dataset of real telecom base-station GNSS timing under spoofing exists. State this
explicitly.

## Processing pipeline

Python-only, no MATLAB. Scripts:
- `agaif-materials/dataset/extract_features.py` — feature extraction across all 24 scenarios.
- `agaif-materials/dataset/sanity_check.py` — stats, sample rows, clean-vs-spoofed plot.
- Environment: `agaif-materials/dataset/.venv` (numpy, pandas, matplotlib, pyarrow).

### Features extracted (per ~1 Hz epoch, from `nav_pvt.csv` timeline)

| Column | Source | Maps to | Notes |
|---|---|---|---|
| `snr_l1_mean/std/min` | `rinex.csv` `snr_L1`, aggregated across tracked satellites | **C/N0** | u-blox reports this field as SNR; treated as a C/N0 proxy. ~1% of raw samples were corrupted logging artifacts (values up to ~4×10⁷) — filtered to the physically valid [0, 60] dB-Hz range before aggregating. |
| `doppler_l1_mean/std` | `rinex.csv` `doppler_L1` | **Doppler shift** | Direct, no derivation needed. |
| `pr_doppler_residual_mean/std` | `rinex.csv`, derived | **Pseudorange residual** | Code-Doppler consistency check: actual per-satellite pseudorange rate (Δpseudorange/Δt) minus the rate predicted from reported Doppler. A standard, ephemeris-free spoofing indicator — genuine GNSS position isn't needed to compute it. Native per-satellite cadence is ~0.2s; residuals were only computed for consecutive epochs ≤1s apart and clipped to \|residual\| < 5000 m/s to exclude satellite-reacquisition discontinuities (a data artifact, not the phenomenon being measured). |
| `clock_drift_proxy_s` | `nav_pvt.csv`, derived | **Clock bias/drift** (attempted) | `Δ(iTOW)/1000 − Δ(real_time)`, i.e. receiver-internal GPS-time progression vs. logging wall clock. **Limitation**: in the scenarios inspected this was flat/zero — iTOW appears quantized to the logging cadence and didn't expose sub-second drift. This dataset has no `NAV-CLOCK`-equivalent field; a real clock-bias/drift feature would need either that message (not present) or a full single-point-positioning re-solve from raw pseudoranges + broadcast ephemeris (ephemeris not included in this dataset) — out of scope here. **Treat this column as unreliable and either drop it or revisit before modeling.** |
| `pos_dev_m` | `nav_pvt.csv` lat/lon/height, derived | Spoofing/meaconing position-jump indicator | Deviation from the scenario's own first-60s median fix. **Only meaningful for `rover_state == "stationary"` scenarios** — for `dynamic` scenarios this conflates real vehicle motion with any spoofing-induced jump and should not be used as-is without a ground-truth trajectory (not included in this dataset). |
| `jam_ind_mean`, `agc_cnt_mean`, `noise_per_ms_mean` | `mon_rf.csv` | RF-level jamming indicator | Most informative for the Jamming/Meaconing categories. |
| `numSV`, `pDOP`, `hAcc`, `vAcc`, `sAcc`, `gSpeed`, `fixType` | `nav_pvt.csv`, passthrough | Receiver-reported solution quality | Included as-is; spoofers/jammers often perturb these too. |

### Labeling

`attack` (0/1) per row, from each scenario's `scenario.json` → `attack_log` (paired
started/end timestamps). **Dataset quirk found and corrected**: `attack_log` timestamps carry
a `Z` (UTC) suffix but are actually **local Bleik/Norway time (CEST, UTC+2 in September
2024)** — confirmed by cross-referencing scenario 2.1.1's own `real_time` range
(06:57:21–07:39:05 UTC) against its attack_log window (09:00:00–09:38:00 "Z"), which only
falls inside the recording once shifted by −2h. This −2h correction (`LOCAL_TZ_OFFSET_HOURS`
in `extract_features.py`) is **an inferred convention, not something documented by the dataset
authors** — worth a caveat if this is scrutinized, and worth double-checking against the
dataset's IEEE paper if time allows before the model-training phase.

### Output

`agaif-materials/dataset/processed/syncguard_features.csv` and `.parquet`:
- **44,639 rows** across 24 scenarios (Spoofing 16,357 / Jamming 13,771 / Meaconing 7,471 /
  Jamming+Spoofing 7,040 rows).
- **34,290 attack rows (77%) vs. 10,349 clean rows (23%)** — meaningful class imbalance to
  account for when training (these are attack-focused recordings with a short clean
  prelude/tail, not balanced clean/attack captures).
- `agaif-materials/dataset/processed/scenario_2.1.1_clean_vs_spoofed.png` — sanity plot,
  stationary spoofing scenario, clean vs. spoofed shaded. C/N0 visibly drops (44→as low as
  32 dB-Hz) during the spoofed window; code-Doppler residual std jumps from near-0 to
  100–700+ m/s; position deviation peaks at ~90m near the end of the spoof window (matching
  the scenario's own "large position and time jump" description) then snaps back after.

## Explicitly not done yet

No model was built or trained, per instructions — this is feature extraction only, for
sanity-check before committing to a modeling approach.
