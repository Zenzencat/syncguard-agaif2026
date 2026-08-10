# SyncGuard — Project Abstract

**Team Lorem Ipsum (HACK_TH_001, Thailand) — AGAIF 2026 Hackathon, Cybersecurity (CY) Track**

*This abstract covers all 10 PDGS Canvas dimensions: C1–C3 (§1), C4 (§3), C5–C7 (§2, §4), C8
(§5), C9–C10 (§6).*

---

## 1. Problem
*(PDGS Canvas C1 — Problem Statement; C2 — Stakeholders & Users; C3 — Spatial Context)*

4G/5G base stations depend on GNSS-disciplined oscillators (GNSSDOs) to hold the strict phase
and frequency synchronization that time-division duplexing, inter-cell coordination, and
handover all require. GNSS spoofing — transmitting fabricated satellite signals to feed a
receiver a false position or, critically, a false time — can silently corrupt this timing
reference. Unlike a hard outage, a spoofed timing source degrades network performance (frame
misalignment, interference between neighbouring cells, handover failures) without an obvious
trigger, making it materially harder for network operations teams to detect and diagnose than
a conventional failure.

This sits squarely inside a gap the AGAIF curriculum's own literature synthesis names directly:
*"Cybersecurity research must increasingly account for spatial variation in digital maturity,
policy capacity, and threat exposure"* (Module 7 Session 7, ASEAN telecom literature review).
GNSS spoofing of timing infrastructure is exactly this kind of threat — its impact is uneven
across a region where base-station hardware generations, receiver authentication support, and
network redundancy vary widely.

**Stakeholders**: telecom network operations and security teams (immediate users of a
detection alert), infrastructure regulators (policy/compliance interest in timing-integrity
assurance), and — downstream — the public and services (emergency communications, financial
transactions, general connectivity) that depend on uninterrupted network timing.
**Spatial context**: base stations are fixed, geographically distributed assets; a spoofing
attack is local to a receiver's radio range, so detection has to run per-site rather than
as a single centralized check — and neighbouring sites in the same area are plausibly
affected together, which is exactly what a per-site-only detector cannot show on its own.
§3 and §4 below describe the spatial layer added to make that visible: real ASEAN tower
locations, with a clearly-labeled *simulated* spoofing-spread overlay used to demonstrate
where, how correlated, and which sites to prioritize (§5 shows the resulting map).

## 2. Proposed Solution
*(PDGS Canvas C5 — GeoAI Intelligence; C6 — GeoAI Solution Design)*

SyncGuard is a lightweight, per-site anomaly-detection layer that monitors a base station's
GNSS receiver observables — signal quality, Doppler behaviour, RF interference indicators, and
solution-quality metrics — and flags jamming, spoofing, or meaconing in near real time,
**without requiring any change to the GNSSDO hardware itself.**

Most currently-deployed base-station GNSS receivers do not support cryptographic signal
authentication (e.g. Galileo OSNMA), and a wholesale receiver-hardware upgrade across a
regional network is not a near-term option in every market. SyncGuard's approach is
retrofit-first: it works from receiver-observable statistics available on off-the-shelf
GNSS modules already in the field, detecting the *statistical signature* of an attack
(signal-quality degradation, pseudorange/Doppler inconsistency, RF interference) rather than
requiring the receiver to cryptographically verify the signal itself. Consistent with the
curriculum's own edge-deployment guidance (Module 7 Session 4 — "Edge Device: prediction at a
cell tower"), the intended deployment target is the base station itself, so detection keeps
functioning locally even if the spoofing event also degrades backhaul connectivity.

## 3. Data Sources
*(PDGS Canvas C4 — Data Requirements)*

**GNSS Dataset Under Jamming, Spoofing, and Meaconing Conditions (JammerTest 2024)**
Sayyaf, M. I., Ortiz, M., & Renaudin, V. (2025). Zenodo, v3. https://doi.org/10.5281/zenodo.15911589
— cite alongside: V. Renaudin, M. I. Sayyaf, F. L. Bourhis and M. Ortiz, *"GNSS Positioning
Under Threat: The Rising Risk to Existing Systems and the Role of Alternative Indoor and
Seamless Navigation Technologies,"* IEEE Journal of Indoor and Seamless Positioning and
Navigation, doi: 10.1109/JISPIN.2025.3629705. License: GNU GPL v3.0 or later, as stated on
the Zenodo record.

The dataset contains real u-blox GNSS receiver logs recorded during Jammertest 2024 at the
Andøya Space Defense test range (Bleik, Norway) — 24 scenarios spanning Jamming, Spoofing,
Meaconing, and combined Jamming+Spoofing attacks across multiple power levels, frequency
bands, and both stationary and vehicle-mounted (dynamic) receiver states. Each scenario
provides per-satellite, per-epoch pseudorange/carrier-phase/Doppler/SNR observables, ~1 Hz
receiver position/velocity/time solutions, and RF-monitor telemetry (jamming indicator, AGC,
noise floor) — i.e., real GNSS-receiver data under real, controlled attacks, not synthetic
data.

**Framing gap, stated explicitly**: this dataset comes from u-blox GNSS receivers/rovers at a
public interference test range, **not from 4G/5G base station timing infrastructure or a
GNSS-disciplined oscillator**. SyncGuard's premise — that spoofing detection built on
receiver-level GNSS observable anomalies generalizes to base-station GNSSDO timing — is a
stated assumption in this submission, not something the data demonstrates directly, because no
public dataset of real telecom base-station GNSS timing under spoofing currently exists. This
is a genuine limitation of the evidence base available for this hackathon timeline, and is
carried forward explicitly into the roadmap below rather than left implicit.

### Spatial layer data — real infrastructure, SIMULATED spread (read this split carefully)

To address the gap named above — the Jammertest 2024 detector has no spatial component, so it
cannot show *where* anomalies occur, whether *neighbouring sites* are correlated, or *which
sites to prioritize* — a second, clearly-separated data source and simulation layer was added.
**The line between what is real and what is simulated is stated explicitly here and is not
softened anywhere else this appears (plots, notebooks, or this document):**

- **REAL**: 136 real PT. Telkomsel cellular base-station sites (`site_id`, `site_name`,
  village/district, lat/long, tower type and construction metadata) in Kubu Raya and
  Pontianak, West Kalimantan, Indonesia — `menaratelepon_ar_50k.csv`, sourced from the AGAIF
  bootcamp's own Module 6 (AD1002) materials. Verified clean: no missing coordinates, no
  duplicate coordinates, one coherent region.
- **REAL**: the severity *scale* used on the map is anchored to the Jammertest-2024 detector's
  own, actual `predict_proba()` output — not invented numbers. Re-running the validated
  Pipeline from §4/§5 in-memory and self-checking it against the published classification
  report reproduces median predicted probability = **0.953** on true attack rows (n=10,940)
  and **0.430** on true clean rows (n=3,137); these become the severity ceiling/floor.
- **SIMULATED**: everything about *where* an attack originates and *how* it spreads. No
  public dataset of real ASEAN base-station GNSS timing under spoofing exists (same gap
  as above), so there is no measured spread to show. Instead, one real tower site (the one
  nearest the cluster's geometric centroid, chosen deterministically rather than hand-picked)
  is designated a simulated epicenter, and a documented distance-decay function —
  `severity = floor + (ceiling - floor) · exp(-distance_km / decay_km)`, calibrated so the
  flagged-anomalous footprint grows from 3 to 21 of the 136 real sites across four
  illustrative time steps — spreads simulated severity to real neighbouring sites. This is a
  simplifying assumption standing in for "nearby infrastructure sharing correlated GNSS
  timing anomalies," not a measured or validated propagation model.

Full methodology and every parameter choice: `agaif-materials/dataset/spatial_layer_notes.md`.
Every column, filename, and plot title this layer produces carries an explicit `SIMULATED`
marker so the distinction cannot be lost by truncation, copy-paste, or a later rename.

## 4. Methodology
*(PDGS Canvas C6 — GeoAI Solution Design; C7 — Technology Stack)*

**Feature engineering** (all Python, no MATLAB or proprietary SDR toolchain — a deliberate
choice for a lightweight, reproducible, edge-deployable pipeline): C/N0 (mean/std/min SNR per
epoch across tracked satellites), Doppler shift (mean/std), a code-Doppler pseudorange
residual (actual per-satellite pseudorange rate vs. the rate predicted from reported Doppler —
an ephemeris-free consistency check that is a standard spoofing indicator in the literature),
RF-monitor jamming/AGC/noise-floor indicators, receiver solution-quality metrics (DOP,
accuracy estimates, satellite count), and a position-deviation proxy (valid for stationary
scenarios only). A clock bias/drift feature was also attempted — derived from receiver-internal
GPS time progression vs. logging wall-clock time — but was dropped after review: it proved flat
and uninformative in the scenarios inspected, since this dataset has no `NAV-CLOCK`-equivalent
message and a proper clock-bias solve would require broadcast ephemeris not included in the
data. We report this honestly rather than carry forward a feature known not to work.

**Model**: a RandomForestClassifier baseline (300 trees, `class_weight='balanced'` to handle
the dataset's 77%/23% attack/clean imbalance via reweighting rather than discarding data),
following the curriculum's own guidance (Module 6 Session 6) that Random Forest is a sound,
interpretable starting point before moving to gradient-boosted alternatives — appropriate given
the hackathon timeline.

**Evaluation split**: by recording, not by row. Rows within one recording are ~1 Hz samples of
a continuous, highly autocorrelated signal; a random row-level split would place near-duplicate
neighbouring rows in both train and test and inflate reported performance. Instead, one
dynamic and one stationary recording were held out per attack type (8 of 24 recordings, 14,077
of 44,639 rows), so every attack-type × receiver-state combination present in the data is
represented in both splits, and the model is genuinely tested on recordings it never saw
during training. Scenario metadata (attack type, receiver state, frequency band, timestamps)
was excluded from the feature matrix itself — a deployed detector would not have this as
ground truth, only the signal measurements.

**Spatial simulation methodology** (see §3 for the real/SIMULATED data split this builds on):
the detector above is not retrained or modified for this step — its held-out-set
`predict_proba()` distribution is reused as-is to set a real severity floor (0.430, median
clean-row probability) and ceiling (0.993, 90th-percentile attack-row probability). A single
real tower site nearest the 136-site cluster's centroid is designated the SIMULATED epicenter,
haversine distance from it to every other real site is computed, and SIMULATED severity is
assigned by exponential distance decay (`decay_km = 2.0`, chosen — not measured — so the
flagged set grows from 3 sites at the first illustrative time step to 21 at the last, a
localized footprint rather than the near-uniform 116/136 an earlier, looser decay constant
produced). Sites are ranked by SIMULATED severity to produce a priority list. This whole step
is implemented in `agaif-materials/dataset/build_spatial_simulation.py`, runs headless with no
network calls, and writes `simulated_spatial_anomaly_SIMULATED.csv` plus the map/spread-over-
time plots referenced in §5.

## 5. Results

| | Precision | Recall | F1 |
|---|---|---|---|
| Clean (0) | 0.759 | 0.644 | 0.697 |
| Attack (1) | 0.902 | 0.941 | 0.921 |

Overall accuracy **87.5%**, ROC-AUC **0.916**, PR-AUC (average precision) **0.968**, on 8
held-out recordings never seen during training.

**Recall by attack type** (of true attack rows): Spoofing+Jamming 98.5%, Spoofing 97.2%,
Meaconing 96.5%, Jamming 87.1% (weakest of the four).

**Two limitations, stated as such rather than minimized**:

1. **Clean-class recall (64.4%) is noticeably weaker than attack-class recall (94.1%)** — a
   meaningful fraction of clean rows are misclassified as attack. The likely cause is domain
   shift: each recording's own "quiet" RF noise floor varies somewhat session-to-session, and
   the held-out recordings' baselines differ slightly from what the model saw in training. This
   points to a concrete next step — per-session or per-site baseline calibration — rather than
   a fundamental flaw in the feature set.
2. **Feature importance leans on u-blox's built-in RF-monitor fields** (AGC count, noise
   floor, jamming indicator together account for the top share of importance) rather than on
   the purpose-built spoofing-specific features (code-Doppler residual, position deviation),
   which rank lower. In effect, this baseline is currently doing well partly by re-deriving
   signal already available in commodity receiver firmware. The differentiated, harder-to-spoof
   signal (code-Doppler consistency, position-jump detection) is present and usable but not yet
   the dominant driver — sharpening its contribution is the priority for the next modeling
   iteration, not a claim we're making about this baseline.

**Scope of claim**: these results demonstrate detectability of jamming/spoofing/meaconing
signatures in real GNSS-receiver observables at a controlled test range. They do not yet
demonstrate performance on production base-station telemetry (see Data Sources framing gap
above) — that validation is future work, not a claim of this submission.

**Spatial evidence — where, neighbouring, and prioritize** (methodology in §4, real/SIMULATED
data split in §3): applying the spatial layer to the 136 real Kubu Raya/Pontianak tower sites
produces `agaif-materials/dataset/spatial_processed/spatial_anomaly_map_SIMULATED.png` (real
site locations, SIMULATED severity and epicenter, top-5 priority sites labeled) and
`spatial_anomaly_spread_over_time_SIMULATED.png` (SIMULATED flagged-site count widening 3 → 6
→ 12 → 21 across four illustrative steps). This is the answer to the three gaps named in §1:
*where* — the map plots SIMULATED severity directly onto real coordinates; *neighbouring* —
sites near the SIMULATED epicenter show correlated SIMULATED severity, falling off with real
distance; *prioritize* — `simulated_spatial_anomaly_SIMULATED.csv` ranks all 136 real sites by
SIMULATED severity. **Scope of claim, stated with the same directness as above**: the tower
locations are real and the severity scale is anchored to the real detector's own confidence
range, but the spread pattern itself is a documented simulation, not a measurement — no public
dataset of real ASEAN base-station GNSS timing under spoofing exists to measure it from. This
is illustrative evidence of what the detector's output *could* look like laid over a real
network, not a demonstrated regional spoofing event.

### Ethics, Privacy & Sustainability
*(PDGS Canvas C8)*

**Claim boundaries**: consistent with the scope-of-claim above, this submission does not
assert that test-range detection performance transfers unchanged to production base-station
timing — that is an explicit assumption pending pilot validation (see Implementation
Roadmap), not a demonstrated result. The two limitations in §5 (weaker clean-class recall,
feature importance leaning on generic RF-monitor signal rather than the purpose-built
spoofing features) are disclosed rather than smoothed over, for the same reason.

**Privacy**: the Jammertest 2024 dataset consists of GNSS receiver and RF-monitor telemetry
from equipment at a controlled test event — no personal data, and no linkage to identifiable
individuals. The intended production data source (a base station's own GNSS receiver logs) is
likewise operator infrastructure telemetry, not customer or subscriber data — though any real
pilot would still require the hosting operator's own data-governance sign-off before ingesting
live logs, which is folded into step (1) of the roadmap below.

**Sustainability**: the retrofit-first design (§2) avoids the resource cost and e-waste of a
GNSS receiver hardware refresh across a regional base-station fleet; edge deployment (§2, §6)
keeps compute and bandwidth footprint low relative to a cloud-round-trip architecture; and the
lightweight, open-source Python stack (§4) — no proprietary SDR/MATLAB licensing — keeps the
barrier to reproduction and adoption low across markets with uneven resourcing, in line with
the curriculum's own equitable-access framing.

## 6. Expected Impact
*(PDGS Canvas C9 — Value & Impact; C10 — Implementation Roadmap)*

Silent timing desynchronization has an outsized, spatially uneven downstream effect: markets
and communities with less network redundancy and fewer backup paths are hit hardest by a
degraded-but-not-down cell site, and per the curriculum's own regional literature synthesis,
this is precisely where "spatial variation in digital maturity [and] policy capacity" makes
threat exposure unequal. A retrofit-first detection layer — one that works with GNSS receivers
already deployed, rather than requiring an authentication-capable hardware refresh across an
entire regional network — is disproportionately useful in exactly the markets least able to
afford a rip-and-replace upgrade cycle.

More broadly, trustworthy GNSS timing is a *precondition* for the predictive-coverage and
hazard-response systems the wider AGAIF curriculum builds toward (e.g. the telecom
hazard-response and network-resilience use case in Module 6 Session 5) — those systems assume
the underlying network telemetry and timing are themselves trustworthy. SyncGuard protects that
precondition rather than sitting on top of it.

**Implementation roadmap**: (1) pilot validation against an operator's own base-station GNSS
logs, to close the framing gap named above with real infrastructure data; (2) field validation
and threshold tuning per deployment site; (3) engineering feasibility assessment for on-site
(edge) deployment; (4) phased operational rollout with the 4-dimension monitoring plan already
mapped from the curriculum (Module 7 Session 4 — System health, Input data validity, Model
performance, Data drift); (5) documentation and provenance package for operator/regulator
handoff.

## 7. AI Usage Disclosure

Claude (Anthropic) was used throughout this project's dataset and modeling work: researching
and evaluating candidate GNSS datasets (including identifying that the originally planned
FGI-JSDR/FGI-GSRx pipeline required MATLAB, which was unavailable, and locating the Jammertest
2024 Zenodo dataset used instead), writing the Python feature-extraction and model-training
code, building the spatial layer described in §3–§5 (selecting the real tower dataset, writing
`build_spatial_simulation.py`, and choosing/documenting the SIMULATED epicenter and
distance-decay methodology), and drafting this abstract from the team's results and decisions.
All code was executed and all outputs — the extracted features, the trained model's metrics,
the spatial-layer outputs, and the claims made in this document — were reviewed and validated
by the team before inclusion, including the real/SIMULATED framing itself. No part of the
dataset, results, or this document was generated without human review. This disclosure is
provided in full per the AI Usage Declaration requirement.
