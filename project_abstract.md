# SyncGuard — Project Abstract

**Team Lorem Ipsum (HACK_TH_014, Thailand) — AGAIF 2026 Hackathon, Cybersecurity (CY) Track**

*This abstract covers all 10 PDGS Canvas dimensions: C1–C3 (§1), C4 (§3), C5–C7 (§2, §4), C8
(§5), C9–C10 (§6).*

**TL;DR**: GNSS spoofing can silently corrupt the timing that 4G/5G base stations depend on,
degrading network performance with no alarm to trigger — and no hardware change is required to
fix it. SyncGuard is a retrofit-first, per-site anomaly detector that flags jamming, spoofing,
and meaconing from GNSS receiver observables already exposed by off-the-shelf modules, now
running as a full explainable service (FastAPI, SHAP, live spatial statistics). On real
JammerTest 2024 attack data, the shipped model reaches 87.7% accuracy and 93.7% attack recall
on recordings never seen in training — with the two honest gaps (weaker clean-class recall, and
an unvalidated jump from test-range receivers to production base-station timing) disclosed
below rather than smoothed over.

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

Since the initial submission, SyncGuard has grown from a batch detector into a running,
explainable service, packaged as a FastAPI application (`api/main.py`) behind SQLite
persistence (`api/db.py`) and containerized with Docker — `docker compose up --build` trains
both model artifacts at image-build time and serves the whole stack from one command. The
model itself is unchanged: the CV-validated 23-feature RandomForest at decision threshold
0.52 (clean recall ~0.667, false-positive rate ~0.333, jamming recall ~0.859 on the fixed
held-out split — full table in §5). What is new is the serving layer around it:

- `POST /score` — scores one telemetry reading and returns an attack probability, a severity
  normalized to the model's own confidence range, and a per-prediction SHAP explanation naming
  which observables drove the call; every scored event is persisted to SQLite.
- `GET /health`, `GET /towers`, `GET /events` — service/model status, the 136 real tower rows,
  and recently scored events, respectively.
- A streaming replay mode (`GET /stream/events`, server-sent events) plays a recording through
  the same live scoring path row-by-row in real recording-time order, for demonstration.
- On top of the tower network, `GET /spatial/autocorrelation` computes established spatial
  statistics — global and local Moran's I (LISA) via PySAL — to answer whether flagged sites
  are spatially clustered right now.
- Alert state is debounced with hysteresis (3 consecutive above-threshold readings to raise an
  alert, 5 below to clear it) so a single threshold flicker never raises an alarm.
- A live dashboard (`GET /dashboard`) consumes these endpoints for the tower map, event log,
  and per-event explain panel.

SHAP explanation is always computed for `/score`; during streaming replay it is speed-gated —
computed live at replay speeds up to 20, and available on-demand (cached after first call) via
`GET /events/{id}/explain` above that speed, so a fast demo replay isn't bottlenecked on
per-row SHAP. Operational characteristics are measured, not estimated: sub-100 ms p50 /
sub-135 ms p99 end-to-end scoring latency (full HTTP request, SHAP included), measured replay
throughput at several requested speeds, and — under concurrent request load during an active
replay — tail latency (p95/p99) that degrades 3–5x while typical-case (p50) latency stays
comparatively stable; full numbers in `OPERATIONAL_METRICS.md`.

**Reliability note**: sklearn's RandomForest under parallel inference is not guaranteed to
return the same prediction twice for a borderline row (floating-point noise near the decision
boundary) — the serving layer forces single-threaded inference
(`n_jobs=1`) in `api/model_service.py` so `/score` is deterministic run-to-run, which matters
for a live judge demo. This is a reliability fix, not a modeling result; full detail in
`ROBUSTNESS_NOTES.md`. Endpoints, architecture, and the full evidence trail are in the
repository README.

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

**A load-bearing assumption this carries, stated explicitly**: the feature set (§4) assumes a
real telecom GNSSDO module exposes the same fields the u-blox survey receiver used for
training exposes — per-satellite C/N0, AGC count, jamming indicator, noise floor. Real
telecom-grade GNSSDO hardware (e.g. Trimble, Furuno) may not expose these fields through any
accessible interface, in which case the feature-extraction pipeline itself, not just the
model, would need rework before pilot validation (step 1 of the roadmap, §6) could even begin.
This is currently unstated anywhere else in this submission and is not something the test-range
dataset can confirm either way.

**A second axis of the same gap — receiver motion.** A base-station GNSSDO is bolted to a
fixed site; the deployment-relevant receiver state is therefore *stationary*, with a stable
sky view and multipath environment. The Jammertest 2024 dataset is roughly half
vehicle-mounted (**13 of 24 recordings are `dynamic`, 11 are `stationary`**), and the dynamic
recordings carry receiver-motion effects — changing satellite geometry, obstruction as the
vehicle passes buildings or terrain, movement-induced multipath — that a fixed installation
does not. Metrics pooled across both states (including the headline results in §5) are thus
not a clean estimate of stationary-deployment performance; a later feature-set investigation
found that the recordings where added signal-quality features misfired were *all* dynamic
(`SPOOFING_FEATURES.md`). The natural correction — evaluate on stationary recordings only —
is blocked by data volume: of the 7 jamming recordings, **only 1 is stationary**, too few to
hold out or cross-validate, so jamming detection cannot be assessed within the correctly
scoped (stationary-only) problem on this dataset. That is a limitation of the available data,
not a negative result about the method; it is documented in full in `STATIONARY_SCOPE.md` and
folded into roadmap step (1) below.

### Spatial layer data — real infrastructure, SIMULATED spread (read this split carefully)

To address the gap named above — the Jammertest 2024 detector has no spatial component, so it
cannot show *where* anomalies occur, whether *neighbouring sites* are correlated, or *which
sites to prioritize* — a second, clearly-separated data source and simulation layer was added.
**The line between what is real and what is simulated is stated explicitly here and is not
softened anywhere else this appears (plots, notebooks, or this document):**

- **REAL**: 136 real tower locations from Indonesian national open data (Kubu Raya/Pontianak;
  operator labels mixed — 38 PT. TELKOMSEL, 30 INDOSAT, 30 PROTELINDO, 17 TBG, 8 XL AXIATA,
  7 DAYAMITRA/Mitratel, 4 STP, 2 HCPT; the `namobj` field is simply "Object Name" in the
  national schema) (`site_id`, `site_name`,
  village/district, lat/long, tower type and construction metadata) in Kubu Raya and
  Pontianak, West Kalimantan, Indonesia — `menaratelepon_ar_50k.csv`. Traced to "Tower
  Telekomunikasi 50K," Kubu Raya Regency's own Open Data portal
  (opendata.kuburayakab.go.id) — a public Indonesian government open-data platform — and
  supplied to this project via the AGAIF bootcamp's Module 6 (AD1002) materials. The
  portal's own listing leaves its license field blank, so this is publicly published
  government infrastructure data, not data under a confirmed open license such as CC-BY; it
  is included in this repository because it is a public download from a government
  transparency portal, not because redistribution rights have been formally confirmed.
  Verified clean: no missing coordinates, no duplicate coordinates, one coherent region.
- **REAL**: the severity *scale* is anchored to the detector's own `predict_proba()` output on
  the held-out set — not invented numbers: **floor 0.430** (median probability on true clean
  rows), **ceiling 0.993** (90th-percentile probability on true attack rows). The same two
  values are used by the live API's severity normalization and by the offline spatial layer.
- **REAL (live spatial statistics)**: the running service computes global and local Moran's I
  (LISA) over whichever real towers currently have a scored severity — standard PySAL
  implementations over real great-circle tower geometry and real `predict_proba`-derived
  severities, not a custom metric. The one element it inherits from the SIMULATED side is the
  same one labelled below: which physical tower each scored event is attributed to
  (deterministic round-robin — the dataset is a single receiver with no real per-tower
  mapping). Every API response and stored row states this. **Framed plainly**: because
  per-event tower attribution is simulated, any specific Moran's I value the service reports
  reflects that simulated attribution, not a measured real-world spatial pattern — we
  demonstrate the method that would surface real clusters once real per-tower data exists, we
  do not claim to have detected a real cluster.
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

Full methodology and every parameter choice: `spatial_layer_notes.md` (offline layer),
`SPATIAL_STATISTICS.md` (live Moran's I / LISA). Every column, filename, and plot title the
offline layer produces carries an explicit `SIMULATED` marker so the distinction cannot be
lost by truncation, copy-paste, or a later rename.

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

**Model**: a RandomForestClassifier (300 trees, `class_weight='balanced'` to handle the
dataset's 77%/23% attack/clean imbalance via reweighting rather than discarding data),
following the curriculum's own guidance (Module 6 Session 6) that Random Forest is a sound,
interpretable starting point before moving to gradient-boosted alternatives — appropriate given
the hackathon timeline. **The shipped detector is this exact forest evaluated at a decision
threshold of 0.52** — selected by 4-fold GroupKFold cross-validation (`ROBUSTNESS_NOTES.md`),
not a separate or retrained model — and §5's results are reported at that operating point; the
untuned 0.50 baseline is in `baseline_model_report.md`.

**Evaluation split**: by recording, not by row. Rows within one recording are ~1 Hz samples of
a continuous, highly autocorrelated signal; a random row-level split would place near-duplicate
neighbouring rows in both train and test and inflate reported performance. Instead, one
dynamic and one stationary recording were held out per attack type (8 of 24 recordings, 14,077
of 44,639 rows), so every attack-type × receiver-state combination present in the data is
represented in both splits, and the model is genuinely tested on recordings it never saw
during training. Scenario metadata (attack type, receiver state, frequency band, timestamps)
was excluded from the feature matrix itself — a deployed detector would not have this as
ground truth, only the signal measurements. Note that this split pools stationary and dynamic
recordings; per the receiver-motion framing gap in §3, the dynamic recordings are not
representative of a fixed base-station installation, and the dataset is too thin in stationary
jamming data (1 recording) to re-run the evaluation on stationary-only data — see
`STATIONARY_SCOPE.md`.

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
is implemented in `build_spatial_simulation.py`, runs headless with no network calls, and
writes `simulated_spatial_anomaly_SIMULATED.csv` plus the map/spread-over-time plots
referenced in §5.

## 5. Results

| | Precision | Recall | F1 |
|---|---|---|---|
| Clean (0) | 0.752 | 0.667 | 0.707 |
| Attack (1) | 0.908 | 0.937 | 0.922 |

Overall accuracy **87.7%**, ROC-AUC **0.916**, PR-AUC (average precision) **0.968**, on 8
held-out recordings never seen during training, at the shipped decision threshold 0.52 (see
§4; the untuned 0.50 baseline is in `baseline_model_report.md`).

**Recall by attack type** (of true attack rows): Spoofing+Jamming 98.5%, Spoofing 97.1%,
Meaconing 96.5%, Jamming 85.8% (weakest of the four).

**Three limitations, stated as such rather than minimized**:

1. **Clean-class recall — 66.7% on the fixed held-out split, and 0.36–0.79 across the four
   GroupKFold folds (mean 0.66 ± 0.20, the honest confidence range) — is well below attack
   recall (93.7%).** A meaningful fraction of clean rows are misclassified as attack. The
   likely cause is domain shift: each recording's own "quiet" RF noise floor varies
   session-to-session, and the held-out recordings' baselines differ slightly from what the
   model saw in training. This points to per-session or per-site baseline calibration, not a
   fundamental flaw in the feature set.
2. **Feature importance leans on u-blox's built-in RF-monitor fields** (AGC count, noise
   floor, jamming indicator together account for the top share of importance) rather than on
   the purpose-built spoofing-specific features (code-Doppler residual, position deviation),
   which rank lower. The detector is doing well partly by re-deriving signal already available
   in commodity receiver firmware; whether the purpose-built signal can be made dominant on
   this dataset is exactly what the six experiments below tested.
3. **The training/test split is 77% attack / 23% clean — the inverse of a real deployment's
   base rate**, where a base station spends the overwhelming majority of its time *not* under
   attack. `class_weight='balanced'` reweights for the imbalance seen during training, but that
   does not fix the base-rate mismatch at deployment: a 33.3% false-positive rate measured
   against a 23%-clean test set implies a much larger absolute volume of false alerts once
   applied to a real, attack-rare traffic mix. This is a known open question about
   real-world alert volume, not something this submission has measured or resolved.

**We tried, rigorously, to close limitation 1.** Six independent experiments targeted the
clean-recall gap — sample reweighting, probability calibration, session-relative feature
normalization, gradient boosting, and two feature-engineering approaches that mine
carrier-phase, dual-frequency, per-constellation and temporal-coherence structure the standard
pipeline discards. Each was held to one bar: a genuine improvement must survive **both** a
rotating 4-fold GroupKFold **and** the fixed held-out split, with no jamming-recall
regression. All six converged on the same conclusion — with 24 recordings (~5 per attack type)
there is not enough independent data for a change to generalize across recording
configurations; a gain on one evaluation is broken by a different recording on the other.
**The ceiling is a data-diversity limit, not a modelling gap, and we can name what would raise
it:** more recordings, more power/band configurations per attack type, and stationary
base-station telemetry specifically. Full evidence trail — six experiments across seven
dedicated notes files (`ROBUSTNESS_NOTES.md`, `CALIBRATION_NOTES.md`, `NORMALIZATION_NOTES.md`,
`GBM_COMPARISON.md`, `SPOOFING_FEATURES.md`, `STATIONARY_SCOPE.md`, `TEMPORAL_COHERENCE.md`) —
is in the repository.

**What we tried and why it's not in the model.** One of the six angles (`SPOOFING_FEATURES.md`)
went looking for spoofing-specific physics in observables the standard extraction pipeline
discards — per-satellite L2-tracking collapse, cross-constellation C/N0 divergence, and related
dual-frequency structure, mined from the raw `rinex.csv` per-satellite records rather than the
aggregated per-epoch table the shipped 23 features are built from. On a rotating GroupKFold
evaluation these held up (jamming +2.9pt, ROC +2.1pt). They did not survive the fixed shipped
TEST split — a 9.1pt jamming-recall and 8.6pt clean-recall regression, concentrated on two
dynamic, degraded-reception recordings — so they were reverted and are **not** part of the
shipped model. We state this as a finding, not a shipped feature: we found evidence that richer
spoofing-specific signal exists in observables the standard pipeline discards; it didn't survive
our fixed validation split, so it's not in the shipped model, but it changes what we believe the
ceiling is if more data becomes available.

**Scope of claim**: these results demonstrate detectability of jamming/spoofing/meaconing
signatures in real GNSS-receiver observables at a controlled test range. They do not yet
demonstrate performance on production base-station telemetry (see Data Sources framing gap
above) — that validation is future work, not a claim of this submission.

**Spatial evidence — where, neighbouring, and prioritize** (methodology in §4, real/SIMULATED
data split in §3): applying the spatial layer to the 136 real Kubu Raya/Pontianak tower sites
produces `spatial_processed/spatial_anomaly_map_SIMULATED.png` (real site locations,
SIMULATED severity and epicenter, top-5 priority sites labeled) and
`spatial_processed/spatial_anomaly_spread_over_time_SIMULATED.png` (SIMULATED flagged-site
count widening 3 → 6 → 12 → 21 across four illustrative steps). This is the answer to the
three gaps named in §1:
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
Roadmap), not a demonstrated result. The three limitations in §5 (weaker clean-class recall,
feature importance leaning on generic RF-monitor signal rather than the purpose-built
spoofing features, and the attack-heavy training/test split's inverted base rate) are
disclosed rather than smoothed over, for the same reason.

**Demand validation**: everything above is a *technical* limitation, disclosed as such. This
submission does not separately validate the demand side — nothing here confirms that a
telecom operator would actually want, trust, or adopt a third-party GNSS anomaly detector on
its base-station infrastructure. That is as open a question as the technical generalization
gap above, and we're naming it explicitly for the same reason: the honesty here should be
even-handed, not concentrated only on what's easiest to measure.

**Privacy**: the Jammertest 2024 dataset consists of GNSS receiver and RF-monitor telemetry
from equipment at a controlled test event — no personal data, and no linkage to identifiable
individuals. The intended production data source (a base station's own GNSS receiver logs) is
likewise operator infrastructure telemetry, not customer or subscriber data — though any real
pilot would still require the hosting operator's own data-governance sign-off before ingesting
live logs, which is folded into step (1) of the roadmap below.

**Sustainability**: the retrofit-first design (§2) avoids the resource cost and e-waste of a
GNSS receiver hardware refresh across a regional base-station fleet; edge deployment (§2, §6)
is designed to keep compute and bandwidth footprint low relative to a cloud-round-trip
architecture — real-world footprint at a live cell site is pending the edge-deployment
feasibility assessment already listed in the roadmap (step 3, §6), not yet a measured result;
and the lightweight, open-source Python stack (§4) — no proprietary SDR/MATLAB licensing —
keeps the barrier to reproduction and adoption low across markets with uneven resourcing, in
line with the curriculum's own equitable-access framing.

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
logs — real infrastructure data, and stationary by nature, closing both axes of the framing
gap named in §3 (receiver type and receiver motion); (2) field validation
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
code, the FastAPI serving layer with SHAP explainability and spatial-statistics integration,
and the six model-improvement experiments summarized in §5; building the spatial layer
described in §3–§5 (selecting the real tower dataset, writing `build_spatial_simulation.py`,
and choosing/documenting the SIMULATED epicenter and distance-decay methodology); and drafting
this abstract from the team's results and decisions.
All code was executed and all outputs — the extracted features, the trained model's metrics,
the spatial-layer outputs, and the claims made in this document — were reviewed and validated
by the team before inclusion, including the real/SIMULATED framing itself. No part of the
dataset, results, or this document was generated without human review. This disclosure is
provided in full per the AI Usage Declaration requirement.
