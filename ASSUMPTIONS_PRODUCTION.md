# Production assumptions — every one, marked validated or unvalidated

Phase 3 of the near-production work.

This document exists so that nobody has to reverse-engineer what SyncGuard is assuming. Each
entry states the assumption, what would make it true, and whether this project has actually
established it. Most have not been. That is the honest position for a system built from one
Norwegian test-range recording and aimed at ASEAN telecom infrastructure, and stating it
plainly is worth more than a demo that implies otherwise.

**Status vocabulary**

| Mark | Meaning |
| --- | --- |
| ✅ **VALIDATED** | Established by evidence in this repository — a measurement, a test, or a check you can re-run. The evidence is named. |
| ⚠️ **PARTIALLY VALIDATED** | Some evidence exists; it does not cover the claim as stated. The gap is named. |
| ❌ **UNVALIDATED** | Assumed. No evidence here either way. |
| 🔴 **KNOWN FALSE / KNOWN GAP** | We know this does not hold, or a defect is known and unfixed. |

**The single most important line in this file:** no real GNSS receiver at a real telecom
site has ever fed this system. Every "live" path in this repo is driven by a replayed
recording or a caller-supplied feature vector. Nothing here is field validation.

---

## 1. Data source and provenance

### 1.1 The training data is what it claims to be
✅ **VALIDATED.** Jammertest 2024 (Zenodo 15911589), Bleik test range, Norway, September 2024.
44,639 feature rows across 24 scenario runs, 10,349 clean / 34,290 attack.
Evidence: `dataset_notes.md`, `extract_features.py`, reproducible from the published dataset.

### 1.2 Attack labels are correctly aligned in time
⚠️ **PARTIALLY VALIDATED.** `scenario.json`'s `attack_log` timestamps are marked `Z` (UTC) but
are actually local CEST; `extract_features.py` shifts them −2 h. The offset was derived by
inspection and cross-checked against observable jamming onset in the RF data.
Gap: derived, not confirmed against an independent authoritative record. A wrong offset would
mislabel rows near window boundaries.
Evidence: `dataset_notes.md`.

### 1.3 The 23 features generalize from a test range to ASEAN telecom infrastructure
❌ **UNVALIDATED**, and this is the assumption the whole product rests on.
The training environment is a Norwegian island test range at ~69°N, one u-blox receiver,
deliberate jamming with known equipment. The target environment is Indonesian telecom base
stations at ~0°, different multipath, different interference, different receivers, different
constellations in view. Nothing in this repo tests transfer.
What would validate it: labeled GNSS observations from ASEAN base-station receivers, with
known interference events. We have none. Phase 3's `/drift` will *measure* the distribution
gap once real input arrives; it cannot tell you whether the model still works across it.

### 1.4 The tower coordinates are real
✅ **VALIDATED.** 136 real Telkomsel tower rows from
`spatial_raw/Module 6_AD1002_Dataset (Tower)/menaratelepon_ar_50k.csv` — real site ids, names
and lat/lon. `tests/test_existing_endpoints.py` asserts the count and key uniqueness.
Caveat that is *not* a caveat about the coordinates: 16 of the 136 share the literal `site_id`
`"tbg"`, a placeholder in the source spreadsheet. `api/spatial.py::load_towers` disambiguates
with a `tower_key`. The coordinates themselves are distinct and real.

### 1.5 Which tower a given event came from
🔴 **KNOWN FALSE for replay; CALLER-SUPPLIED and UNVERIFIED for ingest.**
- **Replay:** entirely **SIMULATED**. The Jammertest recording is one receiver in Norway; the
  towers are real infrastructure in Indonesia. There is no real link, and none can be
  recovered. `api/spatial.py` offers round-robin and a biased-random-walk attributor; both
  are placeholders and both are labelled SIMULATED wherever they surface.
- **Ingest:** the caller states the tower. The service cannot verify that the submitted
  features were observed there. Trusted input, not established fact.

### 1.6 The event layer vs the tower layer
🔴 **KNOWN GAP, disclosed by design.** Tower coordinates are REAL. Which tower an event is
attributed to is SIMULATED (replay) or unverified (ingest). Severities are REAL model output
on REAL recorded telemetry. So the map shows real geography carrying events that were never
observed there. `SPATIAL_STATISTICS.md` and `spatial_layer_notes.md` carry the full framing.

---

## 2. Base-station timing generalization

### 2.1 A base-station GNSS timing receiver behaves like the Jammertest rover under attack
❌ **UNVALIDATED.** The training receiver is a u-blox unit logging at ~1 Hz, mostly stationary,
at a test range. A telecom timing receiver is a different device class with different firmware,
different antennas (often rooftop, with different multipath), a different constellation mix,
and a different job — it cares about time transfer, not position. The features it emits under
the same interference may not look like the features the model learned.
What would validate it: parallel logging from a real timing receiver and a u-blox unit under
the same interference. Not done.

### 2.2 Holdover behaviour
❌ **UNVALIDATED, and not modelled at all.** A real base station under GNSS denial does not
simply lose time: it falls back to a local oscillator (OCXO/rubidium) and holds over, degrading
slowly and gracefully over hours. **The model has never seen a holdover signature**, because the
Jammertest receiver had no such mechanism. Consequences, stated rather than hedged:
- A real station may show *no* symptom for a long time after GNSS is denied, because holdover
  is doing its job. The detector would see degraded GNSS features and alert; whether that
  alert is operationally useful depends on the holdover budget, which the model knows nothing
  about.
- Conversely a holdover *exit* — the resync transient when GNSS returns — is a signature the
  model has never seen and cannot be expected to classify.
This is a structural gap, not a tuning problem. It would need holdover data to close.

### 2.3 The stationary-receiver assumption
⚠️ **PARTIALLY VALIDATED.** `pos_dev_m` is only meaningful for a stationary receiver, and
`STATIONARY_SCOPE.md` documents the scope limit. Base stations *are* stationary, which is the
favourable direction. But the reference-fix definition does not transfer cleanly — see 4.2.

### 2.4 1 Hz observation cadence
⚠️ **PARTIALLY VALIDATED.** MEASURED on the training data: median inter-row gap 1.0 s, p99
1.2 s. `INGESTION_CONTRACT.md` tells collectors to submit at roughly that rate. Nothing
enforces it and nothing tests what a different cadence does to the model's behaviour.

---

## 3. The model

### 3.1 The reported held-out metrics are real
✅ **VALIDATED.** 94.1% attack recall, 87.5% accuracy on the held-out split, produced by
`train_improved_model.py` and recorded in `improved_model_report.md`. Real numbers on real
data. They describe **the Jammertest test-range distribution** and nothing else.

### 3.2 Scoring is deterministic
✅ **VALIDATED.** `predict_proba` runs single-threaded (`rf.n_jobs = 1`) because parallel
per-tree vote aggregation is not bit-reproducible near a threshold — measured directly, ~6 of
14,077 held-out rows flipped between runs before this was forced. `ROBUSTNESS_NOTES.md`.
Exact Tree SHAP is likewise deterministic with no seed.

### 3.3 The 0.52 threshold is appropriate for deployment
❌ **UNVALIDATED for deployment.** It was tuned on the Jammertest held-out split
(`CALIBRATION_NOTES.md`). The right threshold depends on the operational cost ratio of a false
alarm to a missed event at a real NOC, which nobody has stated, and on the real base rate of
interference in the target environment, which nobody has measured. It is a defensible number
for the data it was tuned on and an arbitrary one for anywhere else.

### 3.4 The served artifact is the one that was evaluated
✅ **VALIDATED as of Phase 3.** The model version tag —
`model_version + sha256(model file)[:12] + threshold + sha256(feature list)[:8]` — is reported
by `GET /health`, on every scoring response, and as a `/metrics` gauge label. Any number quoted
from a running service can be traced to the exact artifact that produced it.

---

## 4. Feature extraction

### 4.1 `extract_features.py`'s docstring contradicts its code on RF antenna paths
🔴 **KNOWN DEFECT, documented and not fixed.**
The module docstring says the `mon_rf` features are *"averaged across the two antenna paths"*.
The code does not do that. It loads `jamInd_01/_02`, `agcCnt_01/_02`, `noisePerMS_01/_02` but
aggregates **`_01` only** and never reads the `_02` columns. The `_mean` suffix is the mean over
rows sharing one `real_time` value, not a mean across antenna paths.

**The code is the contract**, because the code is what produced the training data the shipped
model was fit on. `INGESTION_CONTRACT.md` therefore tells collectors to **submit RF block 01
only** — averaging both blocks would feed the model a differently-distributed input than it
learned from.

Not fixed because `extract_features.py` is frozen under this work's constraints, and because
changing it would change the feature distribution and invalidate the shipped model. Anyone who
next touches that file should reconcile the docstring to the code (not the code to the
docstring) unless they are also retraining.
Impact: documentation-only today. It becomes a real bug the moment someone writes a collector
from the docstring instead of the contract.

### 4.2 `pos_dev_m`'s reference fix transfers to a permanent installation
❌ **UNVALIDATED.** In training, `pos_dev_m` is the 3-D distance from the median fix over the
**first 60 seconds of that scenario recording**. A permanently installed tower receiver has no
natural session boundary. Whatever a collector picks — first 60 s after boot, a surveyed
coordinate, a rolling reference — is a design decision this repo has not tested, and a surveyed
fixed coordinate (the obvious production choice) is *not* what the model was trained on.

### 4.3 The C/N0 sanity filter is part of the contract
✅ **VALIDATED as necessary.** Real `snr_L1` samples contain corrupted artifacts up to ~4×10⁷.
`extract_features.py` discards values outside 0–60 before aggregating. A collector that skips
this produces a wildly different `snr_l1_mean` from a single bad sample.

### 4.4 Feature units are the raw Jammertest CSV scales, not SI or UBX-scaled
🔴 **KNOWN TRAP, now partially guarded.** A hand-written, physically sensible vector
(pDOP 1.4, hAcc 2.5 m, numSV 18) scores **"attack" at p = 0.60** — a confident wrong answer with
no error. `pDOP` actually sits near 0.01; `n_sats_l1` near 41, roughly double `numSV`.
Guarded since Phase 3 by input plausibility warnings (`api/plausibility.py`), but only
**partially** — see 6.1.

---

## 5. The serving layer

### 5.1 Ingested events reach the dashboard exactly like replayed ones
✅ **VALIDATED.** Same `ModelService.score()` call, same persistence, same SSE publish.
`tests/test_ingest.py` asserts identical probabilities through `/score` and `/ingest`, and the
path was exercised end to end in a real browser.

### 5.2 The spatial correlation window uses scoring time, not observation time
🔴 **KNOWN LIMITATION, disclosed and not fixed.**
`api/spatial.py`'s 120-second correlation window filters on `created_at` — *when this service
scored the event* — while ingestion stores the caller-stated observation time separately as
`obs_timestamp`. For live 1 Hz ingestion the two track each other and the behaviour is correct.
For a **backlog** they diverge completely: submitting an hour of buffered observations makes
them all correlate against each other within the same 120 s of wall-clock time, regardless of
whether the observations were an hour apart.

Consequence: correlation scores (and therefore the live Moran's I / LISA output) on backfilled
data are **not meaningful**. They describe submission timing, not observation timing.

Not fixed in Phase 1–3 because changing the correlation engine to key on `obs_timestamp` would
change replay's behaviour too (replay has no observation timestamp at all), and that is a
larger redesign than an ingestion adapter should carry. A real deployment ingesting historical
data must fix this before trusting any spatial output.

### 5.3 Per-tower hysteresis state does not survive a restart
🔴 **KNOWN LIMITATION, disclosed and not fixed.**
`TowerHysteresisRegistry` (`api/hysteresis.py`) keeps streak counters **in process memory**. A
service restart resets every tower to `normal`, discarding any in-progress streak. The
resulting `alert_state` **is** persisted on each event row, so `/events` and the dashboard keep
their history — but a tower that was 2 readings into a 3-reading entry streak, or 4 readings
into a 5-reading exit streak, starts over.

Consequence: a restart during a sustained real event delays re-entry into `alerting` by up to
`ALERT_ENTER_STREAK` observations (3 at 1 Hz ≈ 3 s; longer at lower cadence). Restarts during
an incident are exactly when this matters.

Not fixed because a durable streak table needs a policy for how stale a streak may be before it
is discarded — reloading a 6-hour-old streak as if it were current would be worse than
resetting — and nothing in this work measured what that staleness bound should be.

### 5.4 Deduplication makes ingestion safe to retry
✅ **VALIDATED.** Keyed on `(tower_key, obs_timestamp)`, scoped to `source='ingest'`. A repeat
submission is not re-scored, not re-persisted, and returns the original `event_id`. Tested.

### 5.5 Analyst labels are stored, not learned from
✅ **VALIDATED as a property**, not just a claim.
`tests/test_feedback.py::test_feedback_never_changes_a_score` scores a vector, submits five
dismissals, re-scores, and asserts an identical probability and threshold. `FEEDBACK_LOOP.md`.

### 5.6 `/feedback/summary` precision is not a field-precision estimate
✅ **VALIDATED as disclosed.** Analysts choose what to label, so the labeled set is
self-selected and non-random. Every response carries the caveat inline.

### 5.7 SQLite is adequate for the write volume
⚠️ **PARTIALLY VALIDATED.** One shared connection under a lock, WAL mode. Fine at demo volume
(tens of writes/second). Phase 4 measures the scoring path; neither phase measures SQLite under
sustained multi-thousand-tower write load.

---

## 6. Security and operations (Phase 3)

### 6.1 Input plausibility warnings catch bad input
⚠️ **PARTIALLY VALIDATED, and the limit is measured, not guessed.**
Against the exact wrong-scale vector from 4.4, the guard flags **pDOP only — 1 of 4 mis-scaled
fields**:
- `hAcc` 2.5 and `vAcc` 3.5 **cannot be flagged at all**: the training baseline's own p99.9 is
  4,294,967.295, because the real data contains the u-blox "invalid" sentinel (2³²−1 mm). The
  upper bound is meaningless and no value can exceed it.
- `numSV` 18 and `n_sats_l1` 18 are ordinary values inside wide baseline ranges (0–32 and 1–49).

So a per-feature range check catches gross scale errors on features with tight training ranges
and misses everything else. **`input_warnings: []` does not mean the input is correct.**
Cross-feature consistency checks (`n_sats_l1` vs `numSV`, `snr_l1_min ≤ snr_l1_mean`) would
catch more and are **not implemented** — a real gap.
Warnings **never reject input**, deliberately: out-of-range values may be precisely the anomaly
the detector exists to catch. Evidence: `tests/test_ops.py`, including a test that asserts the
guard's own blind spots so they cannot be quietly forgotten.

### 6.2 The bounds margin is a judgement call
✅ **DISCLOSED, not validated.** Bounds are p0.1–p99.9 widened by **25% of that span on each
side**. The 25% is a choice, not a derived quantity: wide enough not to spam on ordinary
out-of-range values, narrow enough to catch order-of-magnitude errors. Stated in
`build_feature_baseline.py`, in `api/feature_baseline.json`, and in this file.

### 6.3 API-key auth protects the service
⚠️ **PARTIALLY VALIDATED — it authenticates a caller, and only when switched on.**
- **Opt-in by design.** Auth is enforced only when `SYNCGUARD_API_KEY` is set. Unset (the
  default, and what `docker compose up` does) **every route is open**. That is a deliberate
  tradeoff for the demo path and is the first thing to change in a real deployment.
- It is a **single shared key**. No per-user identity, no roles, no rotation, no revocation
  short of restarting with a different key. The feedback layer's `analyst` field therefore
  remains unverified free text.
- Comparison is constant-time (`hmac.compare_digest`); the key is never logged, never echoed,
  never placed in a URL. Tested.
- `/health`, `/`, `/dashboard` and the API docs are exempt. `/health` reports liveness, the
  model tag and whether auth is on — no event data, no scores.

### 6.4 The SSE cookie is a reasonable compromise
✅ **DECIDED AND DOCUMENTED**, with costs stated. `EventSource` cannot send custom headers, so
`/stream/events` needed something else. Exempting it was rejected (it pushes the system's
actual output). A `?api_key=` query token was rejected (URLs leak into access logs, proxy logs,
browser history and `Referer`). The dashboard POSTs the key once to `/auth/session` and gets an
opaque `HttpOnly; SameSite=Strict` token back.
Costs, all real: sessions are **in-process** and die on restart; cookies cannot be combined
with wildcard CORS, so a cross-origin dashboard needs `SYNCGUARD_CORS_ORIGINS` set; the cookie
is **not `Secure` by default** because the demo runs on plain-HTTP localhost — set
`SYNCGUARD_COOKIE_SECURE=1` behind TLS. Full reasoning in `api/auth.py`.

### 6.5 There is no transport security
🔴 **KNOWN GAP.** The service speaks plain HTTP. No TLS, no HSTS, no certificate handling. The
API key crosses the wire in clear text unless something in front terminates TLS. Acceptable for
a localhost demo; unacceptable for anything else.

### 6.6 There is no rate limiting or request-size protection beyond the batch cap
🔴 **KNOWN GAP.** `/ingest` caps a batch at 500 observations; nothing limits request *rate*,
concurrent connections, or total body size. An authenticated caller can saturate the
single-threaded scoring path. No lockout on repeated auth failures either — only a metrics
counter.

### 6.7 `/drift` measures distribution shift, not model health
✅ **DISCLOSED.** PSI against the Jammertest baseline. **Real ASEAN input is *expected* to
drift**, so a high PSI is as consistent with "this is different infrastructure" as with
"something broke". The response carries that caveat inline. PSI cut points (0.10 / 0.25) are
the conventional ones, not values derived from this project's data. `fixType` is near-constant
in the baseline and is reported as unbinnable rather than silently dropped.

### 6.8 Metrics and logs are safe to expose
⚠️ **PARTIALLY VALIDATED.** Structured JSON logs emit a fixed field set — method, path, route
template, status, duration, request id — never headers, cookies or bodies, so a credential
cannot reach a log line by accident (tested). `/metrics` uses route *templates*, not concrete
paths, so event ids do not become label cardinality. `/metrics` requires the key when auth is
on. Not validated: no log retention policy, no PII review of `analyst`/`note` free text, which
is stored verbatim and exported in the CSV.

---

## 7. What would actually constitute field validation

None of this has been done. Listing it is the measure of the gap, not a roadmap promise.

1. A real GNSS timing receiver at a real base station, logging the observables
   `INGESTION_CONTRACT.md` specifies, for long enough to characterize normal.
2. Known-truth interference events in that environment — coordinated tests or
   independently corroborated incidents.
3. A collector implementing the contract, validated against `extract_features.py`'s output on
   the same raw input.
4. Held-out evaluation on that data, with the threshold re-tuned against a stated operational
   cost ratio.
5. Holdover characterization — what the features do while the oscillator carries the station,
   and on resync.
6. Multi-site deployment before any spatial correlation claim means anything, since the entire
   spatial layer currently runs on simulated or unverified attribution.

---

## 8. Quick reference

| # | Assumption | Status |
| --- | --- | --- |
| 1.1 | Training data provenance | ✅ |
| 1.2 | Attack label time alignment | ⚠️ |
| 1.3 | Features transfer to ASEAN infrastructure | ❌ |
| 1.4 | Tower coordinates are real | ✅ |
| 1.5 | Per-event tower attribution | 🔴 simulated / unverified |
| 1.6 | Real tower layer + simulated event layer | 🔴 disclosed |
| 2.1 | Base-station receiver behaves like the test rover | ❌ |
| 2.2 | Holdover behaviour | ❌ never modelled |
| 2.3 | Stationary-receiver scope | ⚠️ |
| 2.4 | 1 Hz cadence | ⚠️ |
| 3.1 | Held-out metrics are real | ✅ |
| 3.2 | Deterministic scoring | ✅ |
| 3.3 | 0.52 threshold suits deployment | ❌ |
| 3.4 | Served artifact is the evaluated one | ✅ |
| 4.1 | `extract_features.py` docstring vs code (RF `_01` only) | 🔴 known defect |
| 4.2 | `pos_dev_m` reference fix transfers | ❌ |
| 4.3 | C/N0 filter necessary | ✅ |
| 4.4 | Raw CSV units, not SI | 🔴 known trap |
| 5.1 | Ingest == replay on the dashboard | ✅ |
| 5.2 | Correlation window uses scoring time | 🔴 backfill unusable |
| 5.3 | Hysteresis resets on restart | 🔴 |
| 5.4 | Retry-safe ingestion | ✅ |
| 5.5 | Labels stored, never learned from | ✅ |
| 5.6 | Feedback precision is not field precision | ✅ |
| 5.7 | SQLite write volume | ⚠️ |
| 6.1 | Plausibility warnings catch bad input | ⚠️ 1 of 4 |
| 6.2 | 25% bounds margin | ✅ disclosed |
| 6.3 | API-key auth | ⚠️ opt-in, single shared key |
| 6.4 | SSE session cookie | ✅ decided, costs stated |
| 6.5 | Transport security | 🔴 none |
| 6.6 | Rate limiting | 🔴 none |
| 6.7 | `/drift` measures shift, not health | ✅ disclosed |
| 6.8 | Logs and metrics safe to expose | ⚠️ |
