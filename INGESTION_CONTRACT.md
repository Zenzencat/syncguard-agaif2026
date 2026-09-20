# SyncGuard ingestion contract — `POST /ingest`

Phase 1 of the near-production work. This document is the contract between an external
collector (an edge agent, a lab harness, another service) and the SyncGuard scoring service.

**Status labels used throughout this document, per the project's convention:**

| Label | Meaning |
| --- | --- |
| **REAL** | Real data or a real computation over real data. |
| **MEASURED** | A number this repo actually measured, on stated hardware/inputs. |
| **ESTIMATE** | Extrapolated or derived, with the formula stated. |
| **SIMULATED** | Produced by a model/placeholder, not observed. |
| **CALLER-SUPPLIED** | Asserted by whoever called the API. Trusted, not verified. |

**What this endpoint is not.** It has never been fed by a real GNSS receiver at a real
telecom site. Nothing in this document or in `api/ingest.py` constitutes field validation, and
the service does not operate on live receivers. The endpoint exists so that it *could* be fed
by one; whether the model generalizes to that input is an open question, addressed (as an
explicitly unvalidated assumption) in Phase 3's `ASSUMPTIONS_PRODUCTION.md`.

---

## 1. Why the caller computes the features

The endpoint accepts the **23-feature vector**, already extracted, not raw RINEX/UBX logs.

This was the least invasive of the two options, and the choice is worth stating plainly:

- `extract_features.py` is a **whole-file batch pipeline**. It reads three complete CSVs per
  scenario (`nav_pvt.csv`, `rinex.csv`, `mon_rf.csv`), does per-satellite `groupby`/`diff`
  work across the entire recording, computes a reference fix from the recording's own first
  60 seconds, and joins the three streams with `merge_asof`. It is not an incremental,
  per-window function, and there is no per-window entry point to call.
- Accepting raw observables would therefore have meant writing a **second, streaming
  implementation** of that same logic. Two implementations of a feature definition drift.
  Since the 23 features, the model and the 0.52 threshold are frozen, a drifting second
  extractor is precisely the failure mode to avoid.
- So the service accepts the already-extracted vector, and this document specifies exactly
  how to produce it. Section 3 is the mapping; section 4 is the part that actually bites.

`POST /ingest` and `POST /score` share one Pydantic definition of those 23 fields
(`FeatureVector` in `api/schemas.py`) so the two entry points cannot drift apart either.

---

## 2. Request and response

### Request

```jsonc
{
  "batch_id": "collector-7-2026-09-20T06:00:00Z",   // optional, stored on every resulting row
  "observations": [                                  // 1..500 (MAX_INGEST_BATCH)
    {
      "tower_id": "MPW012",                          // required; must exist — see §5
      "timestamp": "2026-09-20T06:00:00Z",           // required; UTC observation time — see §6
      "fixType": 3.0, "gSpeed": 0.005, "hAcc": 0.183 // ... the 23 features, all optional
    }
  ]
}
```

Every feature field is optional. A missing or `null` feature is imputed by the trained
pipeline's own `SimpleImputer(strategy="median")` — the same imputer, fit on the same training
data, that handled missing values during training. A sparse window is valid input, not an
error. It is still a degraded input, and the imputed median is the *training* median, not
anything about this tower.

### Response (`200`)

```jsonc
{
  "batch_id": "collector-7-...", "received": 24, "scored": 24,
  "duplicates": 0, "out_of_order": 0, "reordered_in_batch": 0,
  "alerts": 2,            // observations whose tower was in alert_state='alerting' afterwards
  "live_explain": false,  // whether SHAP was computed inline — see §8
  "results": [
    {
      "tower_id": "MPW012",                 // the resolved tower_key, may differ from input
      "timestamp": "2026-09-20T06:00:00Z",
      "event_id": 19, "probability": 0.96, "severity": 0.941,
      "predicted_label": "attack", "alert_state": "alerting",
      "correlation_score": 0.0,
      "duplicate": false, "out_of_order": false, "explained": false
    }
  ]
}
```

### Error codes

| Code | When |
| --- | --- |
| `422` | Unknown `tower_id`; missing/invalid `timestamp`; a feature outside its declared range (e.g. `hAcc < 0`, `numSV > 64`); non-numeric feature; empty `observations`; batch larger than 500. |
| `503` | No trained model loaded (run `make train`, restart). |

A batch containing **one** unknown tower is rejected **whole**. Tower resolution happens
before any scoring, so a bad batch is never half-applied. Verified by
`tests/test_ingest.py::test_unknown_tower_rejects_the_whole_batch`.

---

## 3. Mapping: raw receiver observables → the 23 features

Source of truth for every row below is `extract_features.py`. **REAL** — this is what was
actually run over the Jammertest 2024 recordings to produce the training data.

### 3.1 The observation window

One observation = **one `nav_pvt` epoch (~1 Hz)**, with the nearest `rinex` and `mon_rf`
aggregates joined onto it within a **±1 s tolerance** (`pandas.merge_asof(direction="nearest",
tolerance=1s)`). The `nav_pvt` stream is the spine; where no RINEX or RF epoch falls within
1 s, those features are simply absent for that window (and imputed).

**MEASURED** on the real training data: median inter-row gap **1.0 s**, p99 **1.2 s**
(44,639 rows across 24 runs). A collector should submit at roughly 1 Hz per tower. Nothing
enforces this — a different cadence is accepted, and its effect on model behaviour is
**unvalidated**.

### 3.2 From `nav_pvt` — u-blox `UBX-NAV-PVT`, taken at the epoch, no aggregation

| Feature | Raw observable | Transformation |
| --- | --- | --- |
| `fixType` | `UBX-NAV-PVT.fixType` | pass through (0 = no fix … 3 = 3D, 4 = GNSS+DR) |
| `gSpeed` | `UBX-NAV-PVT.gSpeed` | pass through |
| `hAcc` | `UBX-NAV-PVT.hAcc` | pass through |
| `vAcc` | `UBX-NAV-PVT.vAcc` | pass through |
| `sAcc` | `UBX-NAV-PVT.sAcc` | pass through |
| `headAcc` | `UBX-NAV-PVT.headAcc` | pass through |
| `pDOP` | `UBX-NAV-PVT.pDOP` | pass through |
| `numSV` | `UBX-NAV-PVT.numSV` | pass through (satellites **used in the fix**) |
| `velN` | `UBX-NAV-PVT.velN` | pass through |
| `velE` | `UBX-NAV-PVT.velE` | pass through |
| `velD` | `UBX-NAV-PVT.velD` | pass through |
| `pos_dev_m` | `lat`, `lon`, `height` | derived — see below |

`pos_dev_m` is the only derived PVT feature:

1. Take the **median** `lat`, `lon`, `height` over the **first 60 seconds of that receiver's
   session**. This is the reference fix.
2. For each later epoch, convert to metres with a flat-Earth approximation:
   `dlat_m = (lat − ref_lat) × 111320`, `dlon_m = (lon − ref_lon) × 111320 × cos(ref_lat)`,
   `dh_m = height − ref_height`.
3. `pos_dev_m = sqrt(dlat_m² + dlon_m² + dh_m²)`.

Two consequences a collector must handle, both **unvalidated** for a real deployment:

- **The reference fix is per-session, not absolute.** In training, "session" meant "one
  scenario recording". A permanently-installed tower receiver has no natural session
  boundary. Whatever a collector picks — first 60 s after boot, a surveyed coordinate, a
  rolling reference — is a **design decision this repo has not tested**. A surveyed fixed
  coordinate is the obvious production choice and is *not* what the model was trained on.
- **It is meaningful only for stationary receivers.** `STATIONARY_SCOPE.md` covers the scope
  limitation. For a moving receiver this feature is noise.

### 3.3 From `rinex` — per-satellite, per-epoch observables, aggregated across satellites

All of these are computed **per epoch, across every satellite tracked at that epoch**.

| Feature | Raw observable | Transformation |
| --- | --- | --- |
| `n_sats_l1` | `satellite` | **count** of satellite rows at this epoch (every tracked L1 satellite, all constellations — *not* `numSV`) |
| `snr_l1_mean` | `snr_L1` | mean, **after** discarding values `< 0` or `> 60` |
| `snr_l1_std` | `snr_L1` | std dev, same filter |
| `snr_l1_min` | `snr_L1` | minimum, same filter |
| `doppler_l1_mean` | `doppler_L1` | mean |
| `doppler_l1_std` | `doppler_L1` | std dev |
| `pr_doppler_residual_mean` | `pseudorange_L1`, `doppler_L1` | mean of the per-satellite residual below |
| `pr_doppler_residual_std` | `pseudorange_L1`, `doppler_L1` | std dev of the same |

**The C/N0 filter is part of the contract, not a cleanup detail.** A fraction of `snr_L1`
samples in the real logs are corrupted artifacts with values up to ~4×10⁷. They are set to
`NaN` *before* the mean/std/min. A collector that does not apply the same `0 ≤ snr ≤ 60`
filter will produce a wildly different `snr_l1_mean` from one bad sample.

**The code-Doppler residual**, per satellite, in pseudorange-rate units (m/s):

```
λ_L1          = c / 1575.42e6            # c = 299792458.0
actual_rate   = Δpseudorange_L1 / Δt     # consecutive epochs OF THE SAME SATELLITE
predicted_rate = −doppler_L1 × λ_L1
residual      = actual_rate − predicted_rate
```

Set `residual = NaN` when **any** of: `Δt ≤ 0`, `Δt > 1.0 s`, or `|residual| > 5000`. The
`Δt > 1 s` exclusion is not arbitrary — the native per-satellite cadence is ~0.2 s, so a
larger gap means the satellite dropped and re-locked on a discontinuous pseudorange, which
produces a rate artifact rather than a physically meaningful residual. Then take the mean and
std of the surviving residuals across satellites at that epoch.

This is stateful across epochs: a collector must retain the previous epoch's
`pseudorange_L1` and timestamp **per satellite**.

### 3.4 From `mon_rf` — u-blox `UBX-MON-RF`

| Feature | Raw observable | Transformation |
| --- | --- | --- |
| `jam_ind_mean` | `jamInd_01` | mean over rows sharing the epoch timestamp |
| `agc_cnt_mean` | `agcCnt_01` | mean over rows sharing the epoch timestamp |
| `noise_per_ms_mean` | `noisePerMS_01` | mean over rows sharing the epoch timestamp |

> **Discrepancy, stated rather than papered over.** `extract_features.py`'s module docstring
> says these are "averaged across the two antenna paths". The code does not do that: it
> aggregates **`_01` only** and never reads the `_02` columns, even though it loads them. The
> `_mean` suffix is the mean over rows sharing one `real_time` value, not a mean across
> antenna paths.
>
> The **code** is the contract, because the code is what produced the training data the
> shipped model was fit on: **submit RF block 01 only.** Averaging both blocks would feed the
> model a differently-distributed input than it was trained on.
>
> `extract_features.py` is a frozen pipeline script under this work's constraints and was not
> edited. Reconciling the docstring with the code is a follow-up for whoever next touches
> that file, and does not change any shipped number.

### 3.5 What is deliberately **not** a feature

`extract_features.py` also computes `clock_drift_proxy_s` (receiver `iTOW` progression vs.
the logging host's wall clock). It is **not** one of the 23 model features and must not be
submitted — it would be ignored. Scenario metadata (`attack_type`, `scenario_id`,
`rover_state`, `run_id`, ground-truth `attack`) is excluded by design: a deployed detector
does not have it. Ingested events consequently carry **no ground truth**, which is why
`/events` rows from ingestion have `true_attack = null`, unlike replayed rows.

---

## 4. Units and scale — the part that actually bites

**Submit values on exactly the scale the Jammertest CSV columns use.** Do not apply your own
UBX scaling factors on top, and do not "helpfully" convert to SI.

This is not a hypothetical. While writing this phase, the example client and the tests both
started with hand-written, physically sensible values (`pDOP` 1.4, `hAcc` 2.5 m, `numSV` 18,
`n_sats_l1` 18). **Every one of them scored as `attack` with p = 0.60** on the shipped model —
a confident wrong answer, with no error, no warning, and nothing in the response to suggest
the input was malformed. They were replaced with vectors built from the real data's own
medians, which score 0.117 (clean) and 0.960 (attack).

Two scales in particular are not what a GNSS engineer would guess:

- `pDOP` sits around **0.01**, not around 1–2. (`UBX-NAV-PVT.pDOP` has a 0.01 scaling factor;
  whatever the Jammertest logger did, the CSV column is on this scale and the model was
  trained on it.)
- `n_sats_l1` sits around **41**, roughly double `numSV` (~32), because it counts every
  tracked L1 satellite across constellations while `numSV` counts only those used in the fix.

**The service cannot detect a scale error.** There is no plausibility check beyond the
per-field bounds in `api/schemas.py`, and those are wide. A caller must validate its own
extractor against the reference distributions below. Phase 3's `/drift` endpoint will make a
systematic scale error *visible* after the fact, but it will not reject the input.

### Reference distributions — **REAL / MEASURED**

Computed over all 44,639 rows of `processed/syncguard_features.parquet` (24 real Jammertest
2024 runs, both clean and attack). These are the distributions the model was trained on. A
collector's output should be broadly comparable; the ASEAN-deployment caveat in section 9
applies.

| feature | non-null % | p1 | median | p99 |
| --- | ---: | ---: | ---: | ---: |
| `fixType` | 100.0% | 0.0000 | 3.0000 | 3.0000 |
| `gSpeed` | 100.0% | 0.0000 | 0.0160 | 24.9820 |
| `hAcc` | 100.0% | 0.1280 | 0.3050 | 14578.7159 |
| `headAcc` | 100.0% | 0.0001 | 0.1800 | 0.1800 |
| `numSV` | 100.0% | 0.0000 | 32.0000 | 32.0000 |
| `pDOP` | 100.0% | 0.0092 | 0.0109 | 0.9999 |
| `sAcc` | 100.0% | 0.0280 | 0.0650 | 20.0000 |
| `vAcc` | 100.0% | 0.2370 | 0.5640 | 1823.0794 |
| `velD` | 100.0% | −4.1636 | −0.0010 | 2.7091 |
| `velE` | 100.0% | −15.4389 | 0.0000 | 18.6797 |
| `velN` | 100.0% | −14.5675 | −0.0010 | 13.8316 |
| `pos_dev_m` | 100.0% | 0.0165 | 2.4205 | 1550.5141 |
| `n_sats_l1` | 95.1% | 3.0000 | 41.0000 | 48.0000 |
| `snr_l1_mean` | 90.5% | 26.0000 | 41.8929 | 47.1905 |
| `snr_l1_std` | 89.5% | 1.7078 | 4.6064 | 9.4646 |
| `snr_l1_min` | 90.5% | 21.0000 | 28.0000 | 39.0000 |
| `doppler_l1_mean` | 95.1% | −596.5472 | −68.7995 | 346.8335 |
| `doppler_l1_std` | 94.5% | 1.4142 | 2211.6001 | 2501.7050 |
| `pr_doppler_residual_mean` | 94.8% | −174.3196 | 0.4827 | 191.3313 |
| `pr_doppler_residual_std` | 94.3% | 4.1402 | 130.6153 | 738.7386 |
| `jam_ind_mean` | 100.0% | 2.0000 | 9.0000 | 50.0000 |
| `agc_cnt_mean` | 100.0% | 702.0000 | 3861.0000 | 5616.0000 |
| `noise_per_ms_mean` | 100.0% | 41.0000 | 101.0000 | 220.0000 |

Note the ~5–10% non-null rate gap on the RINEX-derived features: that is the ±1 s
`merge_asof` tolerance failing to find a RINEX epoch, in the real training data. Missing
RINEX features are normal, not a collector bug.

---

## 5. `tower_id` resolution

- Resolved against the **real** 136-tower Telkomsel table (`GET /towers`), the same source
  `api/spatial.py::load_towers` reads.
- `tower_key` is tried first, then raw `site_id`. Prefer `tower_key`: **16 of the 136 real
  towers share the literal `site_id` "tbg"**, so a raw `site_id` there resolves
  deterministically to the first matching row, which may not be the tower you meant.
- The response's `results[].tower_id` always reports the resolved `tower_key`.
- Unknown → **422**, whole batch rejected.
- Same resolution order `POST /score` already uses for `tower_site_id`. Unchanged.

**CALLER-SUPPLIED.** The service cannot verify that the submitted features were observed at
the claimed tower. This differs from replay's tower attribution, which is **SIMULATED**
(round-robin over real towers, because the single-receiver Jammertest recording has no
per-tower attribution to recover). Ingested attribution is asserted by the caller and trusted.

---

## 6. Timestamps, ordering, and deduplication

### Timestamps

`timestamp` is the **observation** time, not the submission time. It is stored in a separate
column (`obs_timestamp`) from the service's own `created_at`. Naive timestamps are read as
UTC; offset-aware ones are converted to UTC. All stored values are normalized to
`%Y-%m-%dT%H:%M:%S.%fZ` so lexicographic string order equals chronological order
(`api/ingest.py::iso_utc`).

`created_at` remains "when this service scored it", and the live spatial correlation window
(`api/spatial.py`, 120 s) keys off `created_at`. **Consequence:** submitting a backlog of
hour-old observations does *not* place them in an hour-old correlation window — they
correlate against whatever else was scored in the last 120 s of wall-clock time. This is a
known limitation of routing ingestion through the existing correlation engine unchanged, and
it is **not** fixed in this phase.

### Ordering within a batch

Observations are **sorted by timestamp** before scoring, so per-tower hysteresis always sees a
monotonic sequence regardless of how the batch was assembled. A shuffled batch is not an
error; `reordered_in_batch` reports how many observations arrived out of order relative to an
earlier observation for the same tower, so a collector can see that its feed is shuffled.

### Out-of-order against already-accepted data

An observation **older than the newest already accepted for that tower** is flagged
`out_of_order: true`. It is still **scored and persisted** — nothing is dropped — but it is
deliberately **not fed to per-tower hysteresis**: a reading older than data the sensor has
already moved past is not the "next" reading in that sequence, and advancing a streak counter
with it would corrupt the tower's current state. Its reported `alert_state` is the tower's
current state, unchanged.

Out-of-order is judged **per tower**. A tower with no prior data is never out-of-order,
however old its timestamp.

### Deduplication

The dedup key is **(resolved `tower_key`, `obs_timestamp`)**, scoped to `source='ingest'`
rows. A repeat submission is:

- **not re-scored** (the model is not called),
- **not re-persisted** (no second row),
- answered with `duplicate: true` and the **original** `event_id`,
- **not** passed through hysteresis (`alert_state: null`).

This makes the endpoint safe to retry: a collector that times out mid-request can resubmit
the identical batch. Duplicates within a single batch are caught the same way. `batch_id` is
**not** part of the key — it is traceability metadata only.

---

## 7. Per-tower hysteresis

Ingestion uses `TowerHysteresisRegistry` (`api/hysteresis.py`): **independent streak state per
tower**, with the existing asymmetric thresholds — 3 consecutive above-threshold readings to
enter `alerting`, 5 consecutive below to return to `normal`.

This differs from replay, which keys hysteresis **per recording** rather than per tower, and
the difference is principled. Replay's tower attribution is round-robin: consecutive events
for the same simulated tower are ~136 events apart and have no real temporal relationship, so
debouncing them would smooth noise. Ingested observations do have that relationship — the
caller states which sensor each came from — so per-tower is the correct axis here.

**Limitation:** the registry is in-process and **not persisted**. A service restart resets
every tower to `normal`. The resulting `alert_state` *is* stored on each event row, so
`/events` and the dashboard show history, but the streak itself is lost. Making it durable
would require deciding how stale a streak may be before it is discarded, which nothing in this
phase measures.

---

## 8. Scoring path and explanations

Every ingested observation goes through **`ModelService.score()`** — the same call `POST
/score` makes, the same loaded artifact, the same 23 features, the same **0.52** threshold,
the same **single-threaded** `predict_proba`. Nothing about the model was changed to support
ingestion. `tests/test_ingest.py::test_ingest_uses_the_same_scoring_path_as_score` asserts
that an identical feature vector produces an identical probability through both endpoints.

SHAP explanations follow the same tradeoff replay already makes, applied to batch size:

- batch ≤ **20** observations → explained inline, `live_explain: true`;
- batch > 20 → skipped, `live_explain: false`.

Exact Tree SHAP costs ~45–55 ms/row (`SHAP_EXPLAINABILITY.md`), so a 500-row batch would
otherwise be dominated by explanation cost. Skipped rows are **not** permanently
unexplainable: `GET /events/{id}/explain` computes and caches an explanation on demand from
the row's own stored feature values.

Scoring is synchronous CPU work on a single-threaded path, so `api/ingest.py` yields to the
asyncio event loop every 25 observations. That prevents one large batch from starving the SSE
stream and other requests; it does **not** make scoring concurrent. Phase 4 measures what this
path actually costs — until then, the 500-observation cap is a guardrail, not a benchmark
result.

---

## 9. Where ingested events show up

Ingested events go through the same persistence and publication path as replayed ones, so
they appear with no client-side special-casing in:

| Surface | What it shows |
| --- | --- |
| `GET /events` | the rows, with `source: "ingest"`, `obs_timestamp`, `ingest_batch_id` |
| `GET /events/map` | latest event per tower — the dashboard's map state |
| `GET /events/{id}/explain` | on-demand SHAP |
| `GET /spatial/autocorrelation` | live global Moran's I + per-tower LISA over current severities |
| `GET /stream/events` (SSE) | live push, same envelope shape replay publishes |
| `/dashboard` | map, event log and alert pill, live |

The SSE envelope omits `attack_type` / `true_attack` (both `null`) because ingested data
carries no ground truth. The dashboard renders that as `—`.

**Drift warning, stated up front.** The training baseline is Jammertest 2024 — a Norwegian
test range, one receiver, September 2024. Real ASEAN input from real telecom infrastructure is
**expected** to sit outside these distributions. Phase 3 adds `/drift` to measure it. Until
then, a caller has no automated signal that its input is out of distribution.

---

## 10. Transport

**HTTP only.** `POST /ingest`, JSON, one batch per request.

**Future work, not built:** MQTT (or any broker-based transport) is the obvious fit for a real
edge fleet — intermittent connectivity, QoS-1 delivery, per-tower topics, store-and-forward
from the tower side. None of it exists here. There is no broker, no subscriber, no topic
schema, and no offline queue. The `(tower_id, timestamp)` dedup key was chosen partly because
it is the key an at-least-once broker delivery would need, but that is a design affordance,
not an implementation. Do not read this section as "MQTT support is present."

---

## 11. Quick start

```bash
# 1. Start the service
make serve                  # or: docker compose up --build

# 2. Send a sample batch and print the alerts
python examples/ingest_client.py
```

Minimal `curl` (one observation, one tower — substitute a real `tower_key` from `/towers`):

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "batch_id": "curl-demo",
    "observations": [{
      "tower_id": "MPW012",
      "timestamp": "2026-09-20T06:00:00Z",
      "fixType": 3.0, "gSpeed": 0.026, "hAcc": 0.331, "vAcc": 0.613, "sAcc": 0.078,
      "headAcc": 0.1504, "pDOP": 0.0109, "numSV": 31.0,
      "velN": -0.001, "velE": 0.0, "velD": -0.001, "pos_dev_m": 2.7085,
      "n_sats_l1": 39.0, "snr_l1_mean": 39.4688, "snr_l1_std": 5.2149, "snr_l1_min": 27.0,
      "doppler_l1_mean": -61.3147, "doppler_l1_std": 2096.3066,
      "pr_doppler_residual_mean": 2.8694, "pr_doppler_residual_std": 177.2779,
      "jam_ind_mean": 14.0, "agc_cnt_mean": 3159.0, "noise_per_ms_mean": 102.0
    }]
  }'
```

The feature values in that example are the medians of the real attack-labeled rows —
**SYNTHETIC**, built from real statistics. Interactive API docs: `http://localhost:8000/docs`.

---

## 12. Test coverage

`tests/test_ingest.py` (26 tests) covers the five required cases and more:

| Area | Tests |
| --- | --- |
| Valid batch | scores, persists, appears in `/events` with `source='ingest'`; identical probability to `/score`; `site_id` → `tower_key` resolution; large batch skips inline SHAP but explains lazily |
| Unknown tower | 422; whole batch rejected, nothing persisted |
| Malformed payload | 9 parametrized cases (missing/empty `observations`, missing `tower_id`/`timestamp`, unparseable timestamp, non-numeric feature, range violations, wrong type, over-size batch); plus the positive case that missing features are accepted and imputed |
| Out-of-order | shuffled batch is sorted and reported, not rejected; an observation older than accepted data is flagged, kept, and not advanced through hysteresis; out-of-order is per tower |
| Duplicates | repeat submission returns the original `event_id` and writes no row; duplicate within one batch; key is tower **and** timestamp |
| Hysteresis | per-tower entry after the streak; one tower's streak does not leak into another's; unit-level entry/exit state machine |
| Dashboard state | ingested events reach `/events/map` and the live spatial statistics |

`tests/test_existing_endpoints.py` (10 tests) guards the pre-existing endpoints against the
`FeatureVector` schema refactor.

Run: `python -m pytest`
