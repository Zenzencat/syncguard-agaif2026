# SyncGuard per-prediction explainability — SHAP

`baseline_model_report.md` already reports *global* feature importances and names a real
limitation honestly: the model leans on u-blox's own RF-monitor/jamming-indicator fields
(AGC count, noise floor, jamming indicator) rather than the purpose-built spoofing-specific
features (code-Doppler residual, position deviation). That's an aggregate statement across
the whole held-out set. This adds **per-prediction** explanation — SHAP (SHapley Additive
exPlanations) via `shap.TreeExplainer` over the shipped RandomForest — so "why was *this*
event flagged" is answerable for any individual scored row, not just restated as the global
finding. Implemented in [`api/model_service.py`](api/model_service.py), exposed on
`POST /score` and `GET /events/{id}/explain`, surfaced on the dashboard. Does not retrain or
modify `models/model.joblib` — SHAP explains the existing shipped model exactly as it is.

## Method

**Why TreeExplainer, and why it needs the raw RF, not the Pipeline**: `shap.TreeExplainer`
implements the exact Tree SHAP algorithm (Lundberg et al., 2018) for tree ensembles — no
sampling, no approximation, the right and standard choice for a RandomForest. It does not
understand sklearn `Pipeline` objects, though: it needs `pipeline.named_steps["rf"]`
directly. Imputation therefore happens manually in `ModelService.explain()`
(`self._imputer.transform(X)`) before the array goes to both `rf.predict_proba()` and
`explainer.shap_values()` — the same two-step pattern the rest of `api/model_service.py`
already uses.

**Output shape and sign convention** (confirmed empirically against this SHAP version,
0.52.0, not assumed from docs): `shap_values()` returns a 3D array,
`(n_samples, n_features, n_classes)`. Class index `1` is "attack." Verified via the
additivity property — `sum(shap_values[:, :, 1]) + expected_value[1] == predict_proba(...)`
holds to floating-point precision (~2e-14) on real rows. **Positive SHAP value = pushes the
prediction toward attack; negative = pushes toward clean** — this is what `direction` in
every API response and the dashboard panel means.

**Determinism** (verified directly, same rigor as the RandomForest `n_jobs=1` fix in
`ROBUSTNESS_NOTES.md`): ran the same input through two independent `TreeExplainer` instances,
and through repeated calls on the same instance — bit-identical every time.
`feature_perturbation='tree_path_dependent'` (used here — the faster of the two modes and the
one that needs no background dataset, unlike `'interventional'`) has no sampling and no seed
parameter at all, because it doesn't need one: it's an exact algorithm over the tree
structure itself. It's also independent of the RandomForest's own `n_jobs` setting — SHAP's
tree traversal doesn't route through `predict_proba`, so the existing determinism fix and
this one are unrelated but complementary.

**Top-N, not the full vector**: every response returns the top 5 features by `|shap_value|`,
not all 23 — the full vector is too noisy to read at a glance in a UI or during a live demo.

## The speed/explainability tradeoff — a real systems design point, not a footnote

Exact Tree SHAP over 300 trees costs **~45-55ms per row**, measured directly (single-row and
batched — batching doesn't amortize the cost; it's genuinely a per-row expense, not a
fixed-overhead one). For `POST /score` — one ad-hoc request — that's imperceptible. For
**replay**, it's a real constraint: computed inline on every row, it meaningfully throttles
throughput.

**Correction, made honestly rather than left standing**: an earlier version of this
document estimated replay throughput from the nominal speed multiplier alone ("~20 rows/sec
regardless of speed," "speed=200 finishes 2503 rows in ~15 seconds without SHAP") without
actually measuring it. `OPERATIONAL_METRICS.md` benchmarks it directly and both estimates
were wrong: real measured throughput is **~5.3 rows/sec at speed=10 with live SHAP** and
**~14-16 rows/sec above the cutoff without it** — well below the nominal multiplier either
way, because the replay loop computes-then-sleeps additively (not overlapped) and
`asyncio.sleep()` itself overshoots its nominal duration by ~35-55% on this Windows
deployment (measured root cause, not assumed — see `OPERATIONAL_METRICS.md` §2). At the
*measured* rates, a full 2503-row replay takes on the order of 8 minutes with live SHAP at
speed=10, or ~2.5-3 minutes above the cutoff without it — not the 15 seconds to ~2 minutes
this document originally implied. The speed-gating design below is still the right call (the
relative gap between "live SHAP" and "no live SHAP" throughput is real and large), but the
absolute numbers here were corrected against real measurement, not left as an untested
estimate.

**Design chosen: speed-gated live computation, with lazy on-demand fallback — not an
all-or-nothing choice.**

- `POST /score` always computes SHAP. No exceptions, no gating — this is the ad-hoc,
  single-request path where 50ms is free.
- Replay computes SHAP live only when `speed <= LIVE_EXPLAIN_MAX_SPEED = 20`
  (`api/replay.py`). **Why 20, specifically, not an arbitrary round number**: the dataset
  replays at ~1Hz in real time, so at speed=20 the replay loop's *nominal* per-row delay is
  ~50ms (`1s / 20`) — the same order of magnitude as the SHAP cost, so the relative slowdown
  from adding it stays bounded rather than compounding at higher multipliers where the
  nominal delay is much smaller than 50ms. (The loop's *actual* per-row delay is higher than
  nominal either way — see the correction above and `OPERATIONAL_METRICS.md` §2 for why — so
  "nearly free" was an overstatement; the threshold is still the right one, just for a more
  modest reason than originally claimed.) Below the threshold, live SHAP roughly matches the
  order of magnitude of cost the loop already carries; above it, SHAP would dominate and
  silently override the requested pace far more than it already does.
- Above that speed, live SHAP is skipped entirely (`top_features` stored as `NULL`) and
  replay runs at full requested speed. **Nothing scored this way is permanently
  unexplainable** — `GET /events/{id}/explain` recomputes SHAP on demand from the event's own
  stored `features_json` (persisted for every event regardless of speed), and persists the
  result back onto the row (`EventStore.update_event_top_features`) so asking again doesn't
  recompute. `/replay/status` reports `live_explain: true/false` for the current run so the
  dashboard (and any other consumer) knows which regime it's in.
- Dashboard: clicking an event-log row or map marker shows the stored explanation
  immediately if present; if not, an **Explain** button computes it on demand and renders it
  — the same lazy pattern as the API, one click away.

The alternative designs considered and rejected: computing SHAP unconditionally everywhere
(simple, but throttles every fast demo run to ~20 rows/sec whether or not anyone ever looks at
most of those explanations) or never computing it during replay at all (fast, but means a
judge asking "why was that one flagged" mid-demo always waits, even for the common case of a
slow, explainable-live run). The hybrid costs a little more code (a speed check, a nullable
column, one extra endpoint) for a system that's fast by default at high speed, fully explained
by default at demo-reasonable speed, and never actually loses the ability to explain anything
— it's just deferred, not dropped.

## Sanity check: does the per-event evidence match the global finding? — mostly, with a real exception

Pulled a real sample of scored rows across all four attack types and inspected their SHAP
explanations directly (not just the aggregate feature-importance table).

**Confident predictions confirm the global finding, per-event.** Every attack type's
high-confidence attack rows (proba > 0.9) are dominated by the same handful of RF-monitor
features named in `baseline_model_report.md`:

| Attack type | proba | Top SHAP features |
|---|---|---|
| Jamming | 0.923 | `agc_cnt_mean` (+0.108), `snr_l1_mean` (+0.093), `velN` (+0.052) |
| Meaconing | 0.997 | `snr_l1_mean` (+0.097), `agc_cnt_mean` (+0.091), `snr_l1_std` (+0.046) |
| Spoofing | 0.997 | `snr_l1_mean` (+0.139), `noise_per_ms_mean` (+0.097), `snr_l1_std` (+0.070) |
| Spoofing+Jamming | 0.943 | `agc_cnt_mean` (+0.088), `noise_per_ms_mean` (+0.077), `jam_ind_mean` (+0.063) |

Confidently-correct clean rows mirror this: `agc_cnt_mean`/`snr_l1_mean`/`noise_per_ms_mean`
dominate again, just with negative sign (pushing toward clean). So far, this is exactly what
the global feature-importance table already said — confirmed, not contradicted, at the
per-event level.

**Near-threshold Jamming cases tell a different, more interesting story — the headline
finding of this sanity check, not a footnote.** Every near-threshold case found (probability
within ±0.05 of the 0.52 decision threshold) was a **Jamming** row — itself notable, given
jamming is already established elsewhere in this project (`baseline_model_report.md`,
`ROBUSTNESS_NOTES.md`, `FOLD_ANALYSIS.md`) as the weakest, most heterogeneous attack type.
More importantly, **the usual RF-monitor features often argue the wrong way** in exactly
these cases:

```
attack_type=Jamming true_attack=1 proba=0.473
  noise_per_ms_mean +0.105 (toward attack)
  snr_l1_mean        -0.099 (toward clean)   <- usually the top attack signal; here argues clean
  agc_cnt_mean        -0.099 (toward clean)   <- usually the top attack signal; here argues clean
  doppler_l1_std      +0.063 (toward attack)
  pDOP                +0.045 (toward attack)

attack_type=Jamming true_attack=1 proba=0.557
  noise_per_ms_mean +0.140 (toward attack)
  agc_cnt_mean       -0.087 (toward clean)
  doppler_l1_mean    +0.055 (toward attack)
  snr_l1_mean        +0.030 (toward attack)   <- flips sign vs. the case above
  snr_l1_min         -0.028 (toward clean)

attack_type=Jamming true_attack=1 proba=0.560
  hAcc          +0.110 (toward attack)
  agc_cnt_mean  -0.087 (toward clean)
  snr_l1_mean   -0.075 (toward clean)
  headAcc       +0.065 (toward attack)
  n_sats_l1     +0.065 (toward attack)
```

In these genuinely hard cases, `agc_cnt_mean` and `snr_l1_mean` — the two features that
dominate every confident prediction above — are frequently pushing *toward clean even in true
attack rows*, and it's `noise_per_ms_mean`, `hAcc`, `doppler_l1_mean`, and `n_sats_l1` that end
up deciding the outcome instead. **This is not simply "the model always relies on the
RF-monitor fields" — for the rows that are actually hard, those fields sometimes argue the
wrong direction, and different features become load-bearing.** Reported here exactly as
found, matching this project's standing rule (`ROBUSTNESS_NOTES.md`, `CALIBRATION_NOTES.md`,
`NORMALIZATION_NOTES.md`): a finding gets reported honestly whether or not it flatters the
existing narrative, and this one adds real nuance rather than just restating it.

## Worked examples

**1. Confident attack, correctly flagged** (Spoofing, proba=0.997):
`snr_l1_mean` (+0.139), `noise_per_ms_mean` (+0.097), `snr_l1_std` (+0.070),
`jam_ind_mean` (+0.064), `agc_cnt_mean` (+0.060) — every top feature pushes toward attack,
all from the RF-monitor family. An unambiguous, textbook-confident case.

**2. Confident clean, correctly passed** (Spoofing, proba=0.000):
`agc_cnt_mean` (−0.116), `snr_l1_mean` (−0.112), `noise_per_ms_mean` (−0.073),
`gSpeed` (−0.031), `snr_l1_std` (−0.027) — the mirror image of #1, same features, opposite
sign, opposite conclusion.

**3. Near-threshold, genuinely ambiguous** (Jamming, true label=attack, proba=0.473 — just
below the 0.52 threshold, so this one is actually a miss): `noise_per_ms_mean` (+0.105)
pulls toward attack while `snr_l1_mean` (−0.099) and `agc_cnt_mean` (−0.099) — normally the
two strongest attack signals — both pull toward clean, with `doppler_l1_std` (+0.063) and
`pDOP` (+0.045) providing the rest of the (insufficient) push toward attack. This is the more
useful Q&A example precisely because it isn't confident: it shows the model's internal
disagreement on a row it actually gets wrong, not just a confirmation of what it gets right.

## Where this lives

- `api/model_service.py` — `ModelService.explain()`; the `TreeExplainer` is built once at
  startup alongside the model load.
- `POST /score` (`api/main.py`) — always includes `top_features` in the response.
- `GET /events/{id}/explain` (`api/main.py`, `ExplainResponse` in `api/schemas.py`) — on-demand
  explanation for any already-scored event; returns the cached explanation if one exists
  (`cached: true`), otherwise computes and persists it (`cached: false`).
- `api/db.py` — `top_features_json` column on `scored_events` (nullable; `NULL` means
  "not explained yet," applied via an idempotent `ALTER TABLE` so existing local DBs don't
  break), `get_event()`, `update_event_top_features()`.
- `api/replay.py` — `LIVE_EXPLAIN_MAX_SPEED = 20`; the speed-gating logic described above.
- `syncguard_interactive_summary.html` — click an event-log row or live-map marker to see its
  top features as a horizontal bar list (red = toward attack, teal = toward clean, bar length
  ∝ `|shap_value|`); an **Explain** button appears instead when the event wasn't explained
  live.
- `requirements-api.txt` — `shap==0.52.0`.
- Does not modify `models/model.joblib`, `models/model_baseline.joblib`,
  `train_baseline_model.py`, `train_improved_model.py`, or any deployed threshold.
