# SyncGuard: temporal-coherence features — the data-diversity ceiling, confirmed from a sixth angle

Follow-up to [`SPOOFING_FEATURES.md`](SPOOFING_FEATURES.md) and [`STATIONARY_SCOPE.md`](STATIONARY_SCOPE.md).
Same honest-tradeoffs discipline and the same dual-evaluation harness as those, plus
`ROBUSTNESS_NOTES.md` / `CALIBRATION_NOTES.md` / `NORMALIZATION_NOTES.md` / `GBM_COMPARISON.md`.

`SPOOFING_FEATURES.md` rejected 11 features that reduced `rinex.csv` to **per-epoch spatial
aggregates** (dispersion across satellites at one instant); they misread degraded sky
visibility as attack on dynamic recordings. This experiment attacked a genuinely different
axis: **temporal coherence** — how each satellite's carrier phase, code, and Doppler behave
*over time*. The hypothesis had two halves:

1. A coherent code/carrier divergence (CMC drift) is something a spoofer walking a receiver
   creates but jamming fundamentally cannot fake — jamming raises noise, it does not produce a
   smooth code-vs-carrier ramp. So a CMC-drift feature should **help without costing jamming
   recall.**
2. Temporal-coherence features should **not** fire on degraded reception — a receiver tracking
   few but authentic satellites still has code/carrier locked and smooth Doppler. So they
   should avoid the `1.10.6` / `1.11.7` / `3.2.8` failure mode that sank the last attempt.

**Verdict: reject.** The measured dual evaluation is below. Half the hypothesis held (the
degraded-reception landmine did *not* fire this time), but the feature set produced a null on
rotating GroupKFold and a **−5pt clean-recall regression on the fixed shipped split** — driven
by a *different* recording (`2.1.1`) than last time. Same bar, same outcome: does not clear
both evaluations, not integrated, shipped model untouched.

The real value is the **Step 0 finding** — the hypothesis split cleanly in two, and the
measured result confirms it.

## Step 0 — feasibility (the substantive finding)

### Can per-SV temporal features be computed? Yes, with cycle-slip-aware arc segmentation

- Per-SV cadence is a clean ~0.2 s (5 Hz) on every recording.
- Carrier-phase rate tracks `−doppler_L1` (median residual 0.15–0.41 Hz per recording; all 24
  pass the sign-convention check).
- **Raw CMC (`PR_L1 − λ·carrier_phase_L1`) over time-gap-defined tracking runs is unusable** —
  implausible km/s slopes, because carrier phase slips *without a time gap*. Segmenting arcs on
  **phase-rate discontinuity** (`|Δφ/Δt + doppler|·Δt > 5 cycles`) as well as time gaps fixes
  it: CMC slopes then land at a physical 0.01–30 m/s.
- After slip-aware segmentation, **~89% of SV-epoch rows sit in an arc long enough for a
  trailing 20-epoch (~4 s) window** (per-recording 72–100%); the rest are arc cold-starts →
  NaN → `SimpleImputer(median)`, the existing convention.

### The hypothesis split in two — this is the finding

Probed clean-vs-attack separation (median across satellites, per recording) for each candidate:

| Candidate | Signal | Fires on degraded reception? |
|---|---|---|
| **CMC trailing-window slope** | Strong on Meaconing (10–26×) and Spoofing+Jamming (18–35×). **~none on Jamming** (1 of 4 comparable recordings). **Spoofing: strong on 4 of 8 recordings, absent/inverted on the other 4 — including *both* spoofing recordings in the fixed shipped split** (`2.3.2` 0.035→0.021, `2.1.1` 0.075→0.222). | **No.** `1.10.6` / `1.11.7` clean segments stay 0.017–0.13 m/s. **Half 2 of the hypothesis holds for CMC.** |
| **Reacquisition rate** (trailing-10 s arc-start rate per SV) | **Strong on Spoofing / Meaconing / Spoofing+Jamming — fires on all, including fixed-split `2.3.2` (0.022→0.170) and `2.1.1` (0.000→0.057).** Modest on Jamming. | **YES, catastrophically.** `1.11.7`'s obstructed-sky *clean* segment has a trailing-10 s reacquisition rate of **0.347 — 30× higher than any attack segment anywhere.** This is the exact degraded-reception failure mode that broke `SPOOFING_FEATURES.md`, and `1.11.7` is GroupKFold fold 3. |
| Doppler smoothness (trailing-window Doppler curvature) | Weak — no Jamming/Spoof+Jam signal, ~2× on dynamic Meaconing/Spoofing. | No. Won't cost jamming recall; too weak to carry the set. |
| Tracked-set Jaccard (instantaneous) | **Dead** — 1.000 clean / 1.000 attack on every recording at 5 Hz cadence. | n/a |
| L1/L2 divergence trend | Not deeply probed — L2 only 38–63% coverage, lower priority. | — |

**So: the one feature with the right non-degradation property (CMC slope) has no signal on
the fixed split's spoofing recordings; the one feature with the right signal (reacquisition
rate) reintroduces the `1.11.7` landmine.** Doppler smoothness is too weak to bridge the gap.

### Feature set tested — the landmine-free subset

Per the split above: test only the features that pass the degraded-reception check, and
**deliberately exclude** reacquisition rate (landmine) and instantaneous Jaccard (dead). All
5 are strictly causal — a trailing 20-epoch closed-form linear slope / discrete curvature
*within* a cycle-slip-aware arc, current epoch inclusive, no future rows; cold start → NaN →
imputed. (`extract_temporal_features.py`.)

| feature | definition |
|---|---|
| `sp_cmc_slope_p50` / `sp_cmc_slope_max` | median / max of per-SV \|trailing CMC slope\| across tracked SVs (m/s) |
| `sp_cmc_slope_frac_hi` | fraction of tracked SVs with \|CMC slope\| > 3 m/s |
| `sp_dop_smooth_p50` / `sp_dop_smooth_max` | median / max of per-SV trailing-window std of the 2nd difference of Doppler (Hz); the deployable, vectorizable form of "not a smooth orbital trajectory" (Step 0 used a quadratic-fit residual — the feature is marginal either way) |

Additive only — 5 new columns alongside the shipped 23, written to a separate supplement
(`processed/temporal_features_supplement.parquet`, keyed on the identical `(run_id, real_time)`
nav spine). `extract_features.py` and the shipped parquet are untouched.

## Dual evaluation — measured, not predicted

Both evaluations that disagreed for `SPOOFING_FEATURES.md`, run deliberately. Shipped RF
config (300 trees, `class_weight='balanced'`, `random_state=42`, `n_jobs=1` at predict).
Determinism: every model scored twice, bit-identical (PASS).

### A. Rotating 4-fold GroupKFold (group = recording, `attack_type_balanced_folds`, seed 42), mean ± std, threshold 0.52

| Metric | BASE (23) | BASE+NEW (28) | Δ |
|---|---|---|---|
| Accuracy | 0.829 ± 0.067 | 0.831 ± 0.063 | +0.002 |
| Clean recall | 0.661 ± 0.202 | 0.653 ± 0.186 | −0.008 |
| **Jamming recall** | 0.783 ± 0.103 | 0.796 ± 0.099 | **+0.013** |
| Meaconing recall | 0.970 ± 0.013 | 0.973 ± 0.012 | +0.004 |
| Spoofing recall | 0.953 ± 0.014 | 0.952 ± 0.017 | −0.001 |
| Spoofing+Jamming recall | 0.979 ± 0.012 | 0.979 ± 0.012 | 0.000 |
| Pooled OOF ROC-AUC | 0.847 | 0.849 | +0.002 |

**A null.** Nothing moves beyond fold noise. No jamming regression (small improvement), but no
real gain anywhere either.

### B. Fixed 8-recording shipped TEST split, threshold 0.52

| Metric | BASE (23) | BASE+NEW (28) | Δ |
|---|---|---|---|
| Accuracy | 0.878 | 0.872 | −0.006 |
| **Clean recall** | **0.671** | **0.622** | **−0.050** |
| False positive rate | 0.329 | 0.378 | +0.050 |
| **Jamming recall** | 0.863 | 0.881 | **+0.018** |
| Meaconing recall | 0.964 | 0.964 | −0.001 |
| Spoofing recall | 0.969 | 0.972 | +0.003 |
| Spoofing+Jamming recall | 0.985 | 0.985 | 0.000 |
| ROC-AUC | 0.916 | 0.919 | +0.003 |

**Per-recording clean recall / attack recall (@ t=0.52):**

| Recording | BASE clean / attack | BASE+NEW clean / attack | note |
|---|---|---|---|
| `1.10.6` (Jamming, dyn) | 0.090 / 0.932 | **0.096 / 0.926** | degraded-reception watch — **essentially unchanged** |
| `3.2.8` (Meaconing, dyn) | 0.783 / 0.985 | **0.732 / 0.985** | degraded-reception watch — mild −5pt clean, attack held |
| `1.6.4` (Jamming, stat) | 0.777 / 0.644 | 0.660 / 0.737 | −12pt clean, +9pt attack |
| **`2.1.1` (Spoofing, stat)** | **0.924 / 0.968** | **0.502 / 0.971** | **−42pt clean — drives the aggregate regression** |
| `2.3.2` (Spoofing, dyn) | 0.684 / 0.971 | 0.691 / 0.971 | unchanged |
| `3.2.7` (Meaconing, stat) | 0.951 / 0.940 | 0.954 / 0.939 | unchanged |
| `2.6.3` / `2.6.4` (Spoof+Jam) | 0.460 / 0.751 | 0.496 / 0.736 | ~unchanged |

**What held, and what broke:**

- **The degraded-reception landmine did NOT fire.** `1.10.6` clean recall 0.090 → 0.096
  (unchanged); `3.2.8` clean 0.783 → 0.732 (mild), attack recall held at 0.985. Contrast
  `SPOOFING_FEATURES.md`, where the same two recordings lost 16pt attack recall / 74pt clean
  recall. **Half 2 of the hypothesis is confirmed by measurement, not just reasoning** — a
  temporal-coherence feature set genuinely does not misread poor sky visibility as attack.
- **But a different recording broke.** `2.1.1` (stationary spoofing) clean recall collapses
  0.924 → 0.502 — the temporal features read its clean prelude as attack. `2.1.1` is a
  "power-ramping" spoof; something in its clean-segment CMC-slope / Doppler-curvature profile
  now trips the model. Attack recall on `2.1.1` held (0.968 → 0.971), so this is purely a
  clean→attack false-positive shift on that one recording — but it is a −5pt aggregate
  clean-recall regression on the shipped evaluation.

## SHAP — do the temporal features rank? Two of five, moderately

*(fixed-split BASE+NEW model, exact Tree SHAP, mean\|abs\| over 800 sampled TEST rows; per-fold
Gini from the GroupKFold models. SHAP determinism verified — scored twice, max\|diff\| = 0.0.)*

| Feature | mean\|SHAP\| | SHAP rank (of 28) | per-fold Gini (mean ± std) | Gini rank |
|---|---|---|---|---|
| `sp_dop_smooth_p50` | 0.0242 | **5** | 0.0269 ± 0.0100 | 15 |
| `sp_cmc_slope_p50` | 0.0183 | **11** | 0.0362 ± 0.0091 | 9 |
| `sp_cmc_slope_max` | 0.0103 | 19 | 0.0247 ± 0.0059 | 17 |
| `sp_dop_smooth_max` | 0.0071 | 23 | 0.0086 ± 0.0030 | 26 |
| `sp_cmc_slope_frac_hi` | 0.0033 | 26 | 0.0113 ± 0.0017 | 24 |

Top of the fixed-split ranking: `agc_cnt_mean` (0.086), `snr_l1_mean` (0.072),
`noise_per_ms_mean` (0.048), `jam_ind_mean` (0.025), **`sp_dop_smooth_p50` (0.024)**,
`hAcc`, `velN`, `gSpeed`, … **`sp_cmc_slope_p50` at #11**.

So the model **does use two of the five** — `sp_dop_smooth_p50` (the Doppler-curvature median,
just below `jam_ind_mean`) and `sp_cmc_slope_p50` (the CMC-slope median) — while the `_max` and
`_frac_hi` variants are near-dead. The model picking them up is consistent with `SPOOFING_FEATURES.md`'s
finding (the model will use new signal-quality features when given them); as there, that does
**not** translate into a headline-metric improvement, and on the fixed split it translates into
the `2.1.1` clean-recall regression above.

## Determinism

Every model (both feature sets, both evaluations, per-fold and fixed) scored twice — output
bit-identical (`n_jobs=1` at predict, same discipline as `ROBUSTNESS_NOTES.md`). SHAP values
scored twice — bit-identical (max\|diff\| = 0.0). All PASS.

## Decision: do not integrate

Same bar as every prior experiment: **integrate only if it clears BOTH evaluations with no
jamming or degraded-reception regression.**

- GroupKFold: null — no gain to integrate.
- Fixed split: −5pt aggregate clean recall, −42pt on `2.1.1`. That is a meaningful regression
  on the evaluation the shipped reports and dashboard use.
- Jamming recall and the degraded-reception recordings held on both evaluations this time —
  genuinely different from `SPOOFING_FEATURES.md`, and worth recording — but "held on the two
  things that broke last time" is not a pass when a third thing breaks instead.

The shipped 23-feature model, `extract_features.py`, `processed/syncguard_features.parquet`,
`api/`, and all thresholds are untouched. No model was retrained into the serving path.

## The data-diversity ceiling — a sixth independent angle

This is the sixth distinct approach to move the shipped model that has failed the same way,
for the same root reason: **24 recordings, ~5 per attack type, is not enough independent data
to learn a feature-level improvement that generalizes across recording configurations.**

| # | Attempt | Where it failed |
|---|---|---|
| 1 | Jamming sample reweighting (`ROBUSTNESS_NOTES.md`) | Made TEST clean *and* jamming recall worse — overfit ~5 training jamming recordings |
| 2 | Isotonic / sigmoid calibration (`CALIBRATION_NOTES.md`) | Isotonic bought clean recall at −10.5pt jamming recall |
| 3 | Session-relative normalization (`NORMALIZATION_NOTES.md`) | Oracle version −33.6pt jamming on the one fold it was meant to fix; deployable version did nothing |
| 4 | XGBoost (`GBM_COMPARISON.md`) | −9.5pt jamming recall for +3.1pt clean recall |
| 5 | Per-epoch spatial spoofing features (`SPOOFING_FEATURES.md` + `STATIONARY_SCOPE.md`) | Cleared GroupKFold, −9.2pt jamming / −8.6pt clean on the fixed split (dynamic degraded-reception recordings); stationary-only re-scope blocked by 1 stationary jamming recording |
| **6** | **Temporal-coherence features (this doc)** | **Null on GroupKFold; −5pt clean recall on the fixed split (`2.1.1`). The one hypothesis half that held — no degraded-reception misfire — is real but not sufficient.** |

The consistent signature: a feature idea looks promising on one evaluation, and a *different*
recording breaks it on the other, because with this few recordings per attack type there is no
configuration of held-out data that is simultaneously representative and large enough to trust.
`STATIONARY_SCOPE.md` reached the same conclusion from the deployment-scoping direction. The
shipped model's own honest limitations (`baseline_model_report.md`, weaker clean recall;
leaning on u-blox RF-monitor fields) stand — and the evidence now strongly suggests they will
not be closed by feature engineering on *this* dataset. The productive next step is more
data (more recordings, more configurations per attack type), not more features.

## Where this lives

- `extract_temporal_features.py` — the 5-feature supplementary extractor (slip-aware arcs,
  causal trailing windows). Needs the raw scenario tree (`$SYNCGUARD_RAW_ROOT` / `raw/` /
  `../agaif-materials/dataset/raw`).
- `temporal_coherence_experiment.py` — the dual evaluation (GroupKFold + fixed split +
  per-recording breakdown + SHAP).
- `temporal_feasibility_probe{1,2,3}.py` — the Step-0 probes behind the hypothesis split
  (tracking-run lengths; slip-aware CMC slopes + Doppler curvature; reacquisition rate over
  all 24 recordings).
- `processed/temporal_features_supplement.parquet`, `processed/temporal_shap_ranking.csv` —
  gitignored (regenerate by re-running).
- Does **not** modify `models/model.joblib`, `models/model_baseline.joblib`,
  `extract_features.py`, `processed/syncguard_features.parquet`, `api/`,
  `train_baseline_model.py`, `train_improved_model.py`, or any deployed threshold.
