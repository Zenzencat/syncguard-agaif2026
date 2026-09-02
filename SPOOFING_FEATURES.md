# SyncGuard: spoofing-specific features (carrier-phase + L2 + per-constellation structure)

Same honest-tradeoffs discipline and the same GroupKFold harness as `ROBUSTNESS_NOTES.md`,
`CALIBRATION_NOTES.md`, `NORMALIZATION_NOTES.md`, and `GBM_COMPARISON.md`. Same bar: integrate
only if it is a genuine GroupKFold-validated improvement with **no meaningful jamming-recall
regression, especially on the shipped evaluation**.

**Verdict: DO NOT integrate.** Under 4-fold rotating GroupKFold this looked like the first
clear win of the session — clean recall, FPR, *and* jamming recall all improved, and SHAP +
Gini confirmed the model uses the new carrier-phase / L2 / per-constellation signal. But when
the full 34-feature model was actually retrained and evaluated on the **fixed 8-recording
held-out TEST split** that the shipped reports and dashboard use, it regressed hard: jamming
recall −9.2pt, clean recall −8.6pt, ROC-AUC −1.8pt. The regression is systematic (dynamic
degraded-reception recordings) and mechanistically explained, not fold noise. The shipped
model was reverted and is unchanged. Full story below — the GroupKFold analysis is kept
because the *method* and the SHAP evidence remain valid; only the ship decision changed.

Two standalone scripts, neither touching the shipped model:
- `extract_spoofing_features.py` — supplementary feature extraction from the raw scenario
  `rinex.csv` files → `processed/spoofing_features_supplement.parquet`, keyed on the identical
  `(run_id, real_time)` nav spine as `processed/syncguard_features.parquet`.
- `spoofing_features_experiment.py` — joins the two, runs BASE vs BASE+NEW under the
  established 4-fold `attack_type_balanced_folds` harness (seed 42, group = recording),
  floor-constrained threshold sweep, per-fold reporting, SHAP + Gini importance.
- `spoofing_fold3_dig.py`, `spoofing_lean_check.py` — the two follow-up diagnostics referenced
  below.

## Motivation

`baseline_model_report.md` and `SHAP_EXPLAINABILITY.md` both name the same limitation: the
detector leans on u-blox's own RF-monitor fields (`agc_cnt_mean`, `noise_per_ms_mean`,
`jam_ind_mean`, `snr_l1_mean`) rather than novel spoofing-specific signal. Every fix tried
since — sample reweighting (`ROBUSTNESS_NOTES.md`), isotonic calibration
(`CALIBRATION_NOTES.md`), session-relative normalization (`NORMALIZATION_NOTES.md`), XGBoost
(`GBM_COMPARISON.md`) — re-tuned the *same* information and was rejected for the same
jamming-vs-clean trade-off. This attempt is different: it feeds the model observables the
pipeline currently **discards at `extract_features.py`'s per-epoch aggregation step**.

## Feasibility check (done first, before any implementation)

The raw `rinex.csv` is per-satellite, per-epoch (~5 Hz) and carries three things the current 8
aggregates collapse away:

| Discarded structure | Computable? | What it enables |
|---|---|---|
| `carrier_phase_L1` (100% populated, unused) | **Yes** | carrier-phase-rate vs Doppler consistency residual (phase is ~mm precision vs ~1 m code noise, so this isolates true phase-lock breaks / cycle slips — distinct from the existing *code* `pr_doppler_residual`); code-minus-carrier (CMC) jump detection (a meaconing/spoofing splice produces a discontinuity). |
| `*_L2` columns (~46% of rows, unused) | **Yes** (for ~half the observations, enough for a per-epoch fraction) | a single-frequency L1 spoofer/meaconer can't reproduce L2, so the fraction of satellites still tracked on L2 collapses under spoofing/meaconing — and is roughly inert under pure jamming. |
| constellation prefix in `satellite` (used only as a count) | **Yes** | cross-constellation C/N0 spread — spoofers/meaconers hit some constellations harder than others. |
| **elevation-weighted C/N0** | **No** | no elevation/azimuth anywhere — not in `rinex.csv`, not in `nav_pvt.csv`, no NAV-SAT CSV. Would require parsing the `.ubx` binary (no `pyubx2` in env) or external SP3/ephemeris. Out of scope — the same wall as the dropped `clock_drift_proxy_s`. |
| **pseudorange / RAIM residuals** | **No** | needs satellite positions (broadcast ephemeris / SP3), not in the dataset. |
| receiver clock bias/drift | **No** | no NAV-CLOCK. `nav_pvt.tAcc` exists but is quantized/flat in most recordings — behaves like the dropped clock proxy. |

Verdict of the feasibility check: 3 of the 4 candidate levers are computable; **elevation-
weighted C/N0 is the one hard no.** The genuinely new information is carrier phase + L2 +
per-constellation structure.

## Features built (all additive — 23 → 34, nothing removed)

11 new columns. `sp_*` are per-epoch aggregates from `rinex.csv`; `nsat_l1_roll_std_30` is a
causal trailing-window feature derived in the experiment script from the existing `n_sats_l1`.

| feature | definition |
|---|---|
| `sp_cp_doppler_resid_std`, `sp_cp_doppler_resid_maxabs` | per-epoch spread / max of the per-SV (carrier-phase-rate + Doppler) residual |
| `sp_cmc_l1_step_std`, `sp_cmc_l1_step_maxabs` | per-epoch spread / max of \|Δ(pseudorange_L1 − λ·carrier_phase_L1)\| across SVs |
| `sp_frac_l2_tracked` | fraction of L1-tracked SVs also tracked on L2 |
| `sp_cn0_l1_minus_l2_mean` | mean per-SV (snr_L1 − snr_L2) over dual-tracked SVs |
| `sp_n_const` | number of constellations tracked this epoch |
| `sp_xconst_cn0_spread` | max − min of per-constellation mean C/N0 (≥2 constellations) |
| `sp_frac_low_cn0` | fraction of SVs below 30 dB-Hz |
| `sp_doppler_l1_mad` | median-absolute-deviation of per-SV Doppler (robust dispersion) |
| `nsat_l1_roll_std_30` | trailing 30-epoch rolling std of `n_sats_l1` (shift(1), per recording, cold-start NaN → imputed) |

### Leakage discipline (same as `session_normalization_experiment.py`)

- Every per-epoch statistic is computed strictly within one epoch — no lookahead.
- Carrier-phase / CMC residuals use per-SV *consecutive-epoch* differences (current +
  immediately-preceding epoch only), the same causal construction and `dt ≤ 1 s` reacquisition
  guard as `extract_features.py`'s `pr_doppler_residual`.
- `nsat_l1_roll_std_30` is strictly trailing (`shift(1).rolling(30, min_periods=15)`), built
  from the 1 Hz timeline, not the 5 Hz rinex stream.
- Coverage 85–95% (merge-asof tolerance gaps + cold starts); the rest is NaN → handled by the
  existing `SimpleImputer(median)`, same as `pos_dev_m`.

### One assumption, flagged and checked per recording

u-blox's `carrier_phase_L1` sign convention (`d(phase)/dt ≈ −doppler`) — verified empirically,
per-recording median \|residual\| 0.15–0.39 Hz against a 5 Hz sanity threshold; **all 24
recordings pass.** Any recording violating it has its `sp_cp_*` / `sp_cmc_*` columns NaN'd.

## Result under GroupKFold — a genuine improvement (but see the reversal further down)

> Everything in this section is the **rotating 4-fold GroupKFold** view. It is real and
> reproducible. It is also **not** how the reversal below was found — that used the fixed
> shipped TEST split. Read both.

**GroupKFold mean ± std (4 folds, group = recording), at the shipped threshold 0.52:**

| Metric | BASE (23 feat) | BASE+NEW (34 feat) | Δ |
|---|---|---|---|
| Accuracy | 0.829 ± 0.067 | **0.857 ± 0.053** | **+0.027** |
| Clean recall | 0.661 ± 0.202 | **0.683 ± 0.124** | **+0.022** (std nearly halved) |
| False positive rate | 0.339 ± 0.202 | **0.317 ± 0.124** | **−0.022** |
| Attack recall | 0.906 ± 0.045 | 0.925 ± 0.036 | +0.019 |
| **Jamming recall** | **0.783 ± 0.103** | **0.812 ± 0.088** | **+0.029** |
| Meaconing recall | 0.970 ± 0.013 | 0.964 ± 0.028 | −0.005 |
| Spoofing recall | 0.953 ± 0.014 | 0.954 ± 0.021 | +0.001 |
| Spoofing+Jamming recall | 0.979 ± 0.012 | 0.982 ± 0.005 | +0.003 |

**Pooled out-of-fold, threshold-independent:**

| | BASE | BASE+NEW | Δ |
|---|---|---|---|
| ROC-AUC | 0.847 | **0.868** | **+0.021** |
| PR-AUC | 0.908 | **0.919** | **+0.011** |

The ROC-AUC gain is the load-bearing number — threshold-independent, so this is a genuine
improvement in the model's ability to rank attack vs clean, not a threshold trick. Both feature
sets' floor-constrained threshold sweep independently lands on **t = 0.50**; at that operating
point the jamming-recall gain is larger still (BASE 0.798 ± 0.108 → BASE+NEW 0.837 ± 0.083,
**+0.039**), clean recall +0.014, accuracy +0.029.

Determinism: every fold scored twice, asserted bit-identical (`n_jobs=1` at predict, same
discipline as `ROBUSTNESS_NOTES.md`); SHAP values scored twice, bit-identical.

### The headline finding: this does NOT carry the jamming-vs-clean trade-off

Every prior rejected fix bought a clean-recall / FPR gain with a jamming-recall cost
(reweighting −9pt, isotonic −10.5pt, XGBoost −9.5pt, normalization oracle −33.6pt). **This one
improves clean recall, FPR, *and* jamming recall at once**, with every other attack type flat
or up. That matches the pre-registered physical reasoning: `sp_xconst_cn0_spread` moves under
jamming too (a broadband jammer raises the C/N0 floor unevenly across constellations), while
`sp_frac_l2_tracked` is inert under jamming so it cannot cost jamming recall but is strong
under spoofing / meaconing.

## SHAP + Gini: the model genuinely uses the new signal — with a nuance

Both importance measures (exact Tree SHAP over a full-data fit, mean|abs| on 800 sampled rows;
and RandomForest Gini, per-fold mean ± std) **agree closely**. Top of the combined ranking
(of 34):

| Rank | Feature | mean\|SHAP\| | Gini rank |
|---|---|---|---|
| 1 | `agc_cnt_mean` | 0.082 | 1 |
| 2 | `snr_l1_mean` | 0.072 | 2 |
| 3 | `noise_per_ms_mean` | 0.043 | 3 |
| **4** | **`sp_doppler_l1_mad`** | **0.037** | **4** |
| **5** | **`sp_xconst_cn0_spread`** | **0.029** | **5** |
| **6** | **`sp_frac_low_cn0`** | **0.025** | **6** |
| 7 | `jam_ind_mean` | 0.022 | 7 |
| **8** | **`sp_cn0_l1_minus_l2_mean`** | **0.019** | **11** |
| 9 | `hAcc` | 0.017 | 9 |
| **10** | **`nsat_l1_roll_std_30`** | **0.016** | **15** |
| 11 | `n_sats_l1` | 0.016 | 10 |
| **12** | **`sp_frac_l2_tracked`** | **0.015** | **8** |

**6 of the top 12 are new.** The model isn't ignoring the new features for the same u-blox RF
fields — `sp_doppler_l1_mad`, `sp_xconst_cn0_spread`, and `sp_frac_low_cn0` all rank *above*
`jam_ind_mean`, and the cross-constellation / dual-frequency features displace it down the
list. **The original hypothesis — "give the model new information it doesn't currently have" —
is confirmed** for the cross-constellation + dual-frequency + robust-dispersion group.

**The nuance, reported because it's genuinely instructive:** the 5 **carrier-phase / CMC /
`sp_n_const`** features rank *low* individually — 22nd–30th of 34 by both Gini and SHAP. The
cycle-slip / code-minus-carrier idea, while physically sound and computable, is **not where
the signal is.** But `spoofing_lean_check.py` shows that dropping those 5 (keeping only the 6
that rank) is *worse*, not neutral:

| @ t=0.52 | BASE+NEW (11) | BASE+LEAN (6) |
|---|---|---|
| Jamming recall | **0.812 ± 0.088** | 0.799 ± **0.143** |
| Jamming per-fold | [0.936, 0.803, 0.774, 0.733] | [0.922, 0.911, 0.737, **0.626**] |
| Clean recall | **0.683 ± 0.124** | 0.665 ± 0.145 |
| ROC-AUC / PR-AUC | 0.868 / 0.919 | 0.867 / 0.919 |

Removing the low-ranked features widens jamming-recall variance (std 0.088 → 0.143) and drops
the worst fold by ~11pt. They contribute at the margin on hard jamming rows — consistent with
`SHAP_EXPLAINABILITY.md`'s finding that near-threshold jamming cases lean on non-RF features
that argue against the RF-monitor majority. **Low individual importance ≠ safe to prune here.
Integrate all 11.**

## The one blemish: fold-3 jamming recall regresses — fully explained

Per-fold jamming recall at each set's own best threshold:

| | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Mean |
|---|---|---|---|---|---|
| BASE | 0.842 | 0.812 | 0.644 | **0.892** | 0.798 |
| BASE+NEW | 0.947 | 0.846 | 0.801 | **0.753** | 0.837 |

Folds 0–2 improve substantially (fold 2, the weak one, **+15.7pt**); **fold 3 drops 13.9pt.**
`spoofing_fold3_dig.py` isolates the cause completely: **fold 3's only jamming recording is
`1.11.7`** — the obstructed-sky dynamic drive that `FOLD_ANALYSIS.md` and `NORMALIZATION_NOTES.md`
already identified as pathological, with an **inverted** satellite-count/attack relationship
(its attack window has *more* satellites — 45 — than its own clean segment — 9 — because the
low count during the clean segment is route obstruction, not jamming).

What the new features do to `1.11.7` (@ t=0.52):

| | attack recall | clean recall |
|---|---|---|
| BASE | 0.869 | **0.016** |
| BASE+NEW | **0.733** | **0.174** |

The new features **improve `1.11.7`'s clean recall 10x** (1.6% → 17.4%) — the model finally
recognises part of that obstructed-sky clean segment as clean — but at the cost of its
attack-segment recall (87% → 73%). This is the features **working exactly as designed**,
misfiring on the one recording in the dataset where "degraded reception" correlates with
*clean*, not attack. Every other fold-3 recording gains clean recall with **zero** attack-recall
cost (`2.6.3` 0.75 → 0.90, `3.1.3,3.1.4` 0.38 → 0.55, `2.3.10,2.3.11` 0.70 → 0.75). Net across
all 4 folds, jamming recall is **+2.9pt at the shipped threshold, +3.9pt at the best** — and
its fold-to-fold std *tightens* (0.103 → 0.088).

This is the same one-off `FOLD_ANALYSIS.md` flagged, now surfacing on jamming recall instead of
clean recall, and it's a known, checkable mechanism — not unexplained variance.

## Under GroupKFold this looked like a clear win — then integration broke on the fixed split

Under the 4-fold rotating GroupKFold harness (all 24 recordings), this cleared the bar on
every metric: accuracy +2.7pt, clean recall +2.2pt, FPR −2.2pt, **ROC-AUC +2.1pt
(threshold-independent)**, and **jamming recall +2.9pt** with tighter cross-fold variance.
SHAP + Gini both confirmed the model uses the new signal. On that evidence, integration was
approved.

**It did not survive contact with the shipped evaluation.** `baseline_model_report.md`,
`improved_model_report.md`, `evaluate_models.py`, and the dashboard all report a **fixed
8-recording held-out TEST split** (one dynamic + one stationary recording per attack type),
*not* rotating CV. Retraining the full 34-feature model and evaluating on that exact split:

| Fixed TEST split (improved model) | 23-feat (shipped) | 34-feat | Δ |
|---|---|---|---|
| Accuracy | 0.877 | 0.835 | **−4.2pt** |
| Clean recall | 0.667 | 0.581 | **−8.6pt** |
| **Jamming recall** | **0.859** | **0.767** | **−9.2pt** |
| ROC-AUC | 0.916 | 0.898 | **−1.8pt** |
| Meaconing / Spoofing / Spoof+Jam recall | 0.965 / 0.971 | 0.967 / 0.968 | ~flat |

Per-recording (`diag_split.py`), the damage concentrates on the two **dynamic** recordings in
the split:

| recording | 23-feat attack / clean recall | 34-feat attack / clean recall |
|---|---|---|
| `1.10.6` (Jamming, dynamic) | 0.93 / 0.14 | **0.76** / 0.38 |
| `3.2.8` (Meaconing, dynamic) | 0.99 / **0.79** | 0.99 / **0.05** |

`1.10.6` is the *same failure mode* documented above for `1.11.7`: the new C/N0-dispersion
features push a degraded-reception dynamic-jamming recording toward "clean" — helping its
(near-zero) clean recall, gutting its attack recall. `1.10.6` and `1.11.7` are the same
recording family (High Power dynamic jamming, `E1_E5_E5a…` bands). `3.2.8` is a **new**
regression: the new features make its clean segment read as an attack (clean recall 0.79 →
0.05).

**Why the two evaluations disagree:** rotating GroupKFold puts these pathological
dynamic recordings in the *validation* fold only ~1 fold in 4, and averages their cost against
the larger gains elsewhere — net positive. The fixed shipped split contains **both** of the
worst-affected recordings (and only 2 jamming recordings total, so `1.10.6`'s −16pt attack
recall alone moves aggregate jamming recall −9pt). This is exactly the risk
`ROBUSTNESS_NOTES.md` names: with ~5 jamming recordings in the training pool, 11 extra features
give the model enough capacity to fit training-jamming signatures that don't transfer — and
the dynamic degraded-reception recordings have the inverted "clean segment looks worse than
the attack window" property the new features specifically get wrong.

## Decision: DO NOT integrate as-is

**The shipped model, reports, parquet, and API are unchanged.** The 34-feature build was
retrained, evaluated, and reverted; it is staged (not applied) for reference.

The honest bar was "a genuine GroupKFold-validated improvement with no meaningful jamming
regression, **especially on the shipped evaluation**." It clears the first half and fails the
second: −9.2pt jamming recall and −8.6pt clean recall on the fixed TEST split a judge sees is
a meaningful regression, not fold noise — it is a systematic weakness on dynamic
degraded-reception recordings, reproducible and mechanistically explained.

This lands in the same place as jamming reweighting (`ROBUSTNESS_NOTES.md`), isotonic
calibration (`CALIBRATION_NOTES.md`), session normalization (`NORMALIZATION_NOTES.md`), and
XGBoost (`GBM_COMPARISON.md`): a real signal that does not survive a held-out evaluation
cleanly. The GroupKFold gain is genuine and the SHAP/Gini evidence that the model *uses*
carrier-phase/L2/per-constellation structure is genuine — but neither is sufficient to ship.

### If revisited — concrete leads, each needing its own GroupKFold **and** fixed-split validation

1. **Regularize the RandomForest** (`max_depth`, higher `min_samples_leaf`) to curb the
   overfitting the extra features enable. This changes the shipped model config beyond
   features, so it needs full re-validation on both evaluations — not attempted here.
2. **Exclude the C/N0-dispersion features on dynamic recordings**, or add `rover_state` as a
   feature so the model can learn the dynamic/stationary distinction (currently excluded as
   metadata a deployed detector wouldn't have — but a deployed base-station receiver *is*
   stationary, so a stationary-only variant is defensible).
3. **More clean-but-degraded-reception training examples** — the gap `FOLD_ANALYSIS.md`
   already identified; the new features sharpen the need rather than fixing it.
4. **Reconsider whether the fixed 8-recording split is still the right shipped evaluation.**
   `CALIBRATION_NOTES.md` already argued the GroupKFold mean±std is more trustworthy. Moving
   the shipped headline to GroupKFold would change this verdict — but that is a project
   methodology decision, not something to fold into a feature change.

## Where this lives

- `extract_spoofing_features.py`, `spoofing_features_experiment.py`, `spoofing_fold3_dig.py`,
  `spoofing_lean_check.py` — the GroupKFold experiment (unchanged, still valid).
- `spoofing_fixed_split_diag.py` — the fixed-split per-recording diagnostic behind the
  reversal. Needs a 23-feature and a 34-feature parquet (`processed/syncguard_features_23feat.parquet`
  / `_34feat.parquet`, both gitignored — regenerate the 34-feat one by re-running
  `extract_features.py` with the staged `sp_*` additions).
- The staged-but-not-applied 34-feature build (`extract_features.py` + `api/schemas.py` diffs,
  regenerated models/reports) is kept in the session scratchpad, not in the repo.
- **Unchanged in the repo**: `models/model.joblib`, `models/model_baseline.joblib`,
  `extract_features.py`, `processed/syncguard_features.parquet`, `api/`,
  `train_baseline_model.py`, `train_improved_model.py`, `baseline_model_report.md`,
  `improved_model_report.md`, and every deployed threshold. `evaluate_models.py` confirms the
  restored shipped model matches `ROBUSTNESS_NOTES.md` exactly (jamming 0.872 → 0.859, clean
  0.642 → 0.667).

## Where this lives

- `extract_spoofing_features.py`, `spoofing_features_experiment.py`, `spoofing_fold3_dig.py`,
  `spoofing_lean_check.py` — implementations.
- `processed/spoofing_features_supplement.parquet` — supplementary features (gitignored like
  the rest of `processed/` except the committed base parquet — regenerate by re-running the
  extractor; needs the raw scenario tree, located via `$SYNCGUARD_RAW_ROOT` / `raw/` /
  `../agaif-materials/dataset/raw`).
- `processed/spoofing_comparison_shipped_tbl.csv`, `spoofing_comparison_best_tbl.csv`,
  `spoofing_shap_ranking.csv`, `spoofing_gini_ranking.csv` — the tables above.
- Does **not** modify `models/model.joblib`, `models/model_baseline.joblib`,
  `extract_features.py`, `processed/syncguard_features.parquet`, `api/`,
  `train_baseline_model.py`, `train_improved_model.py`, or any deployed threshold.
