# SyncGuard final model comparison: RandomForest (shipped) vs. XGBoost

Same honest-tradeoffs discipline as `ROBUSTNESS_NOTES.md`, `CALIBRATION_NOTES.md`, and
`NORMALIZATION_NOTES.md`, and the same bar: adopt the challenger only if it's a genuine,
fold-validated improvement with no meaningful regression. It isn't, for a reason directly
consistent with every prior rejection this session — but the comparison surfaces two findings
worth keeping regardless: a real interpretability-cost advantage worth documenting, and a
concrete lead for future work. [`train_gbm_comparison.py`](train_gbm_comparison.py) produces
every number below. **Does not modify `models/model.joblib`, the SHAP explainer, or any
deployed threshold.**

## Why XGBoost, not LightGBM (or both)

XGBoost was chosen as the sole challenger: the most standard, most widely-cited
gradient-boosting library for exactly this kind of comparison, mature `shap.TreeExplainer`
support (directly relevant — see §3), and, confirmed below rather than assumed, a materially
simpler determinism story than RandomForest's. LightGBM is a reasonable secondary comparison
not run here given session scope — a natural next step, not a gap papered over.

## Methodology — identical evaluation harness to the rest of this project

Same 4-fold, group-preserving, attack-type-balanced GroupKFold as `calibration_experiment.py`
and `session_normalization_experiment.py` (same `attack_type_balanced_folds()` logic, same
`seed=42`, same fold membership — no new split introduced). Both models are trained fresh
per fold and evaluated the same way: out-of-fold pooled predictions for a single threshold
sweep (identical floor-constrained methodology to `train_improved_model.py` — best threshold
maximizes clean recall subject to not dropping jamming recall by more than 0.01 below that
model's own threshold-0.5 baseline), then **each model re-evaluated per-fold at its own
chosen threshold** to report GroupKFold mean ± std, not a single point estimate — the same
standard `CALIBRATION_NOTES.md` established for trusting a single split's numbers.

**RandomForest**: reproduces the shipped configuration exactly — 300 trees,
`class_weight='balanced'`, `random_state=42`, `n_jobs=1` at predict time (the determinism
fix from `ROBUSTNESS_NOTES.md`).

**XGBoost**: 300 trees, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`,
`colsample_bytree=0.8`, `tree_method='hist'` — reasonable, commonly-used defaults, not
hand-tuned in either direction (no hyperparameter search run for either model, so neither is
artificially favored). **Class imbalance**: `class_weight='balanced'`'s right XGBoost
equivalent is `scale_pos_weight = count(negative) / count(positive)`, computed fresh per
fold from that fold's own training data only. Note "attack" (label 1) is the *majority*
class here (~77%) — the standard formula naturally produces `scale_pos_weight < 1`
(down-weighting the majority), the correct behavior with no special-casing needed.

## Determinism — verified directly, and a genuinely simpler story than RandomForest's

Same rigor as `ROBUSTNESS_NOTES.md`'s RF investigation, checked explicitly rather than
assumed transferable:

- Two independent XGBoost fits with the same `random_state`, `n_jobs=-1` fit and predict:
  **bit-identical `predict_proba` output** (max abs diff = 0.0).
- Repeated `predict_proba()` calls on the same fitted model: bit-identical.
- **`.predict()` matches `(predict_proba >= 0.5)` exactly** on 4000 held-out rows — 0
  mismatches. RandomForest needed `n_jobs` forced to 1 at predict time specifically because
  this didn't hold otherwise (~6 of 14077 rows flipped — see `ROBUSTNESS_NOTES.md`). XGBoost
  doesn't have this failure mode: `n_jobs=-1` is used throughout for both fit and predict here
  with no workaround needed, confirmed, not assumed.

## 1. Head-to-head (GroupKFold mean ± std, each model at its own best threshold)

| Metric | RandomForest (t=0.50) | XGBoost (t=0.54) | Δ (XGB − RF) |
|---|---|---|---|
| Accuracy | 0.835 ± 0.073 | 0.819 ± 0.063 | −0.016 |
| Clean recall | 0.643 ± 0.204 | **0.674 ± 0.190** | **+0.031** |
| False Positive Rate | 0.357 ± 0.204 | **0.326 ± 0.190** | **−0.031** |
| Attack recall | 0.918 ± 0.038 | 0.886 ± 0.052 | −0.031 |
| **Jamming recall** | **0.812 ± 0.091** | **0.717 ± 0.079** | **−0.095** |
| Meaconing recall | 0.971 ± 0.013 | 0.968 ± 0.013 | −0.003 |
| Spoofing recall | 0.956 ± 0.013 | 0.946 ± 0.024 | −0.011 |
| Spoofing+Jamming recall | 0.979 ± 0.011 | 0.980 ± 0.015 | +0.001 |
| Pooled ROC-AUC | 0.850 | 0.847 | −0.003 |
| Pooled PR-AUC | 0.910 | **0.930** | **+0.020** |

Fit time per fold: RandomForest ~0.8-1.0s, XGBoost ~0.4s (XGBoost trains roughly 2x faster
here too — a minor point next to the SHAP finding below, but a real one).

**The pattern is consistent with every prior rejection this session, not a coincidence.**
XGBoost delivers a genuine clean-recall/FPR improvement (+3.1pt) — the same shape of result
`train_improved_model.py`'s threshold tuning achieved for RF. But it buys that gain with a
**9.5-point jamming recall cost**, the largest single-metric regression of any experiment run
against this GroupKFold harness this session except the rejected normalization oracle (§3 of
`NORMALIZATION_NOTES.md`, −33.6pt). Every other attack type is roughly comparable (within
GroupKFold noise). Pooled PR-AUC modestly favors XGBoost (+2pt) — a genuine, threshold-independent
signal that its raw ranking ability may be slightly better — but the threshold each model
actually operates at (chosen by the identical methodology for both) is what determines the
jamming-recall cost, and that cost is real.

## 2. SHAP explainability — compatible, but not a drop-in identical integration

`shap.TreeExplainer` supports XGBoost natively, confirmed directly (not assumed from docs).
Two things a naive port from the RF integration would get wrong, found by checking rather
than assuming API parity:

- **Output shape differs.** RandomForest's `TreeExplainer.shap_values()` returns a 3D array
  `(n_samples, n_features, n_classes)` (see `SHAP_EXPLAINABILITY.md`). XGBoost's binary
  classifier returns a **2D array** `(n_samples, n_features)` — a single set of values for
  the positive class, not per-class. `ModelService.explain()`'s `sv[0, :, 1]` indexing would
  silently break (or misinterpret) if pointed at an XGBoost model without adjustment.
- **Values are in log-odds (margin) space, not probability space.** Verified via the
  additivity property: `sum(shap_values) + expected_value` does **not** equal
  `predict_proba` directly for XGBoost (confirmed: −6.45 vs. an actual probability of
  0.0016) — it equals the pre-sigmoid margin. `sigmoid(sum(shap_values) + expected_value)`
  does match `predict_proba` (confirmed to 1e-4). RandomForest's output is directly in
  probability space with no such transform needed. Silently skipping this would produce
  SHAP "contributions" that rank features in roughly the same order but are on the wrong
  numeric scale, and the additivity sanity-check this project already relies on
  (`SHAP_EXPLAINABILITY.md`) would silently fail if not accounted for.

**The genuinely good news, and worth keeping independent of the accuracy verdict**: XGBoost's
SHAP cost is **dramatically lower**. Directly comparable to `SHAP_EXPLAINABILITY.md`'s
methodology (single-row, repeated calls):

| | RandomForest (shipped) | XGBoost |
|---|---|---|
| SHAP cost, single-row p50 | ~56ms | **~1.0ms** |
| SHAP cost, batched (300 rows) | — | 0.04ms/row |

**A ~50x speedup**, not a minor difference. Mechanism: exact Tree SHAP's cost scales steeply
with tree depth, and this comparison's XGBoost trees are capped at `max_depth=6` while the
shipped RandomForest's trees are unconstrained (sklearn's default `max_depth=None`, so
considerably deeper). Determinism holds for XGBoost's SHAP values too (bit-identical across
repeated calls, same check as `SHAP_EXPLAINABILITY.md` performed for RF). If XGBoost were
ever adopted, essentially every constraint in `OPERATIONAL_METRICS.md` driven by SHAP's
~50ms/row cost — the `LIVE_EXPLAIN_MAX_SPEED` speed gate, the on-demand `/explain` fallback,
a meaningful share of the measured replay throughput ceiling — would either disappear or
shrink to near-irrelevance.

## 3. Decision: keep the shipped RandomForest

**Do not adopt XGBoost.** Same bar as every other experiment this session: a fix has to
survive contact with the real GroupKFold estimate with no meaningful regression on the metric
this project has consistently protected. A 9.5-point jamming-recall cost for a 3.1-point
clean-recall gain is the same shape of trade already rejected for sample reweighting
(`ROBUSTNESS_NOTES.md`), isotonic calibration (`CALIBRATION_NOTES.md`), and session-relative
normalization's oracle variant (`NORMALIZATION_NOTES.md`) — rejecting it here on the same
grounds is consistency, not a missed opportunity.

**This is not a simple "RF wins, full stop" result, and shouldn't be reported that way.**
Two concrete, evidenced points in XGBoost's favor, worth keeping as documented direction
rather than discarded with the accuracy verdict:

1. **The SHAP speed finding is real and load-bearing for a future iteration.** If a
   jamming-recall-preserving XGBoost configuration were found (see below), adopting it would
   also retire most of `OPERATIONAL_METRICS.md`'s SHAP-driven throughput constraints as a
   side effect — a meaningfully different operational profile, not just an accuracy
   trade-off.
2. **The jamming-recall gap may be a threshold/weighting artifact, not an inherent XGBoost
   weakness** — this comparison used one shared, symmetric methodology (`scale_pos_weight`
   for overall balance, a single floor-constrained threshold sweep) for a fair comparison,
   the same way `train_improved_model.py` initially tried and rejected a single global
   threshold shift for RF before finding a jamming-recall-preserving one via tighter,
   CV-based constraints. The same kind of tightening — e.g. per-attack-type sample weighting
   *for XGBoost specifically* (`ROBUSTNESS_NOTES.md` found this didn't help RF, but XGBoost's
   gradient-based weighting mechanism is different enough from RF's bagging that the same
   conclusion doesn't automatically transfer), or a tighter jamming-floor tolerance in the
   threshold sweep — is a legitimate next step, not attempted here to keep this comparison a
   fair, single-methodology, apples-to-apples benchmark rather than a hand-tuned contest.

## Where this lives

- `train_gbm_comparison.py` — full implementation (GroupKFold harness, both models,
  threshold sweep, per-fold reporting).
- `processed/gbm_comparison_mean.csv`, `processed/gbm_comparison_std.csv` — the head-to-head
  table above (gitignored, like the rest of `processed/` except the committed parquet —
  regenerate by re-running the script).
- Does not modify `models/model.joblib`, `models/model_baseline.joblib`, `api/`,
  `train_baseline_model.py`, `train_improved_model.py`, or any deployed threshold.
