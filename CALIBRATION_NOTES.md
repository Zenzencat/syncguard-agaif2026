# SyncGuard: GroupKFold cross-validation and probability calibration

Follow-up to [`ROBUSTNESS_NOTES.md`](ROBUSTNESS_NOTES.md), same honest-tradeoffs discipline.
Two things were investigated: (1) whether the shipped model's numbers hold up under proper
cross-validation instead of a single 8-recording test split, and (2) whether probability
calibration (`CalibratedClassifierCV`, sigmoid and isotonic) is a genuine improvement.
**Neither model.joblib, model_baseline.joblib, api/, nor the deployed threshold (0.52) have
been touched.** This is a standalone analysis — [`calibration_experiment.py`](calibration_experiment.py)
produces every number below and can be re-run with `python calibration_experiment.py`
(~40 seconds).

## 1. GroupKFold cross-validation: a more trustworthy — and more sobering — estimate

**Setup**: 4-fold, group-preserving K-fold (group = recording, so no recording ever appears
in both train and validation within a fold — the same leakage discipline as the original
split-by-recording) across **all 24 recordings**, not just the 16 the shipped model trained
on. Plain `sklearn.GroupKFold` doesn't consider labels at all; with as few as 4 recordings
for the rarest attack type (Spoofing+Jamming), an unlucky group-to-fold assignment could
leave a fold with zero recordings of that type, making its per-fold recall for that class
meaningless. `attack_type_balanced_folds()` fixes this by round-robining each attack type's
recordings across folds independently — every fold gets a share of every type (verified in
the run log: e.g. fold 3 still gets 1 Spoofing+Jamming recording despite only 4 existing
total). `n_splits=4` is the largest split count where that's guaranteed for the rarest type.

For each fold, the RandomForest is refit from scratch with the **same configuration as the
shipped model** (300 trees, `class_weight='balanced'`, `random_state=42`, median imputation)
on that fold's training recordings, then scored on its held-out recordings. Concatenating all
4 folds' held-out predictions gives one out-of-fold (OOF) prediction per row across all 44639
rows — a genuinely out-of-sample estimate for the *entire* dataset, not just 8 recordings.

**Result — the headline finding, before calibration even enters the picture:**

| | Fold 0 | Fold 1 | Fold 2 | Fold 3 | **Mean ± std** | Pooled (all rows) | evaluate_models.py (single split) |
|---|---|---|---|---|---|---|---|
| Accuracy | 0.893 | 0.860 | 0.852 | 0.734 | 0.835 ± 0.070 | 0.844 | 0.877 |
| Clean recall | 0.787 | 0.699 | 0.794 | **0.360** | **0.660 ± 0.205** | 0.641 | 0.667 |
| FPR | 0.213 | 0.301 | 0.206 | **0.640** | 0.340 ± 0.205 | 0.359 | 0.333 |
| Jamming recall | 0.845 | 0.821 | 0.660 | 0.865 | 0.797 ± 0.094 | 0.793 | 0.859 |
| Meaconing recall | 0.965 | 0.983 | 0.951 | 0.980 | 0.970 ± 0.015 | 0.969 | 0.965 |
| Spoofing recall | 0.945 | 0.942 | 0.956 | 0.973 | 0.954 ± 0.014 | 0.953 | 0.971 |
| Spoofing+Jamming recall | 0.985 | 0.984 | 0.962 | 0.985 | 0.979 ± 0.011 | 0.979 | 0.985 |

(all at the shipped threshold, 0.52; full per-fold table in the script's stdout)

**Two things stand out:**

1. **The pooled CV estimate is close to, but consistently a bit below, the single-split
   number** (accuracy −3.3pt, clean recall −2.6pt, jamming recall −6.5pt) — the single
   8-recording TEST split was mildly optimistic, not wildly biased. This is the "sanity
   check" requested, and it broadly holds up: no red flag here.
2. **The variance across folds is large enough to matter for the pitch.** Clean recall swings
   from 0.360 (fold 3) to 0.794 (fold 2) depending on which recordings happen to be held out
   — a std of 0.205 on a mean of 0.660. Fold 3's collapse (clean recall 0.36, FPR 0.64) means
   more than half of true-clean readings in that fold's held-out recordings were wrongly
   flagged as an attack. This is consistent with — and sharpens — the domain-shift limitation
   `baseline_model_report.md` already named honestly ("each recording's own quiet RF noise
   floor varies somewhat session-to-session"): the effect is large enough that a genuinely new
   recording could plausibly land far worse than the single TEST split suggests. **This
   cross-validated range, not the single-split point estimate, is the more honest number to
   have ready if a judge asks "how confident are you in that recall figure."**

Jamming recall also has real fold-to-fold variance (std 0.094, low of 0.660 in fold 2) — this
matters for what follows, since it means jamming recall is inherently noisier to estimate
than the other three attack types (Meaconing/Spoofing/Spoofing+Jamming all have std ≤ 0.015),
consistent with jamming already being the weakest, most heterogeneous attack type (power
100mW–≥10W, 1–6 frequency bands) in `baseline_model_report.md`.

## 2. Probability calibration

**Setup**: `CalibratedClassifierCV` (`method='sigmoid'` and `method='isotonic'`), fit within
each of the 4 outer folds via a further 3-fold group-preserving *inner* split strictly within
that fold's training recordings — each inner base-RF clone trains on 2 inner folds and is
calibrated on the 3rd, so calibration is always fit on rows disjoint from what that specific
clone trained on, and the outer fold's validation recordings are never touched by any of this
(the standard `CalibratedClassifierCV(cv=<grouped splits>)` pattern, confirmed via the run
log's per-inner-fold attack-type counts — every inner fold still has rows of every type).
Determinism verified explicitly: every fold's predictions were computed twice and diffed
bit-identical (`n_jobs=1` forced on every fitted RF clone, base and calibrated alike, before
any `predict_proba` call) — see the script's "DETERMINISM CHECK" section, all three variants
pass.

### Reliability (Brier score, log-loss, calibration curve)

| | Uncalibrated | Sigmoid | Isotonic |
|---|---|---|---|
| Brier score | **0.1102** | 0.1143 | 0.1134 |
| Log-loss | 0.5178 | **0.3835** | 0.4049 |

Mixed signal by design of the two metrics: log-loss improves markedly with calibration
(penalizes confident-wrong predictions much more heavily than Brier does, and the raw RF
probabilities pile up at the extremes — a known RandomForest characteristic, since
`predict_proba` is literally a vote fraction across 300 trees), while Brier score is actually
very slightly *worse* for both calibrated variants. The reliability diagram
(`processed/calibration_reliability.png`) shows why neither improvement is dramatic: **the raw
uncalibrated model is already reasonably well-calibrated** across most of the probability
range — all three curves track the diagonal closely from 0.2 to 0.9. The one shared weak spot
is the top bin (predicted probability ≈0.95–1.0), where all three variants over-predict
slightly (observed attack fraction 0.80–0.97, not ≈1.0) — calibration doesn't clearly fix this
either; if anything isotonic is slightly worse there than uncalibrated.

### Does a better threshold become available?

| Variant @ threshold | Accuracy | Clean recall | FPR | Jamming recall |
|---|---|---|---|---|
| **Uncalibrated @ shipped 0.52** (current) | 0.844 | 0.641 | 0.359 | 0.793 |
| Uncalibrated @ 0.50 (untuned) | 0.844 | 0.619 | 0.381 | 0.811 |
| Sigmoid @ 0.50 (best available — sweep found nothing better) | 0.840 | 0.588 | 0.412 | 0.818 |
| Isotonic @ 0.50 (untuned) | 0.834 | 0.708 | 0.292 | **0.706** |
| Isotonic @ 0.52 (best under the floor constraint) | 0.836 | 0.730 | 0.270 | **0.700** |

(all on the pooled out-of-fold predictions; "best" uses the same jamming-recall-floor-constrained
sweep as `train_improved_model.py`, floor = each variant's own threshold-0.5 jamming recall
minus 0.01)

**Neither calibration method clears the bar.** Sigmoid is flat-out worse across the board at
every threshold the sweep considered (lower accuracy, lower clean recall, higher FPR, and the
sweep couldn't find anything better than its own untuned 0.50 without breaching the jamming
floor). Isotonic is the more interesting case — clean recall *does* improve substantially
(+8.9pt vs. uncalibrated at 0.52) — but **jamming recall drops to 0.70 even at isotonic's own
untuned threshold of 0.50**, a 10.5-point regression from uncalibrated's 0.811 at the same
threshold, before any threshold tuning is even applied. That's the same shape of result as the
jamming sample-reweighting attempt in `ROBUSTNESS_NOTES.md`: a real, measurable gain on one
metric bought by trading away real ground on the metric this whole exercise exists to protect.
Isotonic regression's flexibility (an unconstrained monotonic step function, unlike sigmoid's
smooth logistic curve) is the likely mechanism — it has more freedom to overfit each inner
fold's ~8,000–13,000-row calibration set, and jamming, with the highest fold-to-fold variance
of any attack type (§1), is exactly the class most exposed to that.

## 3. Decision: do not adopt calibration

**Keep the currently-shipped model (`models/model.joblib`, threshold 0.52) unchanged.**
Neither sigmoid nor isotonic calibration produces a genuine improvement under the GroupKFold
estimate:

- Sigmoid: strictly worse on every operational metric, no better threshold available.
- Isotonic: a real clean-recall gain, but at a real, unacceptable jamming-recall cost —
  exactly the trade `ROBUSTNESS_NOTES.md` already established this project won't make.
- The reliability improvement (log-loss) doesn't translate into better detection performance,
  and the raw model wasn't badly miscalibrated to begin with, so there wasn't a large
  calibration problem to fix.

This is reported plainly rather than adopted, for the same reason the earlier reweighting
attempt was — a fix has to survive contact with a real held-out evaluation, not just look good
on one metric or one split.

**The one recommendation that *is* worth acting on**: use the GroupKFold mean±std numbers in
§1, not the single-split numbers, when characterizing confidence in the pitch — particularly
for clean recall, where the true range (0.36–0.79 across folds) is much wider than the single
point estimate (0.64–0.67) suggests. This doesn't require touching the model at all.

## Where this lives

- `calibration_experiment.py` — full implementation; `attack_type_balanced_folds()`,
  `fold_metrics()`, the outer/inner CV loop, reliability metrics, and the threshold sweep.
- `processed/calibration_reliability.png` — the reliability diagram referenced above (gitignored,
  like the rest of `processed/` except the committed parquet — regenerate by re-running the
  script).
- `processed/calibration_comparison_table.csv` — the full comparison table (same directory,
  same gitignore note).
- Does not modify `models/model.joblib`, `models/model_baseline.joblib`, `api/`,
  `train_baseline_model.py`, `train_improved_model.py`, or any deployed threshold.
