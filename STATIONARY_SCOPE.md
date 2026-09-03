# SyncGuard: scoping the spoofing-feature question to the stationary deployment scenario

Follow-up to [`SPOOFING_FEATURES.md`](SPOOFING_FEATURES.md). Same honest-tradeoffs discipline
as `ROBUSTNESS_NOTES.md` / `CALIBRATION_NOTES.md` / `NORMALIZATION_NOTES.md` /
`GBM_COMPARISON.md`.

`SPOOFING_FEATURES.md` ended by rejecting the 11 carrier-phase / L2 / per-constellation
features: they cleared the rotating-GroupKFold bar but regressed the fixed shipped TEST split
(jamming recall −9.1pt, clean recall −8.6pt), and the regression was **concentrated entirely
on the two dynamic (moving-vehicle) recordings** in that split — `1.10.6` (Jamming, dynamic)
and `3.2.8` (Meaconing, dynamic).

That raised a fair question: **SyncGuard's deployment target is a stationary telecom
base-station GNSS timing receiver, which does not move.** If the recordings where the new
features misfire are all dynamic, they may not represent the deployment scenario at all. The
right move is to re-scope the problem to stationary-only data and honestly re-test — not to
"restrict to stationary to rescue the features," but to ask whether they help *within the
correctly-scoped problem*, with the same two-evaluation discipline (GroupKFold **and** a fixed
held-out split) that caught the artifact last time.

This document records that scoping check. **It did not reach a modeling result** — a
mandatory dataset-size gate failed first. That gate failing is itself the finding.

## 1. Restricting to stationary is a legitimate deployment-realism scoping point — regardless of the feature question

This is worth stating in the project's framing independently of whether the spoofing features
help:

- SyncGuard is pitched as *"a lightweight, per-site anomaly-detection layer that monitors a
  base station's GNSS receiver observables"* (README) — i.e. a **GNSS-disciplined oscillator
  (GNSSDO) at a fixed cell site**. A base-station timing receiver is bolted to a tower or a
  equipment shelter. It does not move.
- The Jammertest 2024 dataset (`dataset_notes.md`) is a mix of **stationary and
  vehicle-mounted (dynamic)** u-blox receivers at a public interference test range. The
  `dataset_notes.md` "Framing gap" section already flags that this is receiver-level GNSS data,
  not base-station GNSSDO timing data — the stationary/dynamic split is a second axis of the
  same gap.
- The features that caused the `SPOOFING_FEATURES.md` regression — cross-constellation C/N0
  spread, low-C/N0 fraction, robust Doppler dispersion — misfire specifically when a **moving**
  receiver drives through varying sky visibility (obstruction, multipath, changing geometry),
  which the model reads as attack-like. A fixed base-station receiver with a clear sky view and
  a stable multipath environment doesn't have that failure mode.
- `FOLD_ANALYSIS.md` already identified the same thing from the other direction: its worst-case
  clean-recall failure traced to `1.11.7`, *"almost certainly a route obstruction rather than
  anything attack-related,"* on a `rover_state='dynamic'` recording.

So: **evaluating and (if it ever shipped) deploying SyncGuard against stationary-receiver data
is the more deployment-faithful choice**, and that is true whether or not the spoofing features
survive. This is a framing point worth carrying into `project_abstract.md` regardless of this
document's outcome.

## 2. But the dataset has only one stationary jamming recording — the central question is unanswerable

`rover_state` (parquet column, set in `extract_features.py` from `scenario.json` →
`attack_parameters["Rover states"]`; the raw directory tree — `Jamming/stationary/…` vs
`Jamming/dynamic/…` — agrees) splits the 24 recordings into **11 stationary, 13 dynamic**.

### Stationary-only recordings and rows, per attack type

| Attack type | Recordings | Rows | Attack rows | Clean rows | (full pool: recordings) |
|---|---|---|---|---|---|
| **Jamming** | **1** | 1,325 | 840 | 485 | (7) |
| Meaconing | 3 | 4,223 | 3,103 | 1,120 | (5) |
| Spoofing | 5 | 11,506 | 9,060 | 2,446 | (8) |
| Spoofing + Jamming | 2 | 3,303 | 2,462 | 841 | (4) |
| **Total** | **11** | 20,357 | 15,465 | 4,892 | (24) |

The single stationary jamming recording is `1.6.4` (Very High Power ≥10W, bands L1/L2/L5).

### Why this fails the evaluation gate

The whole point of the `SPOOFING_FEATURES.md` investigation — and its rejection — was
**jamming recall**: does adding the features cost it? Answering that requires measuring
stationary jamming recall on **held-out** data. With the stationary data available:

- **GroupKFold (group = recording) is undefined for jamming.** One group cannot be split
  across a fold's train and validation halves — `1.6.4` is either wholly in train (jamming
  recall unmeasurable that fold) or wholly in validation (no jamming in training that fold).
  There is no configuration that estimates held-out stationary jamming recall.
- **A fixed held-out split analogous to the shipped one is impossible.** The shipped split
  holds out one recording per attack type. Holding out `1.6.4` leaves **zero** jamming
  recordings in training; not holding it out means there is no held-out jamming to score.
- **No power/band diversity to generalize over.** `ROBUSTNESS_NOTES.md` established that
  jamming recall is unstable precisely because jamming recordings span 100 mW–≥10 W and 1–6
  bands. `1.6.4` is one point in that space (≥10 W, 3 bands). Even if it could be held out, one
  recording says nothing about how the features behave on a *different* jamming configuration —
  which is exactly the generalization failure that sank the full-dataset attempt.
- **Spoofing+Jamming (2 recordings, both Low Power 100 mW / L1)** degenerates to a single
  leave-one-out split: one number, no variance, no way to separate a recording-specific effect
  from a feature effect.
- **Meaconing (3) and Spoofing (5) are thinner than the raw counts suggest**: 3 of the 5
  Spoofing recordings are near-siblings (Medium Power, `E1_E5_L1_L2_L5`); 2 of the 3 Meaconing
  are the same Very High Power family. Effective independent-configuration diversity is lower
  still.

This was a **pre-registered gate**, set before any stationary modeling: *"if stationary-only
leaves too few recordings per attack type to evaluate meaningfully — especially jamming, which
already only had ~5 recordings in the full pool and 2 in the fixed split — then a
stationary-only model can't be validated reliably, and THAT is the answer."* One jamming
recording, none holdable-out, is past that line by a wide margin. Generating BASE-vs-BASE+NEW
numbers on this data anyway is exactly what the gate exists to prevent: they would look like
an answer without being one.

## 3. Conclusion: the data can't tell us — this is a dataset limitation, not a result

**Restricting to stationary is the right deployment-realistic scope. The Jammertest 2024
dataset does not contain enough stationary jamming data to evaluate the spoofing features
within that scope to this project's standard.** That is a limitation of the available data,
not a finding that the features fail when correctly scoped — the correctly-scoped experiment
cannot be run.

Combined with `SPOOFING_FEATURES.md`:

| Evaluation | Verdict |
|---|---|
| Full dataset, rotating GroupKFold | Features look like a clear improvement (jamming +2.9pt, ROC-AUC +2.1pt), model demonstrably uses them (SHAP + Gini) |
| Full dataset, fixed shipped TEST split | Features regress hard (jamming −9.1pt, clean −8.6pt), concentrated on dynamic recordings |
| Stationary-only (deployment scope), either evaluation | **Cannot be run** — 1 jamming recording, none can be held out |

**No evaluation available to this project supports integrating the spoofing features.** The
one that looked favorable (full-dataset GroupKFold) is the one whose fold-distribution artifact
`SPOOFING_FEATURES.md` already showed made the result look better than it was. The shipped
23-feature model stays.

If a future dataset with multiple stationary jamming recordings (different powers and bands)
becomes available, the stationary-scoped experiment in the user's Step 1–4 plan is the right
thing to run — GroupKFold **and** a fixed stationary split, both required, jamming recall the
metric that must not regress on either.

## Where this lives

- No new script — this is a dataset-composition check. The counts above reproduce with a
  one-liner group-by on `processed/syncguard_features.parquet`'s `rover_state` / `attack_type`
  / `attack` columns (the shipped 23-feature parquet; the same counts hold for the staged
  34-feature parquet, which has identical rows).
- `SPOOFING_FEATURES.md` — the full-dataset attempt and its reversal.
- Does **not** modify `models/model.joblib`, `models/model_baseline.joblib`,
  `extract_features.py`, `processed/syncguard_features.parquet`, `api/`,
  `train_baseline_model.py`, `train_improved_model.py`, or any deployed threshold. No model was
  trained for this document.
