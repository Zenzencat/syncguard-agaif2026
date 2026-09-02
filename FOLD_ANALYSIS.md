# SyncGuard: why GroupKFold fold 3 has the worst clean recall

Follow-up to `CALIBRATION_NOTES.md` §1, which found clean recall swinging from 0.36 to 0.79
across the 4 GroupKFold folds. This digs into the worst one. **Diagnostic only** —
[`fold3_investigation.py`](fold3_investigation.py) is read-only analysis; it does not modify
`models/model.joblib`, `api/`, or any deployed threshold. Re-run with
`python fold3_investigation.py` (~15s).

## Confirming which fold, and what's in it

Fold 3 is confirmed as the worst: refitting the exact same model config on fold 3's training
recordings and scoring its held-out recordings reproduces clean recall = 0.3597, matching the
0.360 reported in `CALIBRATION_NOTES.md` exactly. Its 5 held-out recordings:

| Recording | Attack type | Rover state | Clean rows | Misclassified | Rate |
|---|---|---|---|---|---|
| **1.11.7** | Jamming | **dynamic** | 1119 | 1105 | **98.7%** |
| 3.1.3,3.1.4 | Meaconing | dynamic | 954 | 595 | 62.4% |
| 2.6.3 | Spoofing+Jamming | stationary | 498 | 130 | 26.1% |
| 2.3.10,2.3.11 | Spoofing | stationary | 412 | 123 | 29.9% |
| 2.3.2 | Spoofing | dynamic | 136 | 44 | 32.4% |

**This is not spread evenly.** Recording `1.11.7` alone accounts for 55.3% of all 1997
misclassified clean rows in the fold, at a near-total 98.7% failure rate. Two ruled-out
hypotheses first, since they were the obvious candidates: **not** a receiver warm-up effect
(misclassification rate in the first 60s of each recording is 6.7%, vs. 67.5% after — if
anything the opposite of a warm-up problem), and **not** concentrated near attack-window
transition boundaries (60.2% misclassified within 10s of a boundary vs. 64.2% elsewhere —
essentially the same rate). The real mechanisms are feature-level, and there are two distinct
ones, together explaining roughly 91.6% of the fold's misclassified clean rows.

## Mechanism 1 — satellite visibility (accounts for 55.3% of the fold's failures, recording 1.11.7)

`1.11.7`'s "clean" segment has a median of **9 tracked satellites** (`n_sats_l1`), vs. a
median of **43-44** for every other clean segment in this fold and in the training set overall
— an 8x AGC-count drop (702 vs. 5616) and 29x worse horizontal accuracy (5.0m vs. 0.17m) go
with it, all consistent with a receiver that's simply seeing far less open sky, not one under
attack. Confirming that this isn't the attack's doing: **within this same recording, the
attack window has a median of 45 satellites — higher than its own clean segment.** Satellite
visibility recovers when the attack window starts, not the other way around; whatever caused
the low count during the clean segment (most plausibly a stretch of the drive route with
obstructed sky — tree cover, buildings, terrain — since `rover_state='dynamic'` means this is
a moving vehicle) is unrelated to jamming.

**Why the model gets this wrong**: across the *training* set (fold 3's 19 other recordings),
clean rows from dynamic (moving) recordings have a median of 43 satellites — essentially
identical to stationary clean rows (42.8 vs. 43.0) — and only **11 of 3246** dynamic clean
training rows have fewer than 15 satellites. The model has almost no training exposure to
"genuinely clean, but low satellite count" conditions. Meanwhile, across the whole training
set, attack rows *do* have a lower median satellite count than clean rows (39 vs. 43) — a
real, physically sensible pattern, since jamming and spoofing often do reduce the number of
usable satellites. The model has learned "fewer satellites → more likely an attack," which is
a reasonable heuristic in aggregate — and it fires on `1.11.7`'s clean segment for an
unrelated, purely environmental reason it has essentially never seen labeled "clean" before.

`2.3.2` (2.2% of the fold's failures) shows a weaker echo of the same pattern (misclassified
rows: median 27.5 satellites vs. 45 for correctly-classified rows in that recording) — same
mechanism, smaller dose.

## Mechanism 2 — elevated code-Doppler residual baseline (accounts for ~36.3%, recordings 3.1.3,3.1.4 and 2.6.3)

For these two recordings — one dynamic (`3.1.3,3.1.4`), one stationary (`2.6.3`), ruling out
vehicle motion as the cause — satellite count is normal (misclassified vs. correct rows are
statistically indistinguishable: 43 vs. 43 satellites). What differs is
`pr_doppler_residual_std`, the code-Doppler pseudorange-rate consistency residual —
`extract_features.py` built this specifically as "a standard spoofing indicator in the
literature" (actual pseudorange rate vs. Doppler-predicted rate; multipath/spoofing-induced
disagreement should show up here). In these two recordings, it runs high even in the
*correctly-classified* clean rows (median 21.4 and 28.3) relative to the training-set
baseline for the same attack type (other Meaconing clean rows: median 4.9) — roughly 4-6x
elevated as an apparent baseline characteristic of these two specific recording sessions. The
*misclassified* rows push further still (median 106-125, another ~4-5x on top of that). This
reads as a session-specific shift in one of the purpose-built spoofing-detection features
itself, not a receiver-motion artifact (present in both a moving and a stationary recording),
and not explainable from the data at hand beyond "these two sessions' receivers/antennas/RF
environment produced a different code-Doppler noise baseline than the sessions the model
trained on."

`2.3.10,2.3.11` (6.2% of the fold's failures) shows neither mechanism clearly — its
misclassified and correctly-classified rows look similar across every feature checked. This
residual looks like the general session-to-session noise-floor variance
`baseline_model_report.md` already names, without a single additional identifiable cause.

## One-off artifact, or systematic weakness?

**Both, in a specific sense.** The exact recording that triggered mechanism 1 (`1.11.7`'s
obstructed-sky drive segment) is a one-off — a different held-out dynamic recording with good
sky visibility throughout wouldn't reproduce this. But the *underlying gap it exposes* is
systematic: **the training data has almost no examples of legitimately degraded reception
conditions (low satellite count, or an elevated code-Doppler baseline) that are still
labeled clean.** Any future recording — in this dataset or in a real deployment — that
combines "clean" with "unusually poor GNSS geometry or an atypical receiver/RF baseline" is
exposed to the same failure mode, for either of the two reasons found here. This sharpens,
with a concrete mechanism, what `baseline_model_report.md` already disclosed as a suspected
cause ("each recording's own quiet RF noise floor varies somewhat session-to-session") — now
attributable to at least two specific, checkable feature dimensions (satellite
visibility/geometry, and code-Doppler residual baseline) rather than "noise floor" in general.

**Talking-point version for Q&A**: *"Our worst-case fold's clean-recall failure isn't random —
55% of it traces to one specific recording where the receiver saw far fewer satellites during
its clean segment than during its own attack segment (9 vs. 45), almost certainly a route
obstruction rather than anything attack-related, and our training data has almost no examples
of legitimately clean-but-low-visibility conditions to teach the model that distinction. A
further third traces to two recordings with an elevated code-Doppler baseline in exactly the
feature we built as a purpose-built spoofing indicator. Together these explain over 90% of
that fold's misclassifications with concrete, checkable mechanisms — not just 'it's noisier.'"*

## What this doesn't change (yet)

No model or code change is proposed here — this was requested and delivered as diagnostic
only. If pursued, the two concrete next steps this analysis points to are (1) checking
whether `n_sats_l1` and `pr_doppler_residual_std` need session-relative normalization rather
than being fed to the model as raw values, and (2) whether the training set needs more
clean-but-degraded-reception examples specifically (not just more data in general) — but
either would need its own held-out validation before being trusted, the same discipline
applied throughout `ROBUSTNESS_NOTES.md` and `CALIBRATION_NOTES.md`.

## Where this lives

- `fold3_investigation.py` — full implementation.
- `processed/fold3_full_predictions.csv` — every fold-3 validation row with its prediction
  (gitignored, like the rest of `processed/` except the committed parquet — regenerate by
  re-running the script).
- `processed/fold3_feature_comparison.csv`, `processed/fold3_feature_distributions.png` —
  the feature comparison table and histograms referenced above (same gitignore note).
