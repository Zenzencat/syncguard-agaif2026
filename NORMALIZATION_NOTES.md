# SyncGuard: session-relative normalization — does it fix what FOLD_ANALYSIS.md found?

Follow-up to `FOLD_ANALYSIS.md`, same honest-tradeoffs discipline as `ROBUSTNESS_NOTES.md` and
`CALIBRATION_NOTES.md`. Tests whether normalizing `n_sats_l1` and `pr_doppler_residual_std`
relative to their own recording's baseline — added as **new** features alongside the existing
raw ones, nothing removed — fixes the clean-recall failure traced to those two features.
**Neither `models/model.joblib`, `api/`, nor the deployed threshold (0.52) have been
touched.** Standalone analysis — [`session_normalization_experiment.py`](session_normalization_experiment.py)
produces every number below (~14s; the module docstring has the full implementation detail).

## Two variants, and why they had to be kept separate

1. **Offline** (`*_dev_offline`): `value − median(value across the whole recording)`. Uses
   only that recording's own feature values (no labels, no other recordings), but *does* use
   rows that come after the one being normalized — not computable for an early row in a live
   streaming session. Treated strictly as an **oracle ceiling**: if this doesn't help, a
   deployable version won't either; if it does help, that's necessary but not sufficient.
2. **Causal** (`*_dev_causal`): `value − median(value over the trailing 60 rows)`, a true
   rolling window (shift(1).rolling(60, min_periods=30).median()) — fully computable live
   (maintain a rolling buffer per active tower/session), no lookahead.

**Why a rolling window and not the `pos_dev_m`-style "first 60 seconds" reference window**,
which was the first thing tried: checked directly, and it would have been wrong.
`1.11.7` — the recording responsible for 55% of fold 3's failures — has its **first 60 seconds
entirely inside its attack window** (60/60 attack rows), with a *higher* satellite count during
that attack window (46) than during the clean segment that follows (9). A first-60s reference
baseline would be contaminated by the attack itself and would anchor to the wrong regime
entirely. This also ruled out a simple *expanding*-from-session-start window for the same
reason — it would stay biased toward the early attack-period reading for a long time. A
trailing rolling window was the only design that could plausibly adapt to a regime change
partway through a session, which `1.11.7` clearly has.

## Result: the idea has real signal, but it isn't deployable — and even the ceiling has a cost

**Pooled (all 44639 rows, out-of-fold, threshold=0.52):**

| | Accuracy | Clean recall | FPR | Jamming recall | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (current shipped config) | 0.837 | 0.638 | 0.362 | 0.773 | 0.847 |
| **Offline (oracle, not deployable)** | **0.896** | **0.722** | **0.278** | **0.922** | **0.893** |
| Causal, 60-row window (deployable) | 0.843 | 0.651 | 0.375 | 0.795 | 0.853 |
| Causal, 300-row window (deployable) | 0.843 | 0.625 | 0.375 | 0.804 | 0.852 |

**Fold 3 specifically (the direct test — does this fix the recording FOLD_ANALYSIS.md found?):**

| | Accuracy | Clean recall | FPR | Jamming recall |
|---|---|---|---|---|
| Baseline | 0.735 | 0.363 | 0.637 | 0.869 |
| **Offline (oracle)** | 0.773 | **0.515** (+15.1pt) | 0.485 | **0.533 (−33.6pt)** |
| Causal, 60-row window | 0.736 | 0.362 (essentially unchanged) | 0.638 | 0.886 |
| Causal, 300-row window | 0.723 | **0.321 (−4.2pt, worse)** | 0.679 | 0.897 |

Three findings, each load-bearing for the decision below:

1. **The offline/oracle version confirms the underlying idea has real signal** — pooled clean
   recall, jamming recall, and ROC-AUC all improve substantially, and the new features rank
   3rd and 5th (of 25) in feature importance. Session-relative deviation *is* informative when
   computed with full-session hindsight.
2. **But the oracle version doesn't actually fix fold 3's specific problem cleanly — it moves
   the cost onto jamming recall in the exact same recording.** `1.11.7` has an *inverted*
   relationship between satellite count and attack status (its attack window has *more*
   satellites than its clean segment — see `FOLD_ANALYSIS.md`), the opposite of the general
   population pattern (attack rows have lower satellite counts than clean rows overall, mean
   37.1 vs 42.9). A feature that helps the model lean harder on session-relative satellite
   deviation helps it recognize *this recording's* clean segment as session-typical — and
   simultaneously makes it lean the wrong way on *this same recording's* attack segment, which
   looks session-atypical in the "wrong" (clean-like) direction once you're reasoning
   relative to session baselines. Real gain, real cost, same recording — the same shape of
   trade-off `ROBUSTNESS_NOTES.md` and `CALIBRATION_NOTES.md` already rejected fixes for.
3. **The causal/deployable version — the only one that could actually ship — does not
   meaningfully fix fold 3 at either window size tried (60 or 300 rows), and pooled gains are
   small.** At 60 rows, fold-3 clean recall is statistically unchanged (0.363→0.362) and the
   new features rank near the bottom of feature importance (22nd/23rd of 25) — the model
   barely uses them. A wider 300-row window, tried specifically to check whether 60 rows was
   too short to let the rolling baseline "catch up" past `1.11.7`'s attack-to-clean regime
   change, made fold-3 clean recall *worse* (0.363→0.321), not better — even though the new
   features' importance rose (17th/16th of 25), whatever they picked up didn't help this
   specific case. Likely explanation: in most *other* training recordings there's no
   comparable regime shift, so the causal-deviation feature is mostly near-zero noise there,
   and the model has limited experience learning when it should be trusted — exactly the same
   "too few examples of the failure condition to learn from" problem `FOLD_ANALYSIS.md`
   diagnosed for the raw features in the first place, now inherited by the derived ones.

Determinism confirmed for every variant and every fold — each scored twice, bit-identical,
`n_jobs=1` forced before every `predict_proba` call (same discipline as `ROBUSTNESS_NOTES.md`).

## Decision: do not integrate either variant

**Keep the currently-shipped model (`models/model.joblib`, threshold 0.52) unchanged.**

- The causal/deployable version — the only one a live system could actually compute — does not
  clear the bar. It doesn't fix fold 3 (the specific case motivating this work) at either
  window size tested, and its pooled gains (roughly +1pt on most metrics) are too small to
  justify adding two features and their associated NaN-handling to the shipped pipeline.
- The offline/oracle version *would* be a genuine aggregate improvement — but it isn't
  deployable as computed (needs whole-session hindsight a live system doesn't have for early
  rows), and even setting that aside, it fails the "no meaningful regression elsewhere" bar:
  a 33.6-point jamming-recall cost in the exact recording this investigation was about is not
  a trade this project has been willing to make anywhere else.

This is reported as a genuine, evidenced negative result, not a shortcut — the same standard
applied to jamming sample-reweighting and isotonic calibration before it. The one part of the
original hypothesis that *is* now confirmed rather than merely suspected: session-relative
information about `n_sats_l1` and `pr_doppler_residual_std` does carry real signal (the oracle
version proves it) — the open problem is a deployable way to estimate a session's own baseline
quickly enough, and robustly enough to an early regime change, to capture that signal without
also learning to fight the (rarer, but real) recordings where the usual attack/clean direction
inverts. Concrete next steps for anyone picking this up: a shorter minimum-history requirement
combined with a change-point detector (rather than a plain rolling median) to catch a regime
shift faster than 60-300 rows allows; or normalizing by *rover_state* (dynamic vs. stationary)
population statistics instead of per-session ones, which wouldn't have the "attack happened to
occur early" contamination problem at all — neither has been tried here, and either would need
its own GroupKFold validation before being trusted.

## Where this lives

- `session_normalization_experiment.py` — full implementation (default 60-row window); the
  300-row sensitivity check was run from a temporary copy with `ROLLING_WINDOW`/`MIN_HISTORY`
  edited, not kept as a separate file — re-run with those constants changed to reproduce it.
- `processed/normalization_comparison_table.csv`, `processed/normalization_fold3_comparison.csv`
  — the two comparison tables above (gitignored, like the rest of `processed/` except the
  committed parquet — regenerate by re-running the script).
- Does not modify `models/model.joblib`, `models/model_baseline.joblib`, `api/`,
  `train_baseline_model.py`, `train_improved_model.py`, or any deployed threshold.
