# Analyst feedback — what it is, and what it is not

Phase 2 of the near-production work.

---

## The one-sentence version

**Analyst labels are stored for future retraining. No closed loop exists.**

Nothing in this repository reads a label back into the model. No retraining runs, no
threshold adapts, no weights update, no automated action follows from a confirm or a dismiss.
The model that answers your next `/score` call is byte-for-byte the model that answered the
last one, no matter how many events have been labeled in between.

That is not a limitation we are working around and hope to fix quietly. It is the honest
state of the system, and this document exists so that nobody — in a demo, a deck, or a
conversation with a judge — describes it as anything else.

---

## What was built

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/events/{id}/feedback` | Record a verdict: `{"label": "confirmed" \| "dismissed", "note"?, "analyst"?}` |
| `GET` | `/events/{id}/feedback` | Current label for one event (the dashboard reads this on every panel open) |
| `GET` | `/events/{id}/feedback/history` | Every submission for that event, oldest first, including superseded ones |
| `GET` | `/feedback/export` | All labeled events as CSV — the retraining dataset |
| `GET` | `/feedback/summary` | Counts + alert precision over labeled events only |

Plus **Confirm / Dismiss buttons** on the dashboard's event panel, with the current label read
back from the API on every open, and a label tally + precision strip beneath it.

### Storage

Two SQLite tables (`api/db.py`):

- **`event_feedback`** — exactly one row per event: the *current* label, its note, its
  analyst, when it was first labeled, when it was last changed, and a `revision` counter.
  Relabeling upserts this row.
- **`event_feedback_log`** — append-only. Every submission ever made, including superseded
  ones, each carrying the `previous_label` it replaced.

Two tables rather than one, for a reason worth stating: a relabel is **information**, not
noise. An analyst changing their mind about an event is exactly the kind of signal a future
retraining effort would want to weigh — or deliberately exclude — rather than never know
about. Meanwhile the current-label table keeps `/feedback/summary` honest: three submissions
on one event is still **one** labeled event, never three.

Both writes happen in one transaction under the store's shared lock, so the audit log can
never record a submission the current-label table rejected, or vice versa.

### What a label does not touch

`scored_events` is never written by the feedback path. A label cannot change an event's
probability, severity, predicted label, SHAP explanation, or hysteresis state. This is
asserted, not just asserted-in-prose:
`tests/test_feedback.py::test_labels_do_not_alter_the_scored_event_row` compares the full
event row before and after labeling.

---

## The precision numbers, and how not to quote them

`GET /feedback/summary` returns two precision figures. Both are **MEASURED** — real counts of
real labels — and both describe a population you must state when you quote them.

| Field | Denominator |
| --- | --- |
| `predicted_attack_precision` | Labeled events whose `predicted_label` was `attack` (per-reading, threshold 0.52, no debouncing) |
| `hysteresis_alert_precision` | Labeled events whose `alert_state` was `alerting` (3 consecutive above-threshold readings — `api/hysteresis.py`) |

Both are `confirmed / labeled` over that denominator. Both are **`null`**, not `0.0`, when the
denominator is zero — `0.0` would read as "every alert was a false alarm", which is a
completely different claim from "nothing has been labeled yet".

Every response carries this caveat in a `caveat` field, verbatim:

> MEASURED over labeled events only. Analysts choose which events to label, so this is a
> self-selected, non-random subset of all scored events — not a random sample. These figures
> describe that subset and nothing wider. They are NOT the model's field precision, NOT a
> validation result, and NOT comparable to the held-out test metrics in the model reports. No
> retraining uses these labels — see FEEDBACK_LOOP.md.

### Why the selection bias is not a technicality

Analysts label what they look at, and they look at what draws attention. In this system that
means high-severity events, events on the map's hot spots, events during a demo. The labeled
set is therefore skewed toward exactly the events the model was most confident about, in both
directions. A precision figure over that subset can be well above or well below the detector's
true rate, and there is no correction available because the selection process is not recorded
and not random.

Two further reasons the figure is not a field-precision estimate:

1. **The events are not field events.** Every scored event in this system originates from
   either the Jammertest 2024 recordings (replay) or a caller-supplied feature vector
   (`/ingest`). No real GNSS receiver at a real telecom site has ever produced one. Labeling
   them measures agreement with an analyst about *this* data, not performance in ASEAN
   infrastructure.
2. **There is no ground truth to check the analyst against.** For replayed events the dataset
   *does* carry a true label (`true_attack`), and a confirm/dismiss can be compared to it —
   but the summary endpoint deliberately does not do that, because it would conflate "the
   analyst agreed with the model" with "the model was right". Those are different
   measurements and merging them would be the kind of quiet overstatement this project
   avoids. The export CSV carries `true_attack` alongside `feedback_label`, so anyone who
   wants that comparison can compute it explicitly and say what they computed.

---

## The export CSV

`GET /feedback/export` returns one row per labeled event, **48 columns**:

- **7 feedback columns** — `event_id`, `feedback_label`, `feedback_analyst`, `feedback_note`,
  `labeled_at`, `first_labeled_at`, `feedback_revision`
- **18 event columns** — `created_at`, `source`, `run_id`, `scenario_id`, `obs_timestamp`,
  `ingest_batch_id`, `attack_type`, `true_attack`, `probability`, `severity`,
  `predicted_label`, `alert_state`, `model_version`, `tower_site_id`, `tower_site_name`,
  `tower_lat`, `tower_lon`, `correlation_score`
- **23 feature columns** — `feature_<name>`, one per model feature, in the trained artifact's
  own `feature_cols` order

The label reported is the **current** one, with `feedback_revision` saying how many times it
has changed. Notes containing commas, quotes and newlines round-trip correctly (tested).

This file is the artifact a retraining effort would start from. **Producing it is where this
repo stops.** There is no consumer, no scheduled job, no training script that reads it.

---

## What a real closed loop would need, and why none of it is here

Listing this is not a roadmap promise. It is the measure of the distance between what exists
and what "the system learns from analyst feedback" would actually require.

| Needed | Status |
| --- | --- |
| Enough labels for a usable training signal | **Absent.** Tens of labels, not the thousands a 23-feature RandomForest would need to shift meaningfully. |
| A label-quality process — inter-analyst agreement, adjudication of disagreements | **Absent.** `analyst` is free text with no authentication behind it (Phase 3's API key authenticates a *caller*, not a person). Two analysts labeling the same event just overwrite each other, with the audit log as the only record. |
| Correction for the selection bias above | **Absent, and not straightforwardly possible** with the current data. |
| A retraining pipeline with held-out evaluation | **Absent.** `train_improved_model.py` trains from the Jammertest feature table and knows nothing about feedback. It was not modified. |
| A promotion gate — never ship a retrained model that regresses on the frozen test set | **Absent.** |
| Drift monitoring to know whether new labels even describe the same distribution | **Phase 3** adds `/drift`. Not yet present. |
| Rollback, model versioning across retrains, an audit trail of which model made which call | **Partially**: each event stores `model_version`. Nothing else. |

---

## Honest limitations of what *was* built

- **No authentication.** Phase 2 ships no auth at all; anyone who can reach the API can label
  anything. Phase 3 adds an API key, which authenticates a caller, not a person — `analyst`
  remains an unverified free-text attribution hint.
- **Last write wins.** Concurrent labels on one event overwrite each other. The audit log
  preserves what happened, but the current label is simply the most recent submission. There
  is no locking, no merge, no conflict surface.
- **Deleting a label is not possible.** There is no `DELETE`. A mistaken label can be
  corrected by relabeling (which is recorded as a revision), never erased.
- **`analyst` on the dashboard comes from `localStorage`** and is `null` unless set. There is
  no UI to set it in this phase; the field exists so an API caller can populate it.
- **Labels are scoped to `event_id`, which is scoped to the DB.** Wiping `data/syncguard.db`
  discards events and labels together. The export CSV is the only durable form.
- **The dashboard shows the label of one event at a time.** There is no queue, no "show me
  everything unlabeled", no bulk labeling. Labeling 14,000 replayed events through this UI is
  not practical and was not the goal.

---

## Test coverage

`tests/test_feedback.py` — **30 tests**:

| Area | Coverage |
| --- | --- |
| Label | first label, note/analyst optional, persistence and read-back, unlabeled events report `label: null, revision: 0` |
| Relabel | replaces the current label, bumps `revision`, preserves `first_labeled_at`, reports `previous_label`, does not double-count in the summary, append-only history keeps each submission's own analyst |
| Unknown event | 404 on POST, GET and history; a 404 writes no audit-log row |
| Bad payloads | 7 parametrized cases (missing/invalid/empty/null label, wrong case, over-length note and analyst) |
| Export format | CSV headers, `Content-Disposition`, all 23 features in artifact order, only labeled events, current label not the first, notes with commas/quotes/newlines round-trip, empty store is still valid |
| Summary | counts, precision arithmetic in both directions, clean-predicted events excluded from the attack denominator, `null` precision on an empty denominator, caveat present |
| No closed loop | identical vector scores identically before and after many labels; labeling never mutates the scored-events row |

Run: `python -m pytest`

---

## If you are writing a slide or a script about this

Accurate: *"Analysts confirm or dismiss alerts; the labels are stored, exported as CSV, and
are the starting point for retraining that has not been built."*

Not accurate: *"the system learns from analyst feedback"*, *"the model improves over time"*,
*"closed-loop detection"*, *"self-tuning"*, *"human-in-the-loop training"*, or any precision
figure from `/feedback/summary` quoted without the labeled-events-only caveat attached.

Phase 5's `DOC_SYNC_CHECKLIST.md` carries suggested wording for the abstract, deck and
narration script.
