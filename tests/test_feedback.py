"""Tests for the analyst confirm/dismiss layer (Phase 2).

Scope note: these test storage, retrieval, export shape and arithmetic. They deliberately do
NOT test that any label changes the model's behaviour, because nothing does — there is no
retraining and no closed loop (FEEDBACK_LOOP.md). `test_feedback_never_changes_a_score`
below pins that down as a property rather than leaving it as a claim in a document.
"""
import csv
import io
import itertools
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from conftest import requires_model

pytestmark = requires_model

# Median of the real clean-labeled rows -- same vector the other suites use; see
# tests/test_ingest.py for why hand-written values are avoided.
FEATURES = {
    "fixType": 3.0, "gSpeed": 0.005, "hAcc": 0.183, "vAcc": 0.324, "sAcc": 0.047,
    "headAcc": 0.18, "pDOP": 0.0106, "numSV": 32.0, "velN": -0.001, "velE": 0.0,
    "velD": 0.0, "pos_dev_m": 1.5415, "n_sats_l1": 43.0, "snr_l1_mean": 43.4762,
    "snr_l1_std": 3.8068, "snr_l1_min": 33.0, "doppler_l1_mean": -114.6735,
    "doppler_l1_std": 2325.6041, "pr_doppler_residual_mean": 0.0861,
    "pr_doppler_residual_std": 19.9222, "jam_ind_mean": 7.0, "agc_cnt_mean": 5616.0,
    "noise_per_ms_mean": 101.0,
}

ATTACK_FEATURES = {
    "fixType": 3.0, "gSpeed": 0.026, "hAcc": 0.331, "vAcc": 0.613, "sAcc": 0.078,
    "headAcc": 0.1504, "pDOP": 0.0109, "numSV": 31.0, "velN": -0.001, "velE": 0.0,
    "velD": -0.001, "pos_dev_m": 2.7085, "n_sats_l1": 39.0, "snr_l1_mean": 39.4688,
    "snr_l1_std": 5.2149, "snr_l1_min": 27.0, "doppler_l1_mean": -61.3147,
    "doppler_l1_std": 2096.3066, "pr_doppler_residual_mean": 2.8694,
    "pr_doppler_residual_std": 177.2779, "jam_ind_mean": 14.0, "agc_cnt_mean": 3159.0,
    "noise_per_ms_mean": 102.0,
}

# Strictly increasing observation timestamps, well clear of the ingest suite's 2075 block, so
# the two suites can never collide on the (tower_id, timestamp) dedup key.
_TS_BASE = datetime(2085, 1, 1, tzinfo=timezone.utc)
_ts_slots = itertools.count()


def fresh_ts() -> datetime:
    return _TS_BASE + timedelta(hours=next(_ts_slots))


@pytest.fixture
def event_id(client) -> int:
    """A freshly scored event to label."""
    r = client.post("/score", json=FEATURES)
    assert r.status_code == 200, r.text
    return r.json()["event_id"]


@pytest.fixture
def attack_event_id(client, tower_ids) -> int:
    """A freshly ingested event the model called 'attack' -- needed for the precision maths."""
    r = client.post("/ingest", json={"observations": [
        {"tower_id": tower_ids[60], "timestamp": fresh_ts().isoformat(), **ATTACK_FEATURES}
    ]})
    assert r.status_code == 200, r.text
    res = r.json()["results"][0]
    assert res["predicted_label"] == "attack"
    return res["event_id"]


# --------------------------------------------------------------------------- label

def test_label_an_event(client, event_id):
    r = client.post(f"/events/{event_id}/feedback",
                    json={"label": "confirmed", "note": "matches the RF log",
                          "analyst": "iris"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["event_id"] == event_id
    assert body["label"] == "confirmed"
    assert body["note"] == "matches the RF log"
    assert body["analyst"] == "iris"
    assert body["revision"] == 1
    assert body["previous_label"] is None
    assert body["created_at"] == body["updated_at"], "first label sets both timestamps"


def test_label_persists_and_is_readable(client, event_id):
    client.post(f"/events/{event_id}/feedback", json={"label": "dismissed"})
    r = client.get(f"/events/{event_id}/feedback")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == "dismissed"
    assert body["revision"] == 1


def test_unlabeled_event_reports_null_label(client, event_id):
    """The dashboard needs to distinguish 'never labeled' from 'event does not exist'."""
    r = client.get(f"/events/{event_id}/feedback")
    assert r.status_code == 200, r.text
    assert r.json()["label"] is None
    assert r.json()["revision"] == 0


def test_note_and_analyst_are_optional(client, event_id):
    r = client.post(f"/events/{event_id}/feedback", json={"label": "confirmed"})
    assert r.status_code == 200, r.text
    assert r.json()["note"] is None
    assert r.json()["analyst"] is None


# --------------------------------------------------------------------------- relabel

def test_relabel_replaces_and_bumps_revision(client, event_id):
    first = client.post(f"/events/{event_id}/feedback",
                        json={"label": "confirmed", "analyst": "a"}).json()
    second = client.post(f"/events/{event_id}/feedback",
                         json={"label": "dismissed", "analyst": "b",
                               "note": "second look: multipath"}).json()

    assert second["label"] == "dismissed"
    assert second["previous_label"] == "confirmed"
    assert second["revision"] == 2
    assert second["created_at"] == first["created_at"], "first-labeled time is preserved"
    assert second["updated_at"] >= first["updated_at"]

    current = client.get(f"/events/{event_id}/feedback").json()
    assert current["label"] == "dismissed"
    assert current["analyst"] == "b"


def test_relabel_does_not_double_count_in_the_summary(client, event_id):
    before = client.get("/feedback/summary").json()
    client.post(f"/events/{event_id}/feedback", json={"label": "confirmed"})
    client.post(f"/events/{event_id}/feedback", json={"label": "dismissed"})
    client.post(f"/events/{event_id}/feedback", json={"label": "confirmed"})
    after = client.get("/feedback/summary").json()

    assert after["total_labeled"] == before["total_labeled"] + 1, \
        "three submissions on one event is still one labeled event"
    assert after["total_submissions"] == before["total_submissions"] + 3
    assert after["relabeled_events"] == before["relabeled_events"] + 1


def test_relabel_history_is_append_only(client, event_id):
    client.post(f"/events/{event_id}/feedback", json={"label": "confirmed", "analyst": "a"})
    client.post(f"/events/{event_id}/feedback", json={"label": "dismissed", "analyst": "b"})

    r = client.get(f"/events/{event_id}/feedback/history")
    assert r.status_code == 200, r.text
    history = r.json()
    assert len(history) == 2
    assert [h["label"] for h in history] == ["confirmed", "dismissed"]
    assert [h["previous_label"] for h in history] == [None, "confirmed"]
    assert [h["revision"] for h in history] == [1, 2]
    assert [h["analyst"] for h in history] == ["a", "b"], \
        "the superseded submission keeps its own analyst"


# --------------------------------------------------------------------- unknown event

def test_unknown_event_post_is_404(client):
    r = client.post("/events/99999999/feedback", json={"label": "confirmed"})
    assert r.status_code == 404
    assert "99999999" in r.json()["detail"]


def test_unknown_event_get_is_404(client):
    assert client.get("/events/99999999/feedback").status_code == 404
    assert client.get("/events/99999999/feedback/history").status_code == 404


def test_unknown_event_is_not_recorded(client):
    before = client.get("/feedback/summary").json()["total_submissions"]
    client.post("/events/99999999/feedback", json={"label": "confirmed"})
    after = client.get("/feedback/summary").json()["total_submissions"]
    assert after == before, "a 404 must not write an audit-log row"


# --------------------------------------------------------------------- bad payloads

@pytest.mark.parametrize("payload, why", [
    ({}, "missing label"),
    ({"label": "maybe"}, "label not in the enum"),
    ({"label": "CONFIRMED"}, "labels are case-sensitive"),
    ({"label": ""}, "empty label"),
    ({"label": None}, "null label"),
    ({"label": "confirmed", "note": "x" * 2001}, "note over max_length"),
    ({"label": "confirmed", "analyst": "y" * 121}, "analyst over max_length"),
])
def test_malformed_feedback_is_422(client, event_id, payload, why):
    r = client.post(f"/events/{event_id}/feedback", json=payload)
    assert r.status_code == 422, f"{why}: got {r.status_code} {r.text}"


# --------------------------------------------------------------------------- export

def test_export_is_csv_with_the_expected_shape(client, event_id):
    client.post(f"/events/{event_id}/feedback",
                json={"label": "confirmed", "analyst": "iris", "note": "checked"})

    r = client.get("/feedback/export")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    assert ".csv" in r.headers["content-disposition"]

    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert rows, "export should contain the labeled event"

    header = rows[0].keys()
    for col in ("event_id", "feedback_label", "feedback_analyst", "feedback_note",
                "labeled_at", "first_labeled_at", "feedback_revision",
                "source", "probability", "severity", "predicted_label", "model_version"):
        assert col in header, f"missing meta column {col}"

    # All 23 model features are flattened into their own columns, in the artifact's order.
    from api.model_service import ModelService
    feature_cols = ModelService().feature_cols
    assert len(feature_cols) == 23
    expected = [f"feature_{c}" for c in feature_cols]
    assert [c for c in header if c.startswith("feature_")] == expected

    row = next(r_ for r_ in rows if r_["event_id"] == str(event_id))
    assert row["feedback_label"] == "confirmed"
    assert row["feedback_analyst"] == "iris"
    assert row["feedback_note"] == "checked"
    assert float(row["feature_snr_l1_mean"]) == pytest.approx(FEATURES["snr_l1_mean"])


def test_export_contains_only_labeled_events(client, event_id):
    unlabeled = client.post("/score", json=FEATURES).json()["event_id"]
    client.post(f"/events/{event_id}/feedback", json={"label": "dismissed"})

    rows = list(csv.DictReader(io.StringIO(client.get("/feedback/export").text)))
    ids = {r["event_id"] for r in rows}
    assert str(event_id) in ids
    assert str(unlabeled) not in ids


def test_export_reflects_the_current_label_not_the_first(client, event_id):
    client.post(f"/events/{event_id}/feedback", json={"label": "confirmed"})
    client.post(f"/events/{event_id}/feedback", json={"label": "dismissed"})

    rows = list(csv.DictReader(io.StringIO(client.get("/feedback/export").text)))
    row = next(r for r in rows if r["event_id"] == str(event_id))
    assert row["feedback_label"] == "dismissed"
    assert row["feedback_revision"] == "2"


def test_export_survives_commas_and_newlines_in_a_note(client, event_id):
    nasty = 'multipath, not spoofing\nsecond line with "quotes" and, commas'
    client.post(f"/events/{event_id}/feedback", json={"label": "dismissed", "note": nasty})

    rows = list(csv.DictReader(io.StringIO(client.get("/feedback/export").text)))
    row = next(r for r in rows if r["event_id"] == str(event_id))
    assert row["feedback_note"] == nasty, "csv quoting must round-trip the note verbatim"


def test_export_is_parseable_when_nothing_is_labeled(client):
    """Header-only output is still valid CSV -- a consumer shouldn't special-case empty."""
    from api.db import EventStore
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        store = EventStore(os.path.join(d, "empty.db"))
        try:
            assert store.labeled_events() == []
            assert store.feedback_summary()["total_labeled"] == 0
        finally:
            store.close()


# --------------------------------------------------------------------------- summary

def test_summary_counts(client, event_id):
    before = client.get("/feedback/summary").json()
    client.post(f"/events/{event_id}/feedback", json={"label": "confirmed", "analyst": "iris"})
    after = client.get("/feedback/summary").json()

    assert after["confirmed"] == before["confirmed"] + 1
    assert after["dismissed"] == before["dismissed"]
    assert after["total_labeled"] == before["total_labeled"] + 1
    assert after["unlabeled"] == after["total_events"] - after["total_labeled"]
    assert after["distinct_analysts"] >= 1


def test_summary_precision_arithmetic(client, attack_event_id):
    """Confirming an event the model called 'attack' must move the predicted-attack
    precision numerator and denominator together, by exactly one each."""
    before = client.get("/feedback/summary").json()
    client.post(f"/events/{attack_event_id}/feedback", json={"label": "confirmed"})
    after = client.get("/feedback/summary").json()

    assert after["predicted_attack_labeled"] == before["predicted_attack_labeled"] + 1
    assert after["predicted_attack_confirmed"] == before["predicted_attack_confirmed"] + 1
    assert after["predicted_attack_precision"] == pytest.approx(
        after["predicted_attack_confirmed"] / after["predicted_attack_labeled"])


def test_dismissing_an_attack_lowers_precision(client, tower_ids):
    r = client.post("/ingest", json={"observations": [
        {"tower_id": tower_ids[61], "timestamp": fresh_ts().isoformat(), **ATTACK_FEATURES}]})
    eid = r.json()["results"][0]["event_id"]

    before = client.get("/feedback/summary").json()
    client.post(f"/events/{eid}/feedback", json={"label": "dismissed"})
    after = client.get("/feedback/summary").json()

    assert after["predicted_attack_labeled"] == before["predicted_attack_labeled"] + 1
    assert after["predicted_attack_confirmed"] == before["predicted_attack_confirmed"]
    if before["predicted_attack_precision"] is not None:
        assert after["predicted_attack_precision"] <= before["predicted_attack_precision"]


def test_precision_is_over_labeled_events_only(client, event_id):
    """A clean-predicted labeled event must not enter the predicted-attack denominator."""
    before = client.get("/feedback/summary").json()
    ev = client.get(f"/events?limit=2000").json()
    assert next(e for e in ev if e["id"] == event_id)["predicted_label"] == "clean"

    client.post(f"/events/{event_id}/feedback", json={"label": "dismissed"})
    after = client.get("/feedback/summary").json()

    assert after["predicted_attack_labeled"] == before["predicted_attack_labeled"]
    assert after["total_labeled"] == before["total_labeled"] + 1


def test_summary_carries_the_caveat(client):
    body = client.get("/feedback/summary").json()
    caveat = body["caveat"]
    assert "labeled events only" in caveat.lower()
    assert "non-random" in caveat.lower()
    assert "FEEDBACK_LOOP.md" in caveat
    assert "not" in caveat.lower()


def test_summary_precision_is_null_with_no_denominator(client):
    """On a store with no labels at all, precision is null rather than 0.0 or a crash --
    0.0 would read as 'every alert was a false alarm', which is a different claim."""
    from api.db import EventStore
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        store = EventStore(os.path.join(d, "empty.db"))
        try:
            s = store.feedback_summary()
            assert s["predicted_attack_precision"] is None
            assert s["hysteresis_alert_precision"] is None
            assert s["total_labeled"] == 0
        finally:
            store.close()


# --------------------------------------------------------------- no closed loop

def test_feedback_never_changes_a_score(client, tower_ids):
    """The core honesty property of Phase 2, asserted rather than merely documented: an
    identical feature vector scores identically before and after labeling, and after many
    labels. No retraining, no threshold adaptation, no feedback path into the model."""
    first = client.post("/score", json=ATTACK_FEATURES).json()

    for i in range(5):
        r = client.post("/ingest", json={"observations": [
            {"tower_id": tower_ids[70 + i], "timestamp": fresh_ts().isoformat(),
             **ATTACK_FEATURES}]})
        eid = r.json()["results"][0]["event_id"]
        client.post(f"/events/{eid}/feedback",
                    json={"label": "dismissed", "note": "false alarm", "analyst": "iris"})

    second = client.post("/score", json=ATTACK_FEATURES).json()

    assert second["probability"] == first["probability"]
    assert second["decision_threshold"] == first["decision_threshold"] == 0.5200000000000002
    assert second["model_version"] == first["model_version"]
    assert second["predicted_label"] == first["predicted_label"]


def test_labels_do_not_alter_the_scored_event_row(client, event_id):
    """Labeling writes to event_feedback, never back onto scored_events."""
    before = next(e for e in client.get("/events?limit=2000").json() if e["id"] == event_id)
    client.post(f"/events/{event_id}/feedback", json={"label": "confirmed"})
    after = next(e for e in client.get("/events?limit=2000").json() if e["id"] == event_id)
    assert after == before
