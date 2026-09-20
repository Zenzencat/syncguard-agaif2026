"""Tests for POST /ingest (Phase 1 ingestion adapter).

These exercise the endpoint's contract, not the model's accuracy: nothing here asserts a
particular probability or label, because the shipped model, its 23 features and its 0.52
threshold are fixed and out of scope for this phase. What is asserted is that ingestion goes
through the same scoring path, persists, deduplicates, orders, and validates the way
INGESTION_CONTRACT.md says it does.
"""
import itertools
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from conftest import requires_model

pytestmark = requires_model


# Feature vectors used across these tests. Each field is the MEDIAN of the real
# clean-labeled (BASE_FEATURES, n=10,349) or attack-labeled (NOISY_FEATURES, n=34,290) rows in
# processed/syncguard_features.parquet. The numbers are therefore real summary statistics, but
# neither vector is a real observation -- no single receiver epoch produced exactly this
# combination. They are used because they sit unambiguously on opposite sides of the shipped
# 0.52 threshold in the model's own units, which hand-written "plausible" values do not: an
# earlier version of this file used invented numbers (pDOP 1.4, hAcc 2.5) and BOTH vectors
# scored as attack. That is a property of the feature scales, not of the tests -- see
# INGESTION_CONTRACT.md's unit/scale warning.
#
# No test asserts a specific probability. The label-dependent test below verifies it against
# the live /score response rather than hardcoding an expectation.
BASE_FEATURES = {
    "fixType": 3.0, "gSpeed": 0.005, "hAcc": 0.183, "vAcc": 0.324, "sAcc": 0.047,
    "headAcc": 0.18, "pDOP": 0.0106, "numSV": 32.0, "velN": -0.001, "velE": 0.0,
    "velD": 0.0, "pos_dev_m": 1.5415, "n_sats_l1": 43.0, "snr_l1_mean": 43.4762,
    "snr_l1_std": 3.8068, "snr_l1_min": 33.0, "doppler_l1_mean": -114.6735,
    "doppler_l1_std": 2325.6041, "pr_doppler_residual_mean": 0.0861,
    "pr_doppler_residual_std": 19.9222, "jam_ind_mean": 7.0, "agc_cnt_mean": 5616.0,
    "noise_per_ms_mean": 101.0,
}

NOISY_FEATURES = {
    "fixType": 3.0, "gSpeed": 0.026, "hAcc": 0.331, "vAcc": 0.613, "sAcc": 0.078,
    "headAcc": 0.1504, "pDOP": 0.0109, "numSV": 31.0, "velN": -0.001, "velE": 0.0,
    "velD": -0.001, "pos_dev_m": 2.7085, "n_sats_l1": 39.0, "snr_l1_mean": 39.4688,
    "snr_l1_std": 5.2149, "snr_l1_min": 27.0, "doppler_l1_mean": -61.3147,
    "doppler_l1_std": 2096.3066, "pr_doppler_residual_mean": 2.8694,
    "pr_doppler_residual_std": 177.2779, "jam_ind_mean": 14.0, "agc_cnt_mean": 3159.0,
    "noise_per_ms_mean": 102.0,
}


def obs(tower_id: str, ts: datetime, features: dict | None = None) -> dict:
    return {"tower_id": tower_id, "timestamp": ts.isoformat(), **(features or BASE_FEATURES)}


# Test timestamps are strictly increasing, one hour apart per call, starting from a fixed
# far-future instant. Monotonicity matters: several tests reuse the same towers, and
# out-of-order is judged against the newest timestamp already accepted for that tower, so a
# non-monotonic generator would make one test's data look "late" to the next. One hour of
# headroom per call is far more than any single test's own second-scale offsets need. The
# suite runs against a throwaway DB (tests/conftest.py), so these never collide across runs.
_TS_BASE = datetime(2075, 1, 1, tzinfo=timezone.utc)
_ts_slots = itertools.count()


def fresh_ts() -> datetime:
    """A fresh, strictly-later-than-every-previous UTC base time for one test."""
    return _TS_BASE + timedelta(hours=next(_ts_slots))


def post(client, observations, batch_id=None):
    payload = {"observations": observations}
    if batch_id is not None:
        payload["batch_id"] = batch_id
    return client.post("/ingest", json=payload)


# --------------------------------------------------------------------------- valid batch

def test_valid_batch_scores_and_persists(client, tower_ids):
    t0 = fresh_ts()
    batch_id = f"test-{uuid.uuid4().hex[:8]}"
    observations = [obs(tower_ids[i], t0 + timedelta(seconds=i)) for i in range(3)]

    r = post(client, observations, batch_id=batch_id)
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["batch_id"] == batch_id
    assert body["received"] == 3
    assert body["scored"] == 3
    assert body["duplicates"] == 0
    assert body["out_of_order"] == 0
    assert body["reordered_in_batch"] == 0
    assert body["live_explain"] is True  # 3 <= INGEST_LIVE_EXPLAIN_MAX_BATCH
    assert len(body["results"]) == 3

    for res in body["results"]:
        assert res["event_id"] is not None
        assert 0.0 <= res["probability"] <= 1.0
        assert 0.0 <= res["severity"] <= 1.0
        assert res["predicted_label"] in ("attack", "clean")
        assert res["alert_state"] in ("normal", "alerting")
        assert res["duplicate"] is False
        assert res["explained"] is True

    # Persisted and visible through the same /events the dashboard reads.
    ids = {res["event_id"] for res in body["results"]}
    recent = client.get("/events?limit=200").json()
    by_id = {e["id"]: e for e in recent}
    assert ids <= set(by_id), "every ingested event should be in /events"
    for eid in ids:
        assert by_id[eid]["source"] == "ingest"
        assert by_id[eid]["obs_timestamp"] is not None
        assert by_id[eid]["ingest_batch_id"] == batch_id
        assert by_id[eid]["true_attack"] is None  # ingested data carries no ground truth


def test_ingest_uses_the_same_scoring_path_as_score(client, tower_ids):
    """Same feature vector through POST /score and POST /ingest must produce the identical
    probability -- this is the check that ingestion did not fork the scoring path."""
    r_score = client.post("/score", json=NOISY_FEATURES)
    assert r_score.status_code == 200, r_score.text
    score_body = r_score.json()

    r_ingest = post(client, [obs(tower_ids[0], fresh_ts(), NOISY_FEATURES)])
    assert r_ingest.status_code == 200, r_ingest.text
    ingest_res = r_ingest.json()["results"][0]

    assert ingest_res["probability"] == score_body["probability"]
    assert ingest_res["predicted_label"] == score_body["predicted_label"]
    assert ingest_res["severity"] == score_body["severity"]


def test_site_id_resolves_to_tower_key(client):
    """A caller may submit a raw site_id; the response reports the unique tower_key it
    resolved to (16 of the 136 real towers share the site_id 'tbg' -- api/spatial.py)."""
    towers = client.get("/towers").json()
    t = towers[0]
    r = post(client, [obs(t["site_id"], fresh_ts())])
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["tower_id"] == t["tower_key"]


def test_large_batch_skips_inline_explain(client, tower_ids):
    t0 = fresh_ts()
    n = 25  # > INGEST_LIVE_EXPLAIN_MAX_BATCH (20)
    r = post(client, [obs(tower_ids[i % len(tower_ids)], t0 + timedelta(seconds=i))
                      for i in range(n)])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scored"] == n
    assert body["live_explain"] is False
    assert all(res["explained"] is False for res in body["results"])

    # Not permanently unexplainable -- GET /events/{id}/explain computes it lazily.
    eid = body["results"][0]["event_id"]
    exp = client.get(f"/events/{eid}/explain")
    assert exp.status_code == 200, exp.text
    assert exp.json()["cached"] is False
    assert len(exp.json()["top_features"]) == 5


# --------------------------------------------------------------------------- unknown tower

def test_unknown_tower_is_422(client, tower_ids):
    r = post(client, [obs("definitely-not-a-real-tower", fresh_ts())])
    assert r.status_code == 422
    assert "definitely-not-a-real-tower" in r.json()["detail"]


def test_unknown_tower_rejects_the_whole_batch(client, tower_ids):
    """Tower resolution happens up front, so a batch with one bad tower is not half-applied."""
    t0 = fresh_ts()
    before = len(client.get("/events?limit=2000").json())
    r = post(client, [
        obs(tower_ids[0], t0),
        obs("not-a-tower", t0 + timedelta(seconds=1)),
        obs(tower_ids[1], t0 + timedelta(seconds=2)),
    ])
    assert r.status_code == 422
    after = len(client.get("/events?limit=2000").json())
    assert after == before, "no observation from a rejected batch should have been persisted"


# --------------------------------------------------------------------------- malformed

@pytest.mark.parametrize("payload, why", [
    ({}, "missing observations"),
    ({"observations": []}, "empty batch"),
    ({"observations": [{"timestamp": "2026-01-01T00:00:00Z", **BASE_FEATURES}]}, "missing tower_id"),
    ({"observations": [{"tower_id": "x", **BASE_FEATURES}]}, "missing timestamp"),
    ({"observations": [{"tower_id": "x", "timestamp": "not-a-timestamp", **BASE_FEATURES}]}, "bad timestamp"),
    ({"observations": [{"tower_id": "x", "timestamp": "2026-01-01T00:00:00Z",
                        **{**BASE_FEATURES, "numSV": "eighteen"}}]}, "non-numeric feature"),
    ({"observations": [{"tower_id": "x", "timestamp": "2026-01-01T00:00:00Z",
                        **{**BASE_FEATURES, "hAcc": -5.0}}]}, "negative hAcc violates ge=0"),
    ({"observations": [{"tower_id": "x", "timestamp": "2026-01-01T00:00:00Z",
                        **{**BASE_FEATURES, "numSV": 9999.0}}]}, "numSV above le=64"),
    ({"observations": "not-a-list"}, "observations wrong type"),
])
def test_malformed_payload_is_422(client, payload, why):
    r = client.post("/ingest", json=payload)
    assert r.status_code == 422, f"{why}: expected 422, got {r.status_code} {r.text}"


def test_missing_features_are_accepted_and_imputed(client, tower_ids):
    """Features are Optional by design -- the trained pipeline's SimpleImputer handles gaps
    exactly as it did in training (see api/schemas.py). A sparse window is not an error."""
    r = post(client, [{"tower_id": tower_ids[0], "timestamp": fresh_ts().isoformat(),
                       "numSV": 6.0, "jam_ind_mean": 70.0}])
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["event_id"] is not None


def test_batch_size_cap_is_enforced(client, tower_ids):
    from api.schemas import MAX_INGEST_BATCH
    t0 = fresh_ts()
    over = [obs(tower_ids[i % len(tower_ids)], t0 + timedelta(seconds=i))
            for i in range(MAX_INGEST_BATCH + 1)]
    assert client.post("/ingest", json={"observations": over}).status_code == 422


# --------------------------------------------------------------------- out-of-order

def test_out_of_order_within_a_batch_is_sorted_not_rejected(client, tower_ids):
    """A shuffled batch is reported as reordered, fully scored, and fed to hysteresis in
    ascending timestamp order -- nothing is dropped and nothing is flagged out_of_order,
    because the sort already restored the sequence."""
    tower = tower_ids[10]
    t0 = fresh_ts()
    times = [t0 + timedelta(seconds=s) for s in (0, 3, 1, 4, 2)]
    r = post(client, [obs(tower, ts) for ts in times])
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["scored"] == 5
    assert body["reordered_in_batch"] == 2  # the 1s and 2s rows arrived after later ones
    assert body["out_of_order"] == 0
    returned = [res["timestamp"] for res in body["results"]]
    assert returned == sorted(returned), "results are returned in processed (sorted) order"


def test_observation_older_than_accepted_data_is_flagged_but_kept(client, tower_ids):
    """A second batch carrying a timestamp older than what this tower already accepted is
    still scored and persisted, flagged out_of_order, and deliberately NOT advanced through
    hysteresis (its alert_state is the tower's current state)."""
    tower = tower_ids[11]
    t0 = fresh_ts()

    first = post(client, [obs(tower, t0 + timedelta(seconds=s)) for s in (10, 11, 12)])
    assert first.status_code == 200, first.text
    state_after_first = first.json()["results"][-1]["alert_state"]

    late = post(client, [obs(tower, t0 + timedelta(seconds=1))])
    assert late.status_code == 200, late.text
    body = late.json()
    assert body["scored"] == 1
    assert body["out_of_order"] == 1
    res = body["results"][0]
    assert res["out_of_order"] is True
    assert res["event_id"] is not None          # kept, not dropped
    assert res["alert_state"] == state_after_first  # hysteresis not advanced

    assert client.get("/events?limit=200").json()[0]["id"] == res["event_id"]


def test_out_of_order_is_per_tower_not_global(client, tower_ids):
    """A tower with no prior data is never out-of-order, however old its timestamp is
    relative to some other tower's."""
    t0 = fresh_ts()
    assert post(client, [obs(tower_ids[12], t0 + timedelta(seconds=60))]).status_code == 200
    r = post(client, [obs(tower_ids[13], t0)])
    assert r.status_code == 200, r.text
    assert r.json()["out_of_order"] == 0


# --------------------------------------------------------------------- duplicates

def test_duplicate_submission_is_not_rescored(client, tower_ids):
    tower = tower_ids[20]
    ts = fresh_ts()

    first = post(client, [obs(tower, ts)])
    assert first.status_code == 200, first.text
    original_id = first.json()["results"][0]["event_id"]

    again = post(client, [obs(tower, ts)])
    assert again.status_code == 200, again.text
    body = again.json()
    assert body["received"] == 1
    assert body["scored"] == 0
    assert body["duplicates"] == 1
    res = body["results"][0]
    assert res["duplicate"] is True
    assert res["event_id"] == original_id, "a duplicate points at the ORIGINAL event"
    assert res["alert_state"] is None, "hysteresis is not re-run for a duplicate"

    original = client.get("/events?limit=2000").json()
    original_row = next(e for e in original if e["id"] == original_id)
    same_key = [e for e in original
                if e["tower_site_id"] == tower
                and e["obs_timestamp"] == original_row["obs_timestamp"]]
    assert len(same_key) == 1, "no second row was written for the duplicate"


def test_duplicate_within_a_single_batch(client, tower_ids):
    tower = tower_ids[21]
    ts = fresh_ts()
    r = post(client, [obs(tower, ts), obs(tower, ts)])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scored"] == 1
    assert body["duplicates"] == 1
    scored = [x for x in body["results"] if not x["duplicate"]][0]
    dup = [x for x in body["results"] if x["duplicate"]][0]
    assert dup["event_id"] == scored["event_id"]


def test_duplicate_key_is_tower_plus_timestamp(client, tower_ids):
    """The same timestamp at a different tower is a distinct observation, not a duplicate."""
    ts = fresh_ts()
    a = post(client, [obs(tower_ids[22], ts)])
    b = post(client, [obs(tower_ids[23], ts)])
    assert a.status_code == b.status_code == 200
    assert b.json()["duplicates"] == 0
    assert b.json()["results"][0]["event_id"] != a.json()["results"][0]["event_id"]


# --------------------------------------------------------------------- hysteresis

def test_per_tower_hysteresis_enters_alerting_after_the_streak(client, tower_ids):
    """Per-tower hysteresis, unlike replay's per-recording state: a streak at one tower must
    not move another tower's state. Skipped if the synthetic NOISY_FEATURES vector doesn't
    actually score above threshold on the shipped model -- this test is about the state
    machine, not about the model's verdict on hand-made input."""
    from api.hysteresis import ALERT_ENTER_STREAK

    probe = client.post("/score", json=NOISY_FEATURES).json()
    if probe["predicted_label"] != "attack":
        pytest.skip("NOISY_FEATURES does not score as attack on the shipped model; "
                    "hysteresis entry can't be driven through the public API here")

    tower = tower_ids[30]
    other = tower_ids[31]
    t0 = fresh_ts()

    r = post(client, [obs(tower, t0 + timedelta(seconds=i), NOISY_FEATURES)
                      for i in range(ALERT_ENTER_STREAK)])
    assert r.status_code == 200, r.text
    states = [res["alert_state"] for res in r.json()["results"]]
    assert states[-1] == "alerting"
    assert states[0] == "normal", "a single reading must not flip the state"

    clean = post(client, [obs(other, t0 + timedelta(seconds=99))])
    assert clean.json()["results"][0]["alert_state"] == "normal", \
        "one tower's streak must not leak into another tower's state"


def test_tower_hysteresis_registry_is_independent_per_tower():
    """Unit-level check of the state machine itself, with no dependence on what the model
    predicts for any particular input."""
    from api.hysteresis import TowerHysteresisRegistry, ALERT_ENTER_STREAK, ALERT_EXIT_STREAK

    reg = TowerHysteresisRegistry()
    for _ in range(ALERT_ENTER_STREAK):
        state_a = reg.update("tower-a", True)
    assert state_a == "alerting"
    assert reg.state("tower-b") == "normal"

    for _ in range(ALERT_EXIT_STREAK - 1):
        assert reg.update("tower-a", False) == "alerting", "slow to clear, by design"
    assert reg.update("tower-a", False) == "normal"


# --------------------------------------------------------------------- dashboard state

def test_ingested_events_reach_the_map_and_spatial_state(client, tower_ids):
    """The point of routing ingestion through the same store: /events/map (what the dashboard
    renders) and /spatial/autocorrelation (live Moran's I) both pick ingested rows up with no
    special-casing."""
    tower = tower_ids[40]
    r = post(client, [obs(tower, fresh_ts(), NOISY_FEATURES)])
    assert r.status_code == 200, r.text
    event_id = r.json()["results"][0]["event_id"]

    map_state = {e["tower_site_id"]: e for e in client.get("/events/map").json()}
    assert tower in map_state
    assert map_state[tower]["id"] == event_id

    auto = client.get("/spatial/autocorrelation")
    assert auto.status_code == 200
    assert auto.json()["n_towers_scored"] >= 1
