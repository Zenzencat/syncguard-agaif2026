"""Regression guard for the endpoints that existed before the ingestion adapter.

Phase 1 split the 23 feature fields out of TelemetryInput into a shared FeatureVector base
(api/schemas.py) so /score and /ingest cannot drift apart. That is a refactor of a public
request schema, so these tests pin down that /score's accepted payload, its response shape,
and the surrounding endpoints are unchanged by it.
"""
from conftest import requires_model

pytestmark = requires_model

FEATURES = {
    # Median of the real clean-labeled rows in processed/syncguard_features.parquet -- see
    # tests/test_ingest.py for why these are used instead of hand-written values.
    "fixType": 3.0, "gSpeed": 0.005, "hAcc": 0.183, "vAcc": 0.324, "sAcc": 0.047,
    "headAcc": 0.18, "pDOP": 0.0106, "numSV": 32.0, "velN": -0.001, "velE": 0.0,
    "velD": 0.0, "pos_dev_m": 1.5415, "n_sats_l1": 43.0, "snr_l1_mean": 43.4762,
    "snr_l1_std": 3.8068, "snr_l1_min": 33.0, "doppler_l1_mean": -114.6735,
    "doppler_l1_std": 2325.6041, "pr_doppler_residual_mean": 0.0861,
    "pr_doppler_residual_std": 19.9222, "jam_ind_mean": 7.0, "agc_cnt_mean": 5616.0,
    "noise_per_ms_mean": 101.0,
}


def test_health(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["towers_loaded"] == 136  # the real Telkomsel tower table
    assert body["status"] == "ok"


def test_towers_are_the_136_real_ones(client):
    towers = client.get("/towers").json()
    assert len(towers) == 136
    assert len({t["tower_key"] for t in towers}) == 136, "tower_key must be unique"


def test_score_still_accepts_its_original_payload(client):
    r = client.post("/score", json=FEATURES)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"probability", "severity", "predicted_label", "decision_threshold",
                         "model_version", "event_id", "top_features"}
    assert body["decision_threshold"] == 0.5200000000000002  # shipped threshold, unchanged
    assert len(body["top_features"]) == 5


def test_score_still_accepts_its_optional_context_fields(client, tower_ids):
    """receiver_id / tower_site_id live on TelemetryInput, not on the shared FeatureVector
    base -- they must still be accepted here and must still drive spatial correlation."""
    r = client.post("/score", json={**FEATURES, "receiver_id": "rx-1",
                                    "tower_site_id": tower_ids[0]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tower"]["site_id"] == tower_ids[0]
    assert body["correlation"] is not None


def test_score_rejects_an_unknown_tower(client):
    r = client.post("/score", json={**FEATURES, "tower_site_id": "nope"})
    assert r.status_code == 422


def test_ingest_only_fields_are_not_accepted_by_score(client, tower_ids):
    """tower_id/timestamp belong to IngestObservation. Pydantic ignores unknown fields by
    default, so this asserts they are ignored rather than silently repurposed -- /score must
    not start attributing events from an ingest-shaped payload."""
    r = client.post("/score", json={**FEATURES, "tower_id": tower_ids[0],
                                    "timestamp": "2075-01-01T00:00:00Z"})
    assert r.status_code == 200, r.text
    assert r.json()["tower"] is None


def test_replay_runs_listed(client):
    runs = client.get("/replay/runs").json()
    assert runs and all("run_id" in r for r in runs)


def test_events_and_map(client):
    assert client.get("/events?limit=5").status_code == 200
    assert client.get("/events/map").status_code == 200


def test_explain_unknown_event_is_404(client):
    assert client.get("/events/99999999/explain").status_code == 404


def test_dashboard_is_served(client):
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "SyncGuard" in r.text


def test_dashboard_assets_are_local_and_served(client):
    dashboard = client.get("/dashboard").text
    assert 'src="/assets/plotly-2.35.2.min.js"' in dashboard
    assert "https://cdn.plot.ly" not in dashboard

    plotly = client.get("/assets/plotly-2.35.2.min.js")
    assert plotly.status_code == 200
    assert "plotly.js v2.35.2" in plotly.text[:200]
    assert "Licensed under the MIT license" in plotly.text[:300]

    basemap = client.get("/assets/offline_basemap.geojson")
    assert basemap.status_code == 200
    body = basemap.json()
    assert body["type"] == "FeatureCollection"
    assert body["attribution"] == "Natural Earth, public domain"
    assert body["features"], "the offline land/coastline layer must not be empty"
