"""Minimal POST /ingest client: sends one sample batch and prints what came back.

This is the smallest honest illustration of the ingestion contract -- it is NOT an edge
agent, and it does not read a real GNSS receiver. The two feature vectors it sends are
SYNTHETIC: each field is the median of the real clean-labeled (or attack-labeled) rows in the
Jammertest dataset, so the numbers are real summary statistics but no receiver ever emitted
exactly these combinations. INGESTION_CONTRACT.md documents which raw receiver observables a
real edge agent would have to reduce, and how, to produce each of the 23 features this sends.

Usage (with the service running -- `make serve` or `docker compose up`):

    python examples/ingest_client.py
    python examples/ingest_client.py --url http://localhost:8000 --towers 3 --windows 5

Only stdlib is used, so this runs anywhere Python does, with nothing installed.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Example feature vectors.
#
# These are the per-feature MEDIANS of the real Jammertest 2024 rows in
# processed/syncguard_features.parquet -- the clean-labeled rows (n=10,349) and the
# attack-labeled rows (n=34,290) respectively. So the individual numbers are REAL summary
# statistics of real data, but neither vector is a real observation: no single receiver epoch
# ever produced exactly this combination. Treat them as SYNTHETIC inputs built from real
# statistics, which is what makes them useful as an illustration -- they land on the right
# side of the model's 0.52 threshold for the right reasons, rather than by hand-tuning.
#
# Why not hand-written "plausible" numbers: an earlier version of this file used them and
# they scored as "attack" with 0.60 confidence. The units and scales here are the raw u-blox /
# RINEX-derived ones extract_features.py produces (e.g. pDOP ~0.01, not ~1.4; hAcc in the
# tenths of a metre; n_sats_l1 ~43 because it counts every tracked L1 satellite across
# constellations, not just those used in the fix). Submitting a sensible-looking vector in
# the wrong scale gets a confident, wrong answer with no error. See INGESTION_CONTRACT.md.
# ---------------------------------------------------------------------------
NOMINAL = {
    "fixType": 3.0, "gSpeed": 0.005, "hAcc": 0.183, "vAcc": 0.324, "sAcc": 0.047,
    "headAcc": 0.18, "pDOP": 0.0106, "numSV": 32.0, "velN": -0.001, "velE": 0.0,
    "velD": 0.0, "pos_dev_m": 1.5415, "n_sats_l1": 43.0, "snr_l1_mean": 43.4762,
    "snr_l1_std": 3.8068, "snr_l1_min": 33.0, "doppler_l1_mean": -114.6735,
    "doppler_l1_std": 2325.6041, "pr_doppler_residual_mean": 0.0861,
    "pr_doppler_residual_std": 19.9222, "jam_ind_mean": 7.0, "agc_cnt_mean": 5616.0,
    "noise_per_ms_mean": 101.0,
}

DEGRADED = {
    "fixType": 3.0, "gSpeed": 0.026, "hAcc": 0.331, "vAcc": 0.613, "sAcc": 0.078,
    "headAcc": 0.1504, "pDOP": 0.0109, "numSV": 31.0, "velN": -0.001, "velE": 0.0,
    "velD": -0.001, "pos_dev_m": 2.7085, "n_sats_l1": 39.0, "snr_l1_mean": 39.4688,
    "snr_l1_std": 5.2149, "snr_l1_min": 27.0, "doppler_l1_mean": -61.3147,
    "doppler_l1_std": 2096.3066, "pr_doppler_residual_mean": 2.8694,
    "pr_doppler_residual_std": 177.2779, "jam_ind_mean": 14.0, "agc_cnt_mean": 3159.0,
    "noise_per_ms_mean": 102.0,
}


def http_json(url: str, payload: dict | None = None, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if data else "GET",
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def build_batch(tower_ids: list[str], n_windows: int, period_s: int) -> dict:
    """One observation per tower per window, on a fixed cadence, oldest first.

    Timestamps are the *observation* times, not submission times -- the service stores them
    separately from its own created_at and uses them for deduplication and out-of-order
    detection.

    The sequence starts at "now" and steps forward, rather than reaching back into the past,
    so that running this script twice against the same service produces strictly newer
    observations the second time. Reaching backwards would make the second run's early
    windows older than data the service already accepted for those towers, and they would
    (correctly) come back flagged out_of_order -- real behaviour, but it obscures what this
    example is trying to show. A real collector submits genuinely past timestamps and is
    expected to submit them in roughly increasing order per tower.
    """
    t0 = datetime.now(timezone.utc).replace(microsecond=0)
    observations = []
    for w in range(n_windows):
        ts = (t0 + timedelta(seconds=period_s * w)).isoformat()
        for i, tower_id in enumerate(tower_ids):
            # One tower switches to the degraded vector partway through, so the printed
            # output shows the per-tower hysteresis state machine actually entering
            # 'alerting' while the other towers stay 'normal'.
            degraded = (i == 0 and w >= max(n_windows - 4, 0))
            observations.append({
                "tower_id": tower_id,
                "timestamp": ts,
                **(DEGRADED if degraded else NOMINAL),
            })
    return {"batch_id": f"example-{uuid.uuid4().hex[:8]}", "observations": observations}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000", help="SyncGuard base URL")
    ap.add_argument("--towers", type=int, default=3, help="How many real towers to feed")
    ap.add_argument("--windows", type=int, default=5, help="Observation windows per tower")
    ap.add_argument("--period", type=int, default=30,
                    help="Seconds between observation windows")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    try:
        health = http_json(f"{base}/health")
    except urllib.error.URLError as e:
        print(f"Could not reach {base}/health: {e}\n"
              f"Start the service first: `make serve` or `docker compose up --build`.")
        return 1
    if not health.get("model_loaded"):
        print(f"Service is up but has no model loaded ({health}). Run `make train`.")
        return 1
    print(f"Connected: {base}  model_version={health['model_version']}  "
          f"towers_loaded={health['towers_loaded']}")

    # Tower ids come from the service's own /towers, never hardcoded -- tower_key is the
    # unique key (16 of the 136 real towers share the site_id "tbg", see api/spatial.py).
    towers = http_json(f"{base}/towers")
    tower_ids = [t["tower_key"] for t in towers[: args.towers]]
    print(f"Feeding {len(tower_ids)} tower(s): {', '.join(tower_ids)}")

    batch = build_batch(tower_ids, args.windows, args.period)
    print(f"POST /ingest  batch_id={batch['batch_id']}  "
          f"observations={len(batch['observations'])}  [features: SYNTHETIC, built from real clean/attack medians]")

    try:
        resp = http_json(f"{base}/ingest", batch)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}")
        return 1

    print(f"\nreceived={resp['received']}  scored={resp['scored']}  "
          f"duplicates={resp['duplicates']}  out_of_order={resp['out_of_order']}  "
          f"reordered_in_batch={resp['reordered_in_batch']}  "
          f"live_explain={resp['live_explain']}")

    alerting = [r for r in resp["results"] if r.get("alert_state") == "alerting"]
    print(f"\nALERTS (per-tower hysteresis in state 'alerting'): {len(alerting)}")
    if alerting:
        print(f"  {'event':>7}  {'tower':<14} {'sev':>6} {'prob':>6}  label")
        for r in alerting:
            print(f"  {r['event_id']:>7}  {r['tower_id']:<14} {r['severity']:>6.3f} "
                  f"{r['probability']:>6.3f}  {r['predicted_label']}")
    else:
        print("  none -- no tower accumulated enough consecutive above-threshold readings "
              "(ALERT_ENTER_STREAK, see api/hysteresis.py)")

    print(f"\nLast {min(5, len(resp['results']))} results:")
    print(f"  {'event':>7}  {'tower':<14} {'sev':>6} {'prob':>6}  {'label':<7} state")
    for r in resp["results"][-5:]:
        print(f"  {str(r['event_id']):>7}  {r['tower_id']:<14} "
              f"{(r['severity'] or 0):>6.3f} {(r['probability'] or 0):>6.3f}  "
              f"{str(r['predicted_label']):<7} {r['alert_state']}")

    print(f"\nThese events are now in /events, /events/map and the live spatial statistics,\n"
          f"exactly like replayed ones. Open {base}/dashboard to see them on the map.")

    # Re-submitting the identical batch demonstrates deduplication: the service answers with
    # the ORIGINAL event ids and scores nothing again.
    again = http_json(f"{base}/ingest", batch)
    print(f"\nRe-submitted the same batch: scored={again['scored']} "
          f"duplicates={again['duplicates']}  (dedup key is tower_id + timestamp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
