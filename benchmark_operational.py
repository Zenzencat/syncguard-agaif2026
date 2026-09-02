"""Operational realism benchmarks: scoring latency (model-only, SHAP-only, and full /score
over real HTTP) and replay throughput at representative speeds. Produces the real numbers for
OPERATIONAL_METRICS.md -- run against an already-running API (`make serve` or
`uvicorn api.main:app`) pointed at by API_BASE below. Read-only measurement: does not modify
the shipped model, the SHAP explainer, or any deployed threshold.

Usage: python benchmark_operational.py
"""
import sys
import time
import json
import statistics as stats
from pathlib import Path
import numpy as np
import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
API_BASE = os.environ.get("SYNCGUARD_BENCHMARK_API_BASE", "http://127.0.0.1:8000")
N_LATENCY_SAMPLES = 300
THROUGHPUT_SPEEDS = [10, 25, 200]  # one below LIVE_EXPLAIN_MAX_SPEED=20, one just above, one stress-test
THROUGHPUT_WINDOW_S = 15.0


def pctl(values, p):
    return float(np.percentile(values, p))


def report(label, values_ms):
    print(f"{label}: n={len(values_ms)}  "
          f"p50={pctl(values_ms,50):.2f}ms  p95={pctl(values_ms,95):.2f}ms  "
          f"p99={pctl(values_ms,99):.2f}ms  mean={stats.mean(values_ms):.2f}ms  "
          f"min={min(values_ms):.2f}ms  max={max(values_ms):.2f}ms")
    return {"n": len(values_ms), "p50_ms": pctl(values_ms, 50), "p95_ms": pctl(values_ms, 95),
            "p99_ms": pctl(values_ms, 99), "mean_ms": stats.mean(values_ms),
            "min_ms": min(values_ms), "max_ms": max(values_ms)}


def load_test_rows(n):
    """Real held-out TEST recordings -- identical selection to train_baseline_model.py /
    evaluate_models.py, so 'the test set' means the same thing here as everywhere else in
    this project."""
    df = pd.read_parquet("processed/syncguard_features.parquet")
    TEST_SELECTION = [
        ("Jamming", "1.10.6", "dynamic"), ("Jamming", "1.6.4", "stationary"),
        ("Meaconing", "3.2.8", "dynamic"), ("Meaconing", "3.2.7", "stationary"),
        ("Spoofing", "2.3.2", "dynamic"), ("Spoofing", "2.1.1", "stationary"),
        ("Spoofing + Jamming", "2.6.4", "dynamic"), ("Spoofing + Jamming", "2.6.3", "stationary"),
    ]
    test_run_ids = set()
    for atype, sid, rstate in TEST_SELECTION:
        m = df.loc[(df["attack_type"] == atype) & (df["scenario_id"] == sid) &
                    (df["rover_state"] == rstate), "run_id"].unique()
        test_run_ids.add(m[0])
    test_df = df[df["run_id"].isin(test_run_ids)]
    sample = test_df.sample(n=n, random_state=42)
    return sample


results = {}

print("=" * 78)
print("1. SCORING LATENCY")
print("=" * 78)
from api.model_service import ModelService  # noqa: E402

ms = ModelService()
print(f"Loaded {ms.model_version} from {ms.model_path}")

sample = load_test_rows(N_LATENCY_SAMPLES)
feature_rows = [{c: (None if pd.isna(r[c]) else float(r[c])) for c in ms.feature_cols}
                for _, r in sample.iterrows()]

# --- model-only, in-process (no HTTP, no SHAP, no DB write) ---
model_only_ms = []
for feats in feature_rows:
    t0 = time.perf_counter()
    ms.score(feats)
    model_only_ms.append((time.perf_counter() - t0) * 1000)
results["model_only"] = report("Model-only (in-process, no SHAP, no HTTP)", model_only_ms)

# --- SHAP-only, in-process ---
shap_only_ms = []
for feats in feature_rows:
    t0 = time.perf_counter()
    ms.explain(feats)
    shap_only_ms.append((time.perf_counter() - t0) * 1000)
results["shap_only"] = report("SHAP-only (in-process, explain() alone)", shap_only_ms)

# --- full POST /score over real HTTP (model + SHAP + correlation + DB write + serialization) ---
health = requests.get(f"{API_BASE}/health", timeout=5).json()
assert health["model_loaded"], f"API at {API_BASE} has no model loaded -- start it with `make serve` first"
print(f"API health: {health}")

http_full_ms = []
for feats in feature_rows:
    t0 = time.perf_counter()
    r = requests.post(f"{API_BASE}/score", json=feats, timeout=10)
    http_full_ms.append((time.perf_counter() - t0) * 1000)
    assert r.status_code == 200, f"unexpected status {r.status_code}: {r.text[:200]}"
results["http_full_score"] = report("Full POST /score over real HTTP (model+SHAP+correlation+DB+network)", http_full_ms)

overhead_ms = pctl(http_full_ms, 50) - pctl(model_only_ms, 50) - pctl(shap_only_ms, 50)
print(f"\nImplied overhead beyond model+SHAP (network/serialization/correlation/DB write), "
      f"at p50: {overhead_ms:.2f}ms")

print("\n" + "=" * 78)
print("2. REPLAY THROUGHPUT")
print("=" * 78)

def ensure_stopped():
    try:
        requests.post(f"{API_BASE}/replay/stop", timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"  (cleanup: replay/stop request itself failed: {e})")

for speed in THROUGHPUT_SPEEDS:
    ensure_stopped()  # guarantee a clean slate even if a prior iteration errored
    time.sleep(1)
    r = requests.post(f"{API_BASE}/replay/start", params={"speed": speed}, timeout=10)
    assert r.status_code == 200, r.text
    status0 = r.json()
    live_explain = status0["live_explain"]
    try:
        t_start = time.perf_counter()
        time.sleep(THROUGHPUT_WINDOW_S)
        t_elapsed = time.perf_counter() - t_start
        status1 = requests.get(f"{API_BASE}/replay/status", timeout=30).json()
        rows = status1["rows_replayed"] - status0["rows_replayed"]
        throughput = rows / t_elapsed
        print(f"speed={speed:>4}  live_explain={live_explain!s:5}  "
              f"rows_replayed={rows:4d} in {t_elapsed:.2f}s  ->  {throughput:.2f} rows/sec")
        results.setdefault("throughput", []).append({
            "requested_speed": speed, "live_explain": live_explain,
            "rows_replayed": rows, "elapsed_s": round(t_elapsed, 2),
            "rows_per_sec": round(throughput, 2),
        })
    finally:
        ensure_stopped()
    time.sleep(1)  # let the stopped task settle before the next run

print("\n" + "=" * 78)
print("3. CONCURRENT-REQUEST LATENCY DURING LIVE-SHAP REPLAY")
print("=" * 78)
print("SHAP's ~50ms explain() call is synchronous/CPU-bound and runs on the same asyncio")
print("event loop as HTTP request handling -- this measures how much that costs a concurrent")
print("caller, not just replay's own row throughput.")

for speed, label in [(10, "speed<=20 (live SHAP)"), (200, "speed>20 (no live SHAP)")]:
    ensure_stopped()
    time.sleep(1)
    requests.post(f"{API_BASE}/replay/start", params={"speed": speed}, timeout=10)
    time.sleep(1)  # let replay ramp up
    concurrent_ms = []
    try:
        for _ in range(50):
            t0 = time.perf_counter()
            requests.get(f"{API_BASE}/health", timeout=10)
            concurrent_ms.append((time.perf_counter() - t0) * 1000)
    finally:
        ensure_stopped()
    key = f"concurrent_health_during_replay_speed_{speed}"
    results[key] = report(f"GET /health latency while replay running ({label})", concurrent_ms)
    time.sleep(1)

# baseline: same measurement with no replay running, for direct comparison
idle_ms = []
for _ in range(50):
    t0 = time.perf_counter()
    requests.get(f"{API_BASE}/health", timeout=10)
    idle_ms.append((time.perf_counter() - t0) * 1000)
results["concurrent_health_idle_baseline"] = report("GET /health latency, no replay running (baseline)", idle_ms)

Path("processed/operational_benchmark_results.json").write_text(json.dumps(results, indent=2))
print("\nSaved processed/operational_benchmark_results.json")
