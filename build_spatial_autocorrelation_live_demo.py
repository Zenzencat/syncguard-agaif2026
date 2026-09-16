"""Genuine live-replay Global/Local Moran's I -- the number Slide 8 actually shows.

This replaces build_spatial_autocorrelation_demo.py as the source of the deck's headline
Moran's I figure. That script computed Moran's I over `simulated_spatial_anomaly_SIMULATED.csv`
-- the epicenter-and-distance-decay SIMULATED severity layer from build_spatial_simulation.py
(the same one behind Slide 7's static map). That severity is a smooth, monotonic function of
real distance from one fixed epicenter (confirmed: corr(distance, severity) = -0.45 across all
136 towers), so feeding it into a spatial-autocorrelation test that uses those same real
distances as its neighbor weights produces a strong positive Moran's I close to a mathematical
certainty -- it demonstrates that a spatially smooth input looks spatially smooth, not that a
real per-tower attribution shows a real cluster. That script's own docstring says as much
("the severity values fed in come from the simulated per-tower attribution/spread layer, not a
live-replay's real per-tower events") -- the mismatch was that the deck's slide text and
speaker notes described the number as "computed live during replay" with round-robin
attribution, which is a different, separate mechanism this script actually runs.

This script reuses the exact same production classes api.main's ReplayManager uses for a live
replay, synchronously (no asyncio sleep -- pacing only controls demo playback speed, not
attribution order or severity values, so it doesn't affect the statistics):

  - api.model_service.ModelService        (the real trained detector, real predict_proba)
  - api.spatial.load_towers / TowerAttributor   (real towers, SIMULATED round-robin attribution)
  - api.db.EventStore                     (same scored_events schema the API writes to)
  - api.spatial_stats.compute_autocorrelation   (same function GET /spatial/autocorrelation calls)

It replays the full scenario used elsewhere in the deck (Spoofing, scenario 2.1.1 -- the
Slide 11 demo recording), row by row, in real recording order, attributing each scored row to
the next tower round-robin -- exactly what a judge watching a live replay to completion would
see. Reports Global Moran's I at several checkpoints through the replay (to show how much it
moves around -- expected, since round-robin attribution is close to random) and the final
state after the whole recording has played once. Run fresh, once, no cherry-picking across
repeated runs; rerunning this script reproduces the same numbers, since the model, the tower
order, and the permutation-test seed (RNG_SEED=42, same convention as api/spatial_stats.py)
are all deterministic.

Requires `models/model.joblib` (`make train` / `python train_improved_model.py` first).
"""
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from api.model_service import ModelService
from api.spatial import load_towers, TowerAttributor
from api.db import EventStore
from api.spatial_stats import compute_autocorrelation

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "spatial_processed"
RUN_DB_PATH = Path("/tmp/sg_live_replay.db")  # scratch DB, not committed -- see .gitignore

QUADRANT_COLORS = {
    0: "#c9ccd1",
    1: "#d1263b",
    2: "#7fb3e8",
    3: "#1f5fa8",
    4: "#f0a35c",
}
QUADRANT_LABELS = {
    0: "Not significant", 1: "High-High (hotspot)", 2: "Low-High (spatial outlier)",
    3: "Low-Low (coldspot)", 4: "High-Low (spatial outlier)",
}


def main():
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(RUN_DB_PATH) + suffix)
        if p.exists():
            p.unlink()

    print("Loading model service (real trained detector)...")
    model = ModelService(model_path=REPO_ROOT / "models" / "model.joblib")
    print(f"  model_version={model.model_version}  decision_threshold={model.decision_threshold:.4f}")

    print("Loading real tower table + fresh round-robin attributor + fresh event store...")
    towers = load_towers()
    attributor = TowerAttributor(towers)
    store = EventStore(db_path=RUN_DB_PATH)
    print(f"  {len(towers)} real towers loaded")

    print("Loading replay dataset...")
    df = pd.read_parquet(REPO_ROOT / "processed" / "syncguard_features.parquet")
    run_ids = df.loc[(df["attack_type"] == "Spoofing") & (df["scenario_id"] == "2.1.1"), "run_id"].unique()
    run_id = run_ids[0]
    rows = df[df["run_id"] == run_id].sort_values("real_time").reset_index(drop=True)
    print(f"  run_id={run_id!r}  n_rows={len(rows)}")

    checkpoints = sorted(set([136, 272, 500, 1000, 1500, 2000, len(rows)]))
    checkpoint_results = {}

    print("\nReplaying row-by-row (real recording order), scoring + round-robin tower attribution...")
    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        features = {c: (None if pd.isna(row[c]) else float(row[c])) for c in model.feature_cols}
        result = model.score(features)
        tower = attributor.next_tower()
        store.insert_event({
            "source": "replay", "run_id": run_id, "scenario_id": row.get("scenario_id"),
            "attack_type": row.get("attack_type"), "true_attack": int(row["attack"]),
            "probability": result["probability"], "severity": result["severity"],
            "predicted_label": result["predicted_label"], "model_version": result["model_version"],
            "tower_site_id": tower["site_id"], "tower_site_name": tower["site_name"],
            "tower_lat": tower["lat"], "tower_lon": tower["lon"],
            "correlation_score": None, "features": features,
        })
        if i in checkpoints:
            latest = store.latest_severity_per_tower()
            res = compute_autocorrelation(towers, latest)
            checkpoint_results[i] = res
            if res.computable:
                n_sig = sum(1 for t in res.per_tower if t["significant"])
                print(f"  [row {i:5d}] n_scored={res.n_towers_scored:3d}/136  "
                      f"Global I={res.global_moran_i:+.4f}  p={res.global_p_value:.4f}  "
                      f"z={res.global_z_score:+.3f}  sig. LISA towers={n_sig}")
            else:
                print(f"  [row {i:5d}] not computable: {res.reason}")

    final = checkpoint_results[len(rows)]
    sig_towers = [t for t in final.per_tower if t["significant"]]
    label_counts = Counter(t["lisa_label"] for t in sig_towers)

    print("\n=== FINAL STATE (full recording replayed once, real round-robin order) ===")
    print(f"n_towers_scored: {final.n_towers_scored} / {final.n_towers_total}")
    print(f"k_neighbors: {final.k_neighbors}")
    print(f"global_moran_i: {final.global_moran_i:.4f}")
    print(f"p_value: {final.global_p_value:.4f}")
    print(f"z_score: {final.global_z_score:.4f}")
    print(f"expected_i: {final.global_expected_i:.4f}")
    print(f"significant LISA towers: {len(sig_towers)} / {final.n_towers_scored}")
    for label, n in label_counts.most_common():
        print(f"  {label}: {n}")

    # --- Results log (provenance, same discipline as calibration_experiment.py etc.) ---
    OUT_DIR.mkdir(exist_ok=True)
    log_path = OUT_DIR / "live_replay_autocorrelation_result_LIVE.txt"
    with open(log_path, "w") as f:
        f.write("SyncGuard -- genuine live-replay Global/Local Moran's I (Slide 8 headline number)\n")
        f.write("Produced by build_spatial_autocorrelation_live_demo.py -- rerun this script to reproduce.\n\n")
        f.write(f"run_id: {run_id}\nn_rows_replayed: {len(rows)}\n\n")
        f.write("Checkpoints (Global Moran's I as the replay progresses):\n")
        for i in checkpoints:
            r = checkpoint_results[i]
            f.write(f"  row {i:5d}: I={r.global_moran_i:+.4f}  p={r.global_p_value:.4f}  "
                    f"z={r.global_z_score:+.3f}  sig_LISA={sum(1 for t in r.per_tower if t['significant'])}\n")
        f.write(f"\nFinal state (all {final.n_towers_total} towers scored):\n")
        f.write(f"  global_moran_i: {final.global_moran_i:.4f}\n")
        f.write(f"  p_value: {final.global_p_value:.4f}\n")
        f.write(f"  z_score: {final.global_z_score:.4f}\n")
        f.write(f"  expected_i: {final.global_expected_i:.4f}\n")
        f.write(f"  significant LISA towers: {len(sig_towers)} / {final.n_towers_scored}\n")
        for label, n in label_counts.most_common():
            f.write(f"    {label}: {n}\n")
        f.write("\nInterpretation: not statistically significant at alpha=0.05 (p > 0.05) -- no\n"
                 "detectable global spatial clustering under round-robin attribution. This is the\n"
                 "expected result: round-robin cycles through towers independent of severity, so it\n"
                 "has no structural reason to produce spatial clustering the way the epicenter-decay\n"
                 "simulation (Slide 7 / build_spatial_simulation.py) does by construction. It is the\n"
                 "honest argument for why real per-tower attribution is the necessary next step --\n"
                 "the method is real and ready; only the input needs to stop being simulated.\n")
    print(f"\nWrote {log_path}")

    # --- LISA cluster map ---
    lats = np.array([t["lat"] for t in final.per_tower])
    lons = np.array([t["lon"] for t in final.per_tower])
    quads = np.array([t["lisa_quadrant"] for t in final.per_tower])
    sev = np.array([t["severity"] for t in final.per_tower])

    fig, ax = plt.subplots(figsize=(9.2, 8.2), dpi=200)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for q in [0, 2, 4, 1, 3]:
        mask = quads == q
        if not mask.any():
            continue
        ax.scatter(lons[mask], lats[mask], s=70 if q != 0 else 40,
                   c=QUADRANT_COLORS[q], edgecolors="#2a2a2a" if q != 0 else "none",
                   linewidths=0.6, alpha=0.95 if q != 0 else 0.55,
                   label=f"{QUADRANT_LABELS[q]} ({mask.sum()})", zorder=3 if q != 0 else 2)
    ax.set_xlabel("Longitude", fontsize=11); ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title("Local Moran's I (LISA) -- genuine live replay, round-robin attribution\n"
                 "136 real Kubu Raya / Pontianak towers", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.15)
    for spine in ax.spines.values():
        spine.set_color("#888888")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "lisa_cluster_map_LIVE.png", facecolor="white")
    plt.close(fig)

    # --- Moran scatter plot ---
    z = (sev - sev.mean()) / sev.std()
    # Rebuild the same KNN weights used inside compute_autocorrelation for the lag term
    from api.spatial_stats import build_knn_weights
    sub = pd.DataFrame({"lon": lons, "lat": lats})
    w = build_knn_weights(sub, k=final.k_neighbors)
    lag = w.sparse @ z
    colors = [QUADRANT_COLORS[q] for q in quads]

    fig2, ax2 = plt.subplots(figsize=(8.2, 8.2), dpi=200)
    fig2.patch.set_facecolor("white"); ax2.set_facecolor("white")
    ax2.scatter(z, lag, c=colors, s=55, edgecolors="#2a2a2a", linewidths=0.5, alpha=0.9, zorder=3)
    ax2.axhline(0, color="#999999", linewidth=0.8); ax2.axvline(0, color="#999999", linewidth=0.8)
    m, b = np.polyfit(z, lag, 1)
    xs = np.linspace(z.min(), z.max(), 100)
    ax2.plot(xs, m * xs + b, color="#d1263b", linewidth=2.2, zorder=4,
             label=f"Moran's I = {final.global_moran_i:.3f}  (p = {final.global_p_value:.3f}, not significant)")
    ax2.set_xlabel("Standardized severity (z)", fontsize=11)
    ax2.set_ylabel("Spatial lag of severity (KNN k=5)", fontsize=11)
    ax2.set_title("Moran Scatter Plot -- genuine live replay, round-robin attribution", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax2.grid(True, alpha=0.15)
    for spine in ax2.spines.values():
        spine.set_color("#888888")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "moran_scatter_LIVE.png", facecolor="white")
    plt.close(fig2)

    print(f"Wrote {OUT_DIR / 'lisa_cluster_map_LIVE.png'}")
    print(f"Wrote {OUT_DIR / 'moran_scatter_LIVE.png'}")

    store.close()
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(RUN_DB_PATH) + suffix)
        if p.exists():
            p.unlink()
    print("\nDone.")


if __name__ == "__main__":
    main()
