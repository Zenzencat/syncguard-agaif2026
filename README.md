# SyncGuard

**GNSS spoofing / jamming / meaconing detection for telecom base-station timing infrastructure.**

*AGAIF 2026 Hackathon — Cybersecurity Track — Team Lorem Ipsum (HACK_TH_014, Thailand)*

4G/5G base stations depend on GNSS-disciplined oscillators (GNSSDOs) for the phase/frequency
sync that TDD, inter-cell coordination, and handover require. GNSS spoofing feeds a receiver a
false time and silently degrades the network — frame misalignment, inter-cell interference,
handover failures — with no hard-outage trigger to diagnose. SyncGuard is a per-site detector
that flags the *statistical signature* of an attack from receiver observables already exposed by
off-the-shelf GNSS modules (signal quality, Doppler behaviour, RF-interference indicators,
solution-quality metrics) — a **retrofit-first** approach that needs no cryptographic signal
authentication and no GNSSDO hardware change.

Full problem framing, methodology, impact, and AI-usage disclosure: [`project_abstract.md`](project_abstract.md).

---

## What the system does

Not just a model — a running service with an evidence trail:

| Capability | Where | Notes |
|---|---|---|
| **Trained detector** | `train_baseline_model.py`, `train_improved_model.py` | 23-feature RandomForest, attack-vs-clean per ~1 Hz epoch. Split by recording, decision threshold tuned by GroupKFold CV. Real metrics + honest limitations in [`baseline_model_report.md`](baseline_model_report.md) / [`improved_model_report.md`](improved_model_report.md). |
| **FastAPI serving layer** | `api/main.py` | `POST /score` returns probability + severity + SHAP explanation for one telemetry reading. `GET /health`. Sub-100 ms p50 end-to-end over HTTP (measured — [`OPERATIONAL_METRICS.md`](OPERATIONAL_METRICS.md)). |
| **Streaming replay** | `api/replay.py`, `GET /stream/events` (SSE) | Replays a recording row-by-row in real recording-time order through the same scoring path, as an asyncio task in-process — no Kafka, no broker. Drives the live demo. |
| **Per-prediction explainability (SHAP)** | `api/model_service.py`, `GET /events/{id}/explain` | Exact Tree SHAP over the shipped forest, top-5 features by signed contribution. Always computed for `/score`; speed-gated during replay with lazy on-demand fallback. [`SHAP_EXPLAINABILITY.md`](SHAP_EXPLAINABILITY.md). |
| **Real spatial statistics** | `api/spatial_stats.py`, `GET /spatial/autocorrelation` | Global Moran's I + Local Moran's I (LISA) via PySAL (`esda`/`libpysal`) over the live tower network — "is the anomaly pattern spatially clustered right now" + per-tower hotspot/outlier classification. [`SPATIAL_STATISTICS.md`](SPATIAL_STATISTICS.md). |
| **Live distance-weighted correlation** | `api/spatial.py` | Per-event: haversine-distance-weighted aggregate of *other real towers'* recent scored severities in a real time window. |
| **Alert hysteresis** | `api/hysteresis.py` | A tower shows `alerting` only after 3 consecutive above-threshold readings, clears after 5 below — debounced over the recording's real temporal order so one flicker never spams. [`OPERATIONAL_METRICS.md`](OPERATIONAL_METRICS.md) §4. |
| **SQLite persistence** | `api/db.py` | Every scored event (from `/score` or replay) → `data/syncguard.db`, with features, SHAP explanation, and debounced `alert_state`. |
| **Live dashboard** | `syncguard_interactive_summary.html`, `GET /dashboard` | Tower map + event log via SSE, Global Moran's I / p-value headline, LISA cluster rings, hysteresis alert diamonds, click-any-event-to-explain panel. The lower sections work fully offline. |
| **Docker deployment** | `Dockerfile`, `docker-compose.yml` | One command; trains the model at image-build time from the committed feature table; SQLite on a named volume. |

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Model-loaded status, tower count, replay-running flag |
| `POST` | `/score` | Score one telemetry reading → probability, severity, label, SHAP top-features; optional live spatial correlation if `tower_site_id` supplied |
| `GET` | `/events/{id}/explain` | On-demand SHAP for an already-scored event (cached after first call) |
| `GET` | `/towers` | The 136 real tower rows |
| `GET` | `/events?limit=` | Recent scored events |
| `GET` | `/events/map` | Latest event per tower — current map state, with `alert_state` |
| `GET` | `/spatial/autocorrelation` | Global Moran's I + z/p-value + per-tower LISA table, computed fresh from current state |
| `GET` | `/replay/runs` | The 24 replayable recordings |
| `POST` | `/replay/start?run_id=&speed=` | Start streaming replay |
| `POST` | `/replay/stop` · `GET` `/replay/status` | Stop / status (incl. `live_explain`, `alert_state`) |
| `GET` | `/stream/events` | Server-sent-events feed of scored replay events |
| `GET` | `/dashboard` | Serves the interactive HTML |

---

## Quick start

### Docker (the demo path — one command)

```bash
git clone https://github.com/Zenzencat/syncguard-agaif2026.git
cd syncguard-agaif2026
docker compose up --build
```

The image trains both model artifacts at build time from the committed feature table
(`processed/syncguard_features.parquet`) and the committed tower CSV — **the ~1.4 GB raw
dataset is not needed to run the service.** Once healthy, open
**<http://localhost:8000/dashboard>**, click **Connect → Start replay**. SQLite persists to a
named volume across restarts.

### Local (venv)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate  on macOS/Linux
make setup                        # pip install -r requirements.txt -r requirements-api.txt
make train                        # train_baseline_model.py + train_improved_model.py -> models/
make serve                        # uvicorn api.main:app --host 0.0.0.0 --port 8000
```

No `make` on Windows outside Git Bash/WSL — run the commands from the [`Makefile`](Makefile)
directly.

### Hitting the API

```bash
curl -s localhost:8000/health

# score one reading (any subset of the 23 features; missing ones are median-imputed)
curl -s -X POST localhost:8000/score -H 'content-type: application/json' \
  -d '{"snr_l1_mean": 34.0, "agc_cnt_mean": 4200, "jam_ind_mean": 60, "pr_doppler_residual_std": 180}'

# start a fast replay and watch the stream
curl -s -X POST "localhost:8000/replay/start?speed=25"
curl -sN localhost:8000/stream/events
curl -s localhost:8000/spatial/autocorrelation
```

---

## Architecture

```
                    train_baseline_model.py            (23-feat RandomForest, split by recording,
                    train_improved_model.py             threshold tuned by 4-fold GroupKFold CV)
                              │
                              ▼
              models/model.joblib   ──────────────  pipeline + feature list + decision threshold
              (+ model_baseline.joblib)              + severity floor/ceiling + version
                              │
                   ┌──────────┴───────────┐
                   ▼                      ▼
        ModelService.score()      shap.TreeExplainer      (loaded once at startup;
        (n_jobs=1, deterministic)  (exact Tree SHAP)       RF forced single-threaded for
                   │                      │                 run-to-run reproducibility)
                   └──────────┬───────────┘
                              ▼
                        api/main.py  (FastAPI)
             ┌────────────────┼───────────────────────────┐
             ▼                ▼                           ▼
     POST /score      ReplayManager  (asyncio task)   GET /spatial/autocorrelation
             │         reads processed parquet,             │
             │         scores row-by-row                    │  KNN weights over 136 real
             ▼                │                              │  towers → global + local
       EventStore  ◄──────────┤ writes every event           │  Moran's I  (esda/libpysal)
       (SQLite,   ────────────┤                              │
        data/*.db) │          ▼                              │
             │     │     EventBus ──► GET /stream/events (SSE)│
             │     │                                          │
             ▼     ▼                                          ▼
     LiveCorrelationEngine  (distance-weighted        syncguard_interactive_summary.html
      aggregate of recent real-tower severities)       (polls /events/map + /spatial/*,
                                                        subscribes /stream/events)
```

Two model artifacts are built: `model_baseline.joblib` (threshold 0.50, untuned) and
`model.joblib` (threshold 0.52, **what the API serves**). `ModelService` falls back to the
baseline with a loud warning if the tuned artifact is missing.

The offline spatial layer (`build_spatial_simulation.py` → `spatial_processed/`) is a separate,
older view kept for the static dashboard sections — see *REAL vs SIMULATED* below.

---

## The model

**RandomForest** (300 trees, `class_weight='balanced'`, median imputation), 23 features, one
row per **~1 Hz epoch** — the feature table is built on the receiver's ~1 Hz PVT-solution
timeline (`nav_pvt.csv`); the ~5 Hz per-satellite `rinex.csv` observables are aggregated per
epoch and merge-asof'd onto it. Features: per-epoch C/N0 stats, Doppler mean/std, a
code-Doppler pseudorange-rate residual (ephemeris-free spoofing indicator), u-blox RF-monitor
fields (AGC, noise floor, jamming indicator), receiver solution-quality metrics, and a
stationary-only position-deviation proxy. Scenario metadata (attack type, receiver state,
band) is **excluded** — a deployed detector wouldn't have it.

**Evaluation** — split by *recording*, not by row (rows within a recording are ~1 Hz samples of
one autocorrelated signal). 16 recordings / 30,562 rows train; 8 held-out recordings / 14,077
rows test (one dynamic + one stationary per attack type). Numbers below are recomputed directly
from the persisted artifacts by [`evaluate_models.py`](evaluate_models.py):

| Held-out TEST (8 recordings) | Baseline (t=0.50) | **Shipped (t=0.52)** |
|---|---|---|
| Accuracy | 0.875 | **0.877** |
| Clean-class recall | 0.642 | **0.667** |
| False positive rate | 0.358 | **0.333** |
| Attack recall | 0.942 | **0.937** |
| ROC-AUC / PR-AUC | 0.916 / 0.968 | **0.916 / 0.968** |
| Jamming recall | 0.872 | **0.859** |
| Meaconing recall | 0.965 | **0.965** |
| Spoofing recall | 0.973 | **0.971** |
| Spoofing + Jamming recall | 0.985 | **0.985** |

The threshold move (0.50 → 0.52) trades a little attack recall for a −2.5 pt false-positive
rate, chosen by 4-fold GroupKFold CV subject to jamming recall not dropping — [`ROBUSTNESS_NOTES.md`](ROBUSTNESS_NOTES.md).

### Known limitations — stated, not smoothed

- **Clean-class recall (~67%) is well below attack recall (~94%).** Under proper 4-fold
  GroupKFold it swings from 0.36 to 0.79 depending on which recordings are held out (mean
  0.66 ± 0.20) — the honest confidence range, driven by session-to-session variation in each
  recording's quiet RF baseline ([`CALIBRATION_NOTES.md`](CALIBRATION_NOTES.md) §1, [`FOLD_ANALYSIS.md`](FOLD_ANALYSIS.md)).
- **The model leans on u-blox's own RF-monitor fields** (AGC, noise floor, jamming indicator)
  more than the purpose-built spoofing-specific features. It's partly re-deriving signal
  commodity firmware already exposes.
- **These are data limitations, not modelling gaps** — established across six independent
  experiments (next section).

---

## REAL vs SIMULATED — the boundary, stated plainly

This is a strength of the submission and is labelled everywhere it surfaces (API responses, DB
column comments, dashboard UI, every `_SIMULATED`-suffixed file).

**REAL:**

- **Signal-level ground truth** — the detector is trained and evaluated on real u-blox GNSS
  receiver logs recorded under *real* jamming / spoofing / meaconing attacks at a controlled
  interference test range (JammerTest 2024, Andøya Space Defense, Bleik, Norway). Not synthetic.
- **Tower geometry** — 136 real PT. Telkomsel base-station sites (Kubu Raya & Pontianak, West
  Kalimantan) — coordinates, site IDs, names, from the AGAIF bootcamp's Module 6 materials.
- **Severity scale** — anchored to the detector's own `predict_proba()` on the held-out set:
  floor 0.43 (median clean-row probability), ceiling 0.993 (90th-percentile attack-row
  probability). Not invented numbers.
- **Spatial machinery** — haversine / great-circle distances, k-NN spatial weights, and global
  + local Moran's I are standard PySAL implementations applied as the literature defines them,
  over real coordinates and real scored severities.

**SIMULATED (and labelled):**

- **Per-event tower attribution** — which physical tower a given scored event "comes from."
  The dataset is a *single receiver's* log, so there is no real per-tower mapping; events are
  assigned to real towers by deterministic round-robin. Every API response and DB row says so.
- **The distance-decay weighting** (`decay_km = 2.0`) — a documented simplifying assumption for
  "nearby infrastructure sharing correlated timing anomalies," not a measured RF-propagation
  model.
- **`build_spatial_simulation.py`'s offline layer** — a fixed simulated epicenter and a
  simulated spread over four illustrative time steps, on real coordinates.

**Framing gap, carried explicitly into the roadmap:** this is *receiver-level* GNSS data, not
base-station GNSSDO timing data, and no public dataset of the latter under spoofing exists. A
base-station receiver is also *stationary*, while ~half the dataset is vehicle-mounted — see
[`STATIONARY_SCOPE.md`](STATIONARY_SCOPE.md) and [`project_abstract.md`](project_abstract.md) §3.

---

## The experiment record — six negative results, on purpose

The shipped detector's limitations above prompted six independent attempts to improve it, each
held to the same bar: a genuine improvement must survive **both** a rotating 4-fold GroupKFold
*and* the fixed held-out TEST split, with no jamming-recall regression. **All six were
rejected** — and consistently, for the same reason: with 24 recordings (~5 per attack type),
there isn't enough independent data to learn a feature- or tuning-level change that generalizes
across recording configurations. Each idea looks good on one evaluation and a *different*
recording breaks it on the other.

| Attempt | Notes file | Outcome |
|---|---|---|
| Jamming sample reweighting | [`ROBUSTNESS_NOTES.md`](ROBUSTNESS_NOTES.md) | Made TEST clean *and* jamming recall worse |
| Probability calibration (isotonic / sigmoid) | [`CALIBRATION_NOTES.md`](CALIBRATION_NOTES.md) | Isotonic bought clean recall at −10.5 pt jamming recall |
| Session-relative feature normalization | [`NORMALIZATION_NOTES.md`](NORMALIZATION_NOTES.md) | Oracle version −33.6 pt jamming on the target fold; deployable version did nothing |
| XGBoost vs RandomForest | [`GBM_COMPARISON.md`](GBM_COMPARISON.md) | −9.5 pt jamming recall for +3.1 pt clean recall |
| Carrier-phase / L2 / per-constellation features | [`SPOOFING_FEATURES.md`](SPOOFING_FEATURES.md) + [`STATIONARY_SCOPE.md`](STATIONARY_SCOPE.md) | Cleared GroupKFold; −9.2 pt jamming / −8.6 pt clean on the fixed split. Stationary-only re-scope blocked: 1 of 7 jamming recordings is stationary. |
| Per-satellite temporal-coherence features | [`TEMPORAL_COHERENCE.md`](TEMPORAL_COHERENCE.md) | Null on GroupKFold; −5 pt clean recall on the fixed split |

Plus [`FOLD_ANALYSIS.md`](FOLD_ANALYSIS.md) — a root-cause diagnostic of the worst CV fold.

This is a differentiator, not an apology: the ceiling was **established rigorously**, with a
consistent bar and reproducible scripts, rather than assumed. The productive next step is more
data — more recordings, more power/band configurations per attack type — not more features.

---

## Data

**Dataset** — *GNSS Dataset Under Jamming, Spoofing, and Meaconing Conditions (JammerTest
2024)*, Sayyaf, Ortiz & Renaudin (2025), Zenodo v3:
**<https://doi.org/10.5281/zenodo.15911589>**. Download `GNSS_DATASET_JAMMING_SPOOFING.tar.gz`
(375 MB compressed, ~1.4 GB extracted) and extract so `Jamming/`, `Spoofing/`, `Meaconing/`,
`Jamming+Spoofing/` land directly under `raw/` at the repo root. 24 scenarios (Jamming 7,
Spoofing 8, Meaconing 5, Jamming+Spoofing 4), u-blox receiver logs — per-satellite
pseudorange/carrier-phase/Doppler/SNR, ~1 Hz PVT solutions, RF-monitor telemetry.
`extract_features.py` reads the tree from `./raw/` at the repo root. (The standalone
feature-experiment extractors — `extract_spoofing_features.py`, `extract_temporal_features.py`
— also honour `$SYNCGUARD_RAW_ROOT`.)

**Committed so the service runs without the raw download:**
`processed/syncguard_features.parquet` (~5 MB, 44,639 feature rows) and
`spatial_raw/Module 6_AD1002_Dataset (Tower)/menaratelepon_ar_50k.csv` (the real tower
coordinates). Everything else in `processed/` and `models/` is regenerated by the scripts —
see [`.gitignore`](.gitignore).

**To reproduce from scratch:**

```bash
python extract_features.py        # raw scenario logs -> processed/syncguard_features.{csv,parquet}
python sanity_check.py            # stats + the clean-vs-spoofed sanity plot
python train_baseline_model.py    # -> models/model_baseline.joblib, baseline_model_report.md
python train_improved_model.py    # -> models/model.joblib, improved_model_report.md
python evaluate_models.py         # recompute the TEST-set table above from the artifacts
python build_spatial_simulation.py  # offline SIMULATED spatial layer -> spatial_processed/
```

**Feature provenance, quirks, and the abandoned FGI-JSDR/MATLAB plan:** [`dataset_notes.md`](dataset_notes.md).

---

## Licensing

**This repository's code** (`.py` files, this README) — [MIT License](LICENSE).

**The dataset is a separate work, not included here, not covered by that license.** It is
licensed by its authors under the **GNU GPL v3.0 or later** (as stated on the Zenodo record).
If you use it, cite the dataset and its paper: V. Renaudin, M. I. Sayyaf, F. L. Bourhis and
M. Ortiz, *"GNSS Positioning Under Threat: The Rising Risk to Existing Systems and the Role of
Alternative Indoor and Seamless Navigation Technologies,"* IEEE Journal of Indoor and Seamless
Positioning and Navigation, doi:10.1109/JISPIN.2025.3629705.
