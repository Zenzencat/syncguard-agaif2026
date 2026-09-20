# Phase 4 scale notes

Every number below is labeled **MEASURED** (observed in the run in this file), **INFERENCE**
(read from code / reasoned from a MEASURED number), or **ESTIMATE** (extrapolated). Feature
values in every run are **RESAMPLED REAL FEATURES** (rows sampled with replacement from the
real `processed/syncguard_features.parquet`); tower IDs and coordinates are **SYNTHETIC**.
The 136-tower run uses 136 *synthetic* IDs, not the 136 real tower locations.

Raw output: `phase_4_output.txt` (in the phase reports folder, not committed) and the two JSON
summaries next to it. Script: `tools/phase4_scale_benchmark.py`; benchmark-only ASGI wrapper:
`tools/phase4_benchmark_app.py` (reuses the shipped `api.main:app` and lifespan, then swaps the
tower registry for synthetic IDs -- production code paths are otherwise untouched).

## Hardware (MEASURED, printed by the script's machine block)

| Item | Value |
|---|---|
| CPU | Intel Core Ultra 9 275HX, 24 logical CPUs |
| RAM | 31.43 GB |
| OS | Windows 11 (10.0.26200) |
| Python | 3.12.10 |

This is one high-end laptop, run once per configuration (single run each; no repeat-to-repeat
variance is reported -- the two 136/1000 runs that were made agreed within ~2%, but that is two
samples, not a confidence interval).

## Method

- The script starts a fresh uvicorn subprocess per configuration with `SYNCGUARD_DB_PATH` set to
  a temp file (it refuses any non-temp path) and the synthetic registry size in
  `SYNTHETIC_TOWER_COUNT`.
- The client asks the server for a batch (real parquet rows resampled server-side), then times
  **only** the `POST /ingest` call. Timing therefore covers request validation, single-threaded
  scoring (shipped `n_jobs=1` RandomForest, threshold 0.52 -- unchanged), correlation, SQLite
  writes, and `EventBus` publication to one connected `/stream/events` subscriber.
- Scale runs: N synthetic towers, N events (one per tower), batches of 500, inline SHAP off
  (the shipped ingest path only computes it for batches of <=20 observations).
- Batch-size sweep: 136 events at batch sizes 1, 20, 21, 100, 136.
- The whole path is timed as one number; correlation / SQLite / SSE are **not** instrumented
  separately. This script does **not** time model scoring in isolation (an earlier version of its
  docstring and one readout line implied it did; both were removed).
- The runs were made with the Item B1 robustness fix (`/incidents` and `/priority` tolerate a
  tower table with no exposure columns) already in the working tree. It only touches those two
  read endpoints, not `/ingest`, so it does not affect these numbers. Other light editing work
  was done on the machine during the runs but no tests, servers or browsers were running.
- Commands: `--sizes 136,1000`, then `--sizes 10000 --skip-batch-sweep --max-seconds 1800`.
  The N=10,000 run completed in full (the cap was never hit).

## Results (MEASURED)

`/ingest` end to end, batch size 500, inline SHAP off, one SSE subscriber:

| N synthetic towers = events | Elapsed | ms/event | events/s | Batch latency p50 / p99 (batch of 500) | Server working set* |
|---|---|---|---|---|---|
| 136  | 2.3 s (one batch of 136) | 16.99 | 58.9 | 2,290 ms (single batch) | 447 MB |
| 1,000  | 32.8 s | 32.79 | 30.5 | 16,323 / 20,761 ms (n=2) | 474 MB |
| 10,000 | 701.2 s | 70.12 | 14.3 | 36,400 / 41,479 ms (n=20) | 545 MB |

\* Working set of the server process sampled between batches. It is the whole process
(model + SHAP explainer + event store), not per-event cost; the growth from 447 to 545 MB across
a 74x increase in events is small.

Per-batch time at N=10,000 (ms): 12,211 -> 21,304 -> 29,417 -> ... -> ~40,000-41,600 for the last
~12 batches. That is ~24 ms/event for the first batch rising to a plateau of ~80 ms/event.

Batch-size sweep (136 synthetic towers, 136 events, SSE subscriber connected):

| Batch size | p50 / p99 per request | events/s | Inline SHAP |
|---|---|---|---|
| 1   | 68.0 / 97.2 ms | 11.8 | on |
| 20  | 1,370 / 1,555 ms | 13.9 | on |
| 21  | 351.7 / 665.0 ms | 48.3 | off for the first 6 requests (21 each), on for the last (10) |
| 100 | 1,142 / 1,590 ms | 58.1 | off |
| 136 | 2,121 ms (single request) | 62.7 | off |

SSE delivery to the one subscriber (MEASURED): 103 of 136, 220 of 1,000 and 2,200 of 10,000
events arrived for the 500-event-batch runs, i.e. roughly 110 events per 500-event batch reached
the subscriber and the rest were dropped. In the sweep every configuration delivered 136/136
except batch size 136, which delivered 103.

## What the numbers say (INFERENCE unless marked)

- **Per-event cost is not constant; it grows with burst rate.** 17 -> 33 -> 70 ms/event from N=136
  to 1,000 to 10,000. `LiveCorrelationEngine.correlate()` (`api/spatial.py`) loads every scored
  event from the last 120 s of *server wall-clock* time (`SELECT *`, converted to dicts) and does
  per-row pandas `.loc` lookups for each event it scores, so cost per event is proportional to
  the number of events in the trailing 120 s. It plateaus once the run is long enough that the
  window stops filling (~80 ms/event here). This is a property of scoring a burst faster than the
  window length, which is exactly how this benchmark drives it.
- **Inline SHAP is the largest single per-event cost at small batch sizes.** Batch size 1 and 20
  (SHAP on) cost ~68 ms/event; batch size 100 (SHAP off) costs ~11 ms/event on the same 136
  events. The ~57 ms difference is attributed to inline SHAP (the other stages are the same code
  path, but their inputs -- e.g. window size -- differ slightly, so this is an approximation).
- **`/ingest` scores synchronously on the event loop** (INFERENCE from the SSE reader dying on
  a 1 s read timeout during the first run of this benchmark: the stream produced nothing for
  many seconds while a batch was being scored). While a large batch is in flight, that server
  answers nothing else.
- **SSE fan-out drops events under bursts.** `EventBus` uses a bounded queue (`maxsize=100`) per
  subscriber and drops on overflow (`api/replay.py`). A 500-event batch is published in one
  synchronous burst, so ~100 fit and the rest are dropped -- consistent with the ~110/500 seen.
  The database still has every event; only the live push is lossy.

## What these numbers do NOT show

- **No real receivers and no field deployment.** No real GNSS receiver has ever fed this
  system. Feature rows are resampled from a recorded dataset; timestamps and tower IDs are
  synthetic. Nothing here says anything about real traffic patterns, arrival jitter, or
  malformed input from real devices.
- **One high-end laptop, not edge or server hardware.** A 24-thread Core Ultra 9 with 31 GB RAM
  on Windows; a small edge box or a shared VM will be slower, and the single-threaded model call
  benefits from this CPU's single-core speed. The numbers should not be quoted as capacity for
  any other machine.
- **One client, one subscriber, one process.** No concurrent ingest clients, no multiple dashboard
  subscribers, no auth enabled, no network between client and server (loopback).
- **Closed-loop bursts, not steady arrival rates.** The client sends the next batch as soon as
  the previous returns, so this measures *drain-a-backlog* behavior. It is not a sustained
  "N towers each reporting every T seconds" test, and the 70-80 ms/event plateau should not be
  read as the cost at a realistic per-tower reporting rate (where the 120 s window would hold far
  fewer events).
- **Synthetic towers are a regular 0.0001-degree grid**, so distance-decay correlation and
  incident/spatial behavior at N=1,000 / 10,000 are not representative of a real network.
- **The ms/event figures are whole-path numbers.** Correlation, SQLite and SSE are not separated;
  the attributions above are inference from code and from comparing configurations.
- **Not a detector-quality test.** Scores and alerts in these runs were not evaluated; no accuracy
  or latency-to-detection claim follows.
- ESTIMATE only: linear extrapolation `T(N') = N' x c` is *wrong* for this system below the
  plateau (per-event cost rises with burst size), and even above it, it is valid only for the
  same closed-loop burst pattern.

## Known scale limits

1. **In-process hysteresis state resets on restart.** Per-tower alert hysteresis lives in
   process memory (`api/hysteresis.py`); after a restart every tower starts back at `normal`, so
   a tower that was `alerting` silently un-alerts until it accumulates the confirmation readings
   again. It also cannot be shared across multiple worker processes. (Already listed in
   `ASSUMPTIONS_PRODUCTION.md`.)
2. **SQLite as the store.** A single-file DB behind one connection/lock (`api/db.py`) is fine for
   this demo and for the measured runs (10,000 events), but there is a single writer, no
   replication, and `SELECT *` over the correlation window (above) is done in Python. Not
   measured beyond 10,000 events or with concurrent writers.
3. **SSE fan-out.** Bounded 100-slot per-subscriber queue with drop-on-full; publication happens
   on the event loop that is also doing scoring. Under a burst the live view lags and drops
   events (measured above); the dashboard re-reads `/priority` and `/incidents` from the store,
   so state converges, but the raw live log does not show every event. One subscriber was
   measured; fan-out to many was not.
4. **Synchronous scoring on the event loop.** A large `/ingest` batch blocks other requests for
   its duration (seconds to tens of seconds at the measured batch sizes). Small batches (<=20)
   also run inline SHAP (~57 ms/event extra).
5. **Correlation window cost grows with events-per-120 s** (see above): the first thing to
   fix if sustained rates above roughly 10-15 events/s per process matter.
6. **The 120 s window and incident windows key off server insert time (`created_at`)**, not
   observation time, so a delayed or replayed backlog is correlated as if it were simultaneous.
   (Also in `ASSUMPTIONS_PRODUCTION.md`.)
