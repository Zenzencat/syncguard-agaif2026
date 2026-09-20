"""Phase 4 scale benchmark: RESAMPLED REAL FEATURES, SYNTHETIC TOWERS.

The benchmark exercises the real HTTP POST /ingest path against a benchmark-only ASGI
wrapper, including validation, model scoring, optional inline SHAP, correlation, SQLite
persistence, and EventBus SSE publication to one live subscriber.  It measures the whole
path end to end; it does NOT time model scoring in isolation (no in-process
ModelService.score() timings are produced by this script).

The wrapper keeps production api/main.py and the dashboard unchanged.  It only replaces
the tower registry at process startup with synthetic IDs/coordinates so N=1,000 and
N=10,000 can be measured instead of being rejected by the production 136-tower lookup.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import winreg
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API_BASE = os.environ.get("SYNCGUARD_BENCHMARK_API_BASE", "http://127.0.0.1:8011")
PORT = int(API_BASE.rsplit(":", 1)[1].split("/", 1)[0])
PYTHON = Path(sys.executable)
DATA_PATH = ROOT / "processed" / "syncguard_features.parquet"
SERVER_MODULE = "tools.phase4_benchmark_app:app"
SAMPLE_SEED = 20260920
SCALE_COUNTS = (136, 1000, 10000)
SCALE_BATCH_SIZE = 500
BATCH_SWEEP_EVENTS = 136
BATCH_SWEEP_SIZES = (1, 20, 21, 100, 136)
REQUEST_TIMEOUT_S = 900

# The shipped artifact and api/model_service.py define this ordered 23-feature
# contract.  Keeping the client-side list literal avoids loading a second native
# model object just to discover column names.  The live ModelService loaded in
# main() still reports and verifies the shipped feature count/tag.
FEATURE_COLS = [
    "fixType", "gSpeed", "hAcc", "headAcc", "numSV", "pDOP", "sAcc", "vAcc",
    "velD", "velE", "velN", "pos_dev_m", "n_sats_l1", "snr_l1_mean",
    "snr_l1_std", "snr_l1_min", "doppler_l1_mean", "doppler_l1_std",
    "pr_doppler_residual_mean", "pr_doppler_residual_std", "jam_ind_mean",
    "agc_cnt_mean", "noise_per_ms_mean",
]


def pct(values: list[float], p: float) -> float:
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (p / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def stats_ms(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "p50_ms": pct(values, 50),
        "p99_ms": pct(values, 99),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def rss_mb(pid: int) -> float | None:
    """Return a Windows process working-set sample without adding psutil."""

    if os.name != "nt":
        return None
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    handle = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid
    )
    if not handle:
        return None
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), ctypes.sizeof(counters)
        )
        return counters.WorkingSetSize / (1024 * 1024) if ok else None
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def machine_info() -> dict[str, object]:
    cpu_name = platform.processor() or "unknown"
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            cpu_name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except OSError:
        pass
    mem = MEMORYSTATUSEX()
    mem.dwLength = ctypes.sizeof(mem)
    total_ram_gb = None
    if os.name == "nt" and ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
        total_ram_gb = mem.ullTotalPhys / (1024**3)
    return {
        "os": platform.platform(),
        "cpu": cpu_name,
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(total_ram_gb, 2) if total_ram_gb is not None else None,
        "python": sys.version.split()[0],
    }


def safe_unlink(path: Path) -> None:
    """Best-effort delete for temp files.

    On Windows a just-terminated server can still hold its log/DB file handle for a
    moment, so unlink() raises PermissionError.  Leaving a stray temp file behind is
    harmless; crashing the benchmark after all measurements are done is not.
    """

    try:
        path.unlink()
    except (FileNotFoundError, PermissionError):
        pass


def remove_db_family(db_path: Path, strict: bool = False) -> None:
    """Delete the benchmark SQLite file plus its -wal/-shm sidecars.

    strict=True is used before starting a server: a leftover DB would make the next
    run score against stale rows (duplicate batch IDs), so retry briefly and then fail
    loudly instead of silently continuing.  Post-run cleanup stays best-effort.
    """

    for path in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
        if not strict:
            safe_unlink(path)
            continue
        for attempt in range(20):
            safe_unlink(path)
            if not path.exists():
                break
            time.sleep(0.25)
        else:
            raise RuntimeError(f"could not remove stale benchmark DB file: {path}")


def benchmark_batch(
    session: requests.Session,
    start: int,
    count: int,
    tower_count: int,
    seed: int,
    day: int,
    batch_id: str,
) -> dict:
    """Get a deterministic real-row resample before timing the /ingest POST."""

    response = session.get(
        f"{API_BASE}/_phase4/batch",
        params={
            "start": start,
            "count": count,
            "tower_count": tower_count,
            "seed": seed,
            "day": day,
            "batch_id": batch_id,
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("feature_provenance") != "RESAMPLED REAL FEATURES, SYNTHETIC TOWERS":
        raise RuntimeError(f"unexpected benchmark provenance: {body.get('feature_provenance')}")
    return body


class SSEReader:
    def __init__(self, url: str):
        self.url = url
        self.stop_event = threading.Event()
        self.connected = threading.Event()
        self.thread = threading.Thread(target=self._run, name="phase4-sse-reader", daemon=True)
        self.events = 0
        self.error: str | None = None

    def start(self) -> None:
        self.thread.start()
        if not self.connected.wait(timeout=10):
            raise RuntimeError(f"SSE subscriber did not connect: {self.error}")

    def _run(self) -> None:
        try:
            with requests.get(
                self.url,
                stream=True,
                # Long read timeout: the shipped /ingest scores synchronously on the event
                # loop, so the stream can legitimately go silent for many seconds during a
                # big batch.  A short timeout made the reader die mid-run and report 0
                # events while the summary still claimed an active subscriber.
                timeout=(10, 120),
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()
                self.connected.set()
                for raw in response.iter_lines(decode_unicode=True):
                    if self.stop_event.is_set():
                        break
                    line = raw or ""
                    if line.startswith("data: ") and line != "data: {}":
                        self.events += 1
        except Exception as exc:  # connection close during cleanup is expected
            if not self.stop_event.is_set():
                self.error = repr(exc)
                self.connected.set()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3)


def start_server(tower_count: int, db_path: Path) -> tuple[subprocess.Popen, Path]:
    remove_db_family(db_path, strict=True)
    env = os.environ.copy()
    env["SYNTHETIC_TOWER_COUNT"] = str(tower_count)
    env["SYNCGUARD_DB_PATH"] = str(db_path)
    env["SYNCGUARD_LOG_LEVEL"] = "WARNING"
    site_packages = ROOT / ".venv" / "Lib" / "site-packages"
    python_path = os.pathsep.join(
        [str(ROOT), str(site_packages), env.get("PYTHONPATH", "")]
    )
    env["PYTHONPATH"] = python_path
    log_path = Path(tempfile.gettempdir()) / f"syncguard_phase4_server_{os.getpid()}_{tower_count}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(PYTHON), "-m", "uvicorn", SERVER_MODULE,
         "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    log_handle.close()
    deadline = time.monotonic() + 90
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            last_error = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"benchmark server exited with {proc.returncode}:\n{last_error}")
        try:
            health = requests.get(f"{API_BASE}/health", timeout=3)
            if health.ok and health.json().get("model_loaded"):
                actual = health.json().get("towers_loaded")
                if actual != tower_count:
                    raise RuntimeError(f"server reported towers_loaded={actual}, expected {tower_count}")
                return proc, log_path
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(0.25)
    stop_server(proc)
    raise RuntimeError(f"benchmark server did not become healthy: {last_error}")


def stop_server(proc: subprocess.Popen) -> None:
    """Stop the server and only return once the OS has reaped the process.

    terminate() -> wait(10) -> kill() -> wait() guarantees the child has exited (and so
    released its log file / SQLite handles) before any caller tries to delete them.
    """

    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def post_ingest_scale(
    session: requests.Session,
    tower_count: int,
    db_path: Path,
    max_seconds: float | None = None,
) -> dict[str, object]:
    proc, log_path = start_server(tower_count, db_path)
    sse = SSEReader(f"{API_BASE}/stream/events")
    try:
        sse.start()
        request_ms: list[float] = []
        peak_rss = rss_mb(proc.pid)
        total_start = time.perf_counter()
        responses = []
        events_done = 0
        for start in range(0, tower_count, SCALE_BATCH_SIZE):
            payload = benchmark_batch(
                session, start, min(SCALE_BATCH_SIZE, tower_count - start),
                tower_count, SAMPLE_SEED + tower_count, 20,
                f"phase4-scale-{tower_count}-{start}",
            )
            t0 = time.perf_counter()
            response = session.post(
                f"{API_BASE}/ingest", json=payload, timeout=REQUEST_TIMEOUT_S
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            response.raise_for_status()
            body = response.json()
            if body["scored"] != len(payload["observations"]) or body["duplicates"]:
                raise RuntimeError(f"unexpected ingest response: {body}")
            request_ms.append(elapsed_ms)
            responses.append(body)
            events_done += len(payload["observations"])
            sample_rss = rss_mb(proc.pid)
            if sample_rss is not None:
                peak_rss = max(peak_rss or sample_rss, sample_rss)
            elapsed_total = time.perf_counter() - total_start
            print(
                f"  [N={tower_count}] batch@{start}: {elapsed_ms:.0f} ms "
                f"({elapsed_ms / len(payload['observations']):.1f} ms/event); "
                f"{events_done}/{tower_count} events, {elapsed_total:.0f}s elapsed",
                flush=True,
            )
            if max_seconds is not None and elapsed_total > max_seconds and events_done < tower_count:
                print(f"  [N={tower_count}] TIME CAP {max_seconds:.0f}s reached -- stopping early; "
                      "result below is PARTIAL", flush=True)
                break
        total_s = time.perf_counter() - total_start
        time.sleep(0.25)
        result = {
            "completed": events_done == tower_count,
            "events_completed": events_done,
            "per_batch_ms": [round(x, 1) for x in request_ms],
            "tower_count": tower_count,
            "synthetic_tower_ids": tower_count,
            "feature_provenance": "RESAMPLED REAL FEATURES, SYNTHETIC TOWERS",
            "batch_size": SCALE_BATCH_SIZE,
            "inline_shap": False,
            "n_batches": len(request_ms),
            "batch_latency": stats_ms(request_ms),
            "total_elapsed_s": total_s,
            "events": events_done,
            "events_per_sec": events_done / total_s,
            "server_rss_start_mb": rss_mb(proc.pid),
            "server_rss_peak_mb": peak_rss,
            "sse_events_received": sse.events,
            "sse_error": sse.error,
            "response_live_explain_values": sorted({b["live_explain"] for b in responses}),
            "db_path": str(db_path),
        }
        return result
    finally:
        sse.stop()
        stop_server(proc)
        safe_unlink(log_path)


def post_batch_sweep(
    session: requests.Session,
    batch_size: int,
    db_path: Path,
) -> dict[str, object]:
    proc, log_path = start_server(BATCH_SWEEP_EVENTS, db_path)
    sse = SSEReader(f"{API_BASE}/stream/events")
    try:
        sse.start()
        request_ms: list[float] = []
        peak_rss = rss_mb(proc.pid)
        offset = 0
        live_values = []
        total_start = time.perf_counter()
        while offset < BATCH_SWEEP_EVENTS:
            count = min(batch_size, BATCH_SWEEP_EVENTS - offset)
            payload = benchmark_batch(
                session, offset, count, BATCH_SWEEP_EVENTS,
                SAMPLE_SEED + 777, 21,
                f"phase4-batch-{batch_size}-{offset}",
            )
            t0 = time.perf_counter()
            response = session.post(
                f"{API_BASE}/ingest", json=payload, timeout=REQUEST_TIMEOUT_S
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            response.raise_for_status()
            body = response.json()
            if body["scored"] != len(payload["observations"]) or body["duplicates"]:
                raise RuntimeError(f"unexpected batch response: {body}")
            request_ms.append(elapsed_ms)
            live_values.append(body["live_explain"])
            offset += count
            sample_rss = rss_mb(proc.pid)
            if sample_rss is not None:
                peak_rss = max(peak_rss or sample_rss, sample_rss)
        total_s = time.perf_counter() - total_start
        time.sleep(0.25)
        return {
            "tower_count": BATCH_SWEEP_EVENTS,
            "batch_size": batch_size,
            "feature_provenance": "RESAMPLED REAL FEATURES, SYNTHETIC TOWERS",
            "n_batches": len(request_ms),
            "batch_latency": stats_ms(request_ms),
            "events": BATCH_SWEEP_EVENTS,
            "total_elapsed_s": total_s,
            "events_per_sec": BATCH_SWEEP_EVENTS / total_s,
            "inline_shap": sorted(set(live_values)),
            "server_rss_peak_mb": peak_rss,
            "sse_events_received": sse.events,
            "sse_error": sse.error,
        }
    finally:
        sse.stop()
        stop_server(proc)
        safe_unlink(log_path)


def print_result(label: str, result: dict[str, object]) -> None:
    print(f"\n{label}")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sizes",
        default=",".join(str(n) for n in SCALE_COUNTS),
        help="comma-separated synthetic tower counts to run (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-batch-sweep",
        action="store_true",
        help="skip the N=136 batch-size sweep (useful for a large-N-only run)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="per-size wall-clock cap; stop after the batch that crosses it and mark the "
             "result PARTIAL (default: no cap)",
    )
    parser.add_argument(
        "--summary-path",
        default=str(ROOT / "processed" / "phase4_scale_benchmark_results.json"),
        help="where to write the raw JSON summary (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    try:
        args.sizes = tuple(int(x) for x in args.sizes.split(",") if x.strip())
    except ValueError:
        parser.error("--sizes must be a comma-separated list of integers")
    if not args.sizes or any(n < 1 for n in args.sizes):
        parser.error("--sizes must contain positive integers")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    db_env = os.environ.get("SYNCGUARD_DB_PATH")
    if not db_env:
        raise SystemExit("SYNCGUARD_DB_PATH must point to a temporary benchmark DB")
    db_path = Path(db_env).resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    if temp_root not in db_path.parents:
        raise SystemExit(f"Refusing non-temporary SYNCGUARD_DB_PATH: {db_path}")

    info = machine_info()
    print("SYNCGUARD PHASE 4 SCALE BENCHMARK — A5")
    print("RESAMPLED REAL FEATURES, SYNTHETIC TOWERS")
    print(f"API_BASE={API_BASE}")
    print(f"SYNCGUARD_DB_PATH={db_path}")
    print("The production dashboard and api/main.py were not edited.")
    print("Each HTTP run has one live /stream/events subscriber; SSE publication is included.")
    print("Synthetic tower IDs/coordinates are benchmark-only; model rows are sampled from the real parquet table.")
    print("MACHINE")
    print(json.dumps(info, indent=2, sort_keys=True))

    print("MODEL_TAG=reported by /health for each benchmark server")
    print(f"FEATURE_COUNT={len(FEATURE_COLS)}")
    print("SINGLE_THREADED_RF_N_JOBS=1 (shipped api/model_service.py)")

    all_results: dict[str, object] = {"machine": info, "scale": [], "batch_sweep": []}
    with requests.Session() as session:
        for tower_count in args.sizes:
            print(f"\n=== SCALE N={tower_count} TOWERS ===")
            all_results["scale"].append({"tower_count": tower_count})
            http_result = post_ingest_scale(session, tower_count, db_path, args.max_seconds)
            print_result("HTTP /ingest E2E (batch size 500; inline SHAP off; SSE subscriber active)", http_result)
            all_results["scale"][-1]["http_ingest"] = http_result

        print("\n=== BATCH-SIZE SWEEP N=136 SYNTHETIC TOWERS ===")
        for batch_size in BATCH_SWEEP_SIZES:
            result = post_batch_sweep(session, batch_size, db_path)
            print_result(f"HTTP /ingest BATCH_SIZE={batch_size} (SSE subscriber active)", result)
            all_results["batch_sweep"].append(result)

    scale = all_results["scale"]
    print("\n=== BOTTLENECK READOUT ===")
    print("MEASURED: /ingest timings include request validation, single-threaded scoring, correlation, SQLite writes, and EventBus SSE enqueue to one subscriber.")
    print("MEASURED: inline SHAP is enabled only when a batch has <=20 observations in the shipped ingest path; larger batches skip it.")
    for item in scale:
        e2e_per_event = item["http_ingest"]["total_elapsed_s"] * 1000 / item["http_ingest"]["events"]
        partial = "" if item["http_ingest"]["completed"] else (
            f" [PARTIAL: {item['http_ingest']['events']}/{item['tower_count']} events, time cap]")
        print(
            f"N={item['tower_count']}{partial}: /ingest E2E={e2e_per_event:.2f} ms/event; "
            f"batch p50={item['http_ingest']['batch_latency']['p50_ms']:.2f} ms; "
            f"batch p99={item['http_ingest']['batch_latency']['p99_ms']:.2f} ms"
        )
    print("INFERENCE: the measured /ingest time includes the full HTTP, validation, scoring, correlation, SQLite, and SSE-publish path; this benchmark does not instrument those internals separately.")
    print("ESTIMATE FORMULA: for a fixed per-event cost c measured at N, T_est(N') = N' * c; this is a linear extrapolation and does not claim production behavior beyond the measured range.")
    print("NO FIELD CLAIM: no real receivers, real tower telemetry, or deployment traffic were used.")

    summary_path = Path(args.summary_path)
    summary_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved raw benchmark summary: {summary_path}")
    remove_db_family(db_path)
    print(f"Removed temporary benchmark DB family: {db_path}")
    return 0


if __name__ == "__main__":
    # The bundled Python 3.12 runtime plus the repo's separately-created native
    # wheels can raise a Windows application error during interpreter teardown
    # after SHAP/model objects have already been used successfully.  All output
    # and files are complete before this point, so flush and exit without the
    # problematic native finalizer path.
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
