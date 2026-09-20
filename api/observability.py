"""Structured JSON logging, request ids, and Prometheus metrics.

No new dependencies. The metrics registry below is a few hundred lines of plain Python rather
than `prometheus_client`, matching the project's existing dependency-light posture (the same
reasoning that made api/db.py stdlib sqlite3 with no ORM). It implements the subset of the
Prometheus text exposition format this service actually emits -- counters, gauges and
histograms with labels -- and nothing else.

Nothing here ever logs a credential. The request logger emits a fixed set of fields; headers,
cookies and request bodies are not among them, so an API key cannot reach a log line by
accident. See api/auth.py.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from contextvars import ContextVar

# Propagated through the request lifecycle so every log line emitted while handling a request
# carries the same id, without threading a parameter through every call site.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

REQUEST_ID_HEADER = "X-Request-ID"

# Fields json.dumps must not choke on and that always appear. Anything else a caller attaches
# via `extra={...}` is merged in, so long as it is JSON-serializable.
_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info", "thread",
    "threadName", "taskName",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Structured from the start rather than regex-parsed later."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        try:
            return json.dumps(payload, default=str)
        except (TypeError, ValueError):
            # A log line must never take the process down. Fall back to a minimal record.
            return json.dumps({"ts": payload["ts"], "level": payload["level"],
                               "logger": payload["logger"], "message": payload["message"],
                               "log_serialization_error": True})


def configure_logging(level: str | None = None) -> logging.Logger:
    """Install the JSON formatter on the root logger and on uvicorn's own loggers.

    SYNCGUARD_LOG_LEVEL overrides the default INFO. uvicorn.access is silenced because this
    module's own middleware emits a richer, structured equivalent -- leaving both on would
    double-log every request in two different formats.
    """
    level_name = (level or os.environ.get("SYNCGUARD_LOG_LEVEL") or "INFO").upper()
    resolved = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved)

    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers = [handler]
        lg.propagate = False
        lg.setLevel(resolved)

    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False
    access.disabled = True

    return logging.getLogger("syncguard")


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape_label_value(str(v))}"' for k, v in labels)
    return "{" + inner + "}"


class MetricsRegistry:
    """Counters, gauges and histograms, rendered in Prometheus text exposition format.

    Guarded by one lock: the scoring path is single-threaded but uvicorn may still serve
    requests from more than one task, and a torn read of a histogram would produce buckets
    that do not sum to their own count.
    """

    # Seconds. Chosen around what this service actually does: sub-millisecond bookkeeping
    # endpoints, ~5-50ms scoring, ~50ms+ when SHAP runs inline, and multi-second batches.
    DEFAULT_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, dict[tuple, float]] = {}
        self._gauges: dict[str, dict[tuple, float]] = {}
        self._histograms: dict[str, dict[tuple, dict]] = {}
        self._help: dict[str, tuple[str, str]] = {}  # name -> (type, help text)

    def register(self, name: str, metric_type: str, help_text: str) -> None:
        with self._lock:
            self._help[name] = (metric_type, help_text)

    @staticmethod
    def _key(labels: dict | None) -> tuple:
        return tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))

    def inc(self, name: str, labels: dict | None = None, value: float = 1.0) -> None:
        key = self._key(labels)
        with self._lock:
            series = self._counters.setdefault(name, {})
            series[key] = series.get(key, 0.0) + value

    def set_gauge(self, name: str, value: float, labels: dict | None = None) -> None:
        key = self._key(labels)
        with self._lock:
            self._gauges.setdefault(name, {})[key] = float(value)

    def observe(self, name: str, value: float, labels: dict | None = None,
                buckets: tuple[float, ...] | None = None) -> None:
        key = self._key(labels)
        bounds = buckets or self.DEFAULT_BUCKETS
        with self._lock:
            series = self._histograms.setdefault(name, {})
            entry = series.get(key)
            if entry is None:
                entry = {"buckets": bounds, "counts": [0] * len(bounds), "sum": 0.0, "count": 0}
                series[key] = entry
            for i, upper in enumerate(entry["buckets"]):
                if value <= upper:
                    entry["counts"][i] += 1
            entry["sum"] += value
            entry["count"] += 1

    def render(self) -> str:
        """Prometheus text format (version 0.0.4). Histogram buckets are emitted cumulatively,
        as the format requires, with the mandatory +Inf bucket equal to the observation count."""
        lines: list[str] = []
        with self._lock:
            counters = {n: dict(s) for n, s in self._counters.items()}
            gauges = {n: dict(s) for n, s in self._gauges.items()}
            histograms = {n: {k: dict(v) for k, v in s.items()}
                          for n, s in self._histograms.items()}
            helps = dict(self._help)

        emitted: set[str] = set()

        def header(name: str, default_type: str) -> None:
            metric_type, help_text = helps.get(name, (default_type, name))
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {metric_type}")
            emitted.add(name)

        for name in sorted(counters):
            header(name, "counter")
            for key, value in sorted(counters[name].items()):
                lines.append(f"{name}{_format_labels(key)} {value:g}")

        for name in sorted(gauges):
            header(name, "gauge")
            for key, value in sorted(gauges[name].items()):
                lines.append(f"{name}{_format_labels(key)} {value:g}")

        for name in sorted(histograms):
            header(name, "histogram")
            for key, entry in sorted(histograms[name].items()):
                # entry["counts"] is ALREADY cumulative: observe() increments every bucket
                # whose upper bound the value falls under, not just the one it lands in.
                # Summing again here would double-count and produce finite buckets larger
                # than the +Inf bucket, which is invalid Prometheus.
                for upper, count in zip(entry["buckets"], entry["counts"]):
                    le = ("+Inf" if upper == float("inf")
                          else f"{upper:g}")
                    lines.append(
                        f"{name}_bucket{_format_labels(key + (('le', le),))} {count}")
                lines.append(
                    f"{name}_bucket{_format_labels(key + (('le', '+Inf'),))} {entry['count']}")
                lines.append(f"{name}_sum{_format_labels(key)} {entry['sum']:g}")
                lines.append(f"{name}_count{_format_labels(key)} {entry['count']}")

        # A registered metric with no samples yet still gets its HELP/TYPE lines. Without
        # this, /metrics omits a counter entirely until its first increment, so a scraper
        # cannot tell "nothing has happened yet" from "this build does not have that metric",
        # and an alert rule referencing it looks broken rather than quiet.
        for name in sorted(helps):
            if name not in emitted:
                metric_type, help_text = helps[name]
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {metric_type}")

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Test-only. Production never clears metrics -- counters are monotonic by contract."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


METRICS = MetricsRegistry()

METRICS.register("syncguard_requests_total", "counter",
                 "HTTP requests handled, by route template, method and status class.")
METRICS.register("syncguard_request_duration_seconds", "histogram",
                 "HTTP request duration in seconds, by route template and method.")
METRICS.register("syncguard_scores_total", "counter",
                 "Model scoring calls, by originating path and predicted label.")
METRICS.register("syncguard_alerts_total", "counter",
                 "Scored events whose alert_state was 'alerting' after hysteresis, by source.")
METRICS.register("syncguard_input_warnings_total", "counter",
                 "Input plausibility warnings raised, by feature and direction. A warning "
                 "never rejects input -- see api/plausibility.py.")
METRICS.register("syncguard_ingest_observations_total", "counter",
                 "Observations submitted to POST /ingest, by outcome "
                 "(scored | duplicate | out_of_order).")
METRICS.register("syncguard_feedback_total", "counter",
                 "Analyst labels submitted, by label and whether it replaced an earlier one.")
METRICS.register("syncguard_auth_failures_total", "counter",
                 "Requests rejected for a missing or invalid API key, by reason. Never "
                 "includes any part of a presented credential.")
METRICS.register("syncguard_events_persisted_total", "counter",
                 "Rows written to scored_events, by source.")
METRICS.register("syncguard_model_info", "gauge",
                 "Always 1. Labels carry the model version tag -- see GET /health.")
METRICS.register("syncguard_build_info", "gauge",
                 "Always 1. Labels carry service metadata.")
