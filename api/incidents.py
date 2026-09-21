"""Groups alerting events into incidents for the NOC tab's incident queue.

The correlation engine (api/spatial.py::LiveCorrelationEngine) computes a real
distance-weighted correlation SCORE per event; it does not group events into discrete
incidents. This module adds a simple, explicitly-approved grouping rule on top of it:

  1. Chain flagged events (api/exposure.py::is_flagged), sorted by time, into one incident
     when two consecutive events are BOTH no more than INCIDENT_WINDOW_SECONDS apart in time
     AND no more than INCIDENT_DISTANCE_KM apart in space (their towers' real haversine
     distance -- same tower always counts as distance 0 regardless of stored coordinates).
     Single-linkage chaining: each event only has to satisfy both checks against the previous
     event added to the current group, not every earlier member.

     Time-only grouping was tried first and rejected: under SIMULATED round-robin replay
     attribution, a sustained alert scores events across many different towers within
     milliseconds of each other, so a time-only window never breaks and the whole scenario
     collapses into one incident spanning all towers. Requiring both dimensions fixes that,
     and pairs with EpicenterRandomWalk (api/spatial.py), an alternative replay
     attribution mode that concentrates events near a fixed simulated epicenter instead of
     scattering them round-robin, so nearby-in-time events are also plausibly nearby in space.
     This is still a demo heuristic, not a real clustering algorithm: real deployments would
     group by shared timing source or region, not raw distance.
  2. The chain also breaks across two different non-null replay run_ids, even if both other
     checks pass -- two separate replay runs are never merged into one incident just because
     they were scored close together in wall-clock time. run_id is only the RECORDING, so two
     replays of the same scenario share it; the per-start `replay_session` (api/replay.py)
     breaks the chain between those too, so re-running a scenario opens a new incident instead
     of extending -- and inheriting the Dismissed/Confirmed status of -- the previous run's.
     Rows written before that column existed have no session and fall back to the run_id rule.

Which events: GET /incidents feeds this every flagged event in the store (up to
api/db.py::INCIDENT_EVENT_CAP, newest first), not the newest N events of any kind. An
incident's ID is INC-<id of its first event>, so a sliding window used to both drop old
incidents and RENAME long-running ones as their first events slid out of it. Over the full
flagged history the ID is a pure function of stored rows: events are appended in time order,
so a later event can only extend the last incident or start a new one -- it can never change
an existing incident's first event, and IDs are never reassigned.

Which clock: every timestamp here is `created_at`, the server's insert time for the scored
row -- NOT the original recording's observation timestamp. This is the same clock
LiveCorrelationEngine's window already uses (see api/spatial.py's module docstring and the
"known limitations" note in ASSUMPTIONS_PRODUCTION.md: correlation/incident windows key off
created_at, not observation time). During replay, created_at is the wall-clock moment the
row was written while streaming at the chosen speed multiplier -- sped up, not the original
file's timestamps. The dashboard labels this "recording time" for replay sessions; outside
replay (POST /score, POST /ingest) it is simply when the row was stored.

INCIDENT_WINDOW_SECONDS defaults to CORRELATION_WINDOW_SECONDS (120s, api/spatial.py);
INCIDENT_DISTANCE_KM defaults to 5.0. Both are independently env-overridable
(SYNCGUARD_INCIDENT_WINDOW_SECONDS / SYNCGUARD_INCIDENT_DISTANCE_KM) for tuning without
touching the correlation engine.

Status is derived from the existing analyst feedback labels (api/db.py's event_feedback
table) on any event in the incident -- no schema change, no new label added:
  - any member event labeled 'confirmed'                       -> "Confirmed"
  - else any member event labeled 'dismissed' (none confirmed) -> "Dismissed"
  - else                                                        -> "New"

Confirm/Dismiss from the incident queue is written against ONE representative event -- the
incident's peak-severity event, `top_event_id` -- never fanned out to every member event. The
incident detail additionally reports how many of the incident's events already carry a label
(`labeled_event_count` of `event_count`), since a single incident-level action does not label
every reading in a multi-event incident.

Population exposure: per-tower `pop_2km` estimates overlap between neighbouring towers (same
convention as api/exposure.py) and must never be summed across an incident's towers. This
module reports only `max_pop_2km` (the single highest per-tower estimate among the incident's
towers) plus the tower count -- never a total.
"""
from __future__ import annotations

import os
from datetime import datetime

from api.exposure import is_flagged
from api.spatial import CORRELATION_WINDOW_SECONDS, haversine_km

INCIDENT_WINDOW_SECONDS = int(
    os.environ.get("SYNCGUARD_INCIDENT_WINDOW_SECONDS", CORRELATION_WINDOW_SECONDS)
)
INCIDENT_DISTANCE_KM = float(os.environ.get("SYNCGUARD_INCIDENT_DISTANCE_KM", 5.0))


def _parse_ts(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _within_distance(a: dict, b: dict, distance_km: float) -> bool:
    if a.get("tower_site_id") and a.get("tower_site_id") == b.get("tower_site_id"):
        return True
    lat1, lon1, lat2, lon2 = a.get("tower_lat"), a.get("tower_lon"), b.get("tower_lat"), b.get("tower_lon")
    if None in (lat1, lon1, lat2, lon2):
        return False  # unknown location -- fail closed, do not merge on an unverifiable guess
    return float(haversine_km(lat1, lon1, lat2, lon2)) <= distance_km


def build_incidents(events: list[dict], feedback_by_event: dict[int, dict],
                     pop_2km_by_tower: dict[str, float] | None = None,
                     window_seconds: int = INCIDENT_WINDOW_SECONDS,
                     distance_km: float = INCIDENT_DISTANCE_KM) -> list[dict]:
    """`events` newest-or-any order, each a scored_events row (dict). `feedback_by_event` maps
    event_id -> feedback record dict with a 'label' key (may be None/missing -- unlabeled).
    `pop_2km_by_tower` maps tower_site_id -> ESTIMATED people within 2km (exposure proxy),
    used only to report `max_pop_2km` (never a sum) -- omit it and that field is null."""
    flagged = [e for e in events if e.get("created_at") and is_flagged(e)[0]]
    flagged.sort(key=lambda e: e["created_at"])

    groups: list[list[dict]] = []
    current: list[dict] = []
    prev_event: dict | None = None
    current_run_id = None
    current_session = None
    for e in flagged:
        ts = _parse_ts(e["created_at"])
        run_id = e.get("run_id")
        session = e.get("replay_session")
        join = False
        if current and prev_event is not None:
            prev_ts = _parse_ts(prev_event["created_at"])
            gap_ok = (ts - prev_ts).total_seconds() <= window_seconds
            dist_ok = _within_distance(prev_event, e, distance_km)
            run_ok = not (run_id is not None and current_run_id is not None and run_id != current_run_id)
            session_ok = not (session is not None and current_session is not None
                              and session != current_session)
            join = gap_ok and dist_ok and run_ok and session_ok
        if current and not join:
            groups.append(current)
            current = []
            current_run_id = None
            current_session = None
        current.append(e)
        if run_id is not None:
            current_run_id = run_id
        if session is not None:
            current_session = session
        prev_event = e
    if current:
        groups.append(current)

    incidents = [_summarize(group, feedback_by_event, pop_2km_by_tower) for group in groups]

    # Most severe first; newest (latest end_time) first within a severity tie.
    incidents.sort(key=lambda inc: (inc["severity"], inc["end_time"]), reverse=True)
    return incidents


def _summarize(group: list[dict], feedback_by_event: dict[int, dict],
               pop_2km_by_tower: dict[str, float] | None) -> dict:
    group_sorted = sorted(group, key=lambda e: e["created_at"])
    first, last = group_sorted[0], group_sorted[-1]
    start_ts, end_ts = _parse_ts(first["created_at"]), _parse_ts(last["created_at"])

    towers = {}
    for e in group_sorted:
        tid = e.get("tower_site_id")
        if tid and tid not in towers:
            towers[tid] = e.get("tower_site_name")

    max_pop_2km = None
    if pop_2km_by_tower:
        pops = [pop_2km_by_tower[tid] for tid in towers if pop_2km_by_tower.get(tid) is not None]
        if pops:
            max_pop_2km = max(pops)

    labels = [feedback_by_event.get(e["id"], {}).get("label") for e in group_sorted]
    if "confirmed" in labels:
        status = "Confirmed"
    elif "dismissed" in labels:
        status = "Dismissed"
    else:
        status = "New"
    labeled_event_count = sum(1 for l in labels if l is not None)

    most_severe = max(group_sorted, key=lambda e: e.get("severity", 0.0))
    sources = {e.get("source") for e in group_sorted if e.get("source")}

    return {
        "incident_id": f"INC-{first['id']}",
        "event_ids": [e["id"] for e in group_sorted],
        "top_event_id": most_severe["id"],
        "towers": [{"site_id": tid, "site_name": name} for tid, name in towers.items()],
        "tower_attribution_simulated": "replay" in sources,
        "max_pop_2km": max_pop_2km,
        "start_time": first["created_at"],
        "end_time": last["created_at"],
        "clock": "created_at (server insert time; 'recording time' during replay -- see "
                 "api/incidents.py docstring, not the original file's observation timestamp)",
        "duration_seconds": max(0.0, (end_ts - start_ts).total_seconds()),
        "severity": round(float(most_severe.get("severity", 0.0)), 4),
        "attack_type": most_severe.get("attack_type"),
        "status": status,
        "event_count": len(group_sorted),
        "labeled_event_count": labeled_event_count,
        "grouping_note": (
            f"Events chained into one incident when consecutive alerting events are within "
            f"{INCIDENT_WINDOW_SECONDS}s AND {INCIDENT_DISTANCE_KM:g}km of each other "
            f"(single-linkage), and never across two different replay runs. Grouped by time "
            f"and straight-line distance only; real deployments would group by shared timing "
            f"source or region. Population, if shown, is the highest single tower's estimate, "
            f"never a sum across this incident's towers."
        ),
    }
