"""Exposure -> Prioritize: attaches ESTIMATED nearby population to the real tower records, and
ranks the towers that are currently flagged. See PRIORITIZE_NOTES.md for why the design is
gate-then-rank rather than severity x population.

What is what:
  - Tower locations: REAL (136 real telecom tower locations, Kubu Raya/Pontianak, operator
    labels mixed).
  - Population: ESTIMATE. Meta Data for Good / CIESIN High Resolution Population Density
    Maps (HDX, CC BY 4.0), circa 2020, summed over the raster pixels within 1 km and 2 km of
    each tower. An exposure proxy -- NOT "people served".
  - Severity / alert state used as the gate: whatever the store holds for each tower's latest
    event. On the live demo path that is SIMULATED attribution (replay assigns events to
    towers) on real tower locations; for /ingest it is the caller-claimed tower. Neither is a
    verified per-tower measurement.

Per-tower populations OVERLAP (neighbouring buffers count the same people), so they must never
be summed or totaled. Nothing in this module produces a total, on purpose.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("syncguard")

EXPOSURE_CSV = (Path(__file__).resolve().parent.parent / "spatial_processed"
                / "tower_exposure_HRSL.csv")

# Decimal places used to build the (lat, lon) join key. The source coordinates carry 6
# decimals (~0.1 m), so this only absorbs float-repr noise; it cannot merge distinct towers.
COORD_DECIMALS = 6

POPULATION_SOURCE = ("Meta Data for Good / CIESIN High Resolution Population Density Maps "
                     "(HDX, CC BY 4.0), circa 2020, estimated")
EXPOSURE_LABEL = "exposure proxy, not people served"
SEVERITY_LABEL = "SIMULATED severity gate -> ESTIMATED nearby population (exposure proxy)"
SUMMING_NOTE = ("Per-tower populations overlap (nearby towers' buffers contain the same "
                "people), so they must not be summed or totaled. This response deliberately "
                "carries no total.")

# Fallback alert threshold quoted in the response text only. The gate itself never applies a
# threshold: it reads the predicted_label / alert_state the store recorded at scoring time,
# which used the shipped model's own decision threshold.
DEFAULT_THRESHOLD = 0.52

GATE_HYSTERESIS = "hysteresis"
GATE_THRESHOLD = "threshold"


def _coord_key(lat: float, lon: float) -> tuple[float, float]:
    return (round(float(lat), COORD_DECIMALS), round(float(lon), COORD_DECIMALS))


def attach_exposure(towers: pd.DataFrame, csv_path: Path | None = None) -> pd.DataFrame:
    """Return `towers` with `pop_1km` and `pop_2km` columns (NaN where missing).

    Joins on (lat, lon) -- NEVER on site_id: 16 of the 136 rows share the placeholder site_id
    "tbg", so a site_id join would fan out. Coordinates are unique per tower in both tables.

    A missing CSV is not fatal: the columns come back all-NaN, /priority still runs (every
    tower ranks with pop_2km null) and a warning is logged. A (lat, lon) key that appears more
    than once in the exposure table cannot be attributed to one tower, so it is treated as
    missing rather than guessed.
    """
    out = towers.copy()
    out["pop_1km"] = np.nan
    out["pop_2km"] = np.nan

    path = Path(csv_path) if csv_path else EXPOSURE_CSV
    if not path.exists():
        log.warning("exposure table not found; pop_1km/pop_2km will be null",
                    extra={"path": str(path)})
        return out

    exp = pd.read_csv(path)
    needed = {"lat", "lon", "pop_within_1km", "pop_within_2km"}
    if not needed.issubset(exp.columns):
        log.warning("exposure table lacks required columns; pop_1km/pop_2km will be null",
                    extra={"path": str(path), "missing": sorted(needed - set(exp.columns))})
        return out

    exp["_key"] = [_coord_key(a, b) for a, b in zip(exp["lat"], exp["lon"])]
    ambiguous = exp["_key"].duplicated(keep=False)
    if ambiguous.any():
        log.warning("exposure rows with duplicate (lat, lon) ignored",
                    extra={"n_rows": int(ambiguous.sum())})
        exp = exp.loc[~ambiguous]
    lookup = exp.set_index("_key")[["pop_within_1km", "pop_within_2km"]]

    keys = [_coord_key(a, b) for a, b in zip(out["lat"], out["lon"])]
    out["pop_1km"] = [lookup["pop_within_1km"].get(k, np.nan) for k in keys]
    out["pop_2km"] = [lookup["pop_within_2km"].get(k, np.nan) for k in keys]
    return out


def records_with_nulls(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe records: NaN becomes None (a bare NaN is not valid JSON), and
    numpy scalars become plain Python ones."""
    obj = df.astype(object).where(df.notna(), None)
    return obj.to_dict(orient="records")


def _pop_int(value) -> int | None:
    """Round an estimated population to a whole number. None/NaN stay None -- a missing
    estimate is never reported as 0. (A genuine 0.0 estimate is kept as 0.)"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(round(float(value)))


def is_flagged(event: dict) -> tuple[bool, str]:
    """The gate. Uses the hysteresis alert state when the event carries one, else the
    per-reading threshold verdict.

    - alert_state present ('normal'/'alerting'): flagged iff 'alerting'. Set on replay events
      (per-recording hysteresis) and /ingest events (per-tower hysteresis).
    - alert_state NULL (e.g. POST /score, which is stateless): flagged iff predicted_label is
      'attack', i.e. probability >= the model's decision threshold (0.52) at scoring time.
    """
    state = event.get("alert_state")
    if state is not None:
        return state == "alerting", GATE_HYSTERESIS
    return event.get("predicted_label") == "attack", GATE_THRESHOLD


def pop_2km_by_tower(towers: pd.DataFrame) -> dict[str, float]:
    """tower_key -> ESTIMATED people within 2 km, for the towers that have an estimate.

    Tolerates a tower table with no exposure columns at all (one that never went through
    attach_exposure, e.g. a fresh benchmark server without the exposure CSV): that is "no
    population data available", so it returns {} rather than raising KeyError.
    """
    if "pop_2km" not in towers.columns or "tower_key" not in towers.columns:
        return {}
    return towers.set_index("tower_key")["pop_2km"].dropna().to_dict()


def rank_priority(latest_by_tower: dict[str, dict], towers: pd.DataFrame,
                  threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Gate-then-rank over the latest event per tower.

    1. Gate: keep only towers whose latest event is flagged (is_flagged).
    2. Rank: order the survivors by pop_2km, descending. Severity is carried as a secondary
       field and breaks ties. Towers with no population estimate sort last (still listed --
       a flagged tower is not hidden because its exposure is unknown).

    Returns the /priority payload. There is no total in it: no population sum, no "people
    affected" figure. `n_flagged` counts towers, not people.
    """
    by_key = towers.set_index("tower_key")
    rows, n_events, basis_counts = [], 0, {GATE_HYSTERESIS: 0, GATE_THRESHOLD: 0}

    for tower_key, ev in latest_by_tower.items():
        n_events += 1
        flagged, basis = is_flagged(ev)
        if not flagged:
            continue
        basis_counts[basis] += 1
        if tower_key in by_key.index:
            t = by_key.loc[tower_key]
            site_id, site_name = t["site_id"], t["site_name"]
            lat, lon = float(t["lat"]), float(t["lon"])
            # .get: a tower table without exposure columns (e.g. a benchmark/synthetic
            # registry that never went through attach_exposure) means "no data", not a 500.
            pop_1km, pop_2km = _pop_int(t.get("pop_1km")), _pop_int(t.get("pop_2km"))
        else:
            # An event attributed to a tower_key the table does not know. Cannot happen via
            # /ingest (unknown towers are rejected) or replay; kept visible rather than dropped.
            site_id, site_name = tower_key, ev.get("tower_site_name")
            lat, lon = ev.get("tower_lat"), ev.get("tower_lon")
            pop_1km = pop_2km = None
        rows.append({
            "tower_key": tower_key,
            "site_id": site_id,
            "site_name": site_name,
            "lat": lat,
            "lon": lon,
            "pop_2km": pop_2km,
            "pop_1km": pop_1km,
            "population_status": ("population unavailable" if pop_2km is None
                                  else "estimated population available"),
            "severity": ev.get("severity"),
            "probability": ev.get("probability"),
            "alert_state": ev.get("alert_state"),
            "gate_basis": basis,
            "event_id": ev.get("id"),
            "event_source": ev.get("source"),
            "event_created_at": ev.get("created_at"),
        })

    # pop_2km desc, unknown last; then severity desc; then tower_key so order is deterministic.
    rows.sort(key=lambda r: (r["pop_2km"] is None,
                             -(r["pop_2km"] or 0),
                             -(r["severity"] or 0.0),
                             r["tower_key"]))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i

    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "method": (
            "gate-then-rank. Gate: a tower is flagged when its latest event is in hysteresis "
            "state 'alerting'; events with no hysteresis state (e.g. POST /score) fall back to "
            f"the per-reading threshold verdict (probability >= {threshold:g}). Rank: flagged "
            "towers ordered by estimated population within 2 km (pop_2km), descending. "
            "Severity is a secondary field and breaks ties; it does not multiply the ranking."
        ),
        "gate": {
            "primary": "hysteresis alert_state == 'alerting'",
            "fallback": f"predicted_label == 'attack' (probability >= {threshold:g})",
            "n_towers_with_events": n_events,
            "flagged_by_basis": basis_counts,
        },
        "n_flagged": len(rows),
        "ranked_by": "pop_2km",
        "exposure_label": EXPOSURE_LABEL,
        "display_label": SEVERITY_LABEL,
        "severity_provenance": (
            "SIMULATED severity/attribution on real tower locations: on the replay path the "
            "tower each event is attributed to is simulated; on the /ingest path the tower is "
            "caller-claimed and unverified."
        ),
        "population": {
            "quantity": "estimated people within 2 km of the tower location (pop_2km) and "
                        "within 1 km (pop_1km)",
            "vintage": "~2020",
            "source": POPULATION_SOURCE,
            "license": "CC BY 4.0",
            "label": "ESTIMATE",
            "note": EXPOSURE_LABEL,
        },
        "note": SUMMING_NOTE,
        "towers": rows,
    }
    if not rows:
        payload["empty_state"] = (
            "No towers are currently flagged by the severity gate, so there is nothing "
            "to rank. Start a replay or ingest observations to populate current events."
        )
    return payload
