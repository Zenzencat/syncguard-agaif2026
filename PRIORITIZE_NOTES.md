# Prioritize: gate, then rank

## What the endpoint does

`GET /priority` answers an operational question: among towers already flagged by the detector,
which locations should an analyst inspect first?

1. **Gate by detector state.** A tower is included when its latest event has hysteresis state
   `alerting`. If an event has no alert state, the endpoint falls back to its stored
   `predicted_label == "attack"`, produced by the frozen 0.52 decision threshold.
2. **Rank the flagged set by ESTIMATED nearby population.** `pop_2km` sorts descending;
   SIMULATED severity breaks ties. A missing estimate sorts last and is explicitly marked
   `population unavailable`.

This is deliberately not severity multiplied by population. In the available inputs,
ESTIMATED population within 2 km spans about 1,408x (MEASURED minimum 37, maximum 52,159),
while SIMULATED severity spans about 2.3x. Multiplication therefore behaves almost entirely
like population ranking and disguises that fact. Gate-then-rank keeps the detector decision
and the exposure priority legible as two separate steps.

## Provenance and wording

- Tower geometry: **REAL** — 136 real telecom tower locations in Kubu Raya/Pontianak;
  operator labels are mixed.
- Severity and attribution in replay: **SIMULATED** on those real tower locations.
- Population: **ESTIMATE** — Meta Data for Good / CIESIN High Resolution Population Density
  Maps, distributed through HDX under CC BY 4.0, circa 2020.
- The population figure means **estimated people within 2 km (exposure proxy)**. It does not
  mean people served, subscribers, customers, or verified impact.

The 1 km and 2 km buffers overlap heavily. The same person can be near several towers, so
per-tower estimates must never be summed or reported as a population total. The API therefore
returns no such total.

## Stale-alert behavior

The endpoint uses each tower's latest stored event and does not apply an age filter. If that
latest event remains `alerting`, the tower remains flagged even when the event is old. This is
intentional for the current prototype: it prevents an unresolved alert from disappearing due
only to elapsed wall-clock time. An analyst or a later normal streak must clear it. The current
hysteresis registry is in-process, so restarting the service resets streak state.

## Population limitations

- The estimates are based on roughly 2019–2020 population inputs, not a current census.
- HRSL allocates census totals approximately uniformly across detected building pixels.
  Aggregates over a buffer are useful; a single pixel should not be treated as a precise
  household or occupancy measurement.
- Nearby population is an exposure proxy, not the population served by a tower.
- Buffer overlap makes cross-tower summation invalid.
- Missing population does not hide a flagged tower; it is ranked last and labeled
  `population unavailable`.
- There is no real per-tower GNSS interference telemetry in this project. `/priority` is not
  field validation and does not establish real affected population.

## Update 2026-09-21: the flagged set is the session's alerted towers

The "Stale-alert behavior" above (gate on each tower's latest event) disagreed with the dashboard
map once a recording returned to normal: replay hysteresis is per recording, so towers that
received no later event kept an old `alerting` event and were counted as flagged while the map
showed nothing alerting. `GET /priority` now uses the rule the dashboard uses:

- **Flagged set = towers that alerted at any point in the current session** (a stored event with
  hysteresis state `alerting`, or, with no hysteresis state such as `POST /score`, a threshold
  `attack` verdict). "Session" is what the event store holds; `POST /demo/reset` clears it.
- Each row says whether the tower is **alerting now** (`alerting_now: true`, `state: "alerting"`)
  or has **cleared** (`state: "cleared"`). Alerting now = the tower's latest event is `alerting`
  and, for replay events, the most recent replay event is still `alerting`. Ingest hysteresis is
  per tower, so an ingest tower's own latest state is its current state.
- The payload adds `n_alerting_now` and `n_cleared`; `n_flagged` is the whole session set.
- **Ranking is unchanged**: `pop_2km` descending, severity breaking ties. A cleared tower is ranked
  by the same exposure proxy; it is marked, not demoted.
- An event with no hysteresis state (`POST /score`) is never "alerting now", matching the map, so a
  tower whose only flagged event is a threshold verdict shows as `cleared`.
