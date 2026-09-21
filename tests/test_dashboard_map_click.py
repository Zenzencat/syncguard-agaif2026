"""NOC map: clicking an alerting marker selects its incident (also mid-replay), and the
refresh path no longer redraws the map / re-fetches faster than a human can use it.

Static checks against the served dashboard script, same convention as
tests/test_priority_refresh_throttle.py (the dashboard is a single HTML file with inline JS and
this repo has no JS test runner). The behavioural side -- clicking a marker while a replay is
streaming -- was verified in a real browser; see the Item C report.

Why these exist: renderLiveMap() used to run a full Plotly.react() on EVERY SSE message, and
/priority + /incidents were re-fetched every 200-300ms, so during replay the map was redrawn
many times a second and a mouse-down/up on a small marker often missed or hit a marker that had
just been replaced.
"""
from __future__ import annotations

import re


def _script(html: str) -> str:
    return "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))


def _function_body(js: str, name: str) -> str:
    """Text of `function name(...){ ... }` via brace matching (bodies here contain nested
    braces, so a `[^}]*` regex would stop too early)."""
    m = re.search(r"function\s+" + re.escape(name) + r"\s*\([^)]*\)\s*\{", js)
    assert m, f"expected function {name}() to exist"
    depth, i = 1, m.end()
    while depth and i < len(js):
        depth += {"{": 1, "}": -1}.get(js[i], 0)
        i += 1
    return js[m.end():i - 1]


def test_refresh_throttle_is_1500ms_leading_and_trailing(client):
    js = _script(client.get("/dashboard").text)
    assert re.search(r"const REFRESH_THROTTLE_MS\s*=\s*1500\s*;", js)
    for fn, last_run in (("schedulePriorityRefresh", "priorityLastRun"),
                         ("scheduleIncidentRefresh", "incidentLastRun")):
        body = _function_body(js, fn)
        assert "clearTimeout" not in body, f"{fn} must stay a throttle, not a debounce"
        # trailing: an already-pending timer swallows further calls
        assert re.search(r"if\((priority|incident)RefreshTimer\)\s*return", body)
        # leading: waits only for what is left of the interval since the last real run
        assert "REFRESH_THROTTLE_MS" in body and last_run in body
        assert "Date.now()" in body


def test_priority_and_incident_fetches_skip_unchanged_payloads(client):
    js = _script(client.get("/dashboard").text)
    for fn, hash_var in (("refreshPriority", "lastPriorityHash"),
                         ("fetchIncidents", "lastIncidentsHash")):
        body = _function_body(js, fn)
        assert "hashString(" in body
        assert f"=== {hash_var}" in body, f"{fn} must compare against {hash_var} before rendering"
        # an error must clear the remembered hash so the error state gets repainted over
        assert f"{hash_var} = null" in body


def test_live_map_skips_identical_redraws_and_keeps_user_zoom(client):
    js = _script(client.get("/dashboard").text)
    body = _function_body(js, "renderLiveMap")
    assert "mapHash === lastMapHash" in body
    # the skip must happen BEFORE the (expensive) Plotly.react call
    assert body.index("mapHash === lastMapHash") < body.index("Plotly.react(")
    # user zoom/pan survives redraws until the selected incident changes
    assert "uirevision: 'incident:' + (selectedIncidentId || 'none')" in body


def test_marker_click_selects_incident_and_is_attached_once(client):
    js = _script(client.get("/dashboard").text)
    # Attached once at load (Plotly.react preserves div-level listeners), so redraws that
    # happen mid-replay cannot drop it.
    assert js.count("getElementById('live-map').on('plotly_click'") == 1
    assert "plotly_click" not in _function_body(js, "renderLiveMap")
    m = re.search(r"getElementById\('live-map'\)\.on\('plotly_click'.*?\n  \}\);", js, re.S)
    assert m
    handler = m.group(0)
    assert "incidentForTower(siteId)" in handler and "selectIncident(incident.incident_id)" in handler
    # the alerting-diamond trace must carry customdata, or clicking it yields no site id
    assert re.search(r"alertTrace = alerting\.length \? \{[^;]*customdata: alerting\.map", js, re.S)


def test_selected_incident_towers_are_larger_and_map_zooms_to_them(client):
    js = _script(client.get("/dashboard").text)
    body = _function_body(js, "renderLiveMap")
    assert "isSelectedTower(t) ? 1.7 : 1" in body        # active markers scaled up
    assert "isSelectedTower(t) ? 24 : 14" in body        # alert diamonds scaled up
    assert "incidentZoomRange(selectedTowers)" in body    # zoom to the incident's towers
    assert "function incidentZoomRange(" in js


def test_detail_panel_only_repaints_when_selected_incident_changed(client):
    js = _script(client.get("/dashboard").text)
    body = _function_body(js, "renderIncidentQueue")
    assert "renderedDetailSig" in body and "sig !== renderedDetailSig" in body


def test_click_on_tower_in_selected_incident_keeps_that_incident(client):
    """A tower can sit in several incidents (same epicenter, successive alert bursts). Clicking
    another tower of the incident already selected must not jump to a different incident."""
    js = _script(client.get("/dashboard").text)
    body = _function_body(js, "incidentForTower")
    assert "selectedIncidentId ? incidentsById.get(selectedIncidentId)" in body
    # the "stay on the current one" check must come before the first-match fallback loop
    assert body.index("current.towers.some") < body.index("for(const inc of incidentsById.values())")


def test_map_redraw_is_failsafe_and_only_remembers_successful_draws(client):
    """A Plotly failure ("axis scaling" when the pane has no size yet) used to escape
    renderLiveMap() and abort connect(), leaving the incident queue unloaded -- and the hash
    skip must never treat a FAILED draw as 'already drawn'."""
    js = _script(client.get("/dashboard").text)
    body = _function_body(js, "renderLiveMap")
    react_at = body.index("Plotly.react(")
    assert "try{" in body[:react_at].rsplit("const mapHash", 1)[1]           # react is inside a try
    assert body.index("lastMapHash = mapHash;") > react_at                  # set only after the draw
    assert "lastMapHash = null;" in body[react_at:]                         # cleared on failure
    assert "catch(err)" in body[react_at:]
