"""NOC map: "solid red = alerting NOW", a hollow "cleared" style, legend entries that appear only
while their marker is drawn, and a SESSION popover that names the real attribution mode.

Why: a tower is coloured from its own latest event, and replay hysteresis is per RECORDING, so a
tower that received no event after the recording returned to normal kept a solid red diamond
forever (found by replaying a full run: 4 of 8 towers stayed 'alerting' while the recording ended
normal). Separately, the hotspot / coldspot / outlier legend entries were always shown although
those rings need 15+ scored towers and never render under epicenter attribution.

Display layer only (no incident logic, /incidents, hysteresis or api/ change). Same conventions as
tests/test_plain_language_signals.py: served-HTML source guards, plus the pure helpers executed with
node when it is installed (skipped otherwise).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest


def _html(client) -> str:
    return client.get("/dashboard").text


def _script(html: str) -> str:
    return "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))


def _function_body(js: str, name: str) -> str:
    m = re.search(r"function\s+" + re.escape(name) + r"\s*\([^)]*\)\s*\{", js)
    assert m, f"expected function {name}() to exist"
    depth, i = 1, m.end()
    while depth and i < len(js):
        depth += {"{": 1, "}": -1}.get(js[i], 0)
        i += 1
    return js[m.end():i - 1]


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _run_node(html: str, expr: str):
    m = re.search(r"// ---- alert state BEGIN.*?// ---- alert state END ----", html, re.S)
    assert m, "the alert-state pure block must be present"
    out = subprocess.run(["node", "-e", m.group(0) + f"\nconsole.log(JSON.stringify({expr}));"],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _classify(html, towers, replay_state, ever=()):
    r = _run_node(html, f"(() => {{ const r = classifyAlerting({json.dumps(towers)}, {json.dumps(replay_state)}, "
                        f"new Set({json.dumps(list(ever))})); "
                        f"return {{now: r.alertingNow.map(t => t.site_id), cleared: r.cleared.map(t => t.site_id)}}; }})()")
    return r["now"], r["cleared"]


def _t(site, state, source="replay"):
    return {"site_id": site, "alert_state": state, "source": source}


# ---------------------------------------------------------------- solid red = alerting now (node)

@needs_node
def test_replay_towers_stop_being_solid_red_once_the_recording_returns_to_normal(client):
    html = _html(client)
    towers = [_t("A", "alerting"), _t("B", "alerting"), _t("C", "normal")]
    # recording still alerting: A and B are alerting now
    assert _classify(html, towers, "alerting", ever=["A", "B", "C"]) == (["A", "B"], ["C"])
    # recording has returned to normal: NOTHING is solid red; A and B (last event alerting) are cleared
    now, cleared = _classify(html, towers, "normal", ever=["A", "B", "C"])
    assert now == [] and cleared == ["A", "B", "C"]


@needs_node
def test_cleared_means_alerted_earlier_and_never_alerted_towers_stay_plain(client):
    html = _html(client)
    towers = [_t("A", "normal"), _t("B", "normal"), _t("C", None, "api")]
    # A alerted earlier (member of an incident) -> cleared; B never alerted -> no diamond at all
    assert _classify(html, towers, "normal", ever=["A"]) == ([], ["A"])
    # a POST /score event has no alert_state and is never drawn as alerting or cleared by itself
    assert _classify(html, [_t("C", None, "api")], None) == ([], [])


@needs_node
def test_ingest_towers_use_their_own_latest_state_not_the_replay_recording(client):
    html = _html(client)
    # ingest hysteresis is per tower, so an ingest tower alerting stays alerting even if some
    # unrelated replay recording is currently normal
    assert _classify(html, [_t("I", "alerting", "ingest")], "normal") == (["I"], [])
    assert _classify(html, [_t("I", "normal", "ingest")], "alerting", ever=["I"]) == ([], ["I"])


@needs_node
def test_full_replay_ends_with_no_solid_red_when_the_recording_ends_normal(client):
    """Model of the real defect: events go to a few towers; the recording alerts, then returns to
    normal, but towers C and D never receive another event, so their latest event stays 'alerting'."""
    html = _html(client)
    js = """(() => {
      const latest = {}, ever = new Set(); let replayState = null; const log = [];
      const feed = (site, state) => {
        latest[site] = {site_id: site, alert_state: state, source: 'replay'};
        if(state === 'alerting') ever.add(site);
        replayState = state;
      };
      const snap = () => { const r = classifyAlerting(Object.values(latest), replayState, ever);
                           return {now: r.alertingNow.length, cleared: r.cleared.length}; };
      ['A', 'B', 'C', 'D'].forEach(s => feed(s, 'alerting')); log.push(snap());
      feed('A', 'alerting'); log.push(snap());
      feed('B', 'normal'); feed('A', 'normal'); log.push(snap());
      return log;
    })()"""
    snaps = _run_node(html, js)
    assert snaps[0] == {"now": 4, "cleared": 0}
    assert snaps[1] == {"now": 4, "cleared": 0}
    assert snaps[-1] == {"now": 0, "cleared": 4}, "no tower may be solid red once the recording is normal"


# ---------------------------------------------------------------- legend entries follow the markers (node)

@needs_node
def test_legend_entries_show_only_for_marker_types_that_are_drawn(client):
    html = _html(client)
    vis = lambda **o: _run_node(html, "legendVisibility(" + json.dumps(  # noqa: E731
        {"lisaQuadrants": [], "alertingNowCount": 0, "clearedCount": 0, "selectedCount": 0, **o}) + ")")
    assert vis() == {"hh": False, "ll": False, "outlier": False, "alerting": False, "cleared": False, "selected": False}
    assert vis(lisaQuadrants=[1])["hh"] and not vis(lisaQuadrants=[1])["ll"]
    assert vis(lisaQuadrants=[3])["ll"] and not vis(lisaQuadrants=[3])["hh"]
    # both outlier quadrants (Low-High = 2, High-Low = 4) map to the one outlier entry
    assert vis(lisaQuadrants=[2])["outlier"] and vis(lisaQuadrants=[4])["outlier"]
    v = vis(alertingNowCount=3, clearedCount=2, selectedCount=1)
    assert v["alerting"] and v["cleared"] and v["selected"] and not (v["hh"] or v["ll"] or v["outlier"])


@needs_node
def test_cluster_note_only_when_rings_are_missing_because_too_few_towers_are_scored(client):
    html = _html(client)
    note = lambda d: _run_node(html, f"clustersNeedMoreTowers({json.dumps(d)})")  # noqa: E731
    assert note({"computable": False, "n_towers_scored": 8, "min_required": 15}) is True     # epicenter mode
    assert note({"computable": True, "n_towers_scored": 136, "min_required": 15}) is False   # rings can render
    assert note({"computable": False, "n_towers_scored": 20, "min_required": 15}) is False   # not a count problem
    assert note(None) is False


# ---------------------------------------------------------------- render path / markup

def test_solid_and_hollow_diamonds_are_separate_traces_and_both_stay_clickable(client):
    js = _script(_html(client))
    body = _function_body(js, "renderLiveMap")
    assert "classifyAlerting(active, latestReplayState, everAlerted)" in body
    solid = re.search(r"const alertTrace = alerting\.length \? \{(.*?)\n    \} : null;", body, re.S).group(1)
    hollow = re.search(r"const clearedTrace = cleared\.length \? \{(.*?)\n    \} : null;", body, re.S).group(1)
    assert "symbol:'diamond'," in solid and "ALERTING (hysteresis-confirmed)" in solid
    assert "symbol:'diamond-open'" in hollow and "CLEARED (recently alerted, not alerting now)" in hollow
    assert "#A83232" in hollow                          # the same red, hollow -- meaning is not recoloured
    assert "customdata: cleared.map" in hollow          # the plotly_click handler keys on customdata
    assert "clearedTrace && [" in body                  # part of the redraw-skip hash, or it would never update
    # counts and the header pill still come from the alerting-NOW list
    assert "activeAlertsEl.textContent = alerting.length" in body


def test_legend_markup_entries_are_conditional_and_include_the_cleared_entry(client):
    html = _html(client)
    legend = re.search(r'<div class="lisa-legend">(.*?)</div>', html, re.S).group(1)
    keys = re.findall(r'<span data-legend="(\w+)" hidden', legend)
    assert keys == ["hh", "ll", "outlier", "alerting", "cleared", "selected"]
    assert "Recently alerted, now cleared" in legend and '<i class="dia hollow"></i>' in legend
    assert 'id="legend-cluster-note"' in legend
    # nothing is unconditionally shown any more
    assert not re.search(r"<span>\s*<i class=\"ring", legend)
    assert ".lisa-legend [hidden]{display:none !important;}" in html   # else the flex rule would win over [hidden]


def test_cluster_note_wording_and_wiring(client):
    js = _script(_html(client))
    assert "spatial clusters appear once ${clusterNeed}+ towers are scored" in _function_body(js, "renderLiveMap")
    assert "clusterNeed = clustersNeedMoreTowers(d) ? d.min_required : null;" in _function_body(js, "fetchAutocorrelation")


def test_spatial_analysis_legend_entries_are_always_backed_by_a_drawn_marker(client):
    """The other map: its legend (Epicenter star, top-5 badges) is unconditional because both marker
    types are built on every render, so it needs no show/hide logic."""
    html = _html(client)
    js = _script(html)
    legend = re.search(r'<div class="map-legend">(.*?)</div>', html, re.S).group(1)
    assert "Epicenter" in legend and "Highest simulated severity" in legend
    body = _function_body(js, "buildTraces")
    assert "epicenterTrace" in body and "labelTrace" in body
    assert "data-legend" not in legend


# ---------------------------------------------------------------- SESSION popover attribution

def test_session_popover_no_longer_hardcodes_round_robin(client):
    html = _html(client)
    assert "SIMULATED round-robin" not in html
    assert 'replay tower attribution is SIMULATED<span id="pop-attribution"></span> because' in html


def test_popover_and_tooltip_share_one_attribution_setter(client):
    js = _script(_html(client))
    setter = _function_body(js, "setReplayAttribution")
    assert "replayAttribution = mode" in setter and "getElementById('pop-attribution')" in setter
    assert "attributionNote()" in setter
    # nothing else assigns the mode directly, so the two can never disagree
    assigns = re.findall(r"(?<![\w.])replayAttribution\s*=(?!=)", js)
    assert len(assigns) == 2, "only the declaration and the setter may assign replayAttribution"
    for call in ("setReplayAttribution(params.get('attribution'))",
                 "setReplayAttribution(replayStatus.status !== 'idle' ? replayStatus.attribution : null)"):
        assert call in js


# ---------------------------------------------------------------- live SSE path (regression found by a full replay)

@needs_node
def test_sse_replay_events_are_recognised_from_run_id_because_they_carry_no_source(client):
    """Found by replaying a scenario that ends normal: the LIVE page kept 3 solid red diamonds and
    "3 active alerts" while a reload was correct. The replay SSE envelope has no `source` (ingest's
    does), so the live page treated every tower as non-replay. The source is derived from run_id."""
    html = _html(client)
    src = lambda ev: _run_node(html, f"eventSourceOf({json.dumps(ev)})")  # noqa: E731
    assert src({"event_id": 1, "run_id": "Spoofing/x", "scenario_id": "2.3.2", "alert_state": "normal"}) == "replay"
    assert src({"event_id": 2, "source": "ingest", "batch_id": "b1", "alert_state": "alerting"}) == "ingest"
    assert src({"event_id": 3, "run_id": None, "alert_state": None}) is None      # e.g. a POST /score event
    assert src({"source": "ingest", "run_id": "x"}) == "ingest"                    # an explicit source wins


@needs_node
def test_live_stream_of_a_recording_that_ends_normal_leaves_no_solid_red(client):
    """The exact live sequence, using raw SSE-shaped payloads (no `source`, run_id present)."""
    html = _html(client)
    js = """(() => {
      const latest = {}, ever = new Set(); let replayState = null;
      const feed = ev => { const src = eventSourceOf(ev);
        latest[ev.site] = {site_id: ev.site, alert_state: ev.alert_state, source: src};
        if(ev.alert_state === 'alerting') ever.add(ev.site);
        if(src === 'replay') replayState = ev.alert_state; };
      const mk = (site, state) => ({site, alert_state: state, run_id: 'Spoofing/2.3.2', scenario_id: '2.3.2'});
      ['A', 'B', 'C'].forEach(s => feed(mk(s, 'alerting')));
      const during = classifyAlerting(Object.values(latest), replayState, ever).alertingNow.length;
      feed(mk('A', 'normal')); feed(mk('B', 'normal'));
      const end = classifyAlerting(Object.values(latest), replayState, ever);
      return {during, now: end.alertingNow.length, cleared: end.cleared.length};
    })()"""
    assert _run_node(html, js) == {"during": 3, "now": 0, "cleared": 3}


def test_sse_handler_derives_the_source_instead_of_trusting_ev_source(client):
    js = _script(_html(client))
    m = re.search(r"eventSource\.onmessage = \(msg\) => \{([\s\S]*?)\n    \};", js)
    assert m
    body = m.group(1)
    assert "source: eventSourceOf(ev)" in body
    assert "noteEvent(ev.tower.site_id, eventSourceOf(ev), ev.alert_state, ev.event_id)" in body
