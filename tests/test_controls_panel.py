"""NOC tab: the scenario / speed / connection controls live in a collapsible "Controls" section
at the top of the right-hand panel (they used to be a bar across the top of the page, above the
map). The incident queue / detail markup and logic are unchanged.

Static checks against the served dashboard (no JS runner in this repo); the layout itself was
verified in a real browser at 1920x1080.
"""
from __future__ import annotations

import re

CONTROL_IDS = ("api-status", "api-status-text", "conn-toggle", "conn-advanced", "api-base",
               "btn-connect", "btn-new-session", "run-select", "speed-select", "speed-hint",
               "btn-start", "btn-stop")


def _html(client) -> str:
    return client.get("/dashboard").text


def _controls_block(html: str) -> str:
    start = html.index('<details class="controls-panel" id="controls-panel"')
    end = html.index("</details>", start)
    return html[start:end]


def test_controls_are_a_collapsible_details_at_the_top_of_the_side_column(client):
    html = _html(client)
    side = html.index('<div class="side-col">')
    controls = html.index('<details class="controls-panel" id="controls-panel"')
    queue = html.index('<div class="incident-wrap">')
    assert side < controls < queue
    assert re.search(r'<details class="controls-panel" id="controls-panel" open>', html)  # open by default
    assert "<summary>" in _controls_block(html)


def test_the_old_bar_above_the_map_is_gone(client):
    html = _html(client)
    tab = html.index('id="tab-live"')
    overview = html.index('id="noc-overview"')
    assert 'class="ctl-bar"' not in html[tab:overview], "controls must no longer sit above the map"


def test_every_control_id_is_still_present_exactly_once(client):
    html = _html(client)
    for cid in CONTROL_IDS:
        assert html.count(f'id="{cid}"') == 1, f"#{cid} should exist exactly once"


def test_the_replay_and_connection_controls_are_inside_the_panel(client):
    block = _controls_block(_html(client))
    for cid in ("btn-start", "btn-stop", "run-select", "speed-select", "conn-toggle", "btn-connect"):
        assert f'id="{cid}"' in block, f"#{cid} must be inside the Controls section"


def test_connection_status_pill_stays_visible_when_collapsed(client):
    block = _controls_block(_html(client))
    summary = block[block.index("<summary>"):block.index("</summary>")]
    assert 'id="api-status"' in summary and 'id="api-status-text"' in summary


def test_grid_has_no_row_for_the_removed_bar(client):
    html = _html(client)
    assert re.search(r"\.panel-live\{[^}]*grid-template-rows:auto minmax\(0,1fr\);", html)
    assert re.search(r"\.noc-overview\{grid-column:1 / -1; grid-row:1;", html)
    assert re.search(r"\.map-col\{grid-column:1; grid-row:2;", html)
    assert re.search(r"\.side-col\{grid-column:2; grid-row:2;", html)
    # the split itself is unchanged: map is the wider column
    assert "grid-template-columns:minmax(0,1.55fr) minmax(0,1fr)" in html


def test_collapsed_state_is_remembered_across_reloads(client):
    html = _html(client)
    assert "syncguard_controls_open" in html
    assert "controlsPanel.addEventListener('toggle'" in html


def test_incident_queue_and_detail_are_untouched(client):
    html = _html(client)
    assert html.count('id="incident-queue"') == 1 and html.count('id="incident-detail"') == 1
    assert html.index('id="controls-panel"') < html.index('id="incident-queue"')
