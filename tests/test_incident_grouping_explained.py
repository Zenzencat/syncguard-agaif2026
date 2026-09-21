"""NOC: several red alerting diamonds can be ONE incident in the queue (api/incidents.py chains
alerting events by time AND distance). The dashboard must say so, and must describe the grouping
rule accurately -- the queue help text used to say it was time-only.

Static checks against the served dashboard, same convention as tests/test_dashboard_map_click.py.
"""
from __future__ import annotations

import re


def _html(client) -> str:
    return client.get("/dashboard").text


def test_queue_help_describes_time_and_distance_grouping(client):
    html = _html(client)
    assert "AND close in space" in html
    assert "not a spatial-clustering algorithm" not in html   # the old, inaccurate time-only wording
    assert "ONE incident here" in html                        # states the many-diamonds/one-incident case


def test_alerting_legend_entry_explains_diamonds_versus_incidents(client):
    html = _html(client)
    m = re.search(r'<span data-legend="alerting" hidden title="([^"]*)"><i class="dia"></i>Alerting \(hysteresis-confirmed\)</span>', html)
    assert m, "the Alerting legend entry must carry an explanatory hover title"
    assert "single incident" in m.group(1)


def test_diamond_tooltip_names_its_incident_and_the_redraw_hash_sees_it(client):
    html = _html(client)
    assert "nearby alerting towers are grouped into one incident" in html
    assert "incidentForTower(t.site_id)" in html
    # renderLiveMap() skips identical redraws by hash; the incident id arrives after the marker
    # does, so the tooltip text has to be part of that hash or it would never update.
    assert "alertTrace && [alertTrace.x, alertTrace.y, alertTrace.marker.size, alertTrace.text]" in html
