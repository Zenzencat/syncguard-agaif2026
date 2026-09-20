"""Regression test for a real bug found in review: the header alert-state pill read "normal"
while "active alerts" showed 2, because the pill was set from whichever SSE event's own
alert_state happened to arrive last (single-event/single-recording hysteresis), in a line
that ran AFTER renderLiveMap() had already (correctly) computed the active-alerts count --
so the last write always won and could disagree with the count next to it.

Fix: the pill is now set inside renderLiveMap() itself, derived from the same `alerting`
array that sets the active-alerts count, so they can never structurally disagree.
"""
from __future__ import annotations

import re


def _script(html: str) -> str:
    return "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))


def test_alert_pill_is_set_from_the_active_alerts_list_inside_render_live_map(client):
    js = _script(client.get("/dashboard").text)
    m = re.search(r"const alerting = active\.filter\(t => t\.alert_state === 'alerting'\);"
                  r"([\s\S]*?)const alertTrace", js)
    assert m, "expected the alerting-towers list to be computed in renderLiveMap()"
    body = m.group(1)
    assert "alertPillEl.className" in body and "alerting.length" in body, (
        "the alert pill must be derived from the same `alerting` list as the active-alerts count"
    )


def test_sse_handler_no_longer_sets_the_pill_from_a_single_events_alert_state(client):
    js = _script(client.get("/dashboard").text)
    m = re.search(r"eventSource\.onmessage = \(msg\) => \{([\s\S]*?)\n    \};", js)
    assert m, "expected to find the SSE onmessage handler"
    body = m.group(1)
    assert "alertPillEl.className = 'pill ' + ev.alert_state" not in body, (
        "onmessage must not set the pill from a single event's alert_state -- that line ran "
        "after renderLiveMap() and silently overwrote the correct derived value"
    )
