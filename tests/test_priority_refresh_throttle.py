"""Regression test: schedulePriorityRefresh()/scheduleIncidentRefresh() must be a throttle
(fires at least every ~200-300ms under continuous events), not a plain debounce.

A plain debounce (clearTimeout + setTimeout on every call) never fires while SSE events keep
arriving faster than the delay -- which is exactly what happens during replay -- so "flagged
towers" and the incident queue would stay stuck at their initial value for as long as the
stream kept flowing. MEASURED live during Task 3 fresh-database verification: header active
alerts showed 8 while flagged towers stayed at 0 for the whole replay, until the stream
slowed down. Fixed by only scheduling when nothing is already pending.
"""
from __future__ import annotations

import re


def _script(html: str) -> str:
    return "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))


def test_priority_refresh_does_not_reset_an_already_pending_timer(client):
    js = _script(client.get("/dashboard").text)
    m = re.search(r"function schedulePriorityRefresh\(\)\{([^}]*)\}", js)
    assert m, "expected schedulePriorityRefresh() to exist"
    body = m.group(1)
    assert "clearTimeout" not in body, (
        "schedulePriorityRefresh must not clearTimeout an existing pending refresh -- that "
        "makes it a debounce that never fires under continuous SSE events"
    )
    assert "if(priorityRefreshTimer)return;" in body.replace(" ", "")


def test_incident_refresh_does_not_reset_an_already_pending_timer(client):
    js = _script(client.get("/dashboard").text)
    m = re.search(r"function scheduleIncidentRefresh\(\)\{([^}]*)\}", js)
    assert m, "expected scheduleIncidentRefresh() to exist"
    body = m.group(1)
    assert "clearTimeout" not in body, (
        "scheduleIncidentRefresh must not clearTimeout an existing pending refresh -- that "
        "makes it a debounce that never fires under continuous SSE events"
    )
    assert "if(incidentRefreshTimer)return;" in body.replace(" ", "")
