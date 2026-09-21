"""Header counter: the /priority-sourced tower counter is "Towers alerted this session".

The number is /priority's n_flagged, which is now the set of towers that alerted at any point in the
session (alerting now + cleared) -- the towers the map draws as solid + hollow diamonds -- so the label
says exactly that, and its tooltip points to "Active alerts" for what is alerting now. The Prioritize tab
shows the same set with each tower marked alerting now / cleared. Display only.
"""
from __future__ import annotations

import re

TIP = "Alerting now or earlier this session. Active alerts shows what is alerting now."


def _html(client) -> str:
    return client.get("/dashboard").text


def _script(html: str) -> str:
    return "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))


def test_both_counters_use_the_new_label_and_tooltip(client):
    html = _html(client)
    for element_id in ("flagged-towers", "noc-flagged-towers"):
        m = re.search(r'<div class="v" id="' + element_id + r'">0</div><div class="l" title="([^"]*)">([^<]*)</div>', html)
        assert m, f"{element_id} must carry a label with a tooltip"
        assert m.group(1) == TIP
        assert m.group(2) == "Towers alerted this session"
    assert '<div class="l">flagged towers</div>' not in html
    assert "Towers flagged this session" not in html      # an earlier wording that overstated what the count was


def test_counters_are_still_filled_from_the_priority_payload(client):
    js = _script(_html(client))
    assert "flaggedTowersEl.textContent = payload.n_flagged" in js
    assert "nocFlaggedTowersEl.textContent = payload.n_flagged" in js


def test_prioritize_tab_shows_the_same_set_and_marks_each_tower(client):
    js = _script(_html(client))
    assert "alerted this session" in js and "alerting now" in js       # summary line
    assert "payload.n_alerting_now" in js
    assert "row.alerting_now" in js
    assert 'class="priority-state now">Alerting now' in js
    assert "Cleared — alerted earlier this session" in js
