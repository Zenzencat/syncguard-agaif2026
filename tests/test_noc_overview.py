"""Task 2 item 1: the NOC tab's overview strip (active alerts, flagged towers, silent
towers, data-mode) must never show different numbers than the global header -- both are set
from the same value in the same place in the dashboard's JS, not computed independently.
"""
from __future__ import annotations

import re


def _script(html: str) -> str:
    return "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))


def test_header_and_noc_overview_elements_both_exist(client):
    html = client.get("/dashboard").text
    for element_id in ("active-alerts", "flagged-towers", "data-mode"):
        assert f'id="{element_id}"' in html
    for element_id in ("noc-active-alerts", "noc-flagged-towers", "noc-silent-towers", "noc-data-mode"):
        assert f'id="{element_id}"' in html


def test_active_alerts_header_and_noc_mirror_are_set_from_the_same_expression(client):
    js = _script(client.get("/dashboard").text)
    m = re.search(r"activeAlertsEl\.textContent\s*=\s*([^;]+);\s*"
                  r"if\(nocActiveAlertsEl\)\s*nocActiveAlertsEl\.textContent\s*=\s*([^;]+);", js)
    assert m, "expected activeAlertsEl and nocActiveAlertsEl to be set consecutively from the same expression"
    assert m.group(1).strip() == m.group(2).strip()


def test_flagged_towers_header_and_noc_mirror_are_set_from_the_same_expression(client):
    js = _script(client.get("/dashboard").text)
    m = re.search(r"flaggedTowersEl\.textContent\s*=\s*([^;]+);\s*"
                  r"if\(nocFlaggedTowersEl\)\s*nocFlaggedTowersEl\.textContent\s*=\s*([^;]+);", js)
    assert m, "expected flaggedTowersEl and nocFlaggedTowersEl to be set consecutively from the same expression"
    assert m.group(1).strip() == m.group(2).strip()


def test_set_data_mode_updates_both_header_and_noc_banner(client):
    js = _script(client.get("/dashboard").text)
    m = re.search(r"function setDataMode\(mode\)\{([^}]*)\}", js)
    assert m, "expected a single setDataMode() that updates both banners"
    body = m.group(1)
    assert "dataModeEl.textContent = mode" in body
    assert "nocDataModeEl" in body and "textContent = mode" in body
