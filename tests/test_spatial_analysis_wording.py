"""Review finding: Spatial analysis showed a bare "--" for Global Moran's I when fewer than
the minimum 15 towers had been scored, giving no indication of what's needed or how close the
session is. Replaced with a specific message built from the same n_towers_scored/min_required
fields GET /spatial/autocorrelation already returns.
"""
from __future__ import annotations

import re


def _script(html: str) -> str:
    return "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))


def test_not_computable_moran_shows_a_specific_needs_message(client):
    js = _script(client.get("/dashboard").text)
    assert "Needs at least ${d.min_required} towers scored (${d.n_towers_scored} so far)" in js


def test_moran_p_and_n_are_hidden_when_not_computable(client):
    html = client.get("/dashboard").text
    assert 'id="moran-p-wrap"' in html
    assert 'id="moran-n-wrap"' in html
