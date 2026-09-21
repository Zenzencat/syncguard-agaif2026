"""Regression test for a real bug found in review: the basemap layer's trace data was
always present and correctly built (confirmed live, both themes, 2560x1400 and 1080x900 --
`offlineBasemap` loaded and all 7 basemap traces had non-empty geometry every time), but its
colors were so close to the page background that it was visually indistinguishable from "no
basemap at all" -- e.g. dark theme's land fill (#18283A) was 27 RGB-units from the paper
background (#0E1826), barely a shade darker.

This test parses chartColors() out of the dashboard and asserts a minimum Euclidean RGB
distance between the page background and each basemap layer color, in both themes, so a
color choice this close can't silently ship again. It is a proxy for "visually present", not
a pixel-level render check (no browser automation dependency in this suite).
"""
from __future__ import annotations

import re

MIN_CONTRAST = 50.0  # Euclidean RGB distance; old (broken) values measured ~27-32


def _script(html: str) -> str:
    return "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _distance(a: str, b: str) -> float:
    ar, ag, ab = _hex_to_rgb(a)
    br, bg, bb = _hex_to_rgb(b)
    return ((ar - br) ** 2 + (ag - bg) ** 2 + (ab - bb) ** 2) ** 0.5


def _extract_theme_colors(js: str, light: bool) -> dict:
    m = re.search(r"return light \? \{(.*?)\} : \{(.*?)\};", js, re.S)
    assert m, "expected chartColors() to return light ? {...} : {...}"
    block = m.group(1) if light else m.group(2)
    colors = dict(re.findall(r"(\w+):\s*'(#[0-9A-Fa-f]{6})'", block))
    for key in ("paper", "mapLand", "mapWater", "mapRoad", "mapBuiltup", "mapCoast"):
        assert key in colors, f"missing {key} in {'light' if light else 'dark'} theme colors"
    return colors


def test_basemap_layers_are_visually_distinguishable_from_the_background_light_theme(client):
    js = _script(client.get("/dashboard").text)
    colors = _extract_theme_colors(js, light=True)
    for key in ("mapLand", "mapWater", "mapBuiltup", "mapRoad"):
        dist = _distance(colors["paper"], colors[key])
        assert dist >= MIN_CONTRAST, (
            f"light theme {key} ({colors[key]}) is only {dist:.1f} RGB-units from the page "
            f"background ({colors['paper']}) -- too close to read as a basemap layer"
        )


def test_basemap_layers_are_visually_distinguishable_from_the_background_dark_theme(client):
    js = _script(client.get("/dashboard").text)
    colors = _extract_theme_colors(js, light=False)
    for key in ("mapLand", "mapWater", "mapBuiltup", "mapRoad"):
        dist = _distance(colors["paper"], colors[key])
        assert dist >= MIN_CONTRAST, (
            f"dark theme {key} ({colors[key]}) is only {dist:.1f} RGB-units from the page "
            f"background ({colors['paper']}) -- too close to read as a basemap layer"
        )


def test_water_is_distinguishable_from_land_in_both_themes(client):
    js = _script(client.get("/dashboard").text)
    for light in (True, False):
        colors = _extract_theme_colors(js, light=light)
        dist = _distance(colors["mapLand"], colors["mapWater"])
        assert dist >= 30.0, (
            f"{'light' if light else 'dark'} theme mapWater is only {dist:.1f} RGB-units "
            f"from mapLand -- water bodies would not read as distinct from land"
        )


def test_basemap_layer_is_included_in_both_maps_trace_lists(client):
    """Guards the specific way "the basemap layer is missing from the figure" would actually
    happen in this codebase: someone editing the trace array and dropping the
    `...basemapTraces(c)` spread. The trace data itself is asset-tested in test_basemap.py."""
    js = _script(client.get("/dashboard").text)
    assert "return [...basemapTraces(c)," in js, (
        "Spatial analysis map's buildTraces() must draw the basemap as its bottom layer"
    )
    assert "const traces = [...basemapTraces(c), base," in js, (
        "NOC map's renderLiveMap() must draw the basemap as its bottom layer"
    )
