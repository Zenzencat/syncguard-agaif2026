"""NOC + Spatial analysis maps: wheel/pinch zoom instead of drag-select zoom, and marker halos
that keep markers legible on the colored raster basemap.

Static checks against the served dashboard script (same convention as
tests/test_dashboard_map_click.py -- single HTML file, inline JS, no JS test runner in this
repo). The behavioural side -- real wheel zoom, drag-pan, click-to-select before and after
zooming, on both maps -- was verified in a real browser; see phase_reports/.
"""
from __future__ import annotations

import re


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


def test_both_maps_zoom_with_the_scroll_wheel(client):
    js = _script(client.get("/dashboard").text)
    spatial = re.search(r"Plotly\.react\('map',[^;]*", js)
    noc = re.search(r"Plotly\.react\('live-map',[^;]*", js)
    assert spatial and noc
    assert "scrollZoom:true" in spatial.group(0), "Spatial analysis map must enable scrollZoom"
    assert "scrollZoom:true" in noc.group(0), "NOC map must enable scrollZoom"


def test_both_maps_pan_on_drag_not_drag_select_zoom(client):
    js = _script(client.get("/dashboard").text)
    assert "dragmode:'pan'" in _function_body(js, "severityLayout")      # Spatial analysis map
    assert "dragmode:'pan'" in _function_body(js, "renderLiveMap")       # NOC map


def test_zoom_is_purely_client_side_no_tile_or_map_service(client):
    """Scroll zoom must stay on the embedded raster: no tiled-map trace types and no map-tile
    URLs (the offline-only design, and the 'only 127.0.0.1 contacted' check, depend on it)."""
    html = client.get("/dashboard").text
    for forbidden in ("scattermapbox", "scattermap\"", "choroplethmapbox", "mapbox://",
                      "tile.openstreetmap", "{z}/{x}/{y}", "api.mapbox", "basemaps.cartocdn"):
        assert forbidden not in html, f"dashboard must not reference {forbidden!r}"


def test_halo_traces_never_intercept_hover_or_clicks(client):
    """Halos sit under the markers. They must not carry customdata (the plotly_click handler
    keys on it) and must skip hover, or a click on a halo edge could select nothing / the wrong
    tower."""
    js = _script(client.get("/dashboard").text)
    for fn in ("markerHalo", "ringHalo"):
        body = _function_body(js, fn)
        assert "hoverinfo:'skip'" in body, f"{fn} must not capture hover"
        assert "customdata" not in body, f"{fn} must not carry customdata"


def test_halos_are_drawn_beneath_the_markers_they_back(client):
    js = _script(client.get("/dashboard").text)
    body = _function_body(js, "renderLiveMap")
    order = re.search(r"const traces = \[(.*?)\]\.filter\(Boolean\)", body, re.S).group(1)
    names = [n.strip() for n in order.replace("...basemapTraces(c)", "basemap").split(",") if n.strip()]
    for halo, marker in (("activeHalo", "activeTrace"), ("lisaHalo", "lisaTrace"),
                         ("selectedHalo", "selectedTrace")):
        assert names.index(halo) < names.index(marker), f"{halo} must be drawn before {marker}"
    assert names.index("activeTrace") < names.index("alertTrace")   # diamonds on top of severity circles
    spatial = _function_body(js, "buildTraces")
    assert spatial.index("markerHalo(lons, lats") < spatial.index("towerTrace, epicenterTrace")


def test_alert_diamond_is_fully_opaque(client):
    """The diamond used to inherit a translucent default (rendered #C16C6B instead of its
    #A83232 on the map); the severity circle beneath it is opaque for the same reason."""
    js = _script(client.get("/dashboard").text)
    m = re.search(r"symbol:'diamond', color:'#A83232', opacity:1,", js)
    assert m, "alerting diamonds must set opacity:1 explicitly"


def test_place_labels_can_be_lifted_off_the_markers_that_must_stay_visible(client):
    js = _script(client.get("/dashboard").text)
    assert "function basemapPlaceAnnotations(c, avoid)" in js
    assert "basemapPlaceAnnotations(c, [...alerting, ...cleared, ...lisaSig]" in _function_body(js, "renderLiveMap")
