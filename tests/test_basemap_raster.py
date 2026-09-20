"""The styled COLOR base image (tools/build_basemap_raster.py) embedded in the basemap asset.

The image is a rendering of REAL OpenStreetMap data for the tower bounding box; its colors and
line widths are styling only. It is built once offline and served from the existing same-origin
asset -- no tile server or runtime fetch. Attribution: (c) OpenStreetMap contributors (ODbL).

Static checks against the asset and the served dashboard script (no JS runner in this repo); the
behaviour -- image drawn beneath the WebGL markers, on both maps, light and dark theme -- was
verified in a real browser (see the visual-polish report).
"""
from __future__ import annotations

import base64
import re
import struct

ATTRIBUTION = "(c) OpenStreetMap contributors (ODbL)"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


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


def _asset(client) -> dict:
    return client.get("/assets/offline_basemap.geojson").json()


def test_asset_carries_a_png_raster_that_matches_its_declared_size(client):
    raster = _asset(client)["raster"]
    assert raster["format"] == "png" and raster["encoding"] == "base64"
    png = base64.b64decode(raster["data"])
    assert png[:8] == PNG_MAGIC
    width, height = struct.unpack(">II", png[16:24])          # IHDR
    assert (width, height) == (raster["width"], raster["height"])
    assert width >= 4096, "the base image should be high resolution (>= 4096 px wide)"


def test_raster_covers_exactly_the_vector_basemap_bounding_box(client):
    body = _asset(client)
    raster = body["raster"]
    assert raster["bbox"] == body["bbox"]
    west, south, east, north = raster["bbox"]
    # equirectangular: pixels must be square in degrees, or markers would not line up
    deg_per_px_x = (east - west) / raster["width"]
    deg_per_px_y = (north - south) / raster["height"]
    assert abs(deg_per_px_x - deg_per_px_y) / deg_per_px_x < 0.005


def test_raster_carries_osm_attribution_and_says_it_is_styling_only(client):
    raster = _asset(client)["raster"]
    assert raster["attribution"] == ATTRIBUTION
    assert "styling only" in raster["note"]
    assert "OpenStreetMap" in raster["note"]


def test_minor_place_labels_are_ranked_and_deduplicated(client):
    places = [f["properties"] for f in _asset(client)["features"]
              if f["properties"]["category"] == "place"]
    names = [p["name"].lower() for p in places]
    assert len(names) == len(set(names)), "duplicate place names"
    rank = {p["name"]: p.get("rank") for p in places}
    assert rank["Pontianak"] == 1
    minor = [p for p in places if p.get("rank") == 2]
    assert 10 <= len(minor) <= 40, "a curated subset, not every village"


def test_dashboard_draws_the_raster_on_both_maps_and_falls_back_to_vector(client):
    js = _script(client.get("/dashboard").text)
    assert js.count("images: basemapLayoutImages()") == 2          # NOC map + Spatial analysis map
    images = _function_body(js, "basemapLayoutImages")
    assert "layer:'below'" in images and "xref:'x'" in images and "yref:'y'" in images
    traces = _function_body(js, "basemapTraces")
    assert "if(basemapRaster) return [];" in traces                 # vector traces only as fallback
    decode = _function_body(js, "decodeBasemapRaster")
    assert decode.count("resolve(null)") >= 2                       # decode error / exception -> vector
    assert "img.onerror" in decode
    # a failed decode must never reject: loadBasemapFrom would then fall to Natural Earth
    assert "reject" not in decode


def test_grid_lines_are_hidden_while_the_image_is_shown(client):
    js = _script(client.get("/dashboard").text)
    assert js.count("showgrid: !basemapRaster") == 4                # x and y on both maps


def test_live_map_redraws_when_the_image_arrives(client):
    js = _script(client.get("/dashboard").text)
    body = _function_body(js, "renderLiveMap")
    assert "basemapLayoutImages().length" in body                   # part of the skip-hash


def test_dark_theme_dims_the_image_and_caption_still_credits_osm(client):
    html = client.get("/dashboard").text
    assert re.search(r':root:not\(\[data-theme="light"\]\) #live-map image[^{]*\{filter:brightness', html)
    assert "Basemap: styled OpenStreetMap render, bundled offline" in html
    assert ATTRIBUTION in html


def test_no_new_network_references_in_the_dashboard(client):
    html = client.get("/dashboard").text
    # hostnames only: the About text legitimately mentions the Overpass API by name
    for host in ("tile.openstreetmap.org", "overpass-api.de", "api.mapbox.com", "maptiler.com",
                 "basemaps.cartocdn.com", "stadiamaps.com", "openstreetmap.ru", "kumi.systems"):
        assert host not in html.lower(), f"dashboard must not reference {host}"
