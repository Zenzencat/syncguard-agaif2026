"""Task 1: the OpenStreetMap-derived vector layer of the basemap asset (tools/build_basemap.py ->
assets/offline_basemap.geojson; the dashboard's primary layer is the raster rendered from the same data).

Locations of the 136 towers are REAL. The basemap under them (coastline, water, rivers,
roads, built-up areas, place labels) is REAL OpenStreetMap data for the same bounding box,
fetched once at build time -- not a runtime call. Attribution: (c) OpenStreetMap
contributors, Open Database License (ODbL).
"""
from __future__ import annotations

ATTRIBUTION = "(c) OpenStreetMap contributors (ODbL)"
EXTERNAL_MAP_HOSTS = ("tile.openstreetmap.org", "overpass-api.de", "openstreetmap.ru",
                      "kumi.systems", "{s}.tile.")


def test_basemap_asset_is_served_with_attribution(client):
    r = client.get("/assets/offline_basemap.geojson")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/geo+json")
    body = r.json()
    assert body["type"] == "FeatureCollection"
    assert body["attribution"] == ATTRIBUTION
    assert body["features"]


def test_basemap_asset_stays_under_the_size_budget():
    from api.main import BASEMAP_PATH
    size = BASEMAP_PATH.stat().st_size
    assert size < int(1.5 * 1024 * 1024), f"basemap asset is {size} bytes, over the 1.5MB budget"


def test_basemap_covers_the_expected_categories(client):
    body = client.get("/assets/offline_basemap.geojson").json()
    categories = {f["properties"]["category"] for f in body["features"]}
    assert {"water", "river", "road", "place"}.issubset(categories)


def test_basemap_includes_pontianak_and_the_kapuas_river(client):
    body = client.get("/assets/offline_basemap.geojson").json()
    place_names = {f["properties"].get("name") for f in body["features"]
                   if f["properties"]["category"] == "place"}
    assert "Pontianak" in place_names
    river_names = {f["properties"].get("name") for f in body["features"]
                   if f["properties"]["category"] == "river"}
    assert any(name and "kapuas" in name.lower() for name in river_names)


def test_dashboard_shows_visible_osm_attribution(client):
    html = client.get("/dashboard").text
    assert ATTRIBUTION in html


def test_dashboard_makes_no_reference_to_live_map_tile_hosts(client):
    html = client.get("/dashboard").text
    for host in EXTERNAL_MAP_HOSTS:
        assert host not in html, f"dashboard must not reference {host} at runtime"
