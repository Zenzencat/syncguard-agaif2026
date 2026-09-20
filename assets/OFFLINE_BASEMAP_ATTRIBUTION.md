# Offline basemap attribution

`offline_basemap.geojson` is a small vector basemap (coastline, water bodies, named
rivers/canals, major roads, built-up areas, place labels) for the Kubu Raya/Pontianak
tower-map bounding box (the 136 real tower locations' extent plus a 0.05 degree margin). It
was fetched **once**, with network access, from OpenStreetMap via the Overpass API, then
simplified and clipped into a compact offline asset. See `tools/build_basemap.py` for the
reproducible build script and exact queries.

- Source: OpenStreetMap contributors, via the Overpass API (https://overpass-api.de and
  mirrors)
- License: ODbL (Open Database License) -- (c) OpenStreetMap contributors
- Attribution required and shown: dashboard footer and a map-corner caption, both reading
  "(c) OpenStreetMap contributors (ODbL)"
- Categories: `water`, `coastline`, `river` (named rivers/canals only -- unnamed
  drainage/irrigation ditches, common in this region, are excluded to stay within the size
  budget and stay readable), `road` (motorway/trunk/primary/secondary), `builtup`
  (residential/commercial landuse), `place` (city/town labels, e.g. Pontianak)
- Purpose: geographic orientation beneath the REAL tower coordinates when the demo machine
  has no internet connection. Fetched once, offline, at build time -- the running dashboard
  makes no network calls to Overpass, OpenStreetMap, or any tile service.

`offline_basemap_natural_earth_fallback.geojson` is the original small Natural Earth
land/coastline/rivers layer this basemap replaced. It ships only as a fallback the dashboard
loads if the OSM asset above fails to fetch -- see syncguard_interactive_summary.html's
`loadBasemapFrom`/`offlineBasemapReady`. Natural Earth is public domain.

This is an orientation layer, not a navigation map and not evidence about GNSS
interference. The dashboard uses no network tile service at runtime -- both basemap assets
above are served same-origin from `/assets/`.
