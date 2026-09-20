# Offline basemap attribution

`offline_basemap.geojson` is a small vector basemap (coastline, water bodies, named
rivers/canals, major roads, built-up areas, place labels) plus an embedded color base image (below) for the Kubu Raya/Pontianak
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

## Styled color base image (primary layer)

`offline_basemap.geojson` also carries, under a top-level `raster` member, a single styled
COLOR base image (PNG, base64, 4096 x 4229 px, about 28 m/px) of the same bounding box. It is a
rendering of real OpenStreetMap data -- coastline/sea, water, named rivers, motorway through
residential roads, land cover (forest, wetland, farmland), built-up areas -- produced **once,
offline** by `tools/build_basemap_raster.py`, which fetched the extra layers (minor roads,
residential streets, land cover, village names) from the Overpass API at build time. The colors
and line widths are this project's own styling and carry no information; every feature drawn is
OSM data. The dashboard draws it beneath the tower markers as the primary map layer and falls
back to the vector layer above if the image fails to decode. It is embedded in the existing
asset (rather than added as a new file) because the API serves only a fixed list of asset files.

- Same source, license and attribution as above: (c) OpenStreetMap contributors (ODbL),
  shown in the dashboard footer and map-corner caption. A rendering of ODbL data is a Produced
  Work; the attribution is kept exactly as required.
- Village/suburb names (`place` features with `rank` 2, a curated subset) are drawn by the
  dashboard as vector text on top of the image, not baked into it, so they stay crisp when the
  map zooms to an incident.
- No tile service, font service or network call is involved at runtime. An OPTIONAL live tile
  layer was considered and deliberately not added: the demo must never depend on the network,
  and OpenStreetMap's public tile servers are not meant for application use.

`offline_basemap_natural_earth_fallback.geojson` is the original small Natural Earth
land/coastline/rivers layer this basemap replaced. It ships only as a fallback the dashboard
loads if the OSM asset above fails to fetch -- see syncguard_interactive_summary.html's
`loadBasemapFrom`/`offlineBasemapReady`. Natural Earth is public domain.

This is an orientation layer, not a navigation map and not evidence about GNSS
interference. The dashboard uses no network tile service at runtime -- both basemap assets
above are served same-origin from `/assets/`.
