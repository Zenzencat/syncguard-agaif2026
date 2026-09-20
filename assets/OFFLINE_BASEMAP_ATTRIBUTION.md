# Offline basemap attribution

**The basemap under the tower markers is a raster image rendered from OpenStreetMap data.**
Data (c) OpenStreetMap contributors, licensed under the Open Database License (ODbL) --
https://www.openstreetmap.org/copyright. The dashboard credits it in the map-corner caption
and the page footer ("(c) OpenStreetMap contributors (ODbL)").

`offline_basemap.geojson` carries two things for the Kubu Raya/Pontianak tower-map bounding box
(the 136 real tower locations' extent plus a 0.05 degree margin):

1. **The raster image (primary layer)** -- under a top-level `raster` member, a single styled
   COLOR image (PNG, base64, 4096 x 4229 px, about 28 m/px). It is a rendering of real
   OpenStreetMap data -- coastline/sea, water, named rivers, motorway through residential
   roads, land cover (forest, wetland, farmland), built-up areas -- produced **once, offline**
   by `tools/build_basemap_raster.py`, which fetched the layers (roads, residential streets,
   land cover, village names) from the OpenStreetMap Overpass API at build time. The colors and
   line widths are this project's own styling and carry no information; every feature drawn is
   OSM data. The dashboard draws it beneath the tower markers on both maps.
2. **A vector layer (fallback only)** -- coastline, water bodies, named rivers/canals, major
   roads, built-up areas and place labels, fetched from the same source by
   `tools/build_basemap.py`, simplified and clipped to a compact GeoJSON feature set. The
   dashboard draws it only if the raster image fails to decode. Village/suburb names (`place`
   features with `rank` 2, a curated subset) are always drawn by the dashboard as vector text
   on top of the image, not baked into it, so they stay crisp when the map zooms.

- Source: OpenStreetMap contributors, via the Overpass API (https://overpass-api.de and
  mirrors)
- License: ODbL (Open Database License) -- (c) OpenStreetMap contributors. A rendering of ODbL
  data is a Produced Work; the attribution is kept exactly as required, and the raw OSM
  extracts used to draw it are cached locally in `tools/basemap_cache/` (not part of the
  served asset).
- Attribution required and shown: dashboard footer and a map-corner caption, both crediting
  "(c) OpenStreetMap contributors (ODbL)" and stating that the basemap is a raster rendered
  from OpenStreetMap data
- Vector categories: `water`, `coastline`, `river` (named rivers/canals only -- unnamed
  drainage/irrigation ditches, common in this region, are excluded to stay within the size
  budget and stay readable), `road` (motorway/trunk/primary/secondary), `builtup`
  (residential/commercial landuse), `place` (city/town labels, e.g. Pontianak)
- Purpose: geographic orientation beneath the REAL tower coordinates when the demo machine
  has no internet connection. Fetched once, offline, at build time -- the running dashboard
  makes no network calls to Overpass, OpenStreetMap, or any tile service.
- The raster is embedded in the existing asset (rather than added as a new file) because the
  API serves only a fixed list of asset files. A dedicated asset route is the post-freeze fix.
- No tile service, font service or network call is involved at runtime. An OPTIONAL live tile
  layer was considered and deliberately not added: the demo must never depend on the network,
  and OpenStreetMap's public tile servers are not meant for application use.

`offline_basemap_natural_earth_fallback.geojson` is the original small Natural Earth
land/coastline/rivers layer the OpenStreetMap basemap replaced. It ships only as a last-resort
fallback the dashboard loads if the asset above fails to fetch -- see
syncguard_interactive_summary.html's `loadBasemapFrom`/`offlineBasemapReady`. Natural Earth is
public domain.

This is an orientation layer, not a navigation map and not evidence about GNSS
interference. The dashboard uses no network tile service at runtime -- both basemap assets
above are served same-origin from `/assets/`.
