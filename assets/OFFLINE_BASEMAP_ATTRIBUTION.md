# Offline basemap attribution

`offline_basemap.geojson` contains clipped land/coastline polygons and river centerlines for
the Kubu Raya/Pontianak tower-map bounding box. It was derived from the official Natural
Earth 1:10m physical layers and simplified into a small offline asset.

- Source: Natural Earth, https://www.naturalearthdata.com/
- License: public domain
- Layers: Natural Earth 1:10m land and rivers/lake centerlines, clipped to
  108.8–110.4° E and 1.1° S–0.4° N
- Purpose: geographic orientation beneath the REAL tower coordinates when the demo machine
  has no internet connection

This is an orientation layer, not a navigation map and not evidence about GNSS
interference. The dashboard uses no network tile service at runtime.
