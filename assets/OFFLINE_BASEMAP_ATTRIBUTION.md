# Offline basemap attribution

`offline_basemap.geojson` is a clipped land/coastline polygon for the Kubu Raya/Pontianak
tower-map bounding box. It was derived from the Natural Earth `naturalearth_lowres` country
dataset distributed with the installed Pyogrio test fixtures.

- Source: Natural Earth, https://www.naturalearthdata.com/
- License: public domain
- Layer: Indonesia land boundary/coastline, clipped to 109.05–110.15° E and
  0.95° S–0.20° N
- Purpose: geographic orientation beneath the REAL tower coordinates when the demo machine
  has no internet connection

This is a coarse orientation layer, not a navigation map and not evidence about GNSS
interference. The dashboard uses no network tile service at runtime.
