Population: Meta Data for Good / CIESIN High Resolution Population Density Maps (HDX, CC BY 4.0), circa 2020, estimated.

File: `tower_exposure_HRSL.csv` -- ESTIMATED population within 1 km and 2 km of each of the 136 real telecom tower locations (Kubu Raya/Pontianak), computed from the general-population GeoTIFF (1 arc-second, ~2020). `pop_within_*` are float estimates; `px_within_*` is the count of raster pixels inside the buffer.

Use as an exposure proxy only -- "estimated people within 2 km", not "people served". Per-tower values overlap and must NEVER be summed or totaled. See PRIORITIZE_NOTES.md.

Only the per-tower result table is included here. The raster and the extracted subset are not part of this repository.
