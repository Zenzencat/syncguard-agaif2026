# Proof-of-method validation via GPSJam data

## Why this exists

Every attribution mechanism tried on the Indonesian tower network was undone by a different
structural artifact of the mechanism itself (see `SPATIAL_STATISTICS.md`'s three-mechanism
synthesis). None of that ever tested whether Moran's I / LISA *as implemented in this
codebase* actually detects real spatial clustering — it only ever tested placeholder
attribution data, which was never going to resolve either way. This validates that separately:
the exact same methodology, run on real, independently-documented clustered data, with no
synthetic attribution involved at all.

**Kept fully separate from the Slide 7/8 tower narrative** — different domain (aircraft-
altitude GPS interference, not ground-based telecom spoofing), different geometry (global H3
hex grid, not 136 real towers), different purpose (validate the method, not narrate
SyncGuard's problem). Nothing here touches the deck, the narration script, or its word count.

## Status: complete — both checks run, once each, real data

`gpsjam.org` itself was never reachable from this environment (egress-blocked; see "Access
trail" below) — reported rather than routed around, per this environment's own operating rule
for a 403 policy block. The user fetched the manifest plus seven real daily CSVs
(2024-04-01 through 2024-04-07) from a machine that could reach the site and supplied them
directly; `validate_moran_gpsjam.py` gained a `--data-dir` mode to read them locally instead of
over HTTP, and both checks ran against real data, once each, no repeated tweaking for a better
number. Raw files are committed at `gpsjam_raw/` for provenance.

## Step 1 — confirmed data schema and access trail

`github.com/wiseman/gpsjam.org` (the canonical repo) and `gpsjam.org` itself both returned
blocked/failed connections directly from this environment. A public mirror,
`github.com/guofengji/gpsjam.org`, was reachable via `raw.githubusercontent.com` and — per its
own README — is a straight clone of the canonical site's code, so its `views/index.ejs` (the
page that actually loads and renders the data) was read directly to confirm the real schema,
not inferred from secondhand descriptions:

- **Per-day data**: `GET https://gpsjam.org/data/{YYYY-MM-DD}-h3_4.csv`
  Columns (from the site's own `d3.csv` parsing code, confirmed identical in the real files
  received): `hex`, `count_good_aircraft`, `count_bad_aircraft`. `hex` is an **H3 resolution-4**
  cell index. ~40,000–41,000 hex rows per day, globally.
- **Manifest of available dates**: `GET https://gpsjam.org/data/manifest.csv`
  Columns: `date`, `suspect` (true/false, flags known-incomplete data), `num_bad_aircraft_hexes`,
  `source` (an extra column beyond what `index.ejs` reads — additive, doesn't affect anything
  here). Real manifest received: 1,673 rows, **2022-02-14 through 2026-09-15**.
- **GPSJam's own interference formula**, copied verbatim from their client-side JS
  (`interferenceLevel()` in `index.ejs`) rather than reconstructed from a description:
  ```
  bad_frac = (count_bad_aircraft - 1) / (count_bad_aircraft + count_good_aircraft)
  low  if bad_frac < 0.02
  med  if bad_frac < 0.10
  high otherwise
  ```
  The `-1` is GPSJam's own choice (reads as a small-sample continuity correction) — reused
  verbatim here for fidelity to their own definition, as the continuous severity value fed to
  Moran's I, the same role `predict_proba`-derived severity plays for the SyncGuard towers.
- **License**: CC-BY (per Bellingcat's toolkit documentation of the project).

### Spatial weights: hex contiguity, not KNN

Unlike the 136 irregularly-spaced real Telkomsel towers (where KNN k=5 is the defensible
choice — see `SPATIAL_STATISTICS.md`), H3 hex cells form a regular grid where every interior
cell has exactly 6 edge-adjacent neighbors. There is no rook-vs-queen ambiguity on a hex grid
the way there is on a square grid (every neighbor shares a full edge, not just a corner) — so
first-ring contiguity (`h3.grid_disk(hex, 1)` minus the cell itself) is the single natural,
standard choice, used instead of KNN throughout.

### Access trail (for the record)

Confirmed directly, not assumed:
```
$ curl -sS -o /dev/null -w "HTTP %{http_code}\n" https://gpsjam.org/about
curl: (56) CONNECT tunnel failed, response 403
```
A `CONNECT` failure at the egress proxy — this environment's tooling documents that signature
as an organization-level policy decision, not a transient error. `raw.githubusercontent.com`
was reachable (used for Step 1), but the actual daily interference CSVs are served from
`gpsjam.org/data/` directly — same blocked host, no separate CDN domain to fall back to. A
search for an independently-hosted mirror of the actual daily CSV data found none. Resolved by
the user fetching the files from their own machine and uploading them here.

## Step 2 — proof-of-method: Black Sea / Crimea, 2024-04-01 to 2024-04-07

Real data, run once, this is the result:

```
n_hexes: 69
global_moran_i: +0.6272
p_value: 0.0010          (SIGNIFICANT, alpha=0.05)
z_score: +6.809
expected_i: -0.0147
significant LISA hexes: 32/69 (16 High-High hotspot, 12 Low-Low coldspot,
                                3 High-Low outlier, 1 Low-High outlier)
```

Four hexes (`841e497ffffffff`, `841e5c3ffffffff`, `841e5d1ffffffff`, `842d2adffffffff`) are
graph-boundary islands with zero in-region neighbors — an expected, disclosed limitation of any
contiguity-weights graph cut off by a bounding box, contributing nothing to the statistic
rather than being silently dropped or imputed.

**The method works.** Strong, significant positive spatial autocorrelation, on real
independently-corroborated jamming data, using this project's exact Moran's I / LISA code path
(hex contiguity swapped in for KNN, everything else identical: `RNG_SEED=42`, 999 permutations,
`esda`/`libpysal`). The 16-hex High-High hotspot band sits at roughly 32–34°E, 44.6–46°N —
geographically consistent with the Crimea/Kerch Strait/northern Black Sea corridor EASA's
advisories describe, not scattered at random. This is the answer to the question this whole
exercise exists to ask: Moran's I / LISA, as implemented in this codebase, does detect real
spatial clustering when real per-event ground truth is available. The tower-network results
were never a method failure — they were an attribution-data problem, and this is the
confirmation.

Plots: `spatial_processed/gpsjam_black_sea_crimea_2024-04-01_2024-04-07_lisa_map.png`,
`..._moran_scatter.png`. Per-hex data:
`spatial_processed/gpsjam_black_sea_crimea_2024-04-01_2024-04-07.csv`.

## Step 3 — honest regional check: Kubu Raya / Pontianak, 2024-04-01 to 2024-04-07

```
n_hexes: 0
NOT COMPUTABLE
```

**Zero H3 resolution-4 hexes with any recorded aircraft data — not a filtering bug, checked
directly.** The bounding box's six corner/center points all resolve to hexes within one
res-3 parent's seven res-4 children; every one of those seven hex IDs was grepped directly
against all seven raw daily CSVs and appears in **zero of the 49 file-hex checks (7 hexes × 7
days)**. This is a genuine null result about ADS-B Exchange coverage in this specific airspace
during this specific week, not the "too few hexes for statistical power" concern flagged
before data arrived (that concern doesn't even get to apply here — there isn't a small sample
to run the statistic on below-power, there's no sample at all).

**Read plainly**: this is a preventative system for a region with no currently-documented GNSS
interference incidents on record — consistent with the region not being a known jamming
hotspot, and additionally consistent with the region simply not carrying enough transponder-
equipped air traffic for GPSJam's ADS-B-Exchange-sourced methodology to have any signal there
at all (dense international air-traffic-control-covered corridors and jamming hotspots like the
Black Sea are exactly where ADS-B Exchange coverage and GPSJam's dataset are richest; a
regional West Kalimantan airspace is neither). Neither reading is more or less honest than the
other — both are true, and this doesn't prove interference *isn't* happening, only that this
particular real, independently-sourced dataset has nothing to say about it either way, for this
specific week.

**Not rerun with a wider region or a different week** — that would be exactly the "keep tuning
the mechanism until it says something" the discipline of this whole exercise exists to
avoid. If a wider check (e.g., all of West Kalimantan, or a longer date range) is wanted, that's
a deliberate follow-up decision, not something to default into after seeing a null result.

Per-hex data (empty, for the record): `spatial_processed/gpsjam_kubu_raya_pontianak_2024-04-01_2024-04-07.csv`.

## Where this lives

- `gpsjam_raw/` — the real manifest.csv and seven daily CSVs, as received, committed for
  provenance (same precedent as `spatial_raw/.../menaratelepon_ar_50k.csv`).
- `validate_moran_gpsjam.py` — the full pipeline: fetch (network or `--data-dir` local),
  GPSJam's own severity formula, H3 first-ring contiguity weights, `esda.moran.Moran` /
  `Moran_Local` with this project's exact conventions, plotting. Rerun with:
  ```
  python validate_moran_gpsjam.py proof-of-method --start 2024-04-01 --end 2024-04-07 --data-dir gpsjam_raw
  python validate_moran_gpsjam.py region-check --start 2024-04-01 --end 2024-04-07 --data-dir gpsjam_raw
  ```
  (Omit `--data-dir` to hit `gpsjam.org` directly, from an environment that can reach it.)
- `spatial_processed/gpsjam_*` — per-hex CSVs and plots from both runs.

## Step 4 — deliberately not decided here

This document is the write-up. Whether/how this belongs in the live pitch — a compressed
callout, a backup slide only shown if asked in Q&A, or left as repo/documentation strength
alone — stays a deliberate decision to be made explicitly, not defaulted into. Nothing here has
touched the deck, the narration script, or its word count.
