# Proof-of-method validation via GPSJam data — status: blocked on data access

## Why this exists

Every attribution mechanism tried on the Indonesian tower network was undone by a different
structural artifact of the mechanism itself (see `SPATIAL_STATISTICS.md`'s three-mechanism
synthesis). None of that ever tested whether Moran's I / LISA *as implemented in this
codebase* actually detects real spatial clustering — it only ever tested placeholder
attribution data, which was never going to resolve either way. This task set out to settle
that separately: run the exact same methodology on real, independently-documented clustered
data, with no synthetic attribution involved at all.

**Kept fully separate from the Slide 7/8 tower narrative** — different domain (aircraft-
altitude GPS interference, not ground-based telecom spoofing), different geometry (global H3
hex grid, not 136 real towers), different purpose (validate the method, not narrate
SyncGuard's problem). Nothing here touches the deck, the narration script, or its word count.

## Status: Step 1 complete, Steps 2–3 blocked on network access

**Step 1 (understand the data) is done — confirmed against the site's actual source, not
assumed.** Steps 2 and 3 (the actual runs) could not be executed: `gpsjam.org` is blocked by
this environment's network egress policy. This is documented below rather than routed around,
per this environment's own operating rule for a 403 policy block: report it, don't retry or
work around it.

## Step 1 — confirmed data schema and access trail

`github.com/wiseman/gpsjam.org` (the canonical repo) and `gpsjam.org` itself both returned
blocked/failed connections directly from this environment. A public mirror,
`github.com/guofengji/gpsjam.org`, was reachable via `raw.githubusercontent.com` and — per its
own README — is a straight clone of the canonical site's code, so its `views/index.ejs` (the
page that actually loads and renders the data) was read directly to confirm the real schema,
not inferred from secondhand descriptions:

- **Per-day data**: `GET https://gpsjam.org/data/{YYYY-MM-DD}-h3_4.csv`
  Columns (from the site's own `d3.csv` parsing code): `hex`, `count_good_aircraft`,
  `count_bad_aircraft`. `hex` is an **H3 resolution-4** cell index.
- **Manifest of available dates**: `GET https://gpsjam.org/data/manifest.csv`
  Columns: `date`, `suspect` (true/false, flags known-incomplete data), `num_bad_aircraft_hexes`.
  Data available from **2022-02-14** onward (per Bellingcat's toolkit documentation).
- **GPSJam's own interference formula**, copied verbatim from their client-side JS
  (`interferenceLevel()` in `index.ejs`) rather than reconstructed from a description:
  ```
  bad_frac = (count_bad_aircraft - 1) / (count_bad_aircraft + count_good_aircraft)
  low  if bad_frac < 0.02
  med  if bad_frac < 0.10
  high otherwise
  ```
  The `-1` is GPSJam's own choice (reads as a small-sample continuity correction) — reused
  verbatim here for fidelity to their own definition, as the continuous severity value that
  would feed Moran's I, the same role `predict_proba`-derived severity plays for the SyncGuard
  towers.
- **License**: CC-BY (per Bellingcat's toolkit documentation of the project).

### Spatial weights: hex contiguity, not KNN

Unlike the 136 irregularly-spaced real Telkomsel towers (where KNN k=5 is the defensible
choice — see `SPATIAL_STATISTICS.md`), H3 hex cells form a regular grid where every interior
cell has exactly 6 edge-adjacent neighbors. There is no rook-vs-queen ambiguity on a hex grid
the way there is on a square grid (every neighbor shares a full edge, not just a corner) — so
first-ring contiguity (`h3.grid_disk(hex, 1)` minus the cell itself) is the single natural,
standard choice, used instead of KNN in the prepared script below.

## The blocker: `gpsjam.org` is not reachable from this environment

Confirmed directly, not assumed:
```
$ curl -sS -o /dev/null -w "HTTP %{http_code}\n" https://gpsjam.org/about
curl: (56) CONNECT tunnel failed, response 403
```
This is a `CONNECT` failure at the egress proxy, the same signature this project's tooling
documents as an organization-level policy decision, not a transient error — the operating rule
for that signature is to report the blocked host, not retry or route around it, so that's what
this document does.

`raw.githubusercontent.com` was reachable (used for Step 1 above), but the actual daily
interference CSVs are served from `gpsjam.org/data/` directly (confirmed from `index.ejs`'s
relative `'data/' + h3Filename` URL) — same blocked host, no separate CDN domain to fall back
to. A search for an independently-hosted mirror of the actual daily CSV files (Kaggle, Zenodo,
GitHub) found none; only descriptions of the site, not redistributions of its data.

## A second, independent concern for Step 3 (Kubu Raya / Pontianak) — worth flagging even if
## access is restored

H3 resolution 4 hexagons average **1,770.3 km²** each (confirmed directly: `h3-py`'s own area
function). Kubu Raya Regency is **8,568 km²**; Pontianak city is **~108–118 km²** (both figures
from public sources, not this project's own tower data) — combined, roughly **8,680–8,690
km²**. That's on the order of **5–10 H3 res-4 hexagons**, depending on how the grid happens to
tile the region's irregular boundary. This project's own code already establishes 15 as the
minimum unit count for permutation-test power to mean anything (`MIN_TOWERS_FOR_STATS` in
`api/spatial_stats.py`) — the same threshold is reused here as `MIN_HEXES_FOR_STATS`. A region
this size is likely to land **below that floor at H3 resolution 4**, independent of whether any
interference is actually recorded there.

This means Step 3 has two structurally distinct ways to come back "empty," and they should not
be conflated if it happens: **(a)** genuinely no interference recorded (a real, legitimate,
reportable null result — "no currently-documented GNSS interference incidents in this region,"
exactly as anticipated in the original task), versus **(b)** too few hex cells at this
resolution to run the statistic at all, a grid-resolution artifact that says nothing about
whether interference occurred. If it happens, the prepared script (below) reports which one
occurred rather than a single ambiguous "no result" — and if it's **(b)**, the honest fix is
widening the region (e.g., all of West Kalimantan) for that specific check, not lowering the
statistical-power threshold to force a number out of too little data.

## What's prepared, and what isn't

`validate_moran_gpsjam.py` (repo root) implements the full pipeline: fetch manifest + daily
CSVs, apply GPSJam's own severity formula, build H3 first-ring contiguity weights, run
`esda.moran.Moran` / `Moran_Local` with this project's exact conventions (`RNG_SEED=42`, 999
permutations, `MIN_HEXES_FOR_STATS=15`, an explicit "not computable" result instead of a
misleading early number — mirroring `api/spatial_stats.py` throughout). Two modes:

```
python validate_moran_gpsjam.py proof-of-method --start 2022-05-01 --end 2022-05-07   # Black Sea/Crimea
python validate_moran_gpsjam.py region-check --start 2022-05-01 --end 2022-05-07      # Kubu Raya/Pontianak
```

**What was verified**: the script compiles cleanly; every `h3-py` call used
(`cell_to_latlng`, `grid_disk`, `latlng_to_cell`) was confirmed against the actually-installed
`h3` v4.5.0 API; the `libpysal.weights.W(neighbors_dict)` → `esda.moran.Moran`/`Moran_Local`
call sequence was confirmed mechanically correct against a small abstract synthetic graph (not
GPSJam-shaped data — deliberately, to avoid producing anything that could be mistaken for a
real result).

**What was NOT verified**: the script has never run against a real GPSJam CSV, because no real
GPSJam CSV was reachable from this environment. There is no result to report for Steps 2 or 3
— not a weak one, not a null one, none. Reporting a number here would mean fabricating it, which
this project does not do.

## Decision needed

This is blocked on something outside this session's control. Ways forward, none chosen yet:
1. **Allowlist `gpsjam.org` for this session** (if that's within reach) and rerun both modes —
   the script is ready to go the moment the host is reachable.
2. **Fetch the specific date-range CSVs from a machine that can reach `gpsjam.org`** (a handful
   of files — one `manifest.csv` plus one CSV per day in each date range) and hand them to this
   session to run the analysis against directly, no network access needed for that half.
3. **Drop this validation** and rely on the existing three-mechanism synthesis as the honest
   statement of the method's status.

Whichever it is, Step 4 (write-up placement, and the deliberate decision about whether/how this
belongs in the live pitch) stays exactly as scoped: a documentation/repo strength, not a new
deck slide, and not decided by default.
