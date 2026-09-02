# SyncGuard spatial statistics — global Moran's I & Local Moran's I (LISA)

Adds an established GeoAI method on top of the live spatial layer, directly answering the
mentor feedback this project has already been responding to (see `spatial_layer_notes.md`):
*do neighbouring base stations show correlated anomalies?* Global Moran's I gives one number
for "is the network's current anomaly pattern spatially clustered or random, right now."
Local Moran's I (LISA) gives a per-tower answer: which specific towers sit inside a
statistically significant cluster of correlated anomalies (a hotspot), and which are isolated
outliers flagged against quiet neighbours. Implemented in
[`api/spatial_stats.py`](api/spatial_stats.py), exposed at `GET /spatial/autocorrelation`,
rendered live on the dashboard. **Additive, not a replacement**: `api/spatial.py`'s
hand-rolled distance-weighted live correlation and `build_spatial_simulation.py`'s offline
SIMULATED-epicenter layer are both untouched and still available — this project now has three
distinct spatial views, each answering a different question, none of them faked to agree with
each other.

## REAL vs SIMULATED — read this before the rest

Same discipline as everywhere else in this project (`api/spatial.py`, `spatial_layer_notes.md`):

- **REAL**: the 136 Telkomsel tower coordinates; the k-nearest-neighbor spatial weights built
  from real great-circle distance between them (`libpysal.weights.KNN` with
  `radius=6371.0088`, the identical Earth-radius constant `api/spatial.py`'s `haversine_km`
  already uses); the severity values fed in (real `predict_proba` output from the trained
  detector, via `api/model_service.py`); and the statistical method itself — global Moran's I
  (Moran, 1950) and Local Moran's I / LISA (Anselin, 1995), computed via `esda`/`libpysal`,
  the standard PySAL implementations, not a custom or approximate reimplementation.
- **SIMULATED (inherited, not re-introduced here)**: which physical tower a given scored
  event is attributed to — see `api/spatial.py`'s `TowerAttributor`. This module adds no new
  simulated component; it is 100% real statistical machinery applied to an input that carries
  the same, already-disclosed attribution caveat as every other scored event in this project.
- **Not a claim being made**: that a real spoofing/jamming event, deployed across a real
  multi-tower network, would produce spatial clustering shaped exactly like what this demo
  shows. The tower geometry and the math are real; the *pattern* observed during a replay
  reflects round-robin attribution interacting with whatever severity sequence the replayed
  recording produces, not a measured multi-tower attack. Same "framing gap" honesty
  `project_abstract.md` already applies elsewhere in this project.

## Method

### Spatial weights: k-nearest-neighbors (k=5), not distance-band

The 136 towers are irregularly spaced — dense in urban Pontianak/Sungai Raya, sparse toward
Batu Ampar and Teluk Pakedai (tens of km out). A fixed distance-band threshold either leaves
rural towers with **zero neighbors** ("islands," which Moran's I cannot handle — a
neighborless observation contributes nothing to the statistic and produces an undefined local
score) or gives urban towers an unwieldy neighbor count. K-nearest-neighbors weights
guarantee every tower has exactly `k` neighbors regardless of local density — the standard,
defensible choice for irregular point patterns like this, and what most cellular-tower and
spatial-epidemiology LISA studies actually use, since towers are placed by coverage and
demand, not on a regular grid.

`k=5` is a conventional middle value in the LISA literature (commonly 4–8 for point data) —
large enough for a stable local neighborhood, small enough to stay genuinely *local* rather
than smoothing over the whole network. Weights are row-standardized (each tower's neighbor
weights sum to 1), the standard transform for both Moran's I and LISA.

Distance itself uses `libpysal.weights.KNN.from_array(coords, k=5, radius=6371.0088)` —
libpysal's built-in great-circle mode, not raw-degree Euclidean distance on latitude/longitude
(which would distort real distances). This reuses the identical Earth-radius constant already
established in `api/spatial.py`'s `haversine_km`, so "real distance" means the same thing in
both spatial modules.

### Missing towers: never fabricated

Not every one of the 136 towers has a scored event at a given moment — especially early in a
replay, when round-robin attribution has only cycled through a handful of towers so far. This
module **never imputes a placeholder severity** for a tower with no data; doing so would
invent a third, unlabeled category of data alongside the REAL/SIMULATED split this project
holds itself to. Instead, every computation runs over whichever subset of towers currently has
at least one scored event (via `EventStore.latest_severity_per_tower()`), and refuses to
compute anything below `MIN_TOWERS_FOR_STATS = 15` (the mathematical floor is `k+1 = 6`; 15
gives more meaningful permutation-test power), returning an explicit "not enough data yet,
N/15" result instead of a misleading early number. The k-NN graph itself is also rebuilt fresh
each call from only the towers-with-data — not a fixed 136-tower graph with missing rows
dropped after the fact, which would silently change what "5 nearest neighbors" means for the
towers that remain.

### Global Moran's I

One statistic, `I`, over the whole present-data network: positive and significant means
similar severities cluster together spatially (what a real correlated multi-tower event would
plausibly look like); near zero means no detectable spatial pattern; negative and significant
means a checkerboard-like *dispersion* (high and low severities alternating between
neighbors) — also a real, interpretable finding, not a null result. Significance is assessed
by conditional permutation (999 random reassignments of the observed severities across the
same tower geometry — `esda.moran.Moran(severities, w, permutations=999)`), giving a
pseudo p-value: the fraction of permuted arrangements that produced an equally or more extreme
`I` than the one actually observed.

### Local Moran's I (LISA)

A per-tower version of the same idea. Each tower gets classified into one of four quadrants of
the Moran scatterplot (a tower's own value vs. its spatially-lagged neighbor average),
retained only where `p_value < 0.05` (999 permutations again, this time per tower):

| Quadrant | Meaning here |
|---|---|
| **High-High** | This tower and its neighbors are all scoring high severity — a genuine correlated anomaly hotspot, the direct answer to "do neighbouring stations show correlated anomalies." |
| **Low-Low** | This tower and its neighbors are all quiet — a coldspot, operationally just "normal." |
| **High-Low** | This tower is flagged, but its neighbors are quiet — a spatial outlier: isolated, worth a second look precisely *because* it doesn't match its surroundings. |
| **Low-High** | This tower is quiet, but its neighbors are flagged — the inverse outlier case. |
| *(not significant)* | No detectable local pattern at α=0.05 — the majority of towers, most of the time. |

### Determinism

`esda`'s `Moran` class (version 2.10.0, pinned in `requirements-api.txt`) has **no `seed=`
parameter** — it draws permutations from NumPy's global RNG, so reproducibility requires
reseeding (`np.random.seed(42)`) immediately before instantiation. Verified directly: two
calls reseeded the same way produce bit-identical `I` and `p_sim`. `Moran_Local` *does* accept
`seed=` (and `n_jobs=1`, passed explicitly) directly. Both are seeded with `RNG_SEED = 42`,
the same convention used throughout this project (`ROBUSTNESS_NOTES.md`,
`train_improved_model.py`). This is independent of, and doesn't touch, the RandomForest
`n_jobs=1` determinism fix in `api/model_service.py` — a separate, already-shipped concern.

## Worked examples (real API responses, captured from a live replay of scenario 2.1.1)

**Dispersion, not clustering** — `GET /spatial/autocorrelation` partway through a replay:

```
n_towers_scored: 136   k_neighbors: 5
global_moran_i: -0.091   p_value: 0.033   z_score: -1.673   expected_i: -0.0074
```

A **statistically significant negative** Moran's I (p=0.033, comfortably below α=0.05).
Interpretation: at this moment, severity is *not* randomly scattered across the network — it's
arranged so that a tower's neighbors tend to differ from it more than chance would predict, a
mild checkerboard pattern. The per-tower LISA output for this same snapshot names exactly
which towers: `tbg` (severity 0.05) is a Low-Low coldspot (p=0.012); `MPW022` (severity 0.72)
and a second `tbg`-keyed tower (severity 0.25) are High-Low outliers (p=0.041, p=0.010); four
towers including `16MPW020` and `16MPW023` are Low-High outliers (p=0.017–0.046). This is an
honest result, reported as found — not every snapshot shows a hotspot, and dispersion is a
real, valid Moran's I finding, not a failure to detect one.

**Clustering** — a different snapshot, ~40 seconds into the same replay, produced a 9-tower
**High-High hotspot** (`16KTP009`, `16MPW023`, `16MPW026`, `MPW029`, `MPW036`, `MPW062`,
`MPW071`, `MPW072`, `MPW149`; severities 0.90–0.98; each individually significant at
p=0.003–0.035) while the *global* Moran's I at that same moment was weak and not significant
(I=0.019, p=0.286). This is expected, not a contradiction: **global Moran's I averages over
the entire network, so a real, significant local cluster can exist while the global average
stays diluted by 127 other quiet, non-clustered towers.** This is exactly why both statistics
are reported together, not just the global headline number — LISA is what actually answers
"which neighbouring stations are correlated," and can say so even when the one-number network
summary doesn't yet.

## Where this lives

- `api/spatial_stats.py` — full implementation (weights, global/local Moran's I, LISA
  classification, the REAL/SIMULATED statement in the module docstring).
- `GET /spatial/autocorrelation` (`api/main.py`) — returns global I/p-value/z-score and the
  per-tower LISA table, computed fresh from `EventStore.latest_severity_per_tower()` each
  call. Schema: `AutocorrelationResponse`/`LisaTower` in `api/schemas.py`.
- `syncguard_interactive_summary.html` — live section polls this endpoint every 3 seconds
  while connected; shows Global Moran's I + p-value as headline stats, and rings significant
  LISA towers on the live map (red = High-High hotspot, blue = Low-Low coldspot, purple =
  spatial outlier) on top of the existing severity fill-color — a judge can watch clusters
  form and dissolve as a replay runs.
- `requirements-api.txt` — `libpysal==4.15.0`, `esda==2.10.0` (pure Python; pulls in
  `geopandas`/`shapely` transitively via prebuilt wheels, no GDAL/C-toolchain build step;
  this code path doesn't call geopandas directly, weights are built from a plain array).
- Does not modify `api/spatial.py`, `build_spatial_simulation.py`, `models/model.joblib`, or
  any deployed threshold.
