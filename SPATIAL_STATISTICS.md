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

## Deck headline number correction (was 0.686, now re-derived honestly)

An earlier version of the pitch deck's GeoAI slide (Slide 8) quoted **Global Moran's I =
0.686, p < 0.01, 65/136 towers in a significant cluster**, captioned as "computed live during
replay" with SIMULATED round-robin tower attribution — the same framing this document uses
for `api/spatial_stats.py`. That number did not actually come from that mechanism. It came
from `build_spatial_autocorrelation_demo.py`, a separate offline script that computed Moran's
I directly over `spatial_processed/simulated_spatial_anomaly_SIMULATED.csv` — the
epicenter-and-distance-decay SIMULATED severity layer behind Slide 7's static map (see
`spatial_layer_notes.md`), not a live replay's round-robin-attributed events at all.

That input is a smooth, monotonically decaying function of real distance from one fixed
epicenter (confirmed: `corr(distance_from_epicenter_km, simulated_anomaly_severity_SIMULATED)
= -0.45` across all 136 towers, computed directly from the CSV). Feeding a smooth function of
real distance into a spatial-autocorrelation test whose neighbor weights are built from those
same real distances produces a strong positive Moran's I close to a mathematical certainty —
nearby towers get near-identical severity by construction. It demonstrates that a spatially
smooth input looks spatially smooth; it does not demonstrate anything about the round-robin
mechanism the slide actually described, which is close to random and has no structural reason
to produce clustering.

**Re-derived honestly**: `build_spatial_autocorrelation_live_demo.py` reuses the exact
production classes `api.main`'s live replay uses — `ModelService`, `TowerAttributor` (real
round-robin), `EventStore`, and this module's own `compute_autocorrelation` — and replays the
full Spoofing/2.1.1 recording (2,503 rows, the same scenario as the Slide 11 demo) once,
fresh, start to finish. Result, after all 136 towers had a scored event:

```
global_moran_i: -0.0378   p_value: 0.2610   z_score: -0.6511   expected_i: -0.0074
significant LISA towers: 17 / 136  (3 High-High, 4 Low-Low, 5 High-Low, 5 Low-High)
```

**Not statistically significant** (p=0.261, comfortably above α=0.05) — no detectable global
spatial clustering. Checkpoints through the same replay (rows 136 / 272 / 500 / 1000 / 1500 /
2000) ranged from I=-0.089 (p=0.039, significant dispersion) to I=+0.049 (p=0.132, not
significant), swinging sign multiple times — consistent with round-robin attribution being
close to random and having no structural relationship to real tower geometry, unlike the
epicenter-decay layer above. Full checkpoint log:
`spatial_processed/live_replay_autocorrelation_result_LIVE.txt`; plots:
`spatial_processed/lisa_cluster_map_LIVE.png`, `spatial_processed/moran_scatter_LIVE.png`.

**This was the number the deck used** until the attribution methodology upgrade below
replaced round-robin as Slide 8's primary mechanism; it remains the honest baseline result for
round-robin specifically, and the honest reading was, and still is, the argument for the next
step, not a caveat to bury: the method (KNN weights, global/local Moran's I) is real and
ready. Round-robin attribution has no reason to produce a real pattern, so it doesn't — which
is exactly why real per-tower attribution data is the thing that would let this method say
something meaningful. `build_spatial_autocorrelation_demo.py` and its
`*_SIMULATED.png` outputs remain in the repo (still correctly labeled `SIMULATED`, and its own
docstring was already honest about its input) but must not be used as the source of a
"live replay" claim — that mismatch is what happened here.

## Attribution methodology upgrade: spatially-persistent replaces round-robin

Round-robin is not wrong because it produced a weak result — it is a weak mechanism on its own
terms: memoryless, cycling through towers in fixed order independent of timing, with no
structural relationship to anything real, so it was always going to produce something close to
noise (confirmed above: -0.0378, not significant). That is a fine, honest placeholder, but
`api/spatial.py` now also offers `SpatiallyPersistentAttributor`, a placeholder that encodes
one real, statable, and openly-disclosed assumption instead of none: **sustained attacks tend
to persist and drift locally, not teleport to a random distant tower on every event.** No
ground-truth fix is possible for either mechanism — the Spoofing/2.1.1 recording is a Norway
test-range log, the 136 towers are real Indonesian infrastructure, and there was never a real
per-event link between the two to recover. Nothing below changes that; the goal is a more
defensible heuristic, not a true one.

### Mechanism

A biased random walk over the real tower graph: the first event of a session picks a uniformly
random real tower; each subsequent event stays at the current tower with probability
`STAY_PROBABILITY = 0.7`, otherwise moves to one of its `ATTRIBUTION_K_NEIGHBORS = 5` nearest
real towers (real haversine distance), chosen with probability inversely proportional to
distance. `ATTRIBUTION_K_NEIGHBORS` happens to equal this module's own `K_NEIGHBORS` (both 5,
both landing on the same "conventional middle value, 4–8" reasoning from the Method section
above) but the two are unrelated parameters of two different mechanisms — one is the Moran's I
spatial-weights graph, the other is a movement rule for attribution — and changing one does not
imply changing the other.

### Honest result: run once, then a reproducibility check

Per the same discipline that caught the round-robin mismatch, `build_spatial_autocorrelation_persistent_demo.py`
runs the real production replay mechanism (`ModelService`, `EventStore`,
`compute_autocorrelation`) with `SpatiallyPersistentAttributor` over the same full Spoofing/2.1.1
recording, seed=42 first and reported as-is, then two more fresh seeds (43, 44) to check
stability:

```
seed=42 (primary): global_moran_i=+0.6676  p=0.0010  z=+7.452  n_towers_scored=30/136
seed=43 (rerun):    global_moran_i=+0.7769  p=0.0010  z=+8.252  n_towers_scored=34/136
seed=44 (rerun):    global_moran_i=+0.5936  p=0.0010  z=+7.130  n_towers_scored=38/136
```

Range across 3 seeds: **I in [+0.594, +0.777], all p=0.001 (the minimum possible at 999
permutations), all significant, all the same sign** — stable and reproducible, not a one-off.
Note `n_towers_scored` is far lower than round-robin's 136/136: a local persistent walk visits
far fewer distinct towers over 2,503 events than a mechanism that deterministically cycles
through all of them, which is expected of "drift locally" by design, not a bug.

### Self-initiated control: is this a real spatial finding, or a confound?

A result this strong and this stable warranted scrutiny before reporting it uncritically — the
same discipline that caught the original 0.686 mistake. This recording has extreme real
lag-1 severity autocorrelation (measured directly: **r = 0.9903** — expected, since a sustained
attack occupies a contiguous block of epochs, and this scenario is 91% attack rows). The
persistent attributor's entire design maps *temporal* adjacency to *spatial* adjacency (stay,
or move to a real neighbor, each step) — so temporally-adjacent events, which already share
correlated severity for real reasons, land at spatially-adjacent towers. That alone could
manufacture apparent spatial autocorrelation with no real spatial content at all.

**Control**: the identical seed=42 tower-visit sequence (same spatial walk, same towers, same
order), but each visit paired with a **randomly shuffled** severity instead of the one from
that temporal position — breaking the temporal-adjacency-to-severity-correlation link while
leaving the walk's spatial structure untouched.

```
control (seed=42 walk, severities shuffled, shuffle_seed=123):
  global_moran_i=-0.0834   p=0.3310   (not significant)
```

**The control collapses to near-zero and not significant.** This confirms the strong primary
result is substantially explained by this single-receiver recording's own real temporal
autocorrelation being funneled into apparent spatial autocorrelation by the walk's design — not
independent evidence of a real multi-tower spatial spread pattern. This is a third, distinct
way a placeholder attribution mechanism can manufacture a number, different from the other two
already documented in this project: round-robin (memoryless) destroys real temporal structure
and produces near-zero by construction; the old offline epicenter-decay CSV had no real
severity input at all and produced a strong positive by construction; spatially-persistent
attribution uses 100% real severity and 100% real geometry, and its strong result is still
substantially attributable to the mechanism's own structure — confirmed directly by this
control — not to independent spatial evidence.

### Bottom line

Spatially-persistent attribution is kept as the new primary attribution mode on Slide 8,
alongside round-robin (not deleted, still documented above as the **prior, less-structured
baseline**: -0.0378, not significant), because it is a more defensible placeholder on its own
terms — it encodes a real, statable assumption instead of none, and it demonstrates that the
method responds to attribution structure when structure exists. But the specific number it
produces (I ≈ 0.6–0.8, p=0.001) is not, by itself, proof of real spatial spread: the control
shows it is substantially a consequence of real temporal autocorrelation funneled through
spatial coherence. Both the strength of the result and this caveat belong on the slide together
— exactly as plainly as the round-robin correction states its own caveat, not less. Only real
per-tower attribution data would settle which (if either) reading is correct.

**The meta-finding**: three different attribution mechanisms were tried for this slide across
this project's history, and each was undone by a *different structural artifact of the
mechanism itself* — never a data-quality problem. The offline epicenter-decay CSV guaranteed a
strong positive Moran's I because severity was a smooth function of the same real distances
used as the test's own neighbor weights. Round-robin guaranteed a near-zero result because it's
memoryless, with no structural link to real geometry at all. Spatially-persistent attribution
produced an apparently strong result contaminated by this recording's own real temporal
autocorrelation (lag-1 r=0.99) leaking through the walk's time-to-space design. That pattern is
itself the finding: no synthetic attribution heuristic can cleanly test spatial clustering
here — this requires real per-event ground truth to resolve, exactly what step one of the pilot
roadmap targets.

Full results and reproducibility log:
`spatial_processed/persistent_replay_autocorrelation_result_LIVE.txt`; plots:
`spatial_processed/moran_scatter_PERSISTENT.png` — **use this one for the slide** — it renders
both the real-order line and the shuffled-control line together, making the confound visible
without relying on the caption alone; `spatial_processed/lisa_cluster_map_PERSISTENT.png` is
the per-tower map, kept for reference but not the recommended slide image.

## Where this lives

- `build_spatial_autocorrelation_live_demo.py` — the round-robin baseline: a genuine, fresh,
  once-run live replay through the real `ReplayManager` mechanism (`ModelService`,
  `TowerAttributor` round-robin, `EventStore`, this module's `compute_autocorrelation`). Rerun
  it to reproduce `spatial_processed/*_LIVE.*`. See "Deck headline number correction" above.
- `api/spatial.py::SpatiallyPersistentAttributor` — the primary attribution mechanism as of the
  attribution methodology upgrade: a biased random walk over real tower geometry, alongside
  (not replacing) `TowerAttributor` round-robin.
- `build_spatial_autocorrelation_persistent_demo.py` — reproduces Slide 8's current headline
  number: the same real replay mechanism as above with `SpatiallyPersistentAttributor` swapped
  in, run once (seed=42) then twice more for reproducibility (seeds 43, 44), plus the
  shuffled-severity control. Rerun it to reproduce `spatial_processed/*_PERSISTENT.*` and
  `persistent_replay_autocorrelation_result_LIVE.txt`. See "Attribution methodology upgrade"
  above.
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
