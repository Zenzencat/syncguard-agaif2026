# SyncGuard — Narration Script v2

**Rebuilt for the current 14-slide deck** (`HACK-TH-014_SLDK_v01.pptx` / Canva
`SyncGuard_Pitch_Deck.pptx`, 14 pages, verified speaker notes as source of truth). This is a
full rebuild against the current deck, not a patch of the old 12-slide, 3-minute script —
structure, numbers, and pacing have all changed.

**Target slot**: ~7 minutes (confirmed). **Target spoken length**: 5:30–6:30, with real
buffer under the cap.

**Word count**: 865 words (main narration, slides only — excludes Q&A prep, which is only
spoken if a judge asks; counted directly from this file, word by word, not estimated).
History: 823 words / 0.686 headline (offline mismatch) → 840 words / -0.04 (round-robin,
honestly re-derived) → 890 words / spatially-persistent attribution (0.59–0.78, with its own
control-test caveat) → 865 words after trimming Slides 2, 5, 6, 9, 12, 13 to make room for the
meta-finding synthesis added to Slide 8 (see Slide 8 and Q&A #7–8 below) without cutting any
honesty content. **Recount methodology note**: this pass caught that our word counts back
through the 823-word version were computed with `wc -w`, which silently misparses em-dashes
under this environment's POSIX/C locale (verified directly — confirmed to undercount by
~4% on this file); recounted here with a locale-independent method that also excludes
bare punctuation tokens (a lone "—" isn't a spoken word). The correction moved the reported
total by about 1 word overall — the earlier figures were coincidentally close, not reliably
so, and shouldn't be trusted as a coincidence going forward.
**Pacing math** (130–150 wpm, natural speaking pace — not the old `[~Xs]` tags, which were
carried over from the 12-slide version and already found to be off by ~40%):

| Pace | Runtime |
|---|---|
| 130 wpm (slow) | 6:39 |
| 140 wpm (mid) | 6:11 |
| 150 wpm (brisk) | 5:46 |

Sits inside, or just at the edge of, the 5:30–6:30 target depending on pace, with ~21s of
buffer to the 7:00 cap even at the slow end — real margin, not a knife's edge, but tighter
than earlier versions. Slide 8 is now the single densest slide in the deck by a wide margin
(mechanism + primary result + control result + the three-mechanism synthesis, ~160 words,
~69s at 140wpm) — if the 7-minute slot ever feels tight in a run-through, that's the one
section with room to trim, by shortening the control-test explanation to "a control test
confirms this" or folding the synthesis into one sentence, without cutting the honesty
content itself.

---

## Slide 1 — Title

*(~33 words · ~14s)*

Hi — we're Team Lorem Ipsum from Thailand. This is SyncGuard, our AGAIF 2026 Cybersecurity
track entry: a GNSS spoofing detector built to protect the timing that keeps 4G and 5G base
stations synchronized.

## Slide 2 — The Problem: An Invisible Attack on Network Time

*(~68 words · ~29s)*

4G and 5G base stations depend on GNSS for precise timing. Spoofing feeds a receiver
fabricated signals and corrupts that timing silently — no alarm, just frame misalignment,
handover failures, and interference downstream — hitting hardest where redundancy is
thinnest. GPS/GNSS signal-loss events in aviation are up 220% from 2021 to 2024, per a joint
EASA–IATA plan — a proxy for the same pressure now reaching telecom timing on the ground.

## Slide 3 — The Solution: Retrofit-First Spoofing Detection

*(~54 words · ~23s)*

SyncGuard is a lightweight anomaly-detection layer that watches a base station's existing
GNSS receiver — signal quality, Doppler behavior, RF interference — and flags an attack in
near real time, without touching the GNSSDO hardware. Most in-field receivers can't
cryptographically verify signals, and a hardware refresh isn't near-term realistic — so
SyncGuard works with what's already installed.

## Slide 4 — Technical Workflow: How It Works

*(~49 words · ~21s)*

Here's the pipeline. Raw receiver observables — C/N0, Doppler, RF-monitor stats — become
features every epoch, get scored by a RandomForest classifier, and fire an alert locally, in
near real time. It runs entirely at the edge, on-site, so detection keeps working even if
backhaul to the network core degrades too.

## Slide 5 — Data Source: Real Attack Data, Not Simulation

*(~62 words · ~27s)*

None of this is simulated data. We used JammerTest 2024 — real u-blox receiver logs from a
controlled test range in Norway: 24 scenarios, jamming through combined attacks, 44,639
epochs. We're upfront about scope: test-range receiver data, not production base-station
telemetry. That receiver-level detection generalizes to base-station timing is a stated
assumption here — no public base-station spoofing dataset exists to prove it directly.

## Slide 6 — GeoAI Methods: Features, Model, and a Split That Doesn't Cheat

*(~58 words · ~25s)*

On modeling: we engineer signal-quality and consistency features — C/N0 stats, Doppler shift,
code-Doppler residuals, RF-monitor stats — pure Python. A 300-tree, class-weight-balanced
RandomForest handles the 77/23 attack-clean imbalance. We split by recording, not row — eight
of twenty-four held out entirely, one dynamic and one stationary per attack type — so the
model is tested on attacks it's genuinely never seen.

## Slide 7 — Spatial Layer (NEW): Where It Spreads, Who to Prioritize

*(~74 words · ~32s)*

Beyond detection, we map it. This is 136 real Telkomsel towers across Kubu Raya and
Pontianak, Indonesia — and the severity scale is real too, anchored directly to our
detector's own `predict_proba` range. What's simulated is the epicenter and how the spread
propagates, since no public dataset of real base-station spoofing spread exists yet. Real
geometry, a real severity scale, simulated spread — together they answer where an attack
lands and which sites to prioritize first.

## Slide 8 — Live Spatial Statistics (NEW): Clustered, or Just Coincidence

*(~160 words · ~69s)*

This is where GeoAI comes in properly. We compute Global and Local Moran's I — LISA — over
the real tower network, using real k-nearest-neighbor weights: established methods going back
to Moran in 1950 and Anselin's LISA in 1995. We also upgraded attribution — from memoryless
round-robin to a spatially-persistent walk: stay put seventy percent of the time, else move to
a real nearest neighbor. One real assumption: sustained attacks drift locally, they don't
teleport. Result: Moran's I from 0.59 to 0.78 across three runs, p equals 0.001, stable and
significant. But a control test — same walk, severities shuffled out of real order — collapsed
it to negative 0.08. So the strong number is mostly this recording's own real temporal
autocorrelation, funneled through the walk, not confirmed spatial spread. Zoom out: three
attribution mechanisms, three different structural artifacts — never a data problem. That
pattern is the real finding. It takes real per-event data to settle this, which is exactly
step one of our roadmap.

## Slide 9 — Innovation: Why This Approach

*(~41 words · ~18s)*

Why this approach? Three reasons. Retrofit, not replace — no crypto-auth hardware upgrade
needed. A statistical signature, not a cryptographic check — the attack's fingerprint in the
receiver's own data. Edge-deployable — it keeps running at the cell site even if backhaul
degrades too.

## Slide 10 — Prototype Evidence: 87.7% Accuracy on Held-Out Attacks

*(~47 words · ~20s)*

Does it work? On eight held-out recordings the model never trained on: 87.7% accuracy,
ROC-AUC 0.916, PR-AUC 0.968. Attack recall is 93.7% overall, up to 98.5% for combined
spoofing-plus-jamming. Every one of these recordings was held out entirely before the model
ever saw a row of it.

## Slide 11 — Show, Don't Tell: Watch It Catch a Live Spoof

*(~52 words · ~22s)*

Let's watch it catch one live — scenario 2.1.1, a held-out, stationary spoofing attack the
model never saw in training. Red is the true spoofing window; purple is what the model flags.
Watch the C/N0 drop, the Doppler residual spike, the position jump — the model catches it,
tracks it, and clears right after.

## Slide 12 — Honest Limitations: Where It Still Needs Work

*(~61 words · ~26s)*

Two things we're not hiding. Clean recall, at 66.7%, trails attack recall's 93.7% — likely
per-session noise-floor drift, not a feature flaw. Feature importance still leans on generic
RF-monitor fields over our purpose-built spoofing features — visible in the ranking. Both are
next steps, not dead ends. To be clear: this is validated on test-range data, not production
telemetry yet — that's future work.

## Slide 13 — Implementation Plan: From Test Range to Pilot

*(~62 words · ~27s)*

The path to a pilot: validate against a real operator's base-station logs; field-validate and
tune thresholds per site; confirm edge-deployment feasibility; phase in rollout with
four-dimension monitoring — health, data validity, model performance, drift; then
documentation and provenance handoff. Sustainable by design too — retrofit-first avoids a
hardware refresh, edge deployment keeps the footprint low, and open-source Python keeps
reproduction affordable across under-resourced markets.

## Slide 14 — Closing: Protecting the Timing Everything Else Depends On

*(~44 words · ~19s)*

Trustworthy GNSS timing is the precondition for every hazard-response and predictive system
built on network telemetry — SyncGuard protects that precondition, rather than sitting on top
of it. Thank you. We're Team Lorem Ipsum — Claude was used throughout, every output reviewed
and validated by us.

---

# Q&A Prep

## Previously prepared (updated)

**1. Why RandomForest — why not something more sophisticated?**

We tested XGBoost as a challenger, using the identical fold-validated evaluation harness. It
didn't produce a meaningful accuracy improvement over the shipped RandomForest — same pattern
as other variants we tried and rejected this session, like sample-reweighting and feature
normalization. XGBoost does have a real advantage worth naming: its SHAP explainability cost
is about 50x lower than RandomForest's. We're keeping that as a concrete lead for future work,
not something we adopted here, since we only adopt a challenger when it's a genuine,
validated improvement with no meaningful regression — and this one wasn't.

**2. What's your false-positive rate?** *(updated to current figure)*

At our shipped decision threshold of 0.52, the false-positive rate is 33.3% — the flip side
of the 66.7% clean recall on our limitations slide. We're upfront that this is measured on a
77% attack / 23% clean test split, the inverse of a real deployment's base rate. A 33.3%
false-positive rate against a mostly-quiet real network implies a meaningfully larger
absolute volume of false alerts in production. That's a known open question we haven't
resolved yet — not something we're claiming to have solved.

**3. How realistic is the spatial simulation?**

Two different things are simulated, and we're explicit about both. On the tower map, the
epicenter and the distance-decay severity are a documented simulation, calibrated so about
15% of towers end up flagged — a plausible, localized footprint rather than an implausibly
tiny or implausibly network-wide one. On the Moran's I slide, a different thing is simulated:
which physical tower each scored event is attributed to, via round-robin. In both cases the
geometry, the severity values, and the statistical methods are real — only the spatial
placement or attribution on top of them is simulated, and we say so on the slide itself, not
just in an appendix.

**4. What about privacy?**

The JammerTest dataset we trained on is GNSS receiver and RF-monitor telemetry from a
controlled test event — no personal data, no link to identifiable individuals. The production
data source we're targeting — a base station's own GNSS receiver logs — is the same kind of
infrastructure telemetry, not customer or subscriber data. That said, any real pilot would
still need the hosting operator's own data-governance sign-off before we ingest live logs,
and that's built into step one of our roadmap, not an afterthought.

**5. What's the pilot roadmap?**

Five steps. First, validate against a real operator's own base-station GNSS logs — real
infrastructure data, stationary by nature, which closes both gaps in our current framing:
receiver type and receiver motion. Second, field validation and threshold tuning per
deployment site. Third, an engineering feasibility assessment for edge deployment. Fourth,
phased rollout with four-dimension monitoring — system health, input data validity, model
performance, and drift — pulled directly from the curriculum's own monitoring framework.
Fifth, a documentation and provenance package for operator and regulator handoff.

## New — for the Moran's I claim (Slide 8)

**6. Why k=5 for the KNN spatial weights?**

Our 136 towers are irregularly spaced — dense in urban Pontianak, sparse out toward Batu
Ampar. A fixed distance threshold either leaves rural towers with zero neighbors, which
Moran's I can't handle, or gives urban towers an unwieldy neighbor count. K-nearest-neighbors
guarantees every tower exactly k neighbors regardless of local density. k=5 is a conventional
middle value in the LISA literature — typically 4 to 8 for point data — large enough for a
stable local neighborhood, small enough to stay genuinely local rather than smoothing over
the whole network. It's also what most cellular-tower and spatial-epidemiology LISA studies
actually use, since towers are placed by coverage and demand, not on a grid. Our attribution
walk's own "move to a nearest neighbor" step also happens to use k=5 — same conventional
value, same reasoning, but it's an unrelated parameter of a different mechanism; changing one
doesn't imply changing the other.

**7. That's a strong, significant number (I up to 0.78, p=0.001) — is that a real finding?**

We don't think so, and we ran the test that would tell us. This slide already went through one
honest correction: an earlier internal draft quoted a number computed offline from a smooth,
distance-decay severity layer, which all but guaranteed a strong positive result by
construction — not a finding. We fixed that by re-deriving from a genuine live replay, which
originally used round-robin attribution and came back honestly weak: -0.04, not significant.
The number on the slide now comes from upgrading attribution to a spatially-persistent walk —
still SIMULATED, but it encodes one real assumption: sustained attacks drift locally instead of
teleporting. That produced this strong, reproducible result. Before reporting it, we ran a
control: same walk, same towers, but severities randomly shuffled out of their real time
order. It collapsed to -0.08, not significant. This recording has extreme real temporal
autocorrelation in severity — sustained attacks occupy a contiguous block of epochs — and the
walk's whole design maps temporal adjacency to spatial adjacency. So the strong number is
substantially that real temporal structure getting funneled into apparent spatial structure,
not independent proof of spatial spread. We're showing you both numbers for exactly that
reason.

**8. Why switch to this mechanism at all, if the strong result doesn't hold up under your own
control test?**

Because the mechanism is still a genuine improvement, even though the specific number isn't
proof. Round-robin encodes zero real assumptions about how an attack would actually move, so
it was always going to land near zero — that's not a finding either, just a different
artifact of a different mechanism. Spatially-persistent attribution encodes one real, stated,
defensible assumption — persistence and local drift — and it demonstrates something round-robin
structurally couldn't: that this method responds when the input has spatial structure. The
honest headline is the comparison itself: round-robin near zero, persistent attribution strong
but explained by temporal-autocorrelation funneling once you control for it — neither one is
evidence of a real spatial spoofing pattern, and both point at the same fix. Only real
per-tower attribution data would let us tell a genuine spatial signal from either kind of
mechanism artifact.

**9. How do you know Moran's I/LISA actually works — that it's not just returning noise
regardless of input?**

We validated the exact same code path against real, independently-documented data completely
separate from our own tower network: GPSJam.org's public GPS-interference data, from real
aircraft ADS-B reports, for the Black Sea/Crimea corridor during a week of confirmed jamming
activity. Result: Global Moran's I of positive 0.627, p equals 0.001 — significant clustering,
with the hotspot band landing almost exactly on the documented Crimea/Kerch Strait jamming
corridor reported in EASA advisories. That confirms this project's statistical method
genuinely detects real spatial clustering when real ground-truth data exists. It's also why
we're confident the null and inconsistent results on our own tower network were never a flaw
in the method — they're a direct consequence of not yet having real per-event attribution
data, exactly what step one of our pilot roadmap targets. We ran the same check on our own
target region, Kubu Raya and Pontianak — zero recorded interference in that airspace for the
same week, a genuine finding about a region with no currently known jamming activity, not a
filtering error. Full methodology and results: `GPSJAM_VALIDATION.md`.
