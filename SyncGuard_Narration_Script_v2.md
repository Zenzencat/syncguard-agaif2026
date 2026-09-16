# SyncGuard — Narration Script v2

**Rebuilt for the current 14-slide deck** (`HACK-TH-014_SLDK_v01.pptx` / Canva
`SyncGuard_Pitch_Deck.pptx`, 14 pages, verified speaker notes as source of truth). This is a
full rebuild against the current deck, not a patch of the old 12-slide, 3-minute script —
structure, numbers, and pacing have all changed.

**Target slot**: ~7 minutes (confirmed). **Target spoken length**: 5:30–6:30, with real
buffer under the cap.

**Word count**: 823 words (main narration, slides only — excludes Q&A prep; counted directly
from this file, word by word, not estimated).
**Pacing math** (130–150 wpm, natural speaking pace — not the old `[~Xs]` tags, which were
carried over from the 12-slide version and already found to be off by ~40%):

| Pace | Runtime |
|---|---|
| 130 wpm (slow) | 6:20 |
| 140 wpm (mid) | 5:53 |
| 150 wpm (brisk) | 5:29 |

Sits inside the 5:30–6:30 target across nearly the whole pace range, with ~40s of buffer to
the 7:00 cap even at the slow end — real margin, not a knife's edge.

---

## Slide 1 — Title

*(~32 words · ~14–15s)*

Hi — we're Team Lorem Ipsum from Thailand. This is SyncGuard, our AGAIF 2026 Cybersecurity
track entry: a GNSS spoofing detector built to protect the timing that keeps 4G and 5G base
stations synchronized.

## Slide 2 — The Problem: An Invisible Attack on Network Time

*(~70 words · ~30s)*

4G and 5G base stations depend on GNSS for precise timing. Spoofing feeds a receiver
fabricated signals and corrupts that timing silently — no alarm, just frame misalignment,
handover failures, and interference downstream — hitting hardest where network redundancy is
thinnest. And it's not hypothetical: GPS/GNSS signal-loss events in aviation are up 220% from
2021 to 2024, per a joint EASA–IATA plan — a proxy for the same pressure now reaching telecom
timing on the ground.

## Slide 3 — The Solution: Retrofit-First Spoofing Detection

*(~55 words · ~24s)*

SyncGuard is a lightweight anomaly-detection layer that watches a base station's existing
GNSS receiver — signal quality, Doppler behavior, RF interference — and flags an attack in
near real time, without touching the GNSSDO hardware. Most in-field receivers can't
cryptographically verify signals, and a hardware refresh isn't near-term realistic — so
SyncGuard works with what's already installed.

## Slide 4 — Technical Workflow: How It Works

*(~55 words · ~24s)*

Here's the pipeline. Raw receiver observables — C/N0, Doppler, RF-monitor stats — become
features every epoch, get scored by a RandomForest classifier, and fire an alert locally, in
near real time. It runs entirely at the edge, on-site, so detection keeps working even if
backhaul to the network core degrades too.

## Slide 5 — Data Source: Real Attack Data, Not Simulation

*(~65 words · ~28s)*

None of this is simulated data. We used JammerTest 2024 — real u-blox receiver logs from a
controlled test range in Norway: 24 scenarios, jamming through combined attacks, 44,639
epochs. We're upfront about scope, though: this is test-range receiver data, not production
base-station telemetry. That receiver-level detection generalizes to base-station timing is a
stated assumption here — no public base-station spoofing dataset exists to prove it directly.

## Slide 6 — GeoAI Methods: Features, Model, and a Split That Doesn't Cheat

*(~65 words · ~28s)*

On modeling: we engineer signal-quality and consistency features — C/N0 stats, Doppler shift,
code-Doppler residuals, RF-monitor stats — pure Python. A 300-tree, class-weight-balanced
RandomForest handles the 77/23 attack-clean imbalance. Critically, we split by recording, not
by row — eight of twenty-four held out entirely, one dynamic and one stationary per attack
type — so the model is tested on attacks it's genuinely never seen.

## Slide 7 — Spatial Layer (NEW): Where It Spreads, Who to Prioritize

*(~75 words · ~32s)*

Beyond detection, we map it. This is 136 real Telkomsel towers across Kubu Raya and
Pontianak, Indonesia — and the severity scale is real too, anchored directly to our
detector's own `predict_proba` range. What's simulated is the epicenter and how the spread
propagates, since no public dataset of real base-station spoofing spread exists yet. Real
geometry, a real severity scale, simulated spread — together they answer where an attack
lands and which sites to prioritize first.

## Slide 8 — Live Spatial Statistics (NEW): Clustered, or Just Coincidence

*(~95 words · ~41s)*

This is where GeoAI comes in properly. We compute Global and Local Moran's I — LISA — live,
over the real tower network, using real k-nearest-neighbor weights: established methods going
back to Moran in 1950 and Anselin's LISA in 1995. It answers whether flagged sites are
actually clustered right now, or just scattered noise. This run: Global Moran's I of 0.686, p
under 0.01 — 65 of 136 towers in a significant cluster. The honest caveat: which tower each
event attributes to is currently simulated, round-robin — so this demonstrates the method,
not a detected real-world cluster.

## Slide 9 — Innovation: Why This Approach

*(~50 words · ~22s)*

Why this approach? Three reasons. Retrofit, not replace — no crypto-auth hardware upgrade
needed. A statistical signature, not a cryptographic check — the attack's fingerprint in the
receiver's own data. And edge-deployable — it keeps running at the cell site even when the
attack degrades backhaul too.

## Slide 10 — Prototype Evidence: 87.7% Accuracy on Held-Out Attacks

*(~55 words · ~24s)*

Does it work? On eight held-out recordings the model never trained on: 87.7% accuracy,
ROC-AUC 0.916, PR-AUC 0.968. Attack recall is 93.7% overall, up to 98.5% for combined
spoofing-plus-jamming. Every one of these recordings was held out entirely before the model
ever saw a row of it.

## Slide 11 — Show, Don't Tell: Watch It Catch a Live Spoof

*(~55 words · ~24s)*

Let's watch it catch one live — scenario 2.1.1, a held-out, stationary spoofing attack the
model never saw in training. Red is the true spoofing window; purple is what the model flags.
Watch the C/N0 drop, the Doppler residual spike, the position jump — the model catches it,
tracks it, and clears right after.

## Slide 12 — Honest Limitations: Where It Still Needs Work

*(~70 words · ~30s)*

Two things we're not hiding. Clean recall, at 66.7%, trails attack recall's 93.7% — likely
per-session noise-floor drift, not a feature-set flaw. And feature importance still leans on
generic RF-monitor fields over our purpose-built spoofing features — you can see it in the
ranking. Both are concrete next steps, not dead ends. And to be clear: this is validated on
test-range data, not yet production telemetry — that's future work.

## Slide 13 — Implementation Plan: From Test Range to Pilot

*(~75 words · ~32s)*

The path to a pilot: validate against a real operator's base-station logs; field-validate and
tune thresholds per site; confirm edge-deployment feasibility; phase in rollout with
four-dimension monitoring — health, data validity, model performance, drift; then a
documentation and provenance handoff. It's sustainable by design too — retrofit-first avoids
a hardware refresh, edge deployment keeps the footprint low, and open-source Python keeps the
barrier to reproduction low across unevenly-resourced markets.

## Slide 14 — Closing: Protecting the Timing Everything Else Depends On

*(~45 words · ~20s)*

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
actually use, since towers are placed by coverage and demand, not on a grid.

**7. Why 0.686 — what does that magnitude mean, and does the significance meaningfully hold
given the simulated per-tower attribution?**

Moran's I is bounded roughly between -1 and 1. Near zero means no spatial pattern; positive
means similar values cluster together; negative means checkerboard-like dispersion. 0.686 is
a strong positive value — for comparison, other snapshots from the same live system have
shown values as low as 0.019, or even a significant negative dispersion around -0.09. So
0.686 sits well up toward the clustered end of that range: this run's severity assignment is
far from randomly scattered across the real tower geometry.

On significance: p<0.01 comes from 999 permutation tests — fewer than 1% of random
reassignments of these exact severity values across this exact tower geometry produced as
extreme an I. That's a real, valid statistical result about this input. What it can't tell
you is whether that input reflects a real spoofing event, because the tower attribution
feeding it is simulated. The significance test is honest about the pattern in the data we
gave it — it just can't certify that the data itself is a real-world observation.

**8. Does this spatial-clustering result depend on the epicenter-choice assumption, or would
it hold under a different simulated attribution?**

Good distinction — these are actually two separate simulated components in two different
parts of the deck. The static tower map on the spread slide uses the epicenter-and-decay
simulation. The live Moran's I number comes from a different mechanism: round-robin
attribution of scored events to towers during replay, which doesn't use the epicenter at all.
So no, this result doesn't depend on the epicenter choice. It does depend on the round-robin
attribution, and that's exactly why the number moves between runs — our own documentation
shows a dispersion result of -0.091 in one snapshot and a weak 0.019 in another, from the same
live system. Different simulated attribution sequences will produce different Global Moran's
I values and different cluster memberships. That variability is expected, not a bug — it's
why the slide says "this run" explicitly and frames it as a demonstration of the method, not
a claim about a specific detected pattern.
