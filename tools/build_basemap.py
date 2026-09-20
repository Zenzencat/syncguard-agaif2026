"""One-time, offline-build tool: fetch a small OpenStreetMap vector basemap for the
Kubu Raya/Pontianak tower bounding box via the Overpass API, simplify it, and write it to
assets/offline_basemap.geojson for the dashboard to serve locally (no runtime network calls).

This builds the VECTOR layer, which the dashboard now uses only as a fallback. The dashboard's
primary base layer is a raster image rendered from the same OpenStreetMap data by
tools/build_basemap_raster.py and embedded in this same asset. Data (c) OpenStreetMap
contributors, licensed under the Open Database License (ODbL); attribution is shown in the
dashboard footer and map caption -- see assets/OFFLINE_BASEMAP_ATTRIBUTION.md.

WARNING: this script overwrites the WHOLE asset, including the embedded "raster" member. Re-run
tools/build_basemap_raster.py (with --from-cache) afterwards to put the raster back.

Run once, with network access, whenever the bundled basemap needs rebuilding:
    .venv\\Scripts\\python.exe tools\\build_basemap.py

Not part of the running API/dashboard and not imported by it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from shapely.geometry import LineString, Polygon, box, mapping
from shapely.ops import linemerge, unary_union

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.spatial import load_towers  # noqa: E402

OUT_PATH = REPO_ROOT / "assets" / "offline_basemap.geojson"
ATTRIBUTION_PATH = REPO_ROOT / "assets" / "OFFLINE_BASEMAP_ATTRIBUTION.md"
CACHE_DIR = Path(__file__).resolve().parent / "basemap_cache"

MARGIN_DEG = 0.05
SIMPLIFY_DEG = 0.0005
RIVER_SIMPLIFY_DEG = 0.0008  # coarser: raw waterway fetch is large; only named rivers/canals ship
MAX_BYTES = int(1.5 * 1024 * 1024)

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
USER_AGENT = (
    "SyncGuard-AGAIF2026-basemap-build/1.0 "
    "(one-time offline map build for a hackathon demo; contact https://github.com/Zenzencat/syncguard-agaif2026)"
)


def compute_bbox() -> tuple[float, float, float, float]:
    """(south, west, north, east) = 136 real tower extent + a fixed margin."""
    df = load_towers()
    south = df["lat"].min() - MARGIN_DEG
    north = df["lat"].max() + MARGIN_DEG
    west = df["lon"].min() - MARGIN_DEG
    east = df["lon"].max() + MARGIN_DEG
    return south, west, north, east


def run_query(ql: str, label: str) -> dict:
    last_err: Exception | None = None
    for mirror in MIRRORS:
        for attempt in range(1, 4):
            try:
                resp = requests.post(
                    mirror, data={"data": ql}, headers={"User-Agent": USER_AGENT}, timeout=180
                )
                if resp.status_code == 200:
                    print(f"  [{label}] {mirror} ok ({len(resp.content)} bytes)")
                    return resp.json()
                print(f"  [{label}] {mirror} attempt {attempt}: HTTP {resp.status_code}")
            except requests.RequestException as e:
                last_err = e
                print(f"  [{label}] {mirror} attempt {attempt}: {e}")
            time.sleep(5 * attempt)
        time.sleep(3)
    raise RuntimeError(f"Overpass query '{label}' failed on all mirrors: {last_err}")


def fetch_all(bbox: tuple[float, float, float, float], use_cache: bool = False) -> dict[str, dict]:
    s, w, n, e = bbox
    b = f"{s},{w},{n},{e}"
    queries = {
        "coastline": f'[out:json][timeout:180];way["natural"="coastline"]({b});out geom;',
        "water": (
            "[out:json][timeout:180];"
            f'(way["natural"="water"]({b});way["landuse"="reservoir"]({b});'
            f'relation["natural"="water"]({b}););out geom;'
        ),
        "rivers": (
            "[out:json][timeout:180];"
            f'way["waterway"~"^(river|canal)$"]({b});out geom;'
        ),
        "roads": (
            "[out:json][timeout:180];"
            f'way["highway"~"^(motorway|trunk|primary|secondary)$"]({b});out geom;'
        ),
        "landuse": (
            "[out:json][timeout:180];"
            f'way["landuse"~"^(residential|commercial)$"]({b});out geom;'
        ),
        "places": (
            "[out:json][timeout:180];"
            f'node["place"~"^(city|town)$"]({b});out body;'
        ),
    }
    CACHE_DIR.mkdir(exist_ok=True)
    results = {}
    for label, ql in queries.items():
        cache_path = CACHE_DIR / f"{label}.json"
        if use_cache and cache_path.exists():
            print(f"Using cached {label} ({cache_path.stat().st_size} bytes)")
            results[label] = json.loads(cache_path.read_text(encoding="utf-8"))
            continue
        print(f"Fetching {label}...")
        data = run_query(ql, label)
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        results[label] = data
        time.sleep(4)  # be polite between queries
    return results


def ways_as_lines(elements: list[dict]) -> list[LineString]:
    lines = []
    for el in elements:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"] if pt.get("lon") is not None]
        if len(coords) >= 2:
            lines.append(LineString(coords))
    return lines


def ways_as_lines_with_names(elements: list[dict]) -> list[tuple[LineString, str | None]]:
    out = []
    for el in elements:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"] if pt.get("lon") is not None]
        if len(coords) >= 2:
            out.append((LineString(coords), el.get("tags", {}).get("name")))
    return out


def ways_and_relations_as_polygons(elements: list[dict]) -> list[Polygon]:
    polys = []
    for el in elements:
        if el.get("type") == "way" and "geometry" in el:
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"] if pt.get("lon") is not None]
            if len(coords) >= 4 and coords[0] == coords[-1]:
                try:
                    poly = Polygon(coords)
                    if poly.is_valid and not poly.is_empty:
                        polys.append(poly)
                except Exception:
                    continue
        elif el.get("type") == "relation":
            outers, inners = [], []
            for m in el.get("members", []):
                if "geometry" not in m:
                    continue
                coords = [(pt["lon"], pt["lat"]) for pt in m["geometry"] if pt.get("lon") is not None]
                if len(coords) < 4 or coords[0] != coords[-1]:
                    continue
                (outers if m.get("role") == "outer" else inners).append(coords)
            for outer in outers:
                holes = inners if len(outers) == 1 else []
                try:
                    poly = Polygon(outer, holes)
                    if poly.is_valid and not poly.is_empty:
                        polys.append(poly)
                except Exception:
                    continue
    return polys


def simplify_clip(geom, clip_box):
    if geom is None or geom.is_empty:
        return None
    g = geom.simplify(SIMPLIFY_DEG, preserve_topology=True)
    g = g.intersection(clip_box)
    return g if not g.is_empty else None


def build_features(raw: dict[str, dict], bbox: tuple[float, float, float, float]) -> list[dict]:
    s, w, n, e = bbox
    clip_box = box(w, s, e, n)
    features = []

    coast_lines = ways_as_lines(raw["coastline"]["elements"])
    if coast_lines:
        merged = linemerge(unary_union(coast_lines))
        merged = simplify_clip(merged, clip_box)
        if merged is not None:
            geoms = [merged] if merged.geom_type == "LineString" else list(merged.geoms)
            for g in geoms:
                features.append({"type": "Feature", "properties": {"category": "coastline"}, "geometry": mapping(g)})

    water_polys = ways_and_relations_as_polygons(raw["water"]["elements"])
    if water_polys:
        merged = unary_union(water_polys)
        merged = simplify_clip(merged, clip_box)
        if merged is not None:
            geoms = [merged] if merged.geom_type in ("Polygon",) else list(merged.geoms)
            for g in geoms:
                features.append({"type": "Feature", "properties": {"category": "water"}, "geometry": mapping(g)})

    # The raw waterway=river/canal fetch includes thousands of unnamed farm/drainage
    # ditches (common in this region's rice-paddy landscape) -- too many to ship within the
    # size budget and not what "the main rivers/canals" means for an orientation layer. Keep
    # only named ways, merged per name so a river drawn as many short OSM segments becomes
    # one feature.
    river_lines = ways_as_lines_with_names(raw["rivers"]["elements"])
    by_name: dict[str, list[LineString]] = defaultdict(list)
    for line, name in river_lines:
        if name:
            by_name[name].append(line)
    for name, lines in by_name.items():
        merged = linemerge(unary_union(lines)) if len(lines) > 1 else lines[0]
        merged = merged.simplify(RIVER_SIMPLIFY_DEG, preserve_topology=True).intersection(clip_box)
        if merged.is_empty:
            continue
        geoms = [merged] if merged.geom_type == "LineString" else [g for g in merged.geoms if g.geom_type == "LineString"]
        major = "kapuas" in name.lower()
        for g in geoms:
            features.append({"type": "Feature", "properties": {"category": "river", "name": name, "major": major}, "geometry": mapping(g)})

    road_lines = ways_as_lines(raw["roads"]["elements"])
    for line in road_lines:
        g = simplify_clip(line, clip_box)
        if g is not None:
            features.append({"type": "Feature", "properties": {"category": "road"}, "geometry": mapping(g)})

    landuse_polys = ways_and_relations_as_polygons(raw["landuse"]["elements"])
    if landuse_polys:
        merged = unary_union(landuse_polys)
        merged = simplify_clip(merged, clip_box)
        if merged is not None:
            geoms = [merged] if merged.geom_type in ("Polygon",) else list(merged.geoms)
            for g in geoms:
                features.append({"type": "Feature", "properties": {"category": "builtup"}, "geometry": mapping(g)})

    for el in raw["places"]["elements"]:
        if el.get("type") != "node":
            continue
        name = el.get("tags", {}).get("name")
        if not name:
            continue
        features.append({
            "type": "Feature",
            "properties": {"category": "place", "name": name, "place": el.get("tags", {}).get("place")},
            "geometry": {"type": "Point", "coordinates": [el["lon"], el["lat"]]},
        })

    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-cache", action="store_true",
                         help="Reuse tools/basemap_cache/*.json instead of re-querying Overpass "
                              "(for re-tuning simplification without hitting the network again).")
    args = parser.parse_args()

    bbox = compute_bbox()
    s, w, n, e = bbox
    print(f"Tower-derived bbox (+{MARGIN_DEG} deg margin): south={s:.6f} west={w:.6f} north={n:.6f} east={e:.6f}")

    raw = fetch_all(bbox, use_cache=args.from_cache)
    features = build_features(raw, bbox)
    print(f"Built {len(features)} features.")

    fc = {
        "type": "FeatureCollection",
        "attribution": "(c) OpenStreetMap contributors (ODbL)",
        "source": "OpenStreetMap via Overpass API, fetched for this bounding box",
        "bbox": [w, s, e, n],
        "generated_note": "One-time build; see tools/build_basemap.py. Dashboard makes no runtime network calls.",
        "features": features,
    }

    encoded = json.dumps(fc, separators=(",", ":"))
    size = len(encoded.encode("utf-8"))
    print(f"Encoded size: {size} bytes ({size / 1024:.1f} KB)")
    if size > MAX_BYTES:
        print(f"WARNING: exceeds {MAX_BYTES} byte budget; consider coarser SIMPLIFY_DEG or dropping a category.")

    OUT_PATH.write_text(encoded, encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
