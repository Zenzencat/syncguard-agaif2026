"""One-time, offline-build tool: render a single styled COLOR base image of the Kubu Raya /
Pontianak tower bounding box from OpenStreetMap vector data, and embed it in the bundled
basemap asset (assets/offline_basemap.geojson) as the dashboard's PRIMARY offline map layer.

Why a raster on top of the existing vector build: the vector basemap (tools/build_basemap.py) is
deliberately sparse (gray major roads, a flat land fill, no sea). This renders the same real OSM
data -- plus minor roads, residential streets, land cover and village names fetched once from the
Overpass API -- into a styled image: sea/water blue, green land with darker forest and lighter
farmland, tan built-up areas, and road lines colored and cased by class.

Nothing here runs at dashboard runtime. The dashboard loads the image from the same-origin
asset it already fetches; no tile server, font service or network call is involved, and the image
is not regenerated on the fly. Not imported by the API.

Where the image lives: inside assets/offline_basemap.geojson, under a top-level "raster" member
(GeoJSON permits foreign members). The API only serves a fixed list of asset files, so this
avoids needing a new route, and the existing 1.5 MB asset-size test keeps it honest.
Order of operations if you rebuild everything:
    python tools/build_basemap.py            # vector layer (overwrites the whole asset)
    python tools/build_basemap_raster.py     # adds/refreshes "raster" and the minor place labels
Re-running this script alone is safe and idempotent.

Usage (repo venv, network only needed the first time; caches to tools/basemap_cache/):
    python tools/build_basemap_raster.py [--from-cache] [--width 4096] [--preview out.png]

REAL: every feature drawn is real OpenStreetMap data for the tower bounding box. The STYLE (colors,
line widths, which road classes get a casing) is this script's own and carries no information.
Attribution: the image is a raster rendered from OpenStreetMap data, (c) OpenStreetMap
contributors, Open Database License (ODbL) -- kept in the dashboard footer and map caption.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PathCollection  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from PIL import Image  # noqa: E402
from shapely.geometry import Point, Polygon, box  # noqa: E402
from shapely.geometry.polygon import orient  # noqa: E402
from shapely.ops import linemerge, split, unary_union  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_basemap as bm  # noqa: E402  (fetch helpers, bbox, geometry parsers)

REPO_ROOT = bm.REPO_ROOT
ASSET_PATH = bm.OUT_PATH
CACHE_DIR = bm.CACHE_DIR

# ---- style ---------------------------------------------------------------------------------
# Soft, mid-saturation palette (Voyager-like): light enough that the severity-colored tower
# markers (yellow -> red) and the dark marker outlines stay readable on top of it.
STYLE = {
    "sea":        "#A9CFE8",
    "land":       "#DDE7C6",   # soft green base: unmapped land is mostly peat forest / plantation
    "farmland":   "#EAE4BC",   # tan-yellow: farmland, plantations, meadow
    "wood":       "#B6D598",   # green: forest / wood / scrub
    "wetland":    "#C8E0B8",   # pale green: wetland / grassland
    "builtup":    "#E8D3B5",   # tan: residential / commercial areas
    "builtup_edge": "#D3B994",
    "water":      "#A9CFE8",
    "coast":      "#86B3D3",
}
# highway class -> (casing color, fill color, fill width px at 4096 px wide, z-order)
ROAD_STYLE = {
    "residential":  ("#C2BAA6", "#FFFFFF", 2.0, 1),
    "unclassified": ("#BDB49F", "#FFFFFF", 2.8, 2),
    "tertiary":     ("#BDB29A", "#FFFFFF", 3.6, 3),
    "secondary":    ("#C79F5F", "#FBE6A0", 4.8, 4),
    "primary":      ("#C4894A", "#F8CF6C", 6.2, 5),
    "trunk":        ("#C27838", "#F4A863", 6.8, 6),
    "motorway":     ("#C27838", "#F4A863", 6.8, 6),
}
LINK_OF = {"tertiary_link": "tertiary", "secondary_link": "secondary",
           "primary_link": "primary", "trunk_link": "trunk"}
CASING_EXTRA_PX = 2.0
RIVER_MAJOR_PX, RIVER_MINOR_PX = 10.0, 3.2
REFERENCE_WIDTH = 4096  # style widths above are for this image width; scaled for other widths

# ---- extra Overpass layers (beyond build_basemap.py's six) -----------------------------------
def extra_queries(b: str) -> dict[str, str]:
    return {
        "roads_minor": ("[out:json][timeout:180];"
                        f'way["highway"~"^(tertiary|unclassified|tertiary_link|secondary_link|'
                        f'primary_link|trunk_link)$"]({b});out geom;'),
        "roads_residential": f'[out:json][timeout:240];way["highway"="residential"]({b});out geom;',
        "landcover": ("[out:json][timeout:240];("
                      f'way["natural"~"^(wood|scrub|wetland|grassland)$"]({b});'
                      f'way["landuse"~"^(forest|farmland|orchard|plantation|meadow|grass|paddy)$"]({b});'
                      f'relation["natural"~"^(wood|scrub|wetland)$"]({b});'
                      f'relation["landuse"~"^(forest|farmland|orchard|plantation|meadow)$"]({b}););'
                      "out geom;"),
        "places_minor": ("[out:json][timeout:120];"
                         f'node["place"~"^(suburb|village|hamlet|neighbourhood|quarter|island)$"]({b});'
                         "out body;"),
    }


def fetch_extras(bbox, use_cache: bool) -> dict[str, dict]:
    s, w, n, e = bbox
    out = {}
    for label, ql in extra_queries(f"{s},{w},{n},{e}").items():
        path = CACHE_DIR / f"{label}.json"
        if use_cache and path.exists():
            print(f"Using cached {label} ({path.stat().st_size} bytes)")
            out[label] = json.loads(path.read_text(encoding="utf-8"))
            continue
        print(f"Fetching {label}...")
        out[label] = bm.run_query(ql, label)
        path.write_text(json.dumps(out[label]), encoding="utf-8")
        time.sleep(4)
    return out


# ---- geometry helpers ---------------------------------------------------------------------------
def land_polygon(coastline_elements, bbox) -> Polygon:
    """Land = the piece of the bbox (split by the OSM coastline) that holds the towers, plus the
    interiors of closed coastline rings (islands). OSM coastline has land on its left, but with
    the whole coast clipped by a bbox it is more robust to anchor on the 136 real tower
    locations, which are by definition on land."""
    from api.spatial import load_towers

    s, w, n, e = bbox
    B = box(w, s, e, n)
    lines = bm.ways_as_lines(coastline_elements)
    merged = linemerge(unary_union(lines))
    geoms = [merged] if merged.geom_type == "LineString" else list(merged.geoms)
    pieces = [B]
    for g in geoms:
        nxt = []
        for p in pieces:
            if g.intersects(p) and not g.is_ring:
                try:
                    nxt += list(split(p, g).geoms)
                    continue
                except Exception:  # noqa: BLE001 -- keep the piece unsplit
                    pass
            nxt.append(p)
        pieces = nxt
    towers = load_towers()
    pts = [Point(x, y) for x, y in zip(towers["lon"], towers["lat"])]
    land = [p for p in pieces if any(p.contains(q) for q in pts)]
    for g in geoms:  # islands: closed coastline rings
        if g.is_ring:
            ring_poly = Polygon(g)
            if ring_poly.is_valid and ring_poly.intersects(B):
                land.append(ring_poly.intersection(B))
    return unary_union(land)


def to_paths(geom) -> list[MPath]:
    """Shapely (Multi)Polygon -> matplotlib compound paths with holes cut out."""
    polys = [geom] if geom.geom_type == "Polygon" else [g for g in geom.geoms if g.geom_type == "Polygon"]
    paths = []
    for poly in polys:
        if poly.is_empty:
            continue
        poly = orient(poly, 1.0)  # exterior CCW, holes CW -> holes render as holes
        rings = [np.asarray(poly.exterior.coords)] + [np.asarray(r.coords) for r in poly.interiors]
        verts = np.concatenate(rings)
        codes = np.concatenate([[MPath.MOVETO] + [MPath.LINETO] * (len(r) - 2) + [MPath.CLOSEPOLY]
                                for r in rings])
        paths.append(MPath(verts, codes))
    return paths


def safe_union(polys):
    polys = [p for p in polys if p.is_valid and not p.is_empty]
    if not polys:
        return None
    return unary_union(polys)


def polys_by_tag(elements, classify):
    """Group polygons from Overpass elements by classify(tags) -> str | None."""
    groups: dict[str, list] = {}
    for el in elements:
        kind = classify(el.get("tags", {}))
        if kind is None:
            continue
        groups.setdefault(kind, []).extend(bm.ways_and_relations_as_polygons([el]))
    return groups


def landcover_class(tags: dict) -> str | None:
    nat, lu = tags.get("natural"), tags.get("landuse")
    if nat in ("wood", "scrub") or lu == "forest":
        return "wood"
    if nat in ("wetland", "grassland"):
        return "wetland"
    if lu in ("farmland", "orchard", "plantation", "meadow", "grass", "paddy"):
        return "farmland"
    return None


def road_class(tags: dict) -> str | None:
    hw = tags.get("highway")
    hw = LINK_OF.get(hw, hw)
    return hw if hw in ROAD_STYLE else None


# ---- rendering -----------------------------------------------------------------------------------
def render(bbox, raw, extras, width: int) -> Image.Image:
    s, w, n, e = bbox
    dpp = (e - w) / width                       # degrees per pixel (same on both axes: equirectangular)
    height = int(round((n - s) / dpp))
    scale = width / REFERENCE_WIDTH
    dpi = 100
    pt = lambda px: px * scale * 72.0 / dpi     # style px (at 4096 wide) -> matplotlib points

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.axis("off")
    fig.patch.set_facecolor(STYLE["sea"])
    ax.set_facecolor(STYLE["sea"])
    clip = box(w, s, e, n)

    def fill(geom, color, edge=None, lw_px=0.0, z=1):
        if geom is None or geom.is_empty:
            return
        for path in to_paths(geom.intersection(clip)):
            ax.add_collection(PathCollection([path], facecolors=[color],
                                             edgecolors=[edge or color],
                                             linewidths=pt(lw_px), zorder=z))

    def lines(segs, color, lw_px, z, cap="round"):
        if not segs:
            return
        lc = LineCollection(segs, colors=color, linewidths=pt(lw_px), zorder=z,
                            capstyle=cap, joinstyle="round")
        ax.add_collection(lc)

    # 1. land (everything not land stays sea-blue)
    land = land_polygon(raw["coastline"]["elements"], bbox)
    fill(land, STYLE["land"], z=1)

    # 2. land cover (light farmland under darker wetland/wood)
    lc_groups = polys_by_tag(extras["landcover"]["elements"], landcover_class)
    for kind, z in (("farmland", 2), ("wetland", 3), ("wood", 4)):
        u = safe_union(lc_groups.get(kind, []))
        if u is not None:
            fill(u.intersection(land) if land is not None else u, STYLE[kind], z=z)

    # 3. built-up areas
    builtup = safe_union(bm.ways_and_relations_as_polygons(raw["landuse"]["elements"]))
    fill(builtup, STYLE["builtup"], edge=STYLE["builtup_edge"], lw_px=1.2, z=5)

    # 4. water polygons + rivers
    water = safe_union(bm.ways_and_relations_as_polygons(raw["water"]["elements"]))
    fill(water, STYLE["water"], edge=STYLE["coast"], lw_px=0.8, z=6)
    named_rivers: dict[str, list] = {}
    for line, name in bm.ways_as_lines_with_names(raw["rivers"]["elements"]):
        if name:
            named_rivers.setdefault(name, []).append(line)
    minor_river, major_river = [], []
    for name, ls in named_rivers.items():
        target = major_river if "kapuas" in name.lower() else minor_river
        target += [np.asarray(l.coords) for l in ls]
    lines(minor_river, STYLE["water"], RIVER_MINOR_PX, 7)
    lines(major_river, STYLE["water"], RIVER_MAJOR_PX, 7)

    # 5. coastline: thin darker edge along the land/sea boundary
    if land is not None and not land.is_empty:
        bounds = land.boundary
        segs = []
        for g in ([bounds] if bounds.geom_type == "LineString" else list(bounds.geoms)):
            arr = np.asarray(g.coords)
            # drop the stretches that merely run along the bbox edge
            keep = ~((np.isclose(arr[:, 0], w) | np.isclose(arr[:, 0], e) |
                      np.isclose(arr[:, 1], s) | np.isclose(arr[:, 1], n)))
            run = []
            for pt_xy, k in zip(arr, keep):
                if k:
                    run.append(pt_xy)
                elif len(run) > 1:
                    segs.append(np.asarray(run)); run = []
                else:
                    run = []
            if len(run) > 1:
                segs.append(np.asarray(run))
        lines(segs, STYLE["coast"], 1.6, 8, cap="butt")

    # 6. roads: casing pass, then fill pass; lower classes first so majors sit on top
    by_class: dict[str, list] = {k: [] for k in ROAD_STYLE}
    for src in (raw["roads"], extras["roads_minor"], extras["roads_residential"]):
        for el in src["elements"]:
            cls = road_class(el.get("tags", {}))
            if cls is None or el.get("type") != "way" or "geometry" not in el:
                continue
            xy = [(p["lon"], p["lat"]) for p in el["geometry"] if p.get("lon") is not None]
            if len(xy) >= 2:
                by_class[cls].append(np.asarray(xy))
    order = sorted(ROAD_STYLE, key=lambda k: ROAD_STYLE[k][3])
    for cls in order:
        casing, _, wpx, z = ROAD_STYLE[cls]
        lines(by_class[cls], casing, wpx + CASING_EXTRA_PX, 10 + z * 2)
    for cls in order:
        _, fillc, wpx, z = ROAD_STYLE[cls]
        lines(by_class[cls], fillc, wpx, 11 + z * 2)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    img = Image.fromarray(buf[:, :, :3])
    print(f"Rendered {img.size[0]}x{img.size[1]} px, {(e - w) / img.size[0] * 111320:.0f} m/px at the equator")
    return img


def encode_png(img: Image.Image, colors: int) -> bytes:
    """Palette PNG: flat map colors compress very well once quantised (no dithering, so edges
    stay clean)."""
    q = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    buf = io.BytesIO()
    q.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def minor_places(extras, bbox, max_labels: int = 26, min_sep_deg: float = 0.06) -> list[dict]:
    """A curated, de-overlapped subset of village/suburb names to label as crisp vector text
    (baked-in text would blur when the dashboard zooms to an incident). Suburbs first, then
    villages, nearest-to-many-towers first; greedy minimum spacing so labels never pile up."""
    rank = {"suburb": 0, "quarter": 0, "neighbourhood": 1, "village": 1, "hamlet": 2, "island": 2}
    cands = []
    for el in extras["places_minor"]["elements"]:
        tags = el.get("tags", {})
        name = tags.get("name")
        if el.get("type") != "node" or not name or len(name) > 24:
            continue
        cands.append((rank.get(tags.get("place"), 3), name, el["lon"], el["lat"], tags.get("place")))
    # Where labels matter: near the 136 real towers. Order by place rank, then by how many towers
    # sit within ~6 km (more first), then name for a stable result.
    from api.spatial import load_towers
    tw = load_towers()
    tx, ty = tw["lon"].to_numpy(), tw["lat"].to_numpy()
    near = lambda c: int((np.hypot(tx - c[2], ty - c[3]) < 0.055).sum())
    cands.sort(key=lambda c: (c[0], -near(c), c[1]))
    chosen: list[tuple] = []
    seen_names = {"pontianak", "sungai raya"}  # already labeled as rank-1 places by build_basemap.py
    for c in cands:
        if c[1].lower() in seen_names:
            continue
        if all(max(abs(c[2] - o[2]), abs(c[3] - o[3])) >= min_sep_deg for o in chosen):
            seen_names.add(c[1].lower())
            chosen.append(c)
        if len(chosen) >= max_labels:
            break
    return [{"type": "Feature",
             "properties": {"category": "place", "name": c[1], "place": c[4], "rank": 2},
             "geometry": {"type": "Point", "coordinates": [c[2], c[3]]}} for c in chosen]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-cache", action="store_true", help="reuse tools/basemap_cache/*.json")
    ap.add_argument("--width", type=int, default=4096, help="image width in px (default 4096)")
    ap.add_argument("--colors", type=int, default=64, help="PNG palette size (default 64)")
    ap.add_argument("--preview", type=Path, help="also write the PNG here for inspection")
    ap.add_argument("--no-write", action="store_true", help="render only; do not touch the asset")
    args = ap.parse_args()

    bbox = bm.compute_bbox()
    s, w, n, e = bbox
    print(f"bbox (same as the vector asset): south={s:.6f} west={w:.6f} north={n:.6f} east={e:.6f}")
    raw = bm.fetch_all(bbox, use_cache=args.from_cache)   # the six base layers
    extras = fetch_extras(bbox, use_cache=args.from_cache)  # minor roads, streets, land cover, villages
    img = render(bbox, raw, extras, args.width)
    png = encode_png(img, args.colors)
    print(f"PNG: {len(png) / 1024:.0f} KB ({args.colors} colors)")
    if args.preview:
        args.preview.write_bytes(png)
        print(f"Wrote preview {args.preview}")
    if args.no_write:
        return

    fc = json.loads(ASSET_PATH.read_text(encoding="utf-8"))
    fc["features"] = [f for f in fc["features"]
                      if not (f["properties"].get("category") == "place" and f["properties"].get("rank") == 2)]
    minor = minor_places(extras, bbox)
    for f in fc["features"]:
        if f["properties"].get("category") == "place":
            f["properties"].setdefault("rank", 1)
    fc["features"] += minor
    fc["raster"] = {
        "format": "png", "encoding": "base64", "width": img.size[0], "height": img.size[1],
        "bbox": [w, s, e, n],
        "attribution": "(c) OpenStreetMap contributors (ODbL)",
        "note": ("Styled color rendering of real OpenStreetMap data (coastline, water, rivers, "
                 "roads, land cover, built-up areas) for the tower bounding box, rendered once "
                 "offline by tools/build_basemap_raster.py. Colors and line widths are styling "
                 "only. Equirectangular: pixel x/y map linearly to longitude/latitude."),
        "data": base64.b64encode(png).decode("ascii"),
    }
    encoded = json.dumps(fc, separators=(",", ":"))
    size = len(encoded.encode("utf-8"))
    print(f"Asset size: {size / 1024:.0f} KB ({len(minor)} minor place labels added)")
    if size >= int(1.5 * 1024 * 1024):
        raise SystemExit(f"asset would be {size} bytes, over the 1.5 MB budget enforced by "
                         "tests/test_basemap.py -- lower --width or --colors")
    ASSET_PATH.write_text(encoded, encoding="utf-8")
    print(f"Wrote {ASSET_PATH}")


if __name__ == "__main__":
    main()
