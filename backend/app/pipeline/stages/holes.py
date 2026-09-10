"""Stage 7 — Hole identification, tee-to-green centerlines, and fairway fill.

OSM golf=hole ways (with ref) are the centerlines. OSM greens keep their
identity; segmentation may already have tightened the ring to imagery.
The hole whose pin sits on a green claims it. Holes with a pin and no OSM
green may receive a compact pin-seeded imagery outline. Circles are never
invented. Putting greens move only when OSM tagged them practice. Tees are
exclusive to the nearest hole start. Fairways are one grass-clipped corridor
per hole.
"""

from __future__ import annotations

import math
from typing import Any

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from app.pipeline.context import PipelineContext
from app.services.geometry import (
    area_m2,
    as_geom,
    as_polygons,
    buffer_meters,
    clean_polygon,
    difference_safe,
    distance_meters,
    feature_collection,
    geoms_from_fc,
    iou,
    to_feature,
    union_layer,
)

# A green farther than this from the pin is not that hole's putting surface.
MAX_GREEN_TO_PIN_M = 28.0
TEE_CATCH_M = 45.0
FAIRWAY_HALF_WIDTH_M = 22.0
FAIRWAY_MIN_AREA_M2 = FAIRWAY_HALF_WIDTH_M * 8
FILL_GREEN_AREA_M2 = (90.0, 1800.0)
FILL_GREEN_MIN_COMPACT = 0.50
FILL_GREEN_PIN_M = 8.0
FILL_GREEN_OVERLAP = 0.12


async def run(ctx: PipelineContext) -> None:
    greens = [g for g in _features(ctx, "green") if not _is_tagged_practice({"properties": g["properties"]})]
    tees = _features(ctx, "tee")
    pins = _features(ctx, "pin")
    fairways = geoms_from_fc(ctx.layers.get("fairway"))
    bunkers = _features(ctx, "bunker")
    water = _features(ctx, "water")

    existing = []
    for feat in (ctx.layers.get("hole_centerline") or {}).get("features", []):
        try:
            geom = as_geom(feat)
        except Exception:
            continue
        if geom.geom_type == "LineString" and geom.length > 0:
            existing.append((geom, feat.get("properties") or {}))

    if existing:
        holes = _from_osm_holes(existing, greens, tees, pins, ctx)
        ctx.log(f"Hole identification from {len(existing)} OSM hole ways")
    else:
        holes = _pair_tees_greens(tees, greens, fairways, ctx)
        ctx.log(f"Hole identification via tee–green pairing ({len(holes)} holes)")

    filled, skipped = _fill_missing_greens_from_imagery(ctx, holes)
    if filled:
        ctx.log(f"Filled {len(filled)} missing greens from pin imagery ({filled})")
    if skipped:
        ctx.log(f"Holes still without a green: {skipped}")

    _assign_tees_exclusive(holes, tees)
    _split_practice_greens(ctx, holes)
    _associate(holes, bunkers, "bunkers")
    _associate(holes, water, "hazards")

    centerlines = []
    for hole in holes:
        props = {
            "hole": hole["number"],
            "name": f"Hole {hole['number']}",
            "tee_ids": hole.get("tee_ids", []),
            "green_id": hole.get("green_id"),
            "bunker_ids": hole.get("bunkers", []),
            "hazard_ids": hole.get("hazards", []),
            "source": hole.get("source", "heuristic"),
        }
        _apply_catalog_scorecard(ctx, props, hole["number"])
        centerlines.append(to_feature(hole["centerline"], props))
        _stamp_hole_number(ctx, "green", hole.get("green_id"), hole["number"])
        _stamp_pin_number(ctx, hole.get("pin"), hole["number"])
        for tee_id in hole.get("tee_ids", []):
            _stamp_hole_number(ctx, "tee", tee_id, hole["number"])
        for bunker_id in hole.get("bunkers", []):
            _stamp_hole_number(ctx, "bunker", bunker_id, hole["number"])

    if centerlines:
        ctx.layers["hole_centerline"] = feature_collection(centerlines)
    crossings = _crossing_pairs(holes)
    missing_green = [h["number"] for h in holes if not h.get("green_id")]
    ctx.quality["holes_detected"] = len(centerlines)
    ctx.quality["hole_centerlines_cross"] = crossings
    ctx.quality["greens_imagery_filled"] = filled
    ctx.quality["holes_missing_green"] = missing_green
    ctx.quality["holes_need_review"] = (
        (len(centerlines) not in {9, 18} and len(centerlines) > 0)
        or bool(crossings)
        or bool(missing_green)
    )

    added = _derive_fairways(ctx, holes)
    if added:
        ctx.log(f"Derived {added} fairway corridors from hole centerlines")
        _associate(holes, _features(ctx, "fairway"), "fairways")


def _features(ctx: PipelineContext, layer_id: str) -> list[dict[str, Any]]:
    fc = ctx.layers.get(layer_id) or {"features": []}
    out = []
    for feat in fc.get("features", []):
        try:
            geom = as_geom(feat)
        except Exception:
            continue
        props = feat.get("properties") or {}
        out.append(
            {
                "geom": geom,
                "properties": props,
                "id": props.get("instance_id") or props.get("osm_id"),
            }
        )
    return out


def _from_osm_holes(
    existing: list[tuple[Any, dict]],
    greens: list[dict],
    tees: list[dict],
    pins: list[dict],
    ctx: PipelineContext,
) -> list[dict[str, Any]]:
    numbered = []
    unnumbered = []
    for geom, props in existing:
        ref = _parse_ref(props.get("ref"))
        (numbered if ref else unnumbered).append((geom, props, ref))
    numbered.sort(key=lambda item: item[2] or 0)
    next_num = 1
    used_nums = {item[2] for item in numbered if item[2]}
    drafts: list[dict[str, Any]] = []
    for geom, props, ref in numbered + unnumbered:
        number = ref or next_num
        while number in used_nums and ref is None:
            number += 1
        used_nums.add(number)
        if ref is None:
            next_num = number + 1
        line = _orient_hole(geom, tees, pins)
        drafts.append(
            {
                "number": number,
                "centerline": line,
                "pin": None,
                "green_id": None,
                "tee_ids": [],
                "source": props.get("source") or "osm",
                "bunkers": [],
                "hazards": [],
            }
        )
    _assign_end_pins(drafts, pins)
    _assign_osm_greens(drafts, greens)
    return drafts


def _orient_hole(line: LineString, tees: list[dict], pins: list[dict]) -> LineString:
    """Pin marks the green, so that endpoint is the hole end.

    Tees near a short-hole green must not flip the line backwards (Jamor 8).
    """
    a, b = Point(line.coords[0]), Point(line.coords[-1])
    pin_a = min((distance_meters(p["geom"], a) for p in pins), default=1e9)
    pin_b = min((distance_meters(p["geom"], b) for p in pins), default=1e9)
    if pin_a + 6.0 < pin_b:
        return LineString(list(line.coords)[::-1])
    if pin_b + 6.0 < pin_a:
        return line
    tee_a = sum(1 for t in tees if distance_meters(t["geom"], a) < TEE_CATCH_M)
    tee_b = sum(1 for t in tees if distance_meters(t["geom"], b) < TEE_CATCH_M)
    if tee_b > tee_a:
        return LineString(list(line.coords)[::-1])
    return line


def _orient_to_pin(line: LineString, pin: Point) -> LineString:
    start, end = Point(line.coords[0]), Point(line.coords[-1])
    if distance_meters(start, pin) + 0.5 < distance_meters(end, pin):
        return LineString(list(line.coords)[::-1])
    return line


def _green_contains_pin(green: dict, pin: Point) -> bool:
    try:
        return bool(green["geom"].buffer(1e-7).contains(pin) or distance_meters(green["geom"], pin) < 6)
    except Exception:
        return False


PIN_CATCH_M = 32.0


def _assign_end_pins(holes: list[dict[str, Any]], pins: list[dict]) -> None:
    """Each pin belongs to at most one hole — nearest start or end."""
    pairs: list[tuple[float, dict, Point, int]] = []
    for hole in holes:
        start = Point(hole["centerline"].coords[0])
        end = Point(hole["centerline"].coords[-1])
        hole["_end"] = end
        hole["pin"] = None
        for i, pin in enumerate(pins):
            pt = pin["geom"]
            if pt.geom_type != "Point":
                pt = pt.centroid
            pt = Point(pt.x, pt.y)
            d = min(distance_meters(end, pt), distance_meters(start, pt))
            pairs.append((d, hole, pt, i))
    pairs.sort(key=lambda item: item[0])
    used_holes: set[int] = set()
    used_pins: set[int] = set()
    for dist, hole, pt, pin_i in pairs:
        hid = id(hole)
        if hid in used_holes or pin_i in used_pins:
            continue
        if dist > PIN_CATCH_M:
            continue
        hole["pin"] = pt
        hole["centerline"] = _orient_to_pin(hole["centerline"], pt)
        hole["_end"] = Point(hole["centerline"].coords[-1])
        used_holes.add(hid)
        used_pins.add(pin_i)


def _assign_osm_greens(holes: list[dict[str, Any]], greens: list[dict]) -> None:
    """Pin-in-green first so an earlier hole cannot steal a later hole's surface.

    Imagery fills happen later, only for holes that still have no OSM green.
    """
    osm_greens = [g for g in greens if _is_osm_play({"properties": g.get("properties") or {}})]
    used: set[Any] = set()
    claims: list[tuple[float, dict, dict]] = []
    for hole in holes:
        pin = hole.get("pin")
        if pin is None:
            continue
        for green in osm_greens:
            if not _green_contains_pin(green, pin):
                continue
            claims.append((distance_meters(green["geom"].centroid, pin), hole, green))
    claims.sort(key=lambda item: item[0])
    for _d, hole, green in claims:
        if hole.get("green_id") or green["id"] in used:
            continue
        hole["green_id"] = green["id"]
        used.add(green["id"])

    leftovers: list[tuple[float, dict, dict]] = []
    for hole in holes:
        if hole.get("green_id"):
            continue
        end = hole.get("_end") or Point(hole["centerline"].coords[-1])
        for green in osm_greens:
            if green["id"] in used:
                continue
            leftovers.append((distance_meters(end, green["geom"]), hole, green))
    leftovers.sort(key=lambda item: item[0])
    for dist, hole, green in leftovers:
        if hole.get("green_id") or green["id"] in used:
            continue
        if dist > MAX_GREEN_TO_PIN_M:
            continue
        hole["green_id"] = green["id"]
        used.add(green["id"])


def _green_at_pin(pin: Point, greens: list[dict], used: set[Any]) -> dict | None:
    containing = []
    nearby = []
    for green in greens:
        if green["id"] in used:
            continue
        if _green_contains_pin(green, pin):
            containing.append(green)
        else:
            nearby.append((distance_meters(green["geom"], pin), green))
    if containing:
        return min(containing, key=lambda g: g["geom"].centroid.distance(pin))
    nearby.sort(key=lambda item: item[0])
    if nearby and nearby[0][0] <= MAX_GREEN_TO_PIN_M:
        return nearby[0][1]
    return None


def _pin_green_from_masks(ctx: PipelineContext, pin: Point):
    items = (ctx.masks or {}).get("pin_greens") or []
    best = None
    best_d = 14.0
    for item in items:
        coords = item.get("pin") or []
        if len(coords) < 2:
            continue
        d = distance_meters(pin, Point(float(coords[0]), float(coords[1])))
        if d >= best_d:
            continue
        ring = item.get("coordinates") or []
        if len(ring) < 4:
            continue
        try:
            geom = Polygon([(float(x), float(y)) for x, y in ring])
        except Exception:
            continue
        if geom.is_empty:
            continue
        best = geom
        best_d = d
    return clean_polygon(best) if best is not None else None


def _compactness(geom) -> float:
    try:
        per = geom.length
        a = geom.area
    except Exception:
        return 0.0
    if per <= 0 or a <= 0:
        return 0.0
    return float(4.0 * math.pi * a / (per * per))


def _osm_green_union(ctx: PipelineContext):
    parts = []
    for feat in (ctx.layers.get("green") or {}).get("features", []):
        if not _is_osm_play(feat):
            continue
        try:
            geom = as_geom(feat)
        except Exception:
            continue
        if geom.is_empty:
            continue
        parts.append(geom)
    return unary_union(parts) if parts else None


def _accept_pin_green(geom, pin: Point, osm_union) -> bool:
    """Reject sausages, circles-as-fallback, and blobs that sit on an OSM green."""
    cleaned = clean_polygon(geom) or geom
    if cleaned is None or cleaned.is_empty:
        return False
    area = area_m2(cleaned)
    if area < FILL_GREEN_AREA_M2[0] or area > FILL_GREEN_AREA_M2[1]:
        return False
    if _compactness(cleaned) < FILL_GREEN_MIN_COMPACT:
        return False
    if not (cleaned.buffer(1e-7).contains(pin) or distance_meters(cleaned, pin) < FILL_GREEN_PIN_M):
        return False
    if osm_union is not None and not osm_union.is_empty:
        try:
            inter = cleaned.intersection(osm_union)
            frac = area_m2(inter) / max(area, 1e-6)
        except Exception:
            return False
        if frac >= FILL_GREEN_OVERLAP or iou(cleaned, osm_union) >= 0.18:
            return False
    return True


def _add_imagery_green(ctx: PipelineContext, geom, number: int, sources: list[str] | None = None) -> dict[str, Any]:
    ident = f"imagery-green-{number}"
    props = {
        "instance_id": ident,
        "source": "imagery_pin",
        "golf": "green",
        "hole": number,
        "name": f"Hole {number} green",
        "imagery_sources": sources or [],
    }
    fc = ctx.layers.setdefault("green", feature_collection([]))
    fc.setdefault("features", []).append(to_feature(geom, props))
    return {"geom": geom, "properties": props, "id": ident}


def _fill_missing_greens_from_imagery(
    ctx: PipelineContext, holes: list[dict[str, Any]]
) -> tuple[list[int], list[int]]:
    """Add a pin-grown outline only when OSM has no green for that hole.

    Never invents a circle. Overlap with an OSM green is rejected so hole 8
    cannot land on hole 9's surface.
    """
    filled: list[int] = []
    skipped: list[int] = []
    osm_union = _osm_green_union(ctx)
    used_ids: set[Any] = {h.get("green_id") for h in holes if h.get("green_id")}
    for hole in holes:
        if hole.get("green_id"):
            continue
        pin = hole.get("pin")
        if pin is None:
            skipped.append(hole["number"])
            continue
        recovered = _pin_green_from_masks(ctx, pin)
        if recovered is None or not _accept_pin_green(recovered, pin, osm_union):
            skipped.append(hole["number"])
            continue
        ident = f"imagery-green-{hole['number']}"
        if ident in used_ids:
            skipped.append(hole["number"])
            continue
        sources = []
        for item in (ctx.masks or {}).get("pin_greens") or []:
            coords = item.get("pin") or []
            if len(coords) >= 2 and distance_meters(pin, Point(float(coords[0]), float(coords[1]))) < 14:
                sources = list(item.get("sources") or [])
                break
        added = _add_imagery_green(ctx, recovered, hole["number"], sources)
        hole["green_id"] = added["id"]
        used_ids.add(added["id"])
        filled.append(hole["number"])
    return filled, skipped


def _tees_near(start: Point, tees: list[dict]) -> list[dict]:
    return [t for t in tees if distance_meters(t["geom"], start) < TEE_CATCH_M]


def _back_tee(hole_tees: list[dict], green) -> Point:
    return max(hole_tees, key=lambda t: t["geom"].centroid.distance(green.centroid))["geom"].centroid


def _assign_tees_exclusive(holes: list[dict[str, Any]], tees: list[dict]) -> None:
    """Each tee belongs to at most one hole — the nearest start within TEE_CATCH_M."""
    for hole in holes:
        hole["tee_ids"] = []
    if not holes or not tees:
        return
    starts = {id(hole): Point(hole["centerline"].coords[0]) for hole in holes}
    for tee in tees:
        best = None
        best_d = TEE_CATCH_M
        for hole in holes:
            d = distance_meters(tee["geom"], starts[id(hole)])
            if d < best_d:
                best_d = d
                best = hole
        if best is not None:
            best.setdefault("tee_ids", []).append(tee["id"])


def _crossing_pairs(holes: list[dict[str, Any]]) -> list[list[int]]:
    pairs: list[list[int]] = []
    for i, a in enumerate(holes):
        la = a.get("centerline")
        if la is None or la.is_empty:
            continue
        for b in holes[i + 1 :]:
            lb = b.get("centerline")
            if lb is None or lb.is_empty:
                continue
            try:
                if la.crosses(lb):
                    pairs.append(sorted([int(a["number"]), int(b["number"])]))
            except Exception:
                continue
    return pairs


def _nearest_point(pt: Point, features: list[dict]) -> Point | None:
    if not features:
        return None
    best = min(features, key=lambda f: f["geom"].distance(pt))
    return best["geom"].centroid if best["geom"].geom_type != "Point" else Point(best["geom"].x, best["geom"].y)


def _is_tagged_practice(feat: dict[str, Any]) -> bool:
    props = feat.get("properties") or {}
    if props.get("practice") is True:
        return True
    golf = str(props.get("golf") or "").lower()
    return golf in {"practice", "putting_green", "practice_green"}


def _split_practice_greens(ctx: PipelineContext, holes: list[dict[str, Any]]) -> None:
    assigned = {h.get("green_id") for h in holes}
    fc = ctx.layers.get("green")
    if not fc:
        return
    play, practice = [], []
    dropped = 0
    for feat in fc.get("features", []):
        props = feat.get("properties") or {}
        ident = props.get("instance_id") or props.get("osm_id")
        if _is_tagged_practice(feat):
            props = dict(props)
            props["practice"] = True
            practice.append({**feat, "properties": props})
        elif ident in assigned or _is_osm_play(feat):
            play.append(feat)
        else:
            dropped += 1
    ctx.layers["green"] = feature_collection(play)
    if practice:
        existing = list((ctx.layers.get("putting_green") or {}).get("features") or [])
        ctx.layers["putting_green"] = feature_collection(existing + practice)
        ctx.log(f"Moved {len(practice)} OSM-tagged practice greens to putting")
    if dropped:
        ctx.log(f"Dropped {dropped} unmatched imagery greens (not added to practice)")


def _pair_tees_greens(
    tees: list[dict], greens: list[dict], fairways: list, ctx: PipelineContext
) -> list[dict[str, Any]]:
    greens = [g for g in greens if not _is_tagged_practice({"properties": g.get("properties") or {}})]
    if not greens:
        return []
    pairs: list[tuple[float, dict, dict]] = []
    for tee in tees:
        for green in greens:
            line = LineString([tee["geom"].centroid, green["geom"].centroid])
            score = line.length
            if fairways:
                hits = sum(1 for fw in fairways if fw.intersects(line))
                if hits:
                    score *= 0.35 / hits
            pairs.append((score, tee, green))
    pairs.sort(key=lambda item: item[0])

    used_t: set[Any] = set()
    used_g: set[Any] = set()
    assigned: list[tuple[dict | None, dict]] = []
    for _score, tee, green in pairs:
        if tee["id"] in used_t or green["id"] in used_g:
            continue
        used_t.add(tee["id"])
        used_g.add(green["id"])
        assigned.append((tee, green))

    for green in greens:
        if green["id"] not in used_g:
            assigned.append((None, green))

    origin = _numbering_origin(ctx, tees)
    remaining = assigned[:]
    ordered: list[tuple[dict | None, dict]] = []
    cursor = origin
    while remaining:
        remaining.sort(key=lambda pair: pair[1]["geom"].centroid.distance(cursor))
        nxt = remaining.pop(0)
        ordered.append(nxt)
        cursor = nxt[1]["geom"].centroid

    holes = []
    for i, (tee, green) in enumerate(ordered, start=1):
        start = tee["geom"].centroid if tee else green["geom"].centroid
        end = green["geom"].centroid
        holes.append(
            {
                "number": i,
                "centerline": LineString([start, end]),
                "green_id": green["id"],
                "tee_ids": [tee["id"]] if tee else [],
                "source": "heuristic",
                "bunkers": [],
                "hazards": [],
            }
        )
    return holes


def _numbering_origin(ctx: PipelineContext, tees: list[dict]) -> Point:
    club = union_layer(ctx.layers.get("clubhouse")) or union_layer(ctx.layers.get("building"))
    if club:
        return club.centroid
    if tees:
        return max(tees, key=lambda t: t["geom"].centroid.y)["geom"].centroid
    if ctx.course.get("lat") is not None:
        return Point(ctx.course["lon"], ctx.course["lat"])
    return Point(0, 0)


def _associate(holes: list[dict[str, Any]], features: list[dict], key: str) -> None:
    if not holes or not features:
        return
    for feat in features:
        best = None
        best_d = float("inf")
        pt = feat["geom"].centroid
        for hole in holes:
            d = hole["centerline"].distance(pt)
            if d < best_d:
                best_d = d
                best = hole
        if best is not None:
            best.setdefault(key, []).append(feat["id"])


def _is_osm_play(feat: dict[str, Any]) -> bool:
    """True for OSM (or OSM-refined) play features. Imagery fills are False."""
    props = feat.get("properties") or {}
    src = props.get("source")
    if src in {"imagery", "centerline_buffer", "pin", "imagery_pin"}:
        return False
    if props.get("osm_id") is not None:
        return True
    if src in {"openstreetmap", "osm", "hybrid", "osm_refined"}:
        return True
    return src is None


def _derive_fairways(ctx: PipelineContext, holes: list[dict[str, Any]]) -> int:
    existing_feats = list((ctx.layers.get("fairway") or {}).get("features") or [])
    if not holes:
        return 0

    osm_feats = [f for f in existing_feats if _is_osm_play(f)]
    kept: list[dict] = []
    covered: set[int] = set()
    for feat in osm_feats:
        try:
            geom = as_geom(feat)
        except Exception:
            continue
        hole_num = (feat.get("properties") or {}).get("hole")
        matched = None
        for hole in holes:
            if geom.intersects(buffer_meters(hole["centerline"], 28.0)):
                matched = hole
                break
        if matched is None and hole_num in {h["number"] for h in holes}:
            matched = next(h for h in holes if h["number"] == hole_num)
        props = dict(feat.get("properties") or {})
        if matched is not None:
            props["hole"] = matched["number"]
            covered.add(matched["number"])
        kept.append(to_feature(geom, props))

    added = 0
    for hole in holes:
        if hole["number"] in covered:
            continue
        for feat in _corridor_fairway(ctx, hole):
            kept.append(feat)
            added += 1
            covered.add(hole["number"])

    ctx.layers["fairway"] = feature_collection(kept)
    return added


def _geo_from_masks(ctx: PipelineContext):
    geo_d = (ctx.masks or {}).get("geo") or {}
    bbox = geo_d.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    from app.services.imagery_segment import MosaicGeo

    return MosaicGeo(
        width=int(geo_d.get("width") or 1),
        height=int(geo_d.get("height") or 1),
        west=float(bbox[0]),
        south=float(bbox[1]),
        east=float(bbox[2]),
        north=float(bbox[3]),
        zoom=geo_d.get("zoom"),
        tile_x_min=geo_d.get("tile_x_min"),
        tile_y_min=geo_d.get("tile_y_min"),
        tile_size=int(geo_d.get("tile_size") or 256),
    )


def _fairway_subtract(ctx: PipelineContext):
    parts = []
    for layer_id in ("green", "tee", "bunker", "water", "putting_green", "building", "driving_range"):
        u = union_layer(ctx.layers.get(layer_id))
        if u is not None and not u.is_empty:
            parts.append(u)
    return unary_union(parts) if parts else None


def _sausage_fairways(ctx: PipelineContext, holes: list[dict[str, Any]]) -> list[dict]:
    features = []
    for hole in holes:
        features.extend(_corridor_fairway(ctx, hole))
    return features


def _corridor_fairway(ctx: PipelineContext, hole: dict[str, Any]) -> list[dict]:
    corridor = buffer_meters(hole["centerline"], FAIRWAY_HALF_WIDTH_M)
    grass = _grass_union(ctx)
    if grass is not None and not grass.is_empty:
        try:
            clipped = corridor.intersection(grass)
        except Exception:
            clipped = corridor
        if area_m2(clipped) >= max(area_m2(corridor) * 0.22, FAIRWAY_MIN_AREA_M2):
            corridor = clipped
    subtract = _fairway_subtract(ctx)
    if subtract is not None:
        corridor = difference_safe(corridor, subtract)
    boundary = union_layer(ctx.layers.get("boundary"))
    if boundary is not None:
        corridor = corridor.intersection(boundary)
    polys = sorted(
        (p for p in as_polygons(corridor) if area_m2(p) >= FAIRWAY_MIN_AREA_M2),
        key=area_m2,
        reverse=True,
    )
    if not polys:
        polys = sorted(as_polygons(corridor), key=area_m2, reverse=True)[:1]
    features = []
    if not polys:
        return features
    cleaned = clean_polygon(polys[0]) or polys[0]
    if cleaned is None or cleaned.is_empty:
        return features
    features.append(
        to_feature(
            cleaned,
            {
                "hole": hole["number"],
                "source": "centerline_buffer",
                "golf": "fairway",
                "instance_id": f"fairway-hole-{hole['number']}-1",
            },
        )
    )
    return features


def _grass_union(ctx: PipelineContext):
    masks = (ctx.masks or {}).get("class_masks") or {}
    geo = _geo_from_masks(ctx)
    if geo is None:
        return None
    import numpy as np

    from app.services.imagery_segment import mask_to_polygons

    combined = None
    for key in ("fairway", "soft_grass"):
        raw = masks.get(key)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=bool)
        combined = arr if combined is None else (combined | arr)
    if combined is None or not combined.any():
        return None
    polys = mask_to_polygons(combined, geo, min_area_m2=40.0, max_area_m2=500_000.0)
    return unary_union(polys) if polys else None


def _parse_ref(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    number = int(digits)
    return number if 1 <= number <= 27 else None


def _apply_catalog_scorecard(ctx: PipelineContext, props: dict[str, Any], number: int) -> None:
    course = ctx.course or {}
    scorecard = course.get("catalog_scorecard") or course.get("golfapi_scorecard") or {}
    if not scorecard or number < 1:
        return
    idx = number - 1
    pars = scorecard.get("pars_men") or []
    indexes = scorecard.get("indexes_men") or []
    if idx < len(pars):
        props["par"] = pars[idx]
    if idx < len(indexes):
        props["stroke_index"] = indexes[idx]


def _stamp_hole_number(ctx: PipelineContext, layer_id: str, feature_id: Any, number: int) -> None:
    if feature_id is None:
        return
    fc = ctx.layers.get(layer_id)
    if not fc:
        return
    for feat in fc.get("features", []):
        props = feat.setdefault("properties", {})
        ident = props.get("instance_id") or props.get("osm_id")
        if ident == feature_id:
            props["hole"] = number


def _stamp_pin_number(ctx: PipelineContext, pin: Point | None, number: int) -> None:
    if pin is None:
        return
    fc = ctx.layers.get("pin")
    if not fc:
        return
    for feat in fc.get("features", []):
        try:
            geom = as_geom(feat)
        except Exception:
            continue
        pt = geom if geom.geom_type == "Point" else geom.centroid
        if distance_meters(pin, Point(pt.x, pt.y)) > 1.5:
            continue
        props = feat.setdefault("properties", {})
        props["hole"] = number
        props["ref"] = str(number)
        props["name"] = f"Hole {number}"
        return
