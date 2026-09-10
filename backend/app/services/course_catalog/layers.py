"""Fuse catalog GPS and scorecard into OSM layers."""

from __future__ import annotations

import math
from typing import Any

from shapely.errors import ShapelyError
from shapely.geometry import LineString, MultiPoint, Point, mapping, shape
from shapely.geometry.base import BaseGeometry

from app.services.geometry import (
    area_m2,
    as_geom,
    buffer_meters,
    distance_meters,
    expand_bbox,
    feature_collection,
    to_feature,
)

PIN_REPLACE_M = 40.0
TEE_MATCH_M = 38.0
TEE_BUFFER_M = 7.0
GPS_PAD_M = 130.0
OSM_VS_GPS_AREA_RATIO = 2.2


def bbox_from_points(points: list[dict[str, Any]], pad_m: float = GPS_PAD_M) -> tuple[float, float, float, float] | None:
    hull = clip_geom_from_points(points, pad_m=pad_m)
    if hull is None or hull.is_empty:
        return None
    return expand_bbox(*hull.bounds, pad_deg=0.0004)


def clip_geom_from_points(points: list[dict[str, Any]], pad_m: float = GPS_PAD_M) -> BaseGeometry | None:
    pts = [Point(p["lon"], p["lat"]) for p in points if _valid_ll(p)]
    if not pts:
        return None
    if len(pts) == 1:
        return buffer_meters(pts[0], pad_m)
    geom: BaseGeometry = MultiPoint(pts).convex_hull
    return buffer_meters(geom, pad_m)


def prefer_clip(osm_boundary: BaseGeometry | None, gps_clip: BaseGeometry | None) -> BaseGeometry | None:
    if gps_clip is None or gps_clip.is_empty:
        return osm_boundary
    if osm_boundary is None or osm_boundary.is_empty:
        return gps_clip
    if area_m2(osm_boundary) > OSM_VS_GPS_AREA_RATIO * max(area_m2(gps_clip), 1.0):
        return gps_clip
    return osm_boundary


def fuse_layers(
    layers: dict[str, dict[str, Any]],
    points: list[dict[str, Any]],
    *,
    scorecard: dict[str, Any] | None = None,
    source: str = "catalog",
) -> dict[str, Any]:
    """Mutate layers in place. Returns stats for logging."""
    stats = {
        "pins_catalog": 0,
        "pins_osm_kept": 0,
        "tees_added": 0,
        "greens_numbered": 0,
        "centerlines_added": 0,
        "holes_stamped": 0,
        "bunker_seeds": 0,
        "water_seeds": 0,
        "fairway_seeds": 0,
        "dogleg_seeds": 0,
        "source": source,
    }
    if not points:
        return stats

    pin_pts = _pins_by_hole(points)
    tee_pts = [p for p in points if p["kind"] == "tee" and _valid_ll(p)]
    doglegs = [p for p in points if p["kind"] == "dogleg" and _valid_ll(p)]
    stats["bunker_seeds"] = sum(1 for p in points if p.get("kind") == "bunker" and _valid_ll(p))
    stats["water_seeds"] = sum(1 for p in points if p.get("kind") == "water" and _valid_ll(p))
    stats["fairway_seeds"] = sum(1 for p in points if p.get("kind") == "fairway" and _valid_ll(p))
    stats["dogleg_seeds"] = len(doglegs)

    stats["pins_catalog"] = len(pin_pts)
    layers["pin"] = _merge_pins(layers.get("pin"), pin_pts, stats, source=source)
    stats["greens_numbered"] = _stamp_nearest(layers.get("green"), pin_pts, max_m=45.0)
    stats["tees_added"] = _merge_tees(layers, tee_pts, source=source)

    osm_holes = (layers.get("hole_centerline") or {}).get("features") or []
    if osm_holes:
        stats["holes_stamped"] = _stamp_hole_lines(layers["hole_centerline"], pin_pts, scorecard)
    else:
        lines = _centerlines_from_gps(
            pin_pts, tee_pts, scorecard, source=source, doglegs=doglegs
        )
        if lines:
            layers["hole_centerline"] = feature_collection(lines)
            stats["centerlines_added"] = len(lines)
    return stats


def _pins_by_hole(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: dict[int | str, dict[str, Any]] = {}
    extras: list[dict[str, Any]] = []
    greens = [p for p in points if p["kind"] == "green" and _valid_ll(p)]
    pins = [p for p in points if p["kind"] == "pin" and _valid_ll(p)]
    for point in greens + pins:
        hole = point.get("hole")
        if hole:
            ranked[int(hole)] = point
        else:
            extras.append(point)
    ordered = [ranked[k] for k in sorted(ranked)]
    return ordered or extras or greens or pins


def _merge_pins(
    existing: dict[str, Any] | None,
    catalog_pins: list[dict[str, Any]],
    stats: dict[str, int],
    *,
    source: str,
) -> dict:
    feats = []
    used_osm: set[int] = set()
    osm_feats = list((existing or {}).get("features") or [])
    for point in catalog_pins:
        props = {
            "source": source,
            "kind": point["kind"],
            "hole": point.get("hole"),
        }
        feats.append(to_feature(Point(point["lon"], point["lat"]), props))
    for i, feat in enumerate(osm_feats):
        geom = _try_geom(feat)
        if geom is None:
            continue
        pt = geom if geom.geom_type == "Point" else geom.centroid
        if any(distance_meters(pt, Point(p["lon"], p["lat"])) <= PIN_REPLACE_M for p in catalog_pins):
            continue
        props = dict(feat.get("properties") or {})
        props["source"] = props.get("source") or "openstreetmap"
        feats.append({"type": "Feature", "geometry": feat.get("geometry"), "properties": props})
        used_osm.add(i)
    stats["pins_osm_kept"] = len(used_osm)
    return feature_collection(feats)


def _stamp_nearest(fc: dict[str, Any] | None, pins: list[dict[str, Any]], max_m: float) -> int:
    if not fc or not pins:
        return 0
    stamped = 0
    for feat in fc.get("features") or []:
        geom = _try_geom(feat)
        if geom is None:
            continue
        best = None
        best_d = max_m
        for pin in pins:
            d = distance_meters(geom, Point(pin["lon"], pin["lat"]))
            if d < best_d:
                best_d = d
                best = pin
        if best and best.get("hole"):
            props = feat.setdefault("properties", {})
            props["hole"] = best["hole"]
            props["catalog_hole"] = best["hole"]
            stamped += 1
    return stamped


def _merge_tees(layers: dict[str, dict[str, Any]], tee_pts: list[dict[str, Any]], *, source: str) -> int:
    existing = list((layers.get("tee") or {}).get("features") or [])
    added = 0
    for point in tee_pts:
        pt = Point(point["lon"], point["lat"])
        matched = False
        for feat in existing:
            geom = _try_geom(feat)
            if geom is None:
                continue
            if distance_meters(geom, pt) <= TEE_MATCH_M:
                props = feat.setdefault("properties", {})
                if point.get("hole"):
                    props["hole"] = point["hole"]
                    props["catalog_hole"] = point["hole"]
                matched = True
                break
        if matched:
            continue
        poly = buffer_meters(pt, TEE_BUFFER_M)
        existing.append(
            to_feature(
                poly,
                {
                    "source": source,
                    "hole": point.get("hole"),
                    "instance_id": f"{source}-tee-{point.get('hole') or added}-{added}",
                },
            )
        )
        added += 1
    if existing:
        layers["tee"] = feature_collection(existing)
    return added


def _stamp_hole_lines(
    fc: dict[str, Any],
    pins: list[dict[str, Any]],
    scorecard: dict[str, Any] | None,
) -> int:
    stamped = 0
    used_holes: set[int] = set()
    for feat in fc.get("features") or []:
        geom = _try_geom(feat)
        if geom is None:
            continue
        if geom.geom_type != "LineString" or geom.is_empty:
            continue
        start, end = Point(geom.coords[0]), Point(geom.coords[-1])
        pin = _nearest_point(end, pins, start)
        hole = pin.get("hole") if pin else None
        if hole in used_holes:
            pin = _nearest_point(end, [p for p in pins if p.get("hole") not in used_holes], start)
            hole = pin.get("hole") if pin else hole
        props = feat.setdefault("properties", {})
        if hole:
            props["hole"] = hole
            props["ref"] = str(hole)
            props["catalog_hole"] = hole
            used_holes.add(int(hole))
            _apply_scorecard(props, hole, scorecard)
            stamped += 1
        props["source"] = props.get("source") or "openstreetmap"
    return stamped


def _centerlines_from_gps(
    pins: list[dict[str, Any]],
    tees: list[dict[str, Any]],
    scorecard: dict[str, Any] | None,
    *,
    source: str,
    doglegs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    tees_by_hole: dict[int, list[dict[str, Any]]] = {}
    for tee in tees:
        hole = tee.get("hole")
        if hole:
            tees_by_hole.setdefault(int(hole), []).append(tee)
    doglegs_by_hole: dict[int, list[dict[str, Any]]] = {}
    for bend in doglegs or []:
        hole = bend.get("hole")
        if hole:
            doglegs_by_hole.setdefault(int(hole), []).append(bend)
    lines = []
    for pin in pins:
        hole = pin.get("hole")
        if not hole:
            continue
        green = Point(pin["lon"], pin["lat"])
        candidates = tees_by_hole.get(int(hole)) or []
        tee_pt = _farthest_tee(candidates, green)
        if tee_pt is None:
            continue
        tee = Point(tee_pt["lon"], tee_pt["lat"])
        if tee.equals(green):
            continue
        coords = [tee, *_dogleg_waypoints(tee, green, doglegs_by_hole.get(int(hole))), green]
        props = {
            "hole": int(hole),
            "ref": str(int(hole)),
            "source": source,
            "name": f"Hole {int(hole)}",
        }
        _apply_scorecard(props, int(hole), scorecard)
        lines.append(to_feature(LineString(coords), props))
    return lines


def _dogleg_waypoints(tee: Point, green: Point, bends: list[dict[str, Any]] | None) -> list[Point]:
    if not bends:
        return []
    ordered = sorted(bends, key=lambda row: tee.distance(Point(row["lon"], row["lat"])))
    out: list[Point] = []
    for row in ordered:
        pt = Point(row["lon"], row["lat"])
        if pt.distance(tee) < 1e-7 or pt.distance(green) < 1e-7:
            continue
        out.append(pt)
    return out


def _apply_scorecard(props: dict[str, Any], hole: int, scorecard: dict[str, Any] | None) -> None:
    if not scorecard or hole < 1:
        return
    idx = hole - 1
    pars = scorecard.get("pars_men") or []
    indexes = scorecard.get("indexes_men") or []
    if idx < len(pars):
        props["par"] = pars[idx]
    if idx < len(indexes):
        props["stroke_index"] = indexes[idx]


def _nearest_point(target: Point, points: list[dict[str, Any]], other: Point | None = None) -> dict[str, Any] | None:
    best = None
    best_d = 1e12
    for point in points:
        pt = Point(point["lon"], point["lat"])
        d = distance_meters(target, pt)
        if other is not None:
            d = min(d, distance_meters(other, pt))
        if d < best_d:
            best_d = d
            best = point
    return best


def _farthest_tee(tees: list[dict[str, Any]], green: Point) -> dict[str, Any] | None:
    if not tees:
        return None
    return max(tees, key=lambda t: math.hypot(t["lon"] - green.x, t["lat"] - green.y))


def _valid_ll(point: dict[str, Any]) -> bool:
    try:
        lat, lon = float(point["lat"]), float(point["lon"])
    except (KeyError, TypeError, ValueError):
        return False
    return -90 <= lat <= 90 and -180 <= lon <= 180


def geojson_clip(geom: BaseGeometry | None) -> dict[str, Any] | None:
    if geom is None or geom.is_empty:
        return None
    return mapping(geom)


def geom_from_geojson(value: Any) -> BaseGeometry | None:
    if not value:
        return None
    try:
        if isinstance(value, dict):
            return shape(value)
    except (TypeError, ValueError, ShapelyError):
        return None
    return None


def _try_geom(feat: Any) -> BaseGeometry | None:
    try:
        geom = as_geom(feat)
    except (TypeError, ValueError, ShapelyError):
        return None
    return None if geom is None or geom.is_empty else geom
