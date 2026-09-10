"""AI-identified course polygons, kept separate from OSM/API evidence layers."""

from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
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
from app.services.imagery_segment import (
    FAIRWAY_AREA_M2,
    FRINGE_WIDTH_M,
    TREE_CROWN_M2,
    TREE_MIN_COMPACT,
    AnalysisResult,
    MosaicGeo,
    catalog_seed_points,
    classify_pixels,
    derive_green_fringes,
    extract_seeded,
    grow_fairway_along_line,
    mask_to_polygons,
    vegetation_index,
)
from app.services.imagery_segment import _hole_lines, _line_ref

AI_SOURCE = "ai"
FIRST_CUT_WIDTH_M = 3.8
FIRST_CUT_AREA_M2 = (18.0, 40_000.0)
ROUGH_AREA_M2 = (80.0, 280_000.0)
TREE_EXAMPLE_RADIUS_M = 14.0
TREE_COLOR_MAX_DIST = 38.0
PLAY_NEAR_TREE_M = 70.0


def publish_ai_layers(
    ctx: PipelineContext,
    *,
    result: AnalysisResult | None = None,
    rgb: np.ndarray | None = None,
    geo: MosaicGeo | None = None,
    pin_greens: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Write `ai_*` layers from catalog GPS, OSM priors, and mosaic analysis."""
    course = getattr(ctx, "course", None) or {}
    catalog_points = course.get("catalog_points") or course.get("golfapi_points") or []
    built = build_ai_layers(
        osm_layers=ctx.layers,
        catalog_points=catalog_points,
        result=result,
        rgb=rgb,
        geo=geo,
        pin_greens=pin_greens,
    )
    counts: dict[str, int] = {}
    for layer_id, feats in built.items():
        if not feats:
            continue
        ctx.layers[layer_id] = feature_collection(feats)
        counts[layer_id] = len(feats)
    ctx.quality["ai_layers"] = counts
    if counts:
        summary = ", ".join(f"{k.replace('ai_', '')}={v}" for k, v in counts.items())
        ctx.log(f"AI layers: {summary}")
    else:
        ctx.log("AI layers: none (need mosaics or OSM/API play surfaces)")
    return counts


def build_ai_layers(
    *,
    osm_layers: dict[str, dict],
    catalog_points: list[dict[str, Any]] | None = None,
    result: AnalysisResult | None = None,
    rgb: np.ndarray | None = None,
    geo: MosaicGeo | None = None,
    pin_greens: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    pins = _numbered_points(osm_layers.get("pin"), catalog_points, "pin", "green")
    hazards = _union_layers(osm_layers, "bunker", "water", "building", "clubhouse", "parking")
    if result is not None:
        extra = [g for lid in ("bunker", "water") for g, _p in result.polygons.get(lid, [])]
        hazards = _union_optional(hazards, *extra)
    boundary = union_layer(osm_layers.get("boundary"))

    greens = _ai_greens(result, pin_greens, pins, hazards)
    green_fc = feature_collection(greens) if greens else None
    fringe_src = green_fc or osm_layers.get("green")
    fringes = derive_green_fringes(
        fringe_src,
        subtract=_geom_list(hazards),
        boundary=boundary,
        width_m=FRINGE_WIDTH_M,
        source=AI_SOURCE,
    )
    for feat in fringes:
        props = feat.setdefault("properties", {})
        props["source"] = AI_SOURCE
        props["method"] = "collar"

    fairways = _ai_fairways(result, hazards, pins, osm_layers=osm_layers, geo=geo)
    fairway_fc = feature_collection(fairways) if fairways else osm_layers.get("fairway")
    first_cut = derive_first_cut(
        fairway_fc,
        subtract=_geom_list(hazards) + geoms_from_fc(green_fc or osm_layers.get("green")),
        boundary=boundary,
    )
    waters = _ai_waters(result, catalog_points)
    trees = extract_tree_crowns(
        rgb,
        geo,
        osm_layers,
        catalog_points=catalog_points,
        play_layers=(fairway_fc, green_fc or osm_layers.get("green"), osm_layers.get("tee")),
    )
    rough = extract_rough(
        result,
        geo,
        subtract=(
            geoms_from_fc(green_fc)
            + geoms_from_fc(fairway_fc)
            + geoms_from_fc(feature_collection(first_cut) if first_cut else None)
            + geoms_from_fc(osm_layers.get("tee"))
            + _geom_list(hazards)
        ),
        boundary=boundary,
    )

    return {
        "ai_green": greens,
        "ai_green_fringe": fringes,
        "ai_first_cut": first_cut,
        "ai_rough": rough,
        "ai_fairway": fairways,
        "ai_water": waters,
        "ai_tree": trees,
    }


def derive_first_cut(
    fairways_fc: dict | None,
    *,
    subtract: list[BaseGeometry] | None = None,
    boundary: BaseGeometry | None = None,
    width_m: float = FIRST_CUT_WIDTH_M,
) -> list[dict[str, Any]]:
    """Intermediate rough collar immediately outside each fairway."""
    mask = unary_union([g for g in (subtract or []) if g is not None and not g.is_empty]) if subtract else None
    fairway_geoms = [g for g in geoms_from_fc(fairways_fc) if not g.is_empty]
    out: list[dict[str, Any]] = []
    for i, feat in enumerate((fairways_fc or {}).get("features") or []):
        try:
            fairway = as_geom(feat)
        except Exception:
            continue
        if fairway.is_empty:
            continue
        ring = difference_safe(buffer_meters(fairway, width_m), fairway)
        others = [g for j, g in enumerate(fairway_geoms) if j != i]
        if others:
            ring = difference_safe(ring, unary_union(others))
        if mask is not None and not mask.is_empty:
            ring = difference_safe(ring, mask)
        if boundary is not None:
            ring = ring.intersection(boundary)
        props = dict(feat.get("properties") or {})
        hole = props.get("hole")
        for part_i, part in enumerate(as_polygons(ring), start=1):
            cleaned = clean_polygon(part)
            if cleaned is None or cleaned.is_empty:
                continue
            if area_m2(cleaned) < FIRST_CUT_AREA_M2[0] or area_m2(cleaned) > FIRST_CUT_AREA_M2[1]:
                continue
            cut_props = {
                "source": AI_SOURCE,
                "golf": "first_cut",
                "method": "fairway_collar",
                "instance_id": f"ai-first-cut-{props.get('instance_id') or i + 1}-{part_i}",
            }
            if hole is not None:
                cut_props["hole"] = hole
            out.append(to_feature(cleaned, cut_props))
    return out


def extract_rough(
    result: AnalysisResult | None,
    geo: MosaicGeo | None,
    *,
    subtract: list[BaseGeometry],
    boundary: BaseGeometry | None,
) -> list[dict[str, Any]]:
    if result is None or result.masks is None or geo is None:
        return []
    polys = mask_to_polygons(
        result.masks.soft_grass & ~result.masks.fairway & ~result.masks.green,
        geo,
        min_area_m2=ROUGH_AREA_M2[0],
        max_area_m2=ROUGH_AREA_M2[1],
        min_compactness=0.04,
    )
    mask = unary_union([g for g in subtract if g is not None and not g.is_empty]) if subtract else None
    out: list[dict[str, Any]] = []
    for i, poly in enumerate(polys, start=1):
        geom = poly
        if mask is not None and not mask.is_empty:
            geom = difference_safe(geom, mask)
        if boundary is not None:
            geom = geom.intersection(boundary)
        for part_i, part in enumerate(as_polygons(geom), start=1):
            cleaned = clean_polygon(part)
            if cleaned is None or cleaned.is_empty:
                continue
            if area_m2(cleaned) < ROUGH_AREA_M2[0]:
                continue
            out.append(
                to_feature(
                    cleaned,
                    {
                        "source": AI_SOURCE,
                        "golf": "rough",
                        "method": "soft_grass",
                        "instance_id": f"ai-rough-{i}-{part_i}",
                    },
                )
            )
    return out


def extract_tree_crowns(
    rgb: np.ndarray | None,
    geo: MosaicGeo | None,
    osm_layers: dict[str, dict],
    *,
    catalog_points: list[dict[str, Any]] | None = None,
    play_layers: tuple[dict | None, ...] = (),
) -> list[dict[str, Any]]:
    """Crown polygons. API/OSM tree spots supply the canopy colour examples."""
    if rgb is None or geo is None:
        return []
    examples = _tree_example_points(osm_layers, catalog_points)
    classified = classify_pixels(rgb)
    veg = vegetation_index(rgb)
    canopy = _canopy_mask(rgb, classified, veg)
    if examples:
        canopy = canopy & _color_like_examples(rgb, geo, examples)
        seeded = extract_seeded(
            canopy,
            examples,
            geo,
            radius_m=TREE_EXAMPLE_RADIUS_M,
            min_area_m2=TREE_CROWN_M2[0],
            max_area_m2=TREE_CROWN_M2[1],
        )
    else:
        seeded = []
    polys = mask_to_polygons(
        canopy,
        geo,
        min_area_m2=TREE_CROWN_M2[0],
        max_area_m2=TREE_CROWN_M2[1],
        min_compactness=TREE_MIN_COMPACT,
    )
    for poly in seeded:
        if any(iou(poly, existing) >= 0.25 for existing in polys):
            continue
        polys.append(poly)

    play_parts = []
    for fc in play_layers:
        play_parts.extend(geoms_from_fc(fc))
    for lid in ("fairway", "green", "tee", "hole_centerline"):
        play_parts.extend(geoms_from_fc(osm_layers.get(lid)))
    play = unary_union(play_parts) if play_parts else None
    greens = [g for g in geoms_from_fc(osm_layers.get("green")) if not g.is_empty]
    green_u = unary_union(greens) if greens else None
    woods = [g for g in geoms_from_fc(osm_layers.get("woodland")) if not g.is_empty]
    wood_u = unary_union(woods) if woods else None

    found: list[dict[str, Any]] = []
    for i, poly in enumerate(polys, start=1):
        cleaned = clean_polygon(poly)
        if cleaned is None or cleaned.is_empty:
            continue
        pt = cleaned.centroid
        if green_u is not None and (green_u.contains(pt) or distance_meters(pt, green_u) < 2.0):
            continue
        if play is not None and distance_meters(pt, play) > PLAY_NEAR_TREE_M:
            continue
        if wood_u is not None and wood_u.contains(pt):
            try:
                if distance_meters(pt, wood_u.boundary) > 12:
                    continue
            except Exception:
                continue
        if any(iou(cleaned, as_geom(f)) >= 0.35 for f in found):
            continue
        found.append(
            to_feature(
                cleaned,
                {
                    "source": AI_SOURCE,
                    "natural": "tree",
                    "method": "example_canopy" if examples else "canopy",
                    "instance_id": f"ai-tree-{i}",
                    "crown_m2": round(area_m2(cleaned), 1),
                    "from_api_example": bool(
                        examples and any(distance_meters(pt, ex) <= TREE_EXAMPLE_RADIUS_M for ex in examples)
                    ),
                },
            )
        )
    return found


def _ai_greens(
    result: AnalysisResult | None,
    pin_greens: list[dict[str, Any]] | None,
    pins: list[tuple[Point, int | None]],
    hazards: BaseGeometry | None,
) -> list[dict[str, Any]]:
    pairs: list[tuple[BaseGeometry, dict[str, Any]]] = []
    for item in pin_greens or []:
        geom = item.get("geometry")
        if geom is None or getattr(geom, "is_empty", True):
            continue
        pairs.append((geom, {"source": AI_SOURCE, "golf": "green", "method": "pin_crop"}))
    for geom, props in (result.polygons.get("green") if result else []) or []:
        if any(iou(geom, existing) >= 0.22 for existing, _p in pairs):
            continue
        merged = dict(props)
        merged["source"] = AI_SOURCE
        merged["method"] = merged.get("method") or "imagery"
        merged["golf"] = "green"
        pairs.append((geom, merged))
    out: list[dict[str, Any]] = []
    for i, (geom, props) in enumerate(pairs, start=1):
        if hazards is not None and not hazards.is_empty:
            geom = difference_safe(geom, hazards)
        cleaned = clean_polygon(geom)
        if cleaned is None or cleaned.is_empty:
            continue
        hole = _nearest_hole(cleaned, pins)
        feat_props = dict(props)
        feat_props["instance_id"] = f"ai-green-{i}"
        if hole is not None:
            feat_props["hole"] = hole
        out.append(to_feature(cleaned, feat_props))
    return out


def _ai_fairways(
    result: AnalysisResult | None,
    hazards: BaseGeometry | None,
    pins: list[tuple[Point, int | None]],
    *,
    osm_layers: dict[str, dict],
    geo: MosaicGeo | None,
) -> list[dict[str, Any]]:
    pairs = list((result.polygons.get("fairway") if result else []) or [])
    if not pairs and result is not None and result.masks is not None and geo is not None:
        pairs = _grow_ai_fairways(result, geo, osm_layers, hazards)
    out: list[dict[str, Any]] = []
    for i, (geom, props) in enumerate(pairs, start=1):
        if hazards is not None and not hazards.is_empty:
            geom = difference_safe(geom, hazards)
        cleaned = clean_polygon(geom)
        if cleaned is None or cleaned.is_empty:
            continue
        feat_props = dict(props)
        feat_props["source"] = AI_SOURCE
        feat_props["golf"] = "fairway"
        feat_props["method"] = feat_props.get("method") or "imagery"
        feat_props["instance_id"] = f"ai-fairway-{i}"
        if feat_props.get("hole") is None:
            hole = _nearest_hole(cleaned, pins)
            if hole is not None:
                feat_props["hole"] = hole
        out.append(to_feature(cleaned, feat_props))
    return out


def _grow_ai_fairways(
    result: AnalysisResult,
    geo: MosaicGeo,
    osm_layers: dict[str, dict],
    hazards: BaseGeometry | None,
) -> list[tuple[BaseGeometry, dict[str, Any]]]:
    """Grow fairways from mosaics even when OSM already mapped the hole."""
    lines = _hole_lines(osm_layers.get("hole_centerline"))
    if not lines or result.masks is None:
        return []
    boundary = union_layer(osm_layers.get("boundary"))
    out: list[tuple[BaseGeometry, dict[str, Any]]] = []
    for idx, line in enumerate(lines, start=1):
        grown = grow_fairway_along_line(
            result.masks.fairway,
            result.masks.soft_grass,
            line,
            geo,
            subtract=hazards,
            boundary=boundary,
        )
        if not grown:
            continue
        merged = as_polygons(unary_union(grown))
        merged = sorted(merged, key=area_m2, reverse=True)[:1]
        hole = _line_ref(osm_layers.get("hole_centerline"), idx - 1)
        for poly in merged:
            if area_m2(poly) < FAIRWAY_AREA_M2[0]:
                continue
            out.append((poly, {"source": AI_SOURCE, "golf": "fairway", "hole": hole, "method": "line_grow"}))
    return out


def _ai_waters(
    result: AnalysisResult | None,
    catalog_points: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    seeds = catalog_seed_points(catalog_points, "water")
    out: list[dict[str, Any]] = []
    for i, (geom, props) in enumerate((result.polygons.get("water") if result else []) or [], start=1):
        cleaned = clean_polygon(geom)
        if cleaned is None or cleaned.is_empty:
            continue
        feat_props = dict(props)
        feat_props["source"] = AI_SOURCE
        feat_props["golf"] = "water"
        feat_props["method"] = "imagery"
        feat_props["instance_id"] = f"ai-water-{i}"
        feat_props["catalog_seeded"] = bool(feat_props.get("catalog_seeded")) or any(
            distance_meters(cleaned, seed) <= 24.0 for seed in seeds
        )
        out.append(to_feature(cleaned, feat_props))
    return out


def _tree_example_points(
    osm_layers: dict[str, dict],
    catalog_points: list[dict[str, Any]] | None,
) -> list[Point]:
    examples = catalog_seed_points(catalog_points, "tree")
    for pt in _layer_points(osm_layers.get("tree")):
        if any(distance_meters(pt, old) < 4.0 for old in examples):
            continue
        examples.append(pt)
    return examples


def _canopy_mask(rgb: np.ndarray, classified: Any, veg: np.ndarray) -> np.ndarray:
    hsv = classified.hsv
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    g = rgb[:, :, 1].astype(np.int16)
    r = rgb[:, :, 0].astype(np.int16)
    return (
        (veg > 0.06)
        & (v < 120)
        & (h >= 28)
        & (h <= 100)
        & (s > 28)
        & (g + 4 >= r)
        & ~classified.water
        & ~classified.green
        & ~classified.bunker
    )


def _color_like_examples(rgb: np.ndarray, geo: MosaicGeo, examples: list[Point]) -> np.ndarray:
    samples = _sample_lab(rgb, geo, examples)
    if samples.size == 0:
        return np.ones(rgb.shape[:2], dtype=bool)
    mean = samples.mean(axis=0)
    spread = float(np.linalg.norm(samples - mean, axis=1).max()) if len(samples) > 1 else 18.0
    limit = max(16.0, min(TREE_COLOR_MAX_DIST, spread * 1.85 + 8.0))
    lab = _rgb_to_lab(rgb).astype(np.float32)
    dist = np.linalg.norm(lab - mean, axis=2)
    return dist <= limit


def _sample_lab(rgb: np.ndarray, geo: MosaicGeo, points: list[Point]) -> np.ndarray:
    lab = _rgb_to_lab(rgb)
    h, w = rgb.shape[:2]
    rows: list[np.ndarray] = []
    for pt in points:
        x, y = geo.lonlat_to_pixel(pt.x, pt.y)
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        patch = lab[max(0, iy - 1) : min(h, iy + 2), max(0, ix - 1) : min(w, ix + 2)]
        if patch.size:
            rows.append(patch.reshape(-1, 3))
    if not rows:
        return np.empty((0, 3), np.float32)
    return np.concatenate(rows, axis=0).astype(np.float32)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def _nearest_hole(geom: BaseGeometry, pins: list[tuple[Point, int | None]]) -> int | None:
    if not pins:
        return None
    best = None
    best_d = 45.0
    for pin, hole in pins:
        d = distance_meters(geom, pin)
        if d < best_d:
            best_d = d
            best = hole
    return best


def _numbered_points(
    fc: dict | None,
    catalog_points: list[dict[str, Any]] | None,
    *kinds: str,
) -> list[tuple[Point, int | None]]:
    out: list[tuple[Point, int | None]] = []
    for feat in (fc or {}).get("features") or []:
        try:
            geom = as_geom(feat)
        except Exception:
            continue
        pt = geom if geom.geom_type == "Point" else geom.centroid
        hole = _as_hole((feat.get("properties") or {}).get("hole"))
        out.append((pt, hole))
    for point in catalog_points or []:
        if kinds and point.get("kind") not in kinds:
            continue
        try:
            lat, lon = float(point["lat"]), float(point["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append((Point(lon, lat), _as_hole(point.get("hole"))))
    return out


def _as_hole(value: Any) -> int | None:
    try:
        hole = int(value)
    except (TypeError, ValueError):
        return None
    return hole if hole > 0 else None


def _layer_points(fc: dict | None) -> list[Point]:
    pts: list[Point] = []
    for feat in (fc or {}).get("features") or []:
        try:
            geom = as_geom(feat)
        except Exception:
            continue
        pts.append(geom if geom.geom_type == "Point" else geom.centroid)
    return pts


def _union_layers(layers: dict[str, dict], *ids: str) -> BaseGeometry | None:
    parts = []
    for lid in ids:
        parts.extend(geoms_from_fc(layers.get(lid)))
    return unary_union(parts) if parts else None


def _union_optional(*geoms: BaseGeometry | None) -> BaseGeometry | None:
    present = [g for g in geoms if g is not None and not g.is_empty]
    if not present:
        return None
    return unary_union(present)


def _geom_list(geom: BaseGeometry | None) -> list[BaseGeometry]:
    if geom is None or geom.is_empty:
        return []
    return [geom]
