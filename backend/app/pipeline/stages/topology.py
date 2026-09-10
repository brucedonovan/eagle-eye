"""GIS topology: greens vs bunkers, water vs fairways, buildings vs greens."""

from __future__ import annotations

from shapely.ops import unary_union

from app.pipeline.context import PipelineContext
from app.services.geometry import (
    as_geom,
    difference_safe,
    feature_collection,
    geoms_from_fc,
    to_feature,
    union_layer,
)
from app.services.imagery_segment import derive_green_fringes


async def run(ctx: PipelineContext) -> None:
    violations: list[str] = []

    bunker_u = union_layer(ctx.layers.get("bunker"))
    water_u = union_layer(ctx.layers.get("water"))
    building_u = union_layer(ctx.layers.get("building"))
    club_u = union_layer(ctx.layers.get("clubhouse"))
    built = _union_optional(building_u, club_u)

    if bunker_u and ctx.layers.get("green"):
        ctx.layers["green"] = _subtract(ctx.layers.get("green"), bunker_u, "green∩bunker", violations)
    if water_u and ctx.layers.get("fairway"):
        ctx.layers["fairway"] = _subtract(ctx.layers.get("fairway"), water_u, "water∩fairway", violations)
    if built and ctx.layers.get("green"):
        ctx.layers["green"] = _subtract(ctx.layers.get("green"), built, "building∩green", violations)

    greens = ctx.layers.get("green", {}).get("features", [])
    holes = ctx.layers.get("hole_centerline", {}).get("features", [])
    assigned = [f.get("properties", {}).get("hole") for f in greens]
    assigned_holes = {h for h in assigned if h is not None}
    if holes:
        hole_nums = {f.get("properties", {}).get("hole") for f in holes}
        missing = [n for n in hole_nums if n not in assigned_holes]
        if missing:
            violations.append(f"holes_without_green:{missing}")
        multi = [n for n in assigned_holes if assigned.count(n) > 1]
        if multi:
            violations.append(f"holes_with_multiple_greens:{multi}")
        orphan_greens = sum(1 for n in assigned if n is None)
        if orphan_greens:
            violations.append(f"greens_without_hole:{orphan_greens}")

    ctx.quality["topology_violations"] = violations
    ctx.quality["topology_ok"] = len(violations) == 0
    ctx.log(f"Topology {'clean' if not violations else 'violations: ' + ', '.join(violations)}")

    fringes = _write_green_fringes(ctx)
    if fringes:
        ctx.log(f"Derived {fringes} green fringe collars")


def _write_green_fringes(ctx: PipelineContext) -> int:
    subtract = []
    for layer_id in ("bunker", "water", "building", "clubhouse", "putting_green"):
        subtract.extend(geoms_from_fc(ctx.layers.get(layer_id)))
    boundary = union_layer(ctx.layers.get("boundary"))
    feats = derive_green_fringes(
        ctx.layers.get("green"),
        subtract=subtract,
        boundary=boundary,
    )
    if feats:
        ctx.layers["green_fringe"] = feature_collection(feats)
    ctx.quality["green_fringes"] = len(feats)
    return len(feats)


def _subtract(fc: dict | None, mask, rule: str, violations: list[str]) -> dict | None:
    if not fc:
        return fc
    features = []
    overlaps = 0
    for feat in fc.get("features", []):
        geom = as_geom(feat)
        if geom.intersects(mask):
            overlaps += 1
            geom = difference_safe(geom, mask)
        if geom.is_empty:
            continue
        features.append(to_feature(geom, feat.get("properties") or {}))
    if overlaps:
        violations.append(f"{rule}:{overlaps}")
    return feature_collection(features)


def _union_optional(*geoms):
    present = [g for g in geoms if g is not None and not g.is_empty]
    if not present:
        return None
    return unary_union(present)
