"""Robot navigation layers (Nav2-oriented)."""

from __future__ import annotations

from shapely.geometry import LineString
from shapely.ops import unary_union

from app.pipeline.context import PipelineContext
from app.services.geometry import (
    as_geom,
    feature_collection,
    geoms_from_fc,
    to_feature,
    union_layer,
)


NO_GO_LAYERS = ("water", "lake", "bunker", "woodland", "oob", "building", "clubhouse", "parking")
COVERAGE_LAYERS = ("fairway", "green", "tee", "managed_rough", "driving_range", "putting_green")
OBSTACLE_LAYERS = ("building", "clubhouse", "tree", "fence", "bridge")


async def run(ctx: PipelineContext) -> None:
    if not ctx.request.get("include_navigation", True):
        ctx.log("Navigation generation disabled")
        return

    boundary = union_layer(ctx.layers.get("boundary"))
    no_go = _collect(ctx, NO_GO_LAYERS)
    obstacles = _collect(ctx, OBSTACLE_LAYERS)
    coverage = _collect(ctx, COVERAGE_LAYERS)

    if boundary is not None:
        if no_go:
            nav = boundary.difference(no_go)
        else:
            nav = boundary
        if not nav.is_empty:
            ctx.layers["nav_mesh"] = feature_collection(
                [to_feature(g, {"class": "traversable"}) for g in _polys(nav)]
            )

    if no_go:
        ctx.layers["no_go"] = feature_collection(
            [to_feature(g, {"class": "no_go"}) for g in _polys(no_go)]
        )
    if obstacles:
        ctx.layers["obstacle"] = feature_collection(
            [to_feature(g, {"class": "obstacle"}) for g in _polys(obstacles)]
        )
    if coverage:
        ctx.layers["coverage"] = feature_collection(
            [to_feature(g, {"class": "coverage"}) for g in _polys(coverage)]
        )
        ctx.layers["mowing_sector"] = _mowing_sectors(ctx)

    ctx.layers["waypoint_graph"] = _waypoint_graph(ctx)
    ctx.log("Generated navigation mesh, no-go, coverage, and waypoint graph")


def _collect(ctx: PipelineContext, layer_ids: tuple[str, ...]):
    geoms = []
    for lid in layer_ids:
        geoms.extend(geoms_from_fc(ctx.layers.get(lid)))
    geoms = [g for g in geoms if not g.is_empty]
    return unary_union(geoms) if geoms else None


def _polys(geom):
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        return [g for g in geom.geoms if g.geom_type in {"Polygon", "MultiPolygon"}]
    return []


def _mowing_sectors(ctx: PipelineContext) -> dict:
    features = []
    for layer_id in ("fairway", "green", "tee", "managed_rough"):
        for feat in (ctx.layers.get(layer_id) or {}).get("features", []):
            props = dict(feat.get("properties") or {})
            props["sector"] = layer_id
            props["pattern"] = "boustrophedon"
            features.append({"type": "Feature", "geometry": feat["geometry"], "properties": props})
    return feature_collection(features)


def _waypoint_graph(ctx: PipelineContext) -> dict:
    lines = []
    for feat in (ctx.layers.get("hole_centerline") or {}).get("features", []):
        try:
            geom = as_geom(feat)
        except Exception:
            continue
        if geom.geom_type == "LineString":
            props = dict(feat.get("properties") or {})
            props["edge"] = "hole"
            lines.append(to_feature(geom, props))
    for feat in (ctx.layers.get("cart_path") or {}).get("features", []):
        try:
            geom = as_geom(feat)
        except Exception:
            continue
        if geom.geom_type == "LineString":
            props = dict(feat.get("properties") or {})
            props["edge"] = "cart_path"
            lines.append(to_feature(geom, props))
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                lines.append(to_feature(part, {"edge": "cart_path"}))

    # Connect neighboring hole endpoints so a robot can sequence holes
    centers = []
    for feat in (ctx.layers.get("hole_centerline") or {}).get("features", []):
        try:
            geom = as_geom(feat)
            if geom.geom_type == "LineString" and len(geom.coords) >= 2:
                centers.append((feat.get("properties", {}).get("hole"), geom.coords[0], geom.coords[-1]))
        except Exception:
            continue
    centers.sort(key=lambda item: item[0] or 0)
    for i in range(len(centers) - 1):
        _n, _start, end = centers[i]
        _n2, start2, _end2 = centers[i + 1]
        lines.append(to_feature(LineString([end, start2]), {"edge": "inter_hole", "from": _n, "to": _n2}))
    return feature_collection(lines)
