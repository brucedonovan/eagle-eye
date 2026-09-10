"""Instance polygons (one feature per bunker, green, tee, …)."""

from __future__ import annotations

from shapely.ops import unary_union

from app.pipeline.context import PipelineContext
from app.services.geometry import as_geom, clean_polygon, feature_collection, to_feature


INSTANCE_LAYERS = (
    "bunker",
    "green",
    "tee",
    "building",
    "clubhouse",
    "bridge",
    "water",
    "lake",
    "parking",
    "ai_green",
    "ai_water",
    "ai_tree",
)


async def run(ctx: PipelineContext) -> None:
    exploded = 0
    for layer_id in INSTANCE_LAYERS:
        fc = ctx.layers.get(layer_id)
        if not fc:
            continue
        features = []
        for idx, feat in enumerate(fc.get("features", []), start=1):
            try:
                geom = as_geom(feat)
            except Exception:
                continue
            if geom.geom_type == "MultiPolygon":
                parts = list(geom.geoms)
            elif geom.geom_type == "GeometryCollection":
                parts = [g for g in geom.geoms if g.geom_type in {"Polygon", "MultiPolygon"}]
                if parts:
                    unioned = unary_union(parts)
                    parts = list(unioned.geoms) if unioned.geom_type == "MultiPolygon" else [unioned]
            else:
                parts = [geom]
            for part_i, part in enumerate(parts, start=1):
                cleaned = clean_polygon(part)
                if cleaned is None:
                    continue
                props = dict(feat.get("properties") or {})
                props["instance_id"] = f"{layer_id}-{idx}-{part_i}"
                features.append(to_feature(cleaned, props))
                exploded += 1
        ctx.layers[layer_id] = feature_collection(features)
    ctx.log(f"Instance extraction produced {exploded} polygons")
