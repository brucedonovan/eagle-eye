"""Stage 2 — AOI generation and OSM vector prior download."""

from __future__ import annotations

from app.pipeline.context import PipelineContext
from app.services import overpass
from app.services.geometry import as_geom, expand_bbox, union_layer


async def run(ctx: PipelineContext) -> None:
    if not ctx.bbox:
        raise RuntimeError("Discovery did not produce a bounding box")

    west, south, east, north = expand_bbox(*ctx.bbox, pad_deg=0.0012)
    ctx.bbox = (west, south, east, north)
    payload = await overpass.fetch_aoi(west, south, east, north)
    layers = overpass.elements_to_layers(payload)

    selected = union_layer(ctx.layers.get("boundary")) if ctx.course.get("_boundary_ready") else None
    if selected is None:
        selected = _pick_named_boundary(layers, ctx.course.get("name") or "")
    if selected is not None:
        layers = overpass.clip_layers_to_boundary(layers, selected)
        ctx.layers["boundary"] = layers.get("boundary") or ctx.layers.get("boundary")
        for key, fc in layers.items():
            if key != "boundary":
                ctx.layers[key] = fc
        ctx.bbox = expand_bbox(*selected.bounds, pad_deg=0.0008)
    else:
        ctx.layers.update(layers)
        tight = overpass.feature_bounds(layers)
        if tight:
            ctx.bbox = expand_bbox(*tight, pad_deg=0.0008)

    if ctx.course:
        ctx.course["bbox"] = list(ctx.bbox)

    n_features = sum(len((fc or {}).get("features", [])) for fc in ctx.layers.values())
    ctx.log(
        f"AOI {ctx.bbox}; ingested {n_features} OSM features across {len(ctx.layers)} layers"
        + (f" clipped to {ctx.course.get('boundary_name')}" if selected is not None else "")
    )


def _pick_named_boundary(layers: dict, name: str):
    fc = layers.get("boundary")
    if not fc:
        return None
    best = None
    best_score = -1.0
    for feat in fc.get("features", []):
        geom = as_geom(feat)
        score = overpass.name_score(name, str((feat.get("properties") or {}).get("name") or ""))
        score += min(geom.area * 1000, 1.0)
        if score > best_score:
            best_score = score
            best = geom
    return best
