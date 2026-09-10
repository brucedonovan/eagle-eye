"""AOI generation and OSM vector prior download."""

from __future__ import annotations

from app.pipeline.context import PipelineContext
from app.services import overpass
from app.services.course_catalog import layers as catalog_layers
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
    gps_clip = catalog_layers.geom_from_geojson(_catalog_field(ctx.course, "clip"))
    preferred = catalog_layers.prefer_clip(selected, gps_clip)
    catalog_source = ctx.course.get("catalog_provider") or ctx.course.get("source") or "catalog"
    clip_source = catalog_source if preferred is not None and preferred is gps_clip else "openstreetmap"
    if preferred is not None:
        layers = overpass.clip_layers_to_boundary(layers, preferred)
        if clip_source != "openstreetmap":
            layers["boundary"] = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": _catalog_field(ctx.course, "clip"),
                        "properties": {
                            "name": ctx.course.get("display_name") or ctx.course.get("name"),
                            "source": catalog_source,
                            "role": "selected_course",
                        },
                    }
                ],
            }
        ctx.layers["boundary"] = layers.get("boundary") or ctx.layers.get("boundary")
        for key, fc in layers.items():
            if key != "boundary":
                ctx.layers[key] = fc
        ctx.bbox = expand_bbox(*preferred.bounds, pad_deg=0.0008)
        selected = preferred
    else:
        selected = None
        ctx.layers.update(layers)
        tight = overpass.feature_bounds(layers)
        if tight:
            ctx.bbox = expand_bbox(*tight, pad_deg=0.0008)

    points = _catalog_field(ctx.course, "points") or []
    if points:
        stats = catalog_layers.fuse_layers(
            ctx.layers,
            points,
            scorecard=_catalog_field(ctx.course, "scorecard"),
            source=catalog_source,
        )
        ctx.quality["catalog_fuse"] = stats
        ctx.log(
            f"{catalog_source} overlay: {stats['pins_catalog']} pins, "
            f"{stats['tees_added']} tees added, {stats['centerlines_added']} hole lines, "
            f"{stats['bunker_seeds']} bunker / {stats['water_seeds']} water GPS seeds"
        )

    if ctx.course:
        ctx.course["bbox"] = list(ctx.bbox)

    n_features = sum(len((fc or {}).get("features", [])) for fc in ctx.layers.values())
    clip_note = ""
    if selected is not None:
        label = ctx.course.get("boundary_name") if clip_source == "openstreetmap" else f"{catalog_source} GPS hull"
        clip_note = f" clipped to {label}"
    ctx.log(f"AOI {ctx.bbox}; ingested {n_features} OSM features across {len(ctx.layers)} layers{clip_note}")


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


def _catalog_field(course: dict, field: str):
    return course.get(f"catalog_{field}") or course.get(f"golfapi_{field}")
