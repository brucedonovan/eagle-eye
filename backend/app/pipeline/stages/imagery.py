"""Multi-source imagery download."""

from __future__ import annotations

from shapely.geometry import Point

from app.pipeline.context import PipelineContext
from app.services.geometry import geoms_from_fc
from app.services.imagery import (
    available_sources,
    best_source,
    download_pin_crops,
    download_second_mosaic,
    download_temporal_mosaic,
    download_xyz_mosaic,
)


async def run(ctx: PipelineContext) -> None:
    if not ctx.request.get("include_imagery", True):
        ctx.imagery = {"status": "skipped", "reason": "disabled"}
        ctx.log("Imagery download disabled")
        return
    if not ctx.bbox:
        ctx.imagery = {"status": "skipped", "reason": "no_bbox"}
        return

    west, south, east, north = ctx.bbox
    dest = ctx.work_dir / "imagery" / "mosaic.jpg"
    chosen = best_source()
    catalog = [
        {"id": s.id, "title": s.title, "gsd_m": s.gsd_m, "available": s.available}
        for s in available_sources()
    ]
    result = await download_xyz_mosaic(
        west=west, south=south, east=east, north=north, dest=dest
    )
    result["selected"] = {"id": chosen.id, "title": chosen.title, "gsd_m": chosen.gsd_m}
    result["catalog"] = catalog
    second_dest = ctx.work_dir / "imagery" / "mosaic_second.jpg"
    second = await download_second_mosaic(
        west=west, south=south, east=east, north=north, dest=second_dest, primary=result
    )
    result["second"] = second
    temporal_dest = ctx.work_dir / "imagery" / "mosaic_temporal.jpg"
    temporal = await download_temporal_mosaic(
        west=west, south=south, east=east, north=north, dest=temporal_dest, primary=result
    )
    result["temporal"] = temporal
    pins = _layer_points(ctx.layers.get("pin"))
    osm_greens = geoms_from_fc(ctx.layers.get("green"))
    pin_crops = await download_pin_crops(
        pins=pins,
        osm_greens=osm_greens,
        dest_dir=ctx.work_dir / "imagery" / "pin_crops",
    )
    result["pin_crops"] = pin_crops
    ctx.imagery = result
    ctx.log(f"Imagery {result.get('status')} via {result.get('source') or chosen.id}")
    ctx.log(
        f"Second imagery {second.get('status')} via {second.get('source') or second.get('reason')}"
    )
    ctx.log(
        f"Temporal imagery {temporal.get('status')} via {temporal.get('source') or temporal.get('reason')}"
    )
    ctx.log(
        f"Pin crops {pin_crops.get('status')} via {pin_crops.get('source') or pin_crops.get('reason')} "
        f"ok={pin_crops.get('crops_ok', 0)}/{pin_crops.get('pins_needed', 0)} (refine + fill)"
    )


def _layer_points(fc: dict | None) -> list[Point]:
    pts: list[Point] = []
    for geom in geoms_from_fc(fc):
        if geom.geom_type == "Point":
            pts.append(Point(geom.x, geom.y))
        else:
            pts.append(geom.centroid)
    return pts
