"""Persist layers and write export packages."""

from __future__ import annotations

import json

from app.pipeline.context import PipelineContext
from app.services import exporters


async def run(ctx: PipelineContext) -> None:
    layer_paths = exporters.write_layer_geojsons(ctx.layers, ctx.layer_dir())
    exports = ctx.export_dir()

    ctx.exports["geojson"] = str(exporters.write_geojson(ctx.layers, exports / "course.geojson"))
    ctx.exports["wkt"] = str(exporters.write_wkt(ctx.layers, exports / "course.wkt"))
    ctx.exports["kml"] = str(exporters.write_kml(ctx.layers, exports / "course.kml"))
    ctx.exports["gpx"] = str(exporters.write_gpx(ctx.layers, exports / "course.gpx"))
    ctx.exports["dxf"] = str(exporters.write_dxf(ctx.layers, exports / "course.dxf"))
    try:
        ctx.exports["ros_grid"] = str(exporters.write_ros_grid(ctx.layers, exports / "ros"))
    except Exception as exc:
        ctx.log(f"ROS grid skipped: {exc}")

    try:
        ctx.exports["gpkg"] = str(exporters.write_geopackage(ctx.layers, exports / "course.gpkg"))
    except Exception as exc:
        ctx.log(f"GeoPackage skipped: {exc}")
    try:
        ctx.exports["shp"] = str(exporters.write_shapefile_zip(ctx.layers, exports / "shapefiles.zip"))
    except Exception as exc:
        ctx.log(f"Shapefile skipped: {exc}")

    manifest = {
        "course": ctx.course,
        "bbox": ctx.bbox,
        "layers": {lid: len(fc.get("features", [])) for lid, fc in ctx.layers.items()},
        "layer_paths": layer_paths,
        "exports": ctx.exports,
        "quality": ctx.quality,
        "imagery": {k: v for k, v in ctx.imagery.items() if k != "catalog"},
        "logs": ctx.logs,
    }
    (ctx.work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    ctx.log(f"Wrote exports: {', '.join(ctx.exports)}")
