from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.services import exporters

from app.api.deps import job_work_dir, load_job

router = APIRouter()

CONTENT = {
    "geojson": ("course.geojson", "application/geo+json"),
    "kml": ("course.kml", "application/vnd.google-earth.kml+xml"),
    "gpx": ("course.gpx", "application/gpx+xml"),
    "wkt": ("course.wkt", "text/plain"),
    "dxf": ("course.dxf", "application/dxf"),
    "gpkg": ("course.gpkg", "application/geopackage+sqlite3"),
    "shp": ("shapefiles.zip", "application/zip"),
    "ros_grid": ("ros/ros_occupancy_grid.zip", "application/zip"),
}


@router.get("/download/{job_id}")
async def download(
    job_id: str,
    format: str = "geojson",
    session: AsyncSession = Depends(get_session),
):
    job = await load_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "completed":
        raise HTTPException(409, f"Job is {job.status}")
    if format not in CONTENT:
        raise HTTPException(400, f"Unsupported format: {format}")

    rel, media = CONTENT[format]
    path = job_work_dir(job_id) / "exports" / rel
    if not path.exists() and format in {"gpkg", "shp"}:
        await _rebuild(job_id, format, path)
    if not path.exists():
        raise HTTPException(404, f"{format} export is not available for this job")
    return FileResponse(path, media_type=media, filename=path.name)


async def _rebuild(job_id: str, fmt: str, dest: Path) -> None:
    layer_dir = job_work_dir(job_id) / "layers"
    if not layer_dir.exists():
        return
    layers = {}
    for path in layer_dir.glob("*.geojson"):
        layers[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if fmt == "gpkg":
            exporters.write_geopackage(layers, dest)
        elif fmt == "shp":
            exporters.write_shapefile_zip(layers, dest)
    except Exception:
        return
