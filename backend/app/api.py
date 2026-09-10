from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.catalog import LAYER_BY_ID, LAYERS
from app.config import settings
from app.db import get_session
from app.models import Job
from app.pipeline.orchestrator import run_pipeline
from app.schemas import (
    CourseCreate,
    CourseOut,
    CourseSearchItem,
    CourseSearchOut,
    JobOut,
    LayerOut,
    StatusOut,
)
from app.services import exporters
from app.services.course_catalog import (
    CatalogError,
    CourseCatalogProvider,
    CourseHit,
    SearchResult,
    get_provider,
)

api_router = APIRouter()

DOWNLOAD_TYPES = {
    "geojson": ("course.geojson", "application/geo+json"),
    "kml": ("course.kml", "application/vnd.google-earth.kml+xml"),
    "gpx": ("course.gpx", "application/gpx+xml"),
    "wkt": ("course.wkt", "text/plain"),
    "dxf": ("course.dxf", "application/dxf"),
    "gpkg": ("course.gpkg", "application/geopackage+sqlite3"),
    "shp": ("shapefiles.zip", "application/zip"),
    "ros_grid": ("ros/ros_occupancy_grid.zip", "application/zip"),
}


@api_router.get("/search", response_model=CourseSearchOut)
async def search_courses(
    q: str = Query(default="", description="Club / course name"),
    lat: float | None = None,
    lon: float | None = None,
) -> CourseSearchOut:
    query = q.strip()
    if not query and (lat is None or lon is None):
        raise HTTPException(400, "Provide a search query or lat/lon")

    try:
        provider = get_provider()
    except CatalogError as exc:
        raise HTTPException(400, str(exc)) from exc
    if provider is None:
        raise HTTPException(503, "Course catalog is not configured (missing provider credentials)")

    try:
        result = await provider.search(query, lat=lat, lon=lon)
    except CatalogError as exc:
        raise HTTPException(502, str(exc)) from exc
    return _catalog_search(query, provider, result, [_from_hit(hit) for hit in result.hits[:40]])


@api_router.post("/course", response_model=JobOut, status_code=202)
@api_router.post("/segment", response_model=JobOut, status_code=202)
@api_router.post("/vectorize", response_model=JobOut, status_code=202)
@api_router.post("/navigation", response_model=JobOut, status_code=202)
async def create_job(
    payload: CourseCreate,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> JobOut:
    job = Job(request_json=payload.model_dump_json(), status="queued", stage="queued")
    session.add(job)
    await session.commit()
    await session.refresh(job)
    _enqueue(background, job.id)
    return JobOut(job_id=job.id, status=job.status, stage=job.stage, progress=job.progress, message="Queued")


@api_router.get("/status/{job_id}", response_model=StatusOut)
async def get_status(job_id: str, session: AsyncSession = Depends(get_session)) -> StatusOut:
    job = await load_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job_to_status(job)


@api_router.get("/jobs", response_model=list[JobOut])
async def list_jobs(session: AsyncSession = Depends(get_session)) -> list[JobOut]:
    rows = (await session.scalars(select(Job).order_by(Job.created_at.desc()).limit(25))).all()
    return [
        JobOut(job_id=j.id, status=j.status, stage=j.stage, progress=j.progress, message=j.message)
        for j in rows
    ]


@api_router.get("/layers/catalog")
async def layer_catalog() -> list[dict]:
    return [
        {
            "id": layer.id,
            "title": layer.title,
            "geometry": layer.geometry,
            "group": layer.group,
            "color": layer.color,
            "mvp": layer.mvp,
        }
        for layer in LAYERS
    ]


@api_router.get("/layers/{job_id}")
async def get_layers(job_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    job = await _completed_job(session, job_id)
    layers = {}
    for artifact in job.layers:
        if not artifact.geojson_path:
            continue
        path = Path(artifact.geojson_path)
        if path.exists():
            layers[artifact.layer_id] = json.loads(path.read_text(encoding="utf-8"))
    return {"job_id": job_id, "layers": layers}


@api_router.get("/layers/{job_id}/{layer_id}")
async def get_layer(job_id: str, layer_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    job = await load_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    artifact = next((a for a in job.layers if a.layer_id == layer_id), None)
    if artifact is None or not artifact.geojson_path:
        raise HTTPException(404, "Layer not found")
    path = Path(artifact.geojson_path)
    if not path.exists():
        raise HTTPException(404, "Layer file missing")
    return json.loads(path.read_text(encoding="utf-8"))


@api_router.get("/download/{job_id}")
async def download(
    job_id: str,
    format: str = "geojson",
    session: AsyncSession = Depends(get_session),
):
    await _completed_job(session, job_id)
    if format not in DOWNLOAD_TYPES:
        raise HTTPException(400, f"Unsupported format: {format}")
    rel, media = DOWNLOAD_TYPES[format]
    path = job_work_dir(job_id) / "exports" / rel
    if not path.exists() and format in {"gpkg", "shp"}:
        await _rebuild_gis_export(job_id, format, path)
    if not path.exists():
        raise HTTPException(404, f"{format} export is not available for this job")
    return FileResponse(path, media_type=media, filename=path.name)


async def load_job(session: AsyncSession, job_id: str) -> Job | None:
    result = await session.execute(
        select(Job).options(selectinload(Job.course), selectinload(Job.layers)).where(Job.id == job_id)
    )
    return result.scalar_one_or_none()


async def _completed_job(session: AsyncSession, job_id: str) -> Job:
    job = await load_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "completed":
        raise HTTPException(409, f"Job is {job.status}")
    return job


def job_to_status(job: Job) -> StatusOut:
    course = None
    if job.course:
        bbox = None
        if None not in (job.course.bbox_west, job.course.bbox_south, job.course.bbox_east, job.course.bbox_north):
            bbox = [job.course.bbox_west, job.course.bbox_south, job.course.bbox_east, job.course.bbox_north]
        course = CourseOut(
            id=job.course.id,
            name=job.course.name,
            display_name=job.course.display_name,
            country=job.course.country,
            address=job.course.address,
            lat=job.course.lat,
            lon=job.course.lon,
            bbox=bbox,
            osm_id=job.course.osm_id,
        )
    layers = []
    for artifact in job.layers:
        spec = LAYER_BY_ID.get(artifact.layer_id)
        layers.append(
            LayerOut(
                layer_id=artifact.layer_id,
                title=spec.title if spec else artifact.layer_id,
                geometry=spec.geometry if spec else "polygon",
                group=spec.group if spec else "other",
                color=spec.color if spec else "#cccccc",
                feature_count=artifact.feature_count,
                source=artifact.source,
            )
        )
    quality = json.loads(job.quality_json) if job.quality_json else {}
    exports: list[str] = []
    if job.result_json:
        result = json.loads(job.result_json)
        exports = list((result.get("exports") or {}).keys())
    return StatusOut(
        job_id=job.id,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        message=job.message,
        error=job.error,
        course=course,
        layers=layers,
        quality=quality,
        exports=exports,
    )


def job_work_dir(job_id: str) -> Path:
    return settings.jobs_dir / job_id


def _enqueue(background: BackgroundTasks, job_id: str) -> None:
    if settings.job_backend == "celery":
        from app.workers.celery_app import run_job

        run_job.delay(job_id)
        return
    background.add_task(_run_local, job_id)


async def _run_local(job_id: str) -> None:
    await run_pipeline(job_id)


async def _rebuild_gis_export(job_id: str, fmt: str, dest: Path) -> None:
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


def _catalog_search(
    query: str,
    provider: CourseCatalogProvider,
    result: SearchResult,
    courses: list[CourseSearchItem],
) -> CourseSearchOut:
    return CourseSearchOut(
        query=query,
        source=provider.id,
        cached=result.cached,
        catalog_configured=True,
        catalog_provider=provider.id,
        provider_title=provider.title,
        api_requests_left=result.api_requests_left,
        courses=courses,
    )


def _from_hit(hit: CourseHit) -> CourseSearchItem:
    return CourseSearchItem(
        source=hit.provider,
        club_id=hit.club_id or None,
        club_name=hit.club_name,
        course_id=hit.course_id or None,
        course_name=hit.course_name,
        display_name=hit.display_name,
        city=hit.city,
        state=hit.state,
        country=hit.country,
        address=hit.address,
        lat=hit.lat,
        lon=hit.lon,
        num_holes=hit.num_holes,
        has_gps=hit.has_gps,
        distance_km=hit.distance,
        timestamp_updated=hit.timestamp_updated,
    )
