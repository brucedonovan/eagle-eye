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


@api_router.get("/catalog/cached", response_model=CourseSearchOut)
async def list_cached_courses(session: AsyncSession = Depends(get_session)) -> CourseSearchOut:
    """Fully cached golfapi.io course payloads (courses/ on disk), not search hits."""
    try:
        provider = get_provider()
    except CatalogError as exc:
        raise HTTPException(400, str(exc)) from exc
    if provider is None:
        raise HTTPException(503, "Course catalog is not configured (missing provider credentials)")
    rows = (
        await session.scalars(
            select(Job)
            .options(selectinload(Job.course), selectinload(Job.layers))
            .where(Job.status == "completed")
            .order_by(Job.updated_at.desc())
            .limit(80)
        )
    ).all()
    return CourseSearchOut(
        query="",
        source=provider.id,
        cached=True,
        catalog_configured=True,
        catalog_provider=provider.id,
        provider_title=provider.title,
        api_requests_left=provider.api_requests_left(),
        courses=_chips_from_catalog(provider.list_cached(), rows),
    )


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


def _chips_from_catalog(hits: list[CourseHit], jobs: list[Job], *, limit: int = 24) -> list[CourseSearchItem]:
    job_ids = _latest_job_ids_by_catalog(jobs)
    items: list[CourseSearchItem] = []
    seen: set[str] = set()
    for hit in hits:
        if not hit.course_id or hit.course_id in seen:
            continue
        seen.add(hit.course_id)
        item = _from_hit(hit)
        item.job_id = job_ids.get(hit.course_id)
        items.append(item)
        if len(items) >= limit:
            break
    return items


def _latest_job_ids_by_catalog(jobs: list[Job]) -> dict[str, str]:
    out: dict[str, str] = {}
    for job in jobs:
        item = _from_completed_job(job)
        if item is None or not item.course_id or item.course_id in out:
            continue
        out[item.course_id] = job.id
    return out


def _items_from_completed_jobs(jobs: list[Job], *, limit: int = 24) -> list[CourseSearchItem]:
    items: list[CourseSearchItem] = []
    seen: set[str] = set()
    for job in jobs:
        item = _from_completed_job(job)
        if item is None:
            continue
        key = _vectorized_key(job, item)
        name_key = f"name:{item.display_name.strip().lower()}"
        if key in seen or name_key in seen:
            continue
        seen.add(key)
        seen.add(name_key)
        items.append(item)
        if len(items) >= limit:
            break
    return items


def _vectorized_key(job: Job, item: CourseSearchItem) -> str:
    if item.course_id:
        return f"catalog:{item.course_id}"
    osm_id = job.course.osm_id if job.course else None
    if osm_id:
        return f"osm:{osm_id}"
    return f"name:{item.display_name}"


def _from_completed_job(job: Job) -> CourseSearchItem | None:
    if job.status != "completed":
        return None
    if not job.layers:
        return None
    req = json.loads(job.request_json or "{}") if job.request_json else {}
    meta: dict = {}
    if job.course and job.course.metadata_json:
        try:
            parsed = json.loads(job.course.metadata_json)
            if isinstance(parsed, dict):
                meta = parsed
        except json.JSONDecodeError:
            meta = {}
    result: dict = {}
    if job.result_json:
        try:
            parsed = json.loads(job.result_json)
            if isinstance(parsed, dict):
                result = parsed
        except json.JSONDecodeError:
            result = {}
    ctx_course = result.get("course") if isinstance(result.get("course"), dict) else {}
    display = (
        (job.course.display_name if job.course else None)
        or ctx_course.get("display_name")
        or (job.course.name if job.course else None)
        or ctx_course.get("name")
        or req.get("name")
        or "Vectorized course"
    )
    course_id = str(
        meta.get("catalog_course_id") or req.get("catalog_course_id") or req.get("golfapi_course_id") or ""
    ).strip() or None
    club_id = str(
        meta.get("catalog_club_id") or req.get("catalog_club_id") or req.get("golfapi_club_id") or ""
    ).strip() or None
    club_name = str(meta.get("catalog_club_name") or ctx_course.get("catalog_club_name") or display)
    course_name = str(meta.get("catalog_course_name") or ctx_course.get("catalog_course_name") or display)
    source = str(meta.get("catalog_provider") or ctx_course.get("catalog_provider") or meta.get("source") or "vectorized")
    return CourseSearchItem(
        source=source,
        club_id=club_id,
        club_name=club_name,
        course_id=course_id,
        course_name=course_name,
        display_name=str(display),
        city=meta.get("city") or ctx_course.get("city"),
        state=meta.get("state") or ctx_course.get("state"),
        country=(job.course.country if job.course else None) or meta.get("country") or ctx_course.get("country"),
        address=(job.course.address if job.course else None) or meta.get("address") or ctx_course.get("address"),
        lat=(job.course.lat if job.course else None) or ctx_course.get("lat"),
        lon=(job.course.lon if job.course else None) or ctx_course.get("lon"),
        num_holes=meta.get("catalog_num_holes") or ctx_course.get("catalog_num_holes"),
        has_gps=bool(meta.get("catalog_has_gps") or ctx_course.get("catalog_has_gps")),
        job_id=job.id,
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
