from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.catalog import LAYER_BY_ID
from app.config import settings
from app.models.entities import Job
from app.schemas.api import CourseOut, LayerOut, StatusOut


async def load_job(session: AsyncSession, job_id: str) -> Job | None:
    result = await session.execute(
        select(Job).options(selectinload(Job.course), selectinload(Job.layers)).where(Job.id == job_id)
    )
    return result.scalar_one_or_none()


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
