from __future__ import annotations

import json
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models.entities import Course, Job, LayerArtifact
from app.pipeline.context import PIPELINE_STAGES, PipelineContext
from app.pipeline.stages import (
    aoi,
    discovery,
    export,
    holes,
    imagery,
    instances,
    navigation,
    preprocess,
    segmentation,
    topology,
    vectorize,
)

StageFn = Callable[[PipelineContext], Awaitable[None]]

STAGES: list[tuple[str, StageFn]] = [
    ("discovery", discovery.run),
    ("aoi", aoi.run),
    ("imagery", imagery.run),
    ("preprocess", preprocess.run),
    ("segmentation", segmentation.run),
    ("instances", instances.run),
    ("holes", holes.run),
    ("vectorize", vectorize.run),
    ("topology", topology.run),
    ("navigation", navigation.run),
    ("export", export.run),
]


async def run_pipeline(job_id: str) -> None:
    async with SessionLocal() as session:
        job = await session.get(Job, job_id)
        if job is None:
            return
        request = json.loads(job.request_json)
        work_dir = settings.jobs_dir / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        ctx = PipelineContext(job_id=job_id, work_dir=work_dir, request=request)
        job.status = "running"
        job.stage = "discovery"
        job.message = "Starting pipeline"
        await session.commit()

    try:
        for index, (name, fn) in enumerate(STAGES):
            await _mark(job_id, stage=name, progress=index / len(STAGES), message=f"Running {name}")
            await fn(ctx)
        await _persist_success(job_id, ctx)
    except Exception as exc:
        await _persist_failure(job_id, ctx if "ctx" in locals() else None, exc)


async def _mark(job_id: str, *, stage: str, progress: float, message: str) -> None:
    async with SessionLocal() as session:
        job = await session.get(Job, job_id)
        if job is None:
            return
        job.stage = stage
        job.progress = progress
        job.message = message
        job.status = "running"
        await session.commit()


async def _persist_success(job_id: str, ctx: PipelineContext) -> None:
    async with SessionLocal() as session:
        job = await session.get(Job, job_id)
        if job is None:
            return
        course = await _upsert_course(session, job, ctx)
        job.course_id = course.id if course else job.course_id
        job.status = "completed"
        job.stage = "done"
        job.progress = 1.0
        job.message = "Pipeline complete"
        job.error = None
        job.result_json = json.dumps(
            {
                "course": ctx.course,
                "bbox": ctx.bbox,
                "exports": ctx.exports,
                "layers": {lid: len(fc.get("features", [])) for lid, fc in ctx.layers.items()},
            },
            default=str,
        )
        job.quality_json = json.dumps(ctx.quality, default=str)

        existing = await session.scalars(select(LayerArtifact).where(LayerArtifact.job_id == job_id))
        for row in existing:
            await session.delete(row)
        for layer_id, fc in ctx.layers.items():
            session.add(
                LayerArtifact(
                    job_id=job_id,
                    layer_id=layer_id,
                    feature_count=len(fc.get("features", [])),
                    geojson_path=str(ctx.layer_dir() / f"{layer_id}.geojson"),
                    source="hybrid",
                )
            )
        await session.commit()


async def _persist_failure(job_id: str, ctx: PipelineContext | None, exc: Exception) -> None:
    async with SessionLocal() as session:
        job = await session.get(Job, job_id)
        if job is None:
            return
        job.status = "failed"
        job.error = str(exc).split("\n", 1)[0][:400]
        job.message = job.error
        payload = {"traceback": traceback.format_exc()}
        if ctx:
            payload["logs"] = ctx.logs
            payload["course"] = ctx.course
        job.result_json = json.dumps(payload, default=str)
        await session.commit()


async def _upsert_course(session: Any, job: Job, ctx: PipelineContext) -> Course | None:
    meta = ctx.course
    if not meta:
        return None
    bbox = ctx.bbox or (None, None, None, None)
    if job.course_id:
        course = await session.get(Course, job.course_id)
    else:
        course = Course()
        session.add(course)
    course.name = meta.get("name") or "Unknown course"
    course.display_name = meta.get("display_name")
    course.country = meta.get("country")
    course.address = meta.get("address")
    course.osm_id = str(meta.get("osm_id") or "") or None
    course.lat = meta.get("lat")
    course.lon = meta.get("lon")
    course.bbox_west, course.bbox_south, course.bbox_east, course.bbox_north = bbox
    course.metadata_json = json.dumps(meta, default=str)
    await session.flush()
    return course


def stage_names() -> tuple[str, ...]:
    return PIPELINE_STAGES
