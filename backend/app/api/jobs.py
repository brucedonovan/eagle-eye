from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_session
from app.models.entities import Job
from app.pipeline.orchestrator import run_pipeline
from app.schemas.api import CourseCreate, JobOut, StatusOut

from app.api.deps import job_to_status, load_job

router = APIRouter()


@router.post("/course", response_model=JobOut, status_code=202)
@router.post("/segment", response_model=JobOut, status_code=202)
@router.post("/vectorize", response_model=JobOut, status_code=202)
@router.post("/navigation", response_model=JobOut, status_code=202)
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


@router.get("/status/{job_id}", response_model=StatusOut)
async def get_status(job_id: str, session: AsyncSession = Depends(get_session)) -> StatusOut:
    job = await load_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job_to_status(job)


@router.get("/jobs", response_model=list[JobOut])
async def list_jobs(session: AsyncSession = Depends(get_session)) -> list[JobOut]:
    from sqlalchemy import select

    rows = (await session.scalars(select(Job).order_by(Job.created_at.desc()).limit(25))).all()
    return [
        JobOut(job_id=j.id, status=j.status, stage=j.stage, progress=j.progress, message=j.message)
        for j in rows
    ]


def _enqueue(background: BackgroundTasks, job_id: str) -> None:
    if settings.job_backend == "celery":
        from app.workers.celery_app import run_job

        run_job.delay(job_id)
        return
    background.add_task(_run_local, job_id)


async def _run_local(job_id: str) -> None:
    await run_pipeline(job_id)
