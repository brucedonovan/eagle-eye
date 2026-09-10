from celery import Celery

from app.config import settings

celery = Celery("eagle_eye", broker=settings.redis_url, backend=settings.redis_url)


@celery.task(name="eagle_eye.run_job")
def run_job(job_id: str) -> str:
    import asyncio

    from app.pipeline.orchestrator import run_pipeline

    asyncio.run(run_pipeline(job_id))
    return job_id
