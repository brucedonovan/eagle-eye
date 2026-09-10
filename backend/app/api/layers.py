from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog import LAYERS
from app.db import get_session

from app.api.deps import load_job

router = APIRouter()


@router.get("/layers/catalog")
async def catalog() -> list[dict]:
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


@router.get("/layers/{job_id}")
async def get_layers(job_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    job = await load_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "completed":
        raise HTTPException(409, f"Job is {job.status}")
    layers = {}
    for artifact in job.layers:
        if not artifact.geojson_path:
            continue
        path = Path(artifact.geojson_path)
        if path.exists():
            layers[artifact.layer_id] = json.loads(path.read_text(encoding="utf-8"))
    return {"job_id": job_id, "layers": layers}


@router.get("/layers/{job_id}/{layer_id}")
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
