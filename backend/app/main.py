from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import api_router
from app.catalog import LAYERS, MVP_LAYERS, SEMANTIC_CLASSES
from app.config import settings
from app.db import init_db
from app.services.course_catalog import status as catalog_status
from app.services.imagery import available_sources


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Eagle Eye",
    description="Autonomous golf course vectorization platform",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/capabilities")
async def capabilities() -> dict:
    return {
        "layers": [layer.id for layer in LAYERS],
        "mvp_layers": list(MVP_LAYERS),
        "semantic_classes": list(SEMANTIC_CLASSES),
        "imagery_sources": [
            {"id": s.id, "title": s.title, "gsd_m": s.gsd_m, "available": s.available}
            for s in available_sources()
        ],
        "imagery_pin_zoom": settings.imagery_pin_zoom,
        "job_backend": settings.job_backend,
        "segmentation_backend": settings.segmentation_backend,
        "course_catalog": catalog_status(),
    }
