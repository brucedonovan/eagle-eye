from fastapi import APIRouter

from app.api import exports, jobs, layers, search

api_router = APIRouter()
api_router.include_router(jobs.router, tags=["pipeline"])
api_router.include_router(search.router, tags=["search"])
api_router.include_router(layers.router, tags=["layers"])
api_router.include_router(exports.router, tags=["exports"])
