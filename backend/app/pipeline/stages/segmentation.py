"""Semantic segmentation fused with OSM priors."""

from __future__ import annotations

from app.catalog import SEMANTIC_CLASSES
from app.config import settings
from app.pipeline.context import PipelineContext
from app.services.imagery_segment import analyze_and_fuse


async def run(ctx: PipelineContext) -> None:
    prior = {cls: bool(ctx.layers.get(cls, {}).get("features")) for cls in SEMANTIC_CLASSES if cls != "background"}
    result = analyze_and_fuse(ctx)
    ctx.masks["classes"] = list(SEMANTIC_CLASSES)
    ctx.masks["prior_coverage"] = prior
    ctx.masks["model_weights"] = settings.segmentation_weights or None
    fused = result.backend in {"imagery_fusion", "dual_imagery_fusion"}
    if not fused and settings.segmentation_weights:
        ctx.masks["status"] = "weights_present_not_loaded"
        ctx.log("Segmentation weights configured; OSM prior still used (no mosaic)")
    elif not fused:
        ctx.masks.setdefault("backend", settings.segmentation_backend or "osm_prior")
