"""Image preprocessing (normalization, tile bookkeeping)."""

from __future__ import annotations

from pathlib import Path

from app.pipeline.context import PipelineContext


async def run(ctx: PipelineContext) -> None:
    path = ctx.imagery.get("path")
    if not path or not Path(path).exists():
        ctx.log("No mosaic to preprocess")
        return
    try:
        from PIL import Image, ImageEnhance, ImageOps

        img = Image.open(path).convert("RGB")
        img = ImageOps.autocontrast(img, cutoff=1)
        img = ImageEnhance.Contrast(img).enhance(1.08)
        img = ImageEnhance.Color(img).enhance(1.04)
        out = ctx.work_dir / "imagery" / "mosaic_enhanced.jpg"
        img.save(out, quality=90)
        ctx.imagery["enhanced_path"] = str(out)
        ctx.imagery["preprocess"] = {
            "autocontrast": True,
            "contrast": 1.08,
            "color": 1.04,
            "cloud_detection": "deferred",
            "shadow_removal": "deferred",
            "super_resolution": "optional",
        }
        ctx.log("Enhanced imagery mosaic")
    except Exception as exc:
        ctx.log(f"Preprocess skipped: {exc}")
