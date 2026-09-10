"""Stage 8 — Polygon extraction / cleaning for every catalog layer."""

from __future__ import annotations

from shapely.geometry.base import BaseGeometry

from app.catalog import LAYER_BY_ID
from app.pipeline.context import PipelineContext
from app.services.geometry import (
    LINE_SIMPLIFY_DEG,
    as_geom,
    clean_polygon,
    feature_collection,
    to_feature,
)


async def run(ctx: PipelineContext) -> None:
    cleaned_layers: dict[str, dict] = {}
    valid = 0
    invalid = 0
    for layer_id, fc in ctx.layers.items():
        spec = LAYER_BY_ID.get(layer_id)
        features = []
        for feat in fc.get("features", []):
            try:
                geom = as_geom(feat)
            except Exception:
                invalid += 1
                continue
            geom = _clean_for_spec(geom, spec.geometry if spec else "polygon")
            if geom is None or geom.is_empty:
                invalid += 1
                continue
            if not geom.is_valid:
                invalid += 1
                continue
            features.append(to_feature(geom, feat.get("properties") or {}))
            valid += 1
        if features:
            cleaned_layers[layer_id] = feature_collection(features)
    ctx.layers = cleaned_layers
    total = valid + invalid
    ctx.quality["polygon_validity"] = (valid / total) if total else 1.0
    ctx.quality["features_valid"] = valid
    ctx.quality["features_dropped"] = invalid
    ctx.log(f"Vector cleaning kept {valid} features, dropped {invalid}")


def _clean_for_spec(geom: BaseGeometry, kind: str) -> BaseGeometry | None:
    if kind == "polygon":
        return clean_polygon(geom)
    if kind == "linestring":
        if geom.geom_type not in {"LineString", "MultiLineString"}:
            return None
        return geom.simplify(LINE_SIMPLIFY_DEG, preserve_topology=True)
    if kind == "point":
        if geom.geom_type not in {"Point", "MultiPoint"}:
            return geom.centroid
        return geom
    return clean_polygon(geom)
