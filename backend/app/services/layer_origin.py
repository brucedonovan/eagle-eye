"""Classify vector features as API, OSM, or generated."""

from __future__ import annotations

from typing import Any

API_SOURCES = frozenset({"golfapi", "catalog", "fake"})
OSM_SOURCES = frozenset({"openstreetmap", "osm", "osm_refined"})
AI_SOURCES = frozenset({"ai", "ai_layer", "ai_fusion"})


def origin_bucket(source: Any) -> str:
    raw = str(source or "").strip().lower()
    if raw in API_SOURCES:
        return "api"
    if raw in OSM_SOURCES:
        return "osm"
    if raw in AI_SOURCES:
        return "ai"
    return "generated"


def layer_origin(fc: dict[str, Any] | None) -> str:
    buckets: set[str] = set()
    for feat in (fc or {}).get("features") or []:
        props = feat.get("properties") if isinstance(feat, dict) else None
        if not isinstance(props, dict):
            continue
        buckets.add(origin_bucket(props.get("source")))
    if not buckets:
        return "hybrid"
    if len(buckets) == 1:
        return next(iter(buckets))
    return "mixed"
