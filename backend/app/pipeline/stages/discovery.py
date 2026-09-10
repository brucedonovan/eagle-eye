"""Course discovery from name, coordinates, or AOI."""

from __future__ import annotations

from shapely.geometry import shape

from app.pipeline.context import PipelineContext
from app.services import overpass
from app.services.course_catalog import CatalogError, get_provider
from app.services.course_catalog import layers as catalog_layers
from app.services.geometry import to_feature

MAX_BBOX_SPAN_DEG = 0.08


def _clamp_bbox(
    bbox: tuple[float, float, float, float],
    lon: float | None = None,
    lat: float | None = None,
) -> tuple[float, float, float, float]:
    west, south, east, north = bbox
    if (east - west) <= MAX_BBOX_SPAN_DEG and (north - south) <= MAX_BBOX_SPAN_DEG:
        return bbox
    cx = lon if lon is not None else (west + east) / 2
    cy = lat if lat is not None else (south + north) / 2
    half = MAX_BBOX_SPAN_DEG / 2
    return cx - half, cy - half, cx + half, cy + half


async def run(ctx: PipelineContext) -> None:
    req = ctx.request
    if req.get("aoi"):
        geom = shape(req["aoi"] if req["aoi"].get("type") != "Feature" else req["aoi"]["geometry"])
        west, south, east, north = geom.bounds
        centroid = geom.centroid
        ctx.bbox = (west, south, east, north)
        ctx.course = {
            "name": req.get("name") or "Custom AOI",
            "display_name": req.get("name") or "Custom AOI",
            "lat": centroid.y,
            "lon": centroid.x,
            "source": "aoi",
        }
        ctx.log("AOI provided; skipped geocoding")
        return

    catalog_id = str(req.get("catalog_course_id") or req.get("golfapi_course_id") or "").strip()
    provider_name = req.get("catalog_provider")
    if catalog_id:
        await _discover_from_catalog(ctx, catalog_id, provider_name=provider_name)
        return

    name = (req.get("name") or "").strip()
    lat = _float(req.get("lat"))
    lon = _float(req.get("lon"))
    if name or (lat is not None and lon is not None):
        await _discover_from_catalog_search(
            ctx, name, lat=lat, lon=lon, provider_name=provider_name
        )
        return

    raise RuntimeError("Could not locate a golf course (no name, coordinates, or catalog course id)")


def _require_catalog(name: str | None = None):
    try:
        provider = get_provider(name)
    except CatalogError as exc:
        raise RuntimeError(str(exc)) from exc
    if provider is None:
        raise RuntimeError("Course catalog is not configured (missing provider credentials)")
    return provider


async def _discover_from_catalog_search(
    ctx: PipelineContext,
    name: str,
    *,
    lat: float | None = None,
    lon: float | None = None,
    provider_name: str | None = None,
) -> None:
    provider = _require_catalog(provider_name)
    try:
        result = await provider.search(name, lat=lat, lon=lon)
    except CatalogError as exc:
        raise RuntimeError(str(exc)) from exc
    if not result.hits:
        label = name or (f"{lat:.5f},{lon:.5f}" if lat is not None and lon is not None else "query")
        raise RuntimeError(f"{provider.title} returned no clubs for “{label}”")
    ctx.log(
        f"{provider.title} search {'cache hit' if result.cached else 'fetched'} ({len(result.hits)} courses)"
    )
    best = result.hits[0]
    await _discover_from_catalog(
        ctx,
        best.course_id,
        provider_name=provider.id,
        club_id=best.club_id,
        timestamp_updated=best.timestamp_updated,
    )


async def _discover_from_catalog(
    ctx: PipelineContext,
    course_id: str,
    *,
    provider_name: str | None = None,
    club_id: str | None = None,
    timestamp_updated: int | None = None,
) -> None:
    req = ctx.request
    provider = _require_catalog(provider_name)
    club_id = club_id or req.get("catalog_club_id") or req.get("golfapi_club_id")
    ts = timestamp_updated
    if ts is None:
        ts = req.get("catalog_timestamp_updated")
    if ts is None:
        ts = req.get("golfapi_timestamp_updated")
    record = await provider.load_course(course_id, club_id=club_id, timestamp_updated=ts)
    lat = record.lat or _float(req.get("lat"))
    lon = record.lon or _float(req.get("lon"))
    if (lat is None or lon is None) and record.points:
        lat = sum(p["lat"] for p in record.points) / len(record.points)
        lon = sum(p["lon"] for p in record.points) / len(record.points)
    if lat is None or lon is None:
        raise RuntimeError(f"{provider.title} course {course_id} has no coordinates")

    display = record.display_name
    gps_bbox = catalog_layers.bbox_from_points(record.points)
    gps_clip = catalog_layers.geojson_clip(catalog_layers.clip_geom_from_points(record.points))
    pad = 0.012
    ctx.bbox = gps_bbox or _clamp_bbox((lon - pad, lat - pad, lon + pad, lat + pad), lon, lat)
    ctx.course = {
        "name": req.get("name") or display,
        "display_name": display,
        "lat": lat,
        "lon": lon,
        "country": record.country,
        "address": record.address,
        "website": record.website,
        "telephone": record.telephone,
        "source": provider.id,
        "catalog_provider": provider.id,
        "catalog_course_id": record.course_id,
        "catalog_club_id": record.club_id,
        "catalog_course_name": record.course_name,
        "catalog_club_name": record.club_name,
        "catalog_num_holes": record.num_holes,
        "catalog_has_gps": record.has_gps,
        "catalog_scorecard": record.scorecard,
        "catalog_points": record.points,
        "catalog_clip": gps_clip,
        "catalog_cached": record.cached,
        "city": record.city,
        "state": record.state,
        "postal_code": record.postal_code,
    }
    cache_note = "cache" if record.cached else "network"
    ctx.log(
        f"{provider.title} {display} ({cache_note}; {len(record.points)} GPS points, "
        f"{record.num_holes or '?'} holes)"
    )
    await _snap_to_golf_course(ctx, display)
    if gps_bbox and ctx.bbox:
        osm_span = max(ctx.bbox[2] - ctx.bbox[0], ctx.bbox[3] - ctx.bbox[1])
        gps_span = max(gps_bbox[2] - gps_bbox[0], gps_bbox[3] - gps_bbox[1])
        if osm_span > 1.8 * max(gps_span, 1e-6):
            ctx.bbox = gps_bbox
            ctx.course["_boundary_ready"] = False
            ctx.log(f"Using {provider.title} GPS hull as AOI (tighter than OSM club boundary)")


async def _snap_to_golf_course(ctx: PipelineContext, name: str) -> None:
    lat, lon = ctx.course.get("lat"), ctx.course.get("lon")
    if lat is None or lon is None:
        return
    try:
        courses = await overpass.find_golf_courses(float(lon), float(lat), name)
    except Exception as exc:
        ctx.log(f"Golf-course snap skipped: {exc}")
        return
    if not courses:
        ctx.log("No leisure=golf_course found near the geocode")
        return
    best = courses[0]
    if best["score"] < 0.2 and len(courses) > 1:
        # Fall back to the largest nearby course if names are all weak.
        best = max(courses, key=lambda row: row["geometry"].area)
    ctx.bbox = best["bbox"]
    ctx.course["osm_id"] = str(best["osm_id"])
    ctx.course["osm_type"] = best["osm_type"]
    ctx.course["boundary_name"] = best["name"]
    ctx.course["type"] = "golf_course"
    ctx.course["resolved_from"] = ctx.course.get("resolved_from") or "overpass_golf_course"
    if ctx.course.get("catalog_provider"):
        ctx.course["resolved_from"] = f"{ctx.course['catalog_provider']}+overpass"
    ctx.course["nearby_courses"] = [
        {"name": c["name"], "osm_id": c["osm_id"], "score": round(c["score"], 3)}
        for c in courses[:6]
    ]
    ctx.layers["boundary"] = {
        "type": "FeatureCollection",
        "features": [to_feature(best["geometry"], {"name": best["name"], "source": "openstreetmap"})],
    }
    ctx.course["_boundary_ready"] = True


def _float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
