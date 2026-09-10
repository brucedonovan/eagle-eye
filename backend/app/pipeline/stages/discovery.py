"""Stage 1 — Course discovery from name, coordinates, or AOI."""

from __future__ import annotations

from shapely.geometry import shape

from app.pipeline.context import PipelineContext
from app.services import nominatim, overpass
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

    if req.get("lat") is not None and req.get("lon") is not None:
        lat, lon = float(req["lat"]), float(req["lon"])
        rev = await nominatim.reverse(lat, lon)
        address = (rev or {}).get("address") or {}
        ctx.course = {
            "name": req.get("name") or (rev or {}).get("name") or "Unnamed course",
            "display_name": (rev or {}).get("display_name"),
            "lat": lat,
            "lon": lon,
            "country": address.get("country"),
            "address": (rev or {}).get("display_name"),
            "osm_id": str((rev or {}).get("osm_id") or ""),
            "source": "coordinates",
        }
        parsed = nominatim.parse_bbox(rev) if rev else None
        pad = 0.012
        ctx.bbox = _clamp_bbox(parsed or (lon - pad, lat - pad, lon + pad, lat + pad), lon, lat)
        await _snap_to_golf_course(ctx, req.get("name") or ctx.course["name"])
        ctx.log(f"Reverse-geocoded {lat:.5f},{lon:.5f}")
        return

    name = (req.get("name") or "").strip()
    hits = await nominatim.search_course(name)
    if not hits:
        raise RuntimeError(f"Could not locate a golf course named “{name}”")

    best = hits[0]
    bbox = nominatim.parse_bbox(best)
    if not bbox:
        lon, lat = float(best["lon"]), float(best["lat"])
        pad = 0.012
        bbox = (lon - pad, lat - pad, lon + pad, lat + pad)
    address = best.get("address") or {}
    ctx.bbox = _clamp_bbox(bbox, float(best["lon"]), float(best["lat"]))
    ctx.course = {
        "name": name,
        "display_name": best.get("display_name") or name,
        "lat": float(best["lat"]),
        "lon": float(best["lon"]),
        "country": address.get("country"),
        "address": best.get("display_name"),
        "osm_id": str(best.get("osm_id") or ""),
        "osm_type": best.get("osm_type"),
        "class": best.get("class"),
        "type": best.get("type"),
        "source": "nominatim",
        "candidates": [
            {"name": h.get("display_name"), "lat": h.get("lat"), "lon": h.get("lon")}
            for h in hits[:5]
        ],
    }
    await _snap_to_golf_course(ctx, name)
    ctx.log(f"Discovered {ctx.course.get('boundary_name') or ctx.course['display_name']}")


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
    ctx.course["resolved_from"] = "overpass_golf_course"
    ctx.course["nearby_courses"] = [
        {"name": c["name"], "osm_id": c["osm_id"], "score": round(c["score"], 3)}
        for c in courses[:6]
    ]
    ctx.layers["boundary"] = {
        "type": "FeatureCollection",
        "features": [to_feature(best["geometry"], {"name": best["name"], "source": "openstreetmap"})],
    }
    ctx.course["_boundary_ready"] = True
