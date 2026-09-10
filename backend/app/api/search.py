from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas.api import CourseSearchItem, CourseSearchOut
from app.services import nominatim
from app.services.course_catalog import CatalogError, CourseHit, get_provider

router = APIRouter()


@router.get("/search", response_model=CourseSearchOut)
async def search_courses(
    q: str = Query(default="", description="Club / course name"),
    lat: float | None = None,
    lon: float | None = None,
) -> CourseSearchOut:
    query = q.strip()
    if not query and (lat is None or lon is None):
        raise HTTPException(400, "Provide a search query or lat/lon")

    try:
        provider = get_provider()
    except CatalogError as exc:
        raise HTTPException(400, str(exc)) from exc

    if provider is not None:
        try:
            result = await provider.search(query, lat=lat, lng=lon)
            if result.hits:
                return CourseSearchOut(
                    query=query,
                    source=provider.id,
                    cached=result.cached,
                    golfapi_configured=provider.id == "golfapi",
                    catalog_configured=True,
                    catalog_provider=provider.id,
                    provider_title=provider.title,
                    api_requests_left=result.api_requests_left,
                    courses=[_from_hit(hit) for hit in result.hits[:40]],
                )
            fallback = await _nominatim_hits(query)
            return CourseSearchOut(
                query=query,
                source="nominatim" if fallback else provider.id,
                cached=result.cached,
                golfapi_configured=provider.id == "golfapi",
                catalog_configured=True,
                catalog_provider=provider.id,
                provider_title=provider.title,
                api_requests_left=result.api_requests_left,
                warning=f"{provider.title} returned no clubs; showing OpenStreetMap matches."
                if fallback
                else None,
                courses=fallback,
            )
        except CatalogError as exc:
            fallback = await _nominatim_hits(query)
            if fallback:
                return CourseSearchOut(
                    query=query,
                    source="nominatim",
                    cached=False,
                    catalog_configured=True,
                    catalog_provider=provider.id,
                    provider_title=provider.title,
                    warning=str(exc),
                    courses=fallback,
                )
            raise HTTPException(502, str(exc)) from exc

    courses = await _nominatim_hits(query)
    return CourseSearchOut(
        query=query,
        source="nominatim",
        cached=False,
        catalog_configured=False,
        warning="No course catalog is configured; search is using OpenStreetMap.",
        courses=courses,
    )


def _from_hit(hit: CourseHit) -> CourseSearchItem:
    return CourseSearchItem(
        source=hit.provider,
        club_id=hit.club_id or None,
        club_name=hit.club_name,
        course_id=hit.course_id or None,
        course_name=hit.course_name,
        display_name=hit.display_name,
        city=hit.city,
        state=hit.state,
        country=hit.country,
        address=hit.address,
        lat=hit.lat,
        lon=hit.lon,
        num_holes=hit.num_holes,
        has_gps=hit.has_gps,
        distance_km=hit.distance,
        timestamp_updated=hit.timestamp_updated,
    )


async def _nominatim_hits(query: str) -> list[CourseSearchItem]:
    if not query:
        return []
    try:
        rows = await nominatim.search_course(query, limit=8)
    except nominatim.NominatimError:
        return []
    items: list[CourseSearchItem] = []
    for row in rows[:12]:
        name = str(row.get("display_name") or row.get("name") or query)
        address = row.get("address") or {}
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
        except (KeyError, TypeError, ValueError):
            lat = lon = None
        items.append(
            CourseSearchItem(
                source="nominatim",
                club_name=name,
                course_name=str(row.get("name") or name),
                display_name=name,
                city=address.get("city") or address.get("town") or address.get("village"),
                state=address.get("state"),
                country=address.get("country"),
                address=name,
                lat=lat,
                lon=lon,
            )
        )
    return items
