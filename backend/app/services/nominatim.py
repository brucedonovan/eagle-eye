"""Course geocoding via Nominatim (OpenStreetMap)."""

from __future__ import annotations

import asyncio
import re
import unicodedata
from typing import Any

import httpx

from app.config import settings


class NominatimError(RuntimeError):
    pass


async def search_course(name: str, *, limit: int = 8) -> list[dict[str, Any]]:
    query = name.strip()
    if not query:
        return []
    headers = {"User-Agent": settings.osm_user_agent, "Accept-Language": "pt,en"}
    hits: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        for variant in query_variants(query):
            hits.extend(await _get(client, variant, limit))
            if any(_is_golf_course(row) for row in hits):
                break

    ranked = _dedupe(sorted(hits, key=lambda row: _score(row, query), reverse=True))
    golf_only = [row for row in ranked if _is_golf_course(row)]
    if golf_only:
        return golf_only

    # Official names often differ from OSM (Jamor drops "de Golfe").
    # Geocode a place token, then snap to a nearby leisure=golf_course.
    from app.services import overpass

    try:
        named = await overpass.find_courses_by_name(query)
    except Exception:
        named = []
    if named and named[0]["score"] >= 0.25:
        return [_course_to_hit(named[0])]

    for anchor in ranked[:4]:
        try:
            lon, lat = float(anchor["lon"]), float(anchor["lat"])
            nearby = await overpass.find_golf_courses(lon, lat, query)
        except Exception:
            continue
        if nearby and nearby[0]["score"] >= 0.25:
            return [_course_to_hit(nearby[0])]

    return ranked


async def reverse(lat: float, lon: float) -> dict[str, Any] | None:
    headers = {"User-Agent": settings.osm_user_agent}
    params = {"lat": lat, "lon": lon, "format": "jsonv2", "addressdetails": 1, "extratags": 1}
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        resp = await client.get(f"{settings.nominatim_url}/reverse", params=params)
        resp.raise_for_status()
        data = resp.json()
    return data if isinstance(data, dict) else None


def query_variants(query: str) -> list[str]:
    """Shorter / unaccented forms when the official name is not in Nominatim."""
    out: list[str] = []

    def add(text: str) -> None:
        cleaned = " ".join(text.split()).strip(" ,.-")
        if cleaned and cleaned.lower() not in {item.lower() for item in out}:
            out.append(cleaned)

    add(query)
    add(strip_accents(query))
    shortened = re.sub(
        r"(?i)\b(centro nacional de forma[cç][aã]o de|centro nacional de|national golf academy|clube de golfe)\b",
        "",
        query,
    )
    add(shortened)
    add(strip_accents(shortened))
    tokens = [t for t in _word_tokens(query) if t not in _STOP]
    if tokens:
        add(" ".join(tokens[-2:]))
        add(f"{tokens[-1]} golfe")
        add(f"{tokens[-1]} golf")
        add(tokens[-1])
    return out[:6]


def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


_STOP = {
    "golf",
    "golfe",
    "course",
    "links",
    "the",
    "club",
    "clube",
    "campo",
    "national",
    "nacional",
    "centro",
    "center",
    "of",
    "de",
    "do",
    "da",
    "e",
    "and",
}


def _word_tokens(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in strip_accents(text))
    return [part for part in cleaned.split() if part]


async def _get(client: httpx.AsyncClient, query: str, limit: int) -> list[dict[str, Any]]:
    params = {
        "q": query,
        "format": "jsonv2",
        "addressdetails": 1,
        "extratags": 1,
        "namedetails": 1,
        "limit": limit,
    }
    await asyncio.sleep(1.05)
    resp = await client.get(f"{settings.nominatim_url}/search", params=params)
    if resp.status_code >= 400:
        raise NominatimError(f"Nominatim {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    return data if isinstance(data, list) else []


def _is_golf_course(row: dict[str, Any]) -> bool:
    extra = row.get("extratags") or {}
    if extra.get("leisure") == "golf_course":
        return True
    if row.get("class") == "leisure" and row.get("type") == "golf_course":
        return True
    if row.get("type") == "golf_course":
        return True
    return False


def _score(row: dict[str, Any], query: str) -> float:
    q = query.lower()
    name = str(row.get("display_name") or row.get("name") or "").lower()
    score = float(row.get("importance") or 0)
    if _is_golf_course(row):
        score += 5.0
    if "golf" in name or "golfe" in name:
        score += 1.0
    if q in name:
        score += 1.5
    if row.get("class") in {"office", "building", "place", "highway", "amenity"}:
        score -= 2.0
    if row.get("type") in {"restaurant", "cafe", "hotel", "shop"}:
        score -= 3.0
    return score


def parse_bbox(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = row.get("boundingbox")
    if not bbox or len(bbox) != 4:
        return None
    south, north, west, east = (float(v) for v in bbox)
    return west, south, east, north


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = f"{row.get('osm_type')}:{row.get('osm_id')}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _course_to_hit(course: dict[str, Any]) -> dict[str, Any]:
    west, south, east, north = course["bbox"]
    geom = course["geometry"]
    centroid = geom.centroid
    return {
        "lat": centroid.y,
        "lon": centroid.x,
        "display_name": course["name"],
        "class": "leisure",
        "type": "golf_course",
        "osm_id": course["osm_id"],
        "osm_type": course["osm_type"],
        "boundingbox": [str(south), str(north), str(west), str(east)],
        "extratags": {"leisure": "golf_course"},
        "importance": 0.6,
        "address": {},
    }
