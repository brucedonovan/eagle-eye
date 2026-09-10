"""golfapi.io catalog adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from app.config import settings
from app.services.course_catalog.cache import (
    CacheResult,
    DiskJsonCache,
    as_float,
    as_int,
    int_list,
    opt_str,
    params_key,
    truthy,
)
from app.services.course_catalog.provider import (
    CatalogError,
    CourseCatalogProvider,
    CourseHit,
    CourseRecord,
    SearchResult,
)


def _default_cache_dir() -> Path:
    current = settings.cache_dir / "catalog" / "golfapi"
    legacy = settings.cache_dir / "golfapi"
    if not current.exists() and legacy.exists():
        return legacy
    return current


class GolfApiError(CatalogError):
    pass


@dataclass
class GolfApiClient:
    cache_dir: Path | None = None
    api_key: str | None = None
    base_url: str | None = None
    search_ttl_days: int | None = None
    http_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        root = Path(self.cache_dir or _default_cache_dir())
        self.cache = DiskJsonCache(root)
        self.cache_dir = root
        self.api_key = settings.golfapi_key if self.api_key is None else self.api_key
        self.base_url = (self.base_url or settings.golfapi_base_url).rstrip("/")
        self.search_ttl_days = (
            settings.course_catalog_search_ttl_days if self.search_ttl_days is None else self.search_ttl_days
        )

    @property
    def enabled(self) -> bool:
        return bool((self.api_key or "").strip())

    def last_requests_left(self) -> str | None:
        return self.cache.last_requests_left()

    async def search_clubs(
        self,
        name: str = "",
        *,
        lat: float | None = None,
        lng: float | None = None,
        country: str | None = None,
        city: str | None = None,
        page: int = 1,
    ) -> CacheResult:
        params: dict[str, Any] = {"page": page}
        if name.strip():
            params["name"] = name.strip()
        if city:
            params["city"] = city
        if country:
            params["country"] = country
        if lat is not None and lng is not None:
            params["lat"] = round(float(lat), 5)
            params["lng"] = round(float(lng), 5)
            params["measureUnit"] = "km"
        return await self._cached_get("clubs", params, bucket="searches", ttl_days=self.search_ttl_days)

    async def get_course(self, course_id: str, *, timestamp_updated: int | None = None) -> CacheResult:
        return await self._cached_get(
            f"courses/{course_id}",
            {},
            bucket="courses",
            cache_id=str(course_id),
            newer_than=timestamp_updated,
        )

    async def get_coordinates(self, course_id: str, *, timestamp_updated: int | None = None) -> CacheResult:
        return await self._cached_get(
            f"coordinates/{course_id}",
            {},
            bucket="coordinates",
            cache_id=str(course_id),
            newer_than=timestamp_updated,
        )

    async def search_course_hits(
        self,
        query: str,
        *,
        lat: float | None = None,
        lng: float | None = None,
    ) -> tuple[list[CourseHit], CacheResult]:
        result = await self.search_clubs(query, lat=lat, lng=lng)
        hits = flatten_club_search(result.payload)
        hits.sort(key=lambda hit: _hit_score(hit, query), reverse=True)
        return hits, result

    def list_cached_hits(self, *, limit: int = 24) -> list[CourseHit]:
        by_id: dict[str, tuple[str, CourseHit]] = {}

        def add(hit: CourseHit, cached_at: str, *, prefer: bool = False) -> None:
            if not hit.course_id:
                return
            prev = by_id.get(hit.course_id)
            if prev is None or prefer:
                by_id[hit.course_id] = (cached_at, hit)

        for entry in self.cache.iter_entries("searches", ttl_days=self.search_ttl_days):
            cached_at = str(entry.get("cached_at") or "")
            for hit in flatten_club_search(entry.get("payload")):
                add(hit, cached_at)
        for entry in self.cache.iter_entries("courses"):
            cached_at = str(entry.get("cached_at") or "")
            hit = hit_from_course_payload(entry.get("payload"))
            if hit:
                add(hit, cached_at, prefer=True)
        ranked = sorted(by_id.values(), key=lambda row: row[0], reverse=True)
        return [hit for _at, hit in ranked[:limit]]

    async def load_course_bundle(
        self,
        course_id: str,
        *,
        timestamp_updated: int | None = None,
    ) -> dict[str, Any]:
        course = await self.get_course(course_id, timestamp_updated=timestamp_updated)
        payload = course.payload if isinstance(course.payload, dict) else {}
        coords: dict[str, Any] | list[Any] = {}
        coords_cached = True
        if truthy(payload.get("hasGPS")) or as_int(payload.get("numCoordinates")) > 0:
            try:
                coord_res = await self.get_coordinates(course_id, timestamp_updated=timestamp_updated)
                coords = coord_res.payload
                coords_cached = coord_res.cached
            except GolfApiError:
                coords = {}
                coords_cached = False
        return {
            "course": payload,
            "coordinates": coords,
            "points": parse_coordinates(coords),
            "course_cached": course.cached,
            "coordinates_cached": coords_cached,
            "api_requests_left": course.api_requests_left or self.last_requests_left(),
        }

    async def _cached_get(
        self,
        path: str,
        params: dict[str, Any],
        *,
        bucket: str,
        cache_id: str | None = None,
        ttl_days: int | None = None,
        newer_than: int | None = None,
    ) -> CacheResult:
        key = cache_id or params_key(path, params)
        hit = self.cache.get(bucket, key, ttl_days=ttl_days, newer_than=newer_than)
        if hit:
            return hit
        data = await self._http_get(path, params)
        left = data.get("apiRequestsLeft") if isinstance(data, dict) else None
        return self.cache.put(bucket, key, data, path=path, params=params, api_requests_left=left)

    async def _http_get(self, path: str, params: dict[str, Any]) -> Any:
        if not self.enabled:
            raise GolfApiError("golfapi.io key is not configured")
        self.http_calls += 1
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        url = f"{self.base_url}/{path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            resp = await client.get(url, params=params or None)
        if resp.status_code >= 400:
            raise GolfApiError(f"golfapi.io {resp.status_code}: {resp.text[:240]}")
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise GolfApiError("golfapi.io returned non-JSON") from exc


class GolfApiProvider(CourseCatalogProvider):
    id = "golfapi"
    title = "golfapi.io"

    def __init__(self, client: GolfApiClient | None = None) -> None:
        self.client = client or GolfApiClient()

    @property
    def configured(self) -> bool:
        return self.client.enabled

    async def search(
        self,
        query: str,
        *,
        lat: float | None = None,
        lon: float | None = None,
    ) -> SearchResult:
        hits, raw = await self.client.search_course_hits(query, lat=lat, lng=lon)
        return SearchResult(
            provider=self.id,
            title=self.title,
            cached=raw.cached,
            hits=hits,
            api_requests_left=raw.api_requests_left or self.client.last_requests_left(),
        )

    async def load_course(
        self,
        course_id: str,
        *,
        club_id: str | None = None,
        timestamp_updated: int | None = None,
    ) -> CourseRecord:
        bundle = await self.client.load_course_bundle(course_id, timestamp_updated=timestamp_updated)
        payload = bundle["course"] if isinstance(bundle["course"], dict) else {}
        points = bundle["points"]
        lat = as_float(payload.get("latitude"))
        lon = as_float(payload.get("longitude"))
        if (lat is None or lon is None) and points:
            lat = sum(p["lat"] for p in points) / len(points)
            lon = sum(p["lon"] for p in points) / len(points)
        club_name = str(payload.get("clubName") or "").strip()
        course_name = str(payload.get("courseName") or club_name).strip()
        address_parts = [
            str(payload.get("address") or "").strip(),
            str(payload.get("city") or "").strip(),
            str(payload.get("state") or "").strip(),
            str(payload.get("country") or "").strip(),
        ]
        return CourseRecord(
            provider=self.id,
            course_id=str(payload.get("courseID") or course_id),
            club_id=str(payload.get("clubID") or club_id or ""),
            club_name=club_name or "Golf club",
            course_name=course_name or club_name,
            lat=lat,
            lon=lon,
            country=opt_str(payload.get("country") or payload.get("country2")),
            city=opt_str(payload.get("city")),
            state=opt_str(payload.get("state")),
            address=", ".join(part for part in address_parts if part) or opt_str(payload.get("address")),
            postal_code=opt_str(payload.get("postalCode")),
            website=opt_str(payload.get("website")),
            telephone=opt_str(payload.get("telephone")),
            num_holes=as_int(payload.get("numHoles")) or None,
            has_gps=truthy(payload.get("hasGPS")),
            scorecard=scorecard_from_course(payload),
            points=[{"hole": p.get("hole"), "kind": p["kind"], "lat": p["lat"], "lon": p["lon"]} for p in points],
            cached=bool(bundle["course_cached"] and bundle["coordinates_cached"]),
            api_requests_left=bundle.get("api_requests_left"),
        )

    def list_cached(self) -> list[CourseHit]:
        return self.client.list_cached_hits()

    def api_requests_left(self) -> str | None:
        return self.client.last_requests_left()


def flatten_club_search(payload: Any) -> list[CourseHit]:
    if not isinstance(payload, dict):
        return []
    hits: list[CourseHit] = []
    for club in payload.get("clubs") or []:
        if not isinstance(club, dict):
            continue
        club_id = str(club.get("clubID") or club.get("club_id") or "")
        club_name = str(club.get("clubName") or club.get("club_name") or "").strip()
        courses = club.get("courses") or []
        if not courses:
            continue
        for course in courses:
            if not isinstance(course, dict):
                continue
            hits.append(
                CourseHit(
                    provider="golfapi",
                    club_id=club_id,
                    club_name=club_name,
                    course_id=str(course.get("courseID") or course.get("course_id") or ""),
                    course_name=str(course.get("courseName") or course.get("course_name") or club_name).strip(),
                    city=opt_str(club.get("city")),
                    state=opt_str(club.get("state")),
                    country=opt_str(club.get("country")),
                    address=opt_str(club.get("address")),
                    lat=as_float(club.get("latitude") or course.get("latitude")),
                    lon=as_float(
                        club.get("longitude") or club.get("lng") or course.get("longitude")
                    ),
                    num_holes=as_int(course.get("numHoles") or course.get("num_holes")) or None,
                    has_gps=truthy(course.get("hasGPS") or course.get("has_gps")),
                    distance=as_float(club.get("distance")),
                    measure_unit=opt_str(club.get("measureUnit")),
                    timestamp_updated=as_int(course.get("timestampUpdated")) or None,
                )
            )
    return [hit for hit in hits if hit.course_id]


def hit_from_course_payload(payload: Any) -> CourseHit | None:
    if not isinstance(payload, dict):
        return None
    course_id = str(payload.get("courseID") or payload.get("course_id") or "")
    if not course_id:
        return None
    club_name = str(payload.get("clubName") or payload.get("club_name") or "").strip()
    course_name = str(payload.get("courseName") or payload.get("course_name") or club_name).strip()
    return CourseHit(
        provider="golfapi",
        club_id=str(payload.get("clubID") or payload.get("club_id") or ""),
        club_name=club_name or "Golf club",
        course_id=course_id,
        course_name=course_name or club_name,
        city=opt_str(payload.get("city")),
        state=opt_str(payload.get("state")),
        country=opt_str(payload.get("country") or payload.get("country2")),
        address=opt_str(payload.get("address")),
        lat=as_float(payload.get("latitude")),
        lon=as_float(payload.get("longitude")),
        num_holes=as_int(payload.get("numHoles") or payload.get("num_holes")) or None,
        has_gps=truthy(payload.get("hasGPS") or payload.get("has_gps")),
        timestamp_updated=as_int(payload.get("timestampUpdated")) or None,
    )


def parse_coordinates(payload: Any) -> list[dict[str, Any]]:
    rows = _coordinate_rows(payload)
    points: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        lat = as_float(row.get("latitude") or row.get("lat"))
        lon = as_float(row.get("longitude") or row.get("lng") or row.get("lon"))
        if lat is None or lon is None:
            continue
        hole = as_int(row.get("hole") or row.get("holeNumber") or row.get("hole_number") or row.get("no"))
        kind = _poi_kind(
            row.get("poi")
            or row.get("poiType")
            or row.get("poi_type")
            or row.get("type")
            or row.get("locationType")
            or row.get("pointType")
            or row.get("name")
            or ""
        )
        points.append({"hole": hole or None, "kind": kind, "lat": lat, "lon": lon, "raw": row})
    return points


def scorecard_from_course(payload: dict[str, Any]) -> dict[str, Any]:
    tees = []
    for tee in payload.get("tees") or []:
        if not isinstance(tee, dict):
            continue
        lengths = []
        for i in range(1, 19):
            value = as_float(tee.get(f"length{i}"))
            if value is not None:
                lengths.append(value)
        tees.append(
            {
                "tee_id": tee.get("teeID"),
                "name": tee.get("teeName"),
                "color": tee.get("teeColor"),
                "lengths": lengths,
                "rating_men": as_float(tee.get("courseRatingMen")),
                "slope_men": as_int(tee.get("slopeMen")) or None,
                "rating_women": as_float(tee.get("courseRatingWomen")),
                "slope_women": as_int(tee.get("slopeWomen")) or None,
            }
        )
    return {
        "pars_men": int_list(payload.get("parsMen")),
        "indexes_men": int_list(payload.get("indexesMen")),
        "pars_women": int_list(payload.get("parsWomen")),
        "indexes_women": int_list(payload.get("indexesWomen")),
        "measure": payload.get("measure"),
        "num_holes": as_int(payload.get("numHoles")) or None,
        "tees": tees,
    }


def _hit_score(hit: CourseHit, query: str) -> float:
    q = query.strip().lower()
    score = 0.0
    club = hit.club_name.lower()
    course = hit.course_name.lower()
    if q and q in club:
        score += 4.0
    if q and q in course:
        score += 3.0
    for token in q.split():
        if token in club:
            score += 0.8
        if token in course:
            score += 0.6
    if hit.has_gps:
        score += 0.5
    if hit.num_holes in {9, 18}:
        score += 0.3
    if hit.distance is not None:
        score += max(0.0, 2.0 - hit.distance / 5.0)
    return score


def _coordinate_rows(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("coordinates", "pois", "points", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = _coordinate_rows(value)
            if nested:
                return nested
    return []


_POI_MAP = {
    "green": "green",
    "greens": "green",
    "pin": "pin",
    "flag": "pin",
    "flagstick": "pin",
    "tee": "tee",
    "tees": "tee",
    "teebox": "tee",
    "tee box": "tee",
    "fairway": "fairway",
    "bunker": "bunker",
    "sand": "bunker",
    "water": "water",
    "hazard": "water",
    "dogleg": "dogleg",
}


def _poi_kind(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "poi"
    if text in _POI_MAP:
        return _POI_MAP[text]
    for key, kind in _POI_MAP.items():
        if key in text:
            return kind
    return "poi"
