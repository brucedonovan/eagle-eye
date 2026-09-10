"""GolfAPI.io client with disk cache.

Limited API quota: search costs 0.1, club/course/coordinates cost 1.0 each.
Course, club, and coordinate payloads are cached forever and only refetched
when a search result reports a newer timestampUpdated. Club searches are
cached for golfapi_search_ttl_days.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from app.config import settings


class GolfAPIError(RuntimeError):
    pass


def configured() -> bool:
    return bool(settings.golfapi_key.strip())


@dataclass
class CourseHit:
    club_id: str
    club_name: str
    course_id: str
    course_name: str
    city: str | None = None
    state: str | None = None
    country: str | None = None
    address: str | None = None
    num_holes: int | None = None
    has_gps: bool = False
    distance: float | None = None
    measure_unit: str | None = None
    timestamp_updated: int | None = None
    source: str = "golfapi"

    @property
    def display_name(self) -> str:
        course = (self.course_name or "").strip()
        club = (self.club_name or "").strip()
        if not course or course.lower() == club.lower() or course.lower() in club.lower():
            return club or course
        return f"{club} — {course}"


@dataclass
class CacheResult:
    payload: Any
    cached: bool
    api_requests_left: str | None = None


@dataclass
class GolfAPIClient:
    cache_dir: Path | None = None
    api_key: str | None = None
    base_url: str | None = None
    search_ttl_days: int | None = None
    http_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir or (settings.cache_dir / "golfapi"))
        self.api_key = settings.golfapi_key if self.api_key is None else self.api_key
        self.base_url = (self.base_url or settings.golfapi_base_url).rstrip("/")
        self.search_ttl_days = (
            settings.golfapi_search_ttl_days if self.search_ttl_days is None else self.search_ttl_days
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return bool((self.api_key or "").strip())

    def last_requests_left(self) -> str | None:
        meta = _read_json(self.cache_dir / "meta.json")
        if not meta:
            return None
        return meta.get("api_requests_left")

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
        return await self._cached_get(
            "clubs",
            params,
            bucket="searches",
            ttl_days=self.search_ttl_days,
        )

    async def get_club(self, club_id: str, *, timestamp_updated: int | None = None) -> CacheResult:
        return await self._cached_get(
            f"clubs/{club_id}",
            {},
            bucket="clubs",
            cache_id=str(club_id),
            newer_than=timestamp_updated,
        )

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

    async def load_course_bundle(
        self,
        course_id: str,
        *,
        timestamp_updated: int | None = None,
    ) -> dict[str, Any]:
        """Full course + coordinates, always from cache when present."""
        course = await self.get_course(course_id, timestamp_updated=timestamp_updated)
        payload = course.payload if isinstance(course.payload, dict) else {}
        coords: dict[str, Any] | list[Any] = {}
        coords_cached = True
        if _truthy(payload.get("hasGPS")) or _as_int(payload.get("numCoordinates")) > 0:
            try:
                coord_res = await self.get_coordinates(course_id, timestamp_updated=timestamp_updated)
                coords = coord_res.payload
                coords_cached = coord_res.cached
            except GolfAPIError:
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
        key = cache_id or _params_key(path, params)
        path_file = self.cache_dir / bucket / f"{_safe_id(key)}.json"
        entry = _read_json(path_file)
        if entry and _cache_usable(entry, ttl_days=ttl_days, newer_than=newer_than):
            return CacheResult(
                payload=entry.get("payload"),
                cached=True,
                api_requests_left=entry.get("api_requests_left") or self.last_requests_left(),
            )
        data = await self._http_get(path, params)
        left = None
        if isinstance(data, dict):
            left = data.get("apiRequestsLeft")
            if left is not None:
                _write_json(self.cache_dir / "meta.json", {"api_requests_left": str(left)})
        _write_json(
            path_file,
            {
                "cached_at": _now_iso(),
                "path": path,
                "params": params,
                "api_requests_left": left,
                "payload": data,
            },
        )
        return CacheResult(payload=data, cached=False, api_requests_left=str(left) if left is not None else None)

    async def _http_get(self, path: str, params: dict[str, Any]) -> Any:
        if not self.enabled:
            raise GolfAPIError("Golf API key is not configured")
        self.http_calls += 1
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        url = f"{self.base_url}/{path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            resp = await client.get(url, params=params or None)
        if resp.status_code >= 400:
            raise GolfAPIError(f"Golf API {resp.status_code}: {resp.text[:240]}")
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise GolfAPIError("Golf API returned non-JSON") from exc


def flatten_club_search(payload: Any) -> list[CourseHit]:
    """Clubs own courses. Search returns clubs, each with a nested courses list."""
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
            hits.append(
                CourseHit(
                    club_id=club_id,
                    club_name=club_name,
                    course_id="",
                    course_name=club_name,
                    city=_opt_str(club.get("city")),
                    state=_opt_str(club.get("state")),
                    country=_opt_str(club.get("country")),
                    address=_opt_str(club.get("address")),
                    distance=_as_float(club.get("distance")),
                    measure_unit=_opt_str(club.get("measureUnit")),
                    timestamp_updated=_as_int(club.get("timestampUpdated")) or None,
                )
            )
            continue
        for course in courses:
            if not isinstance(course, dict):
                continue
            hits.append(
                CourseHit(
                    club_id=club_id,
                    club_name=club_name,
                    course_id=str(course.get("courseID") or course.get("course_id") or ""),
                    course_name=str(course.get("courseName") or course.get("course_name") or club_name).strip(),
                    city=_opt_str(club.get("city")),
                    state=_opt_str(club.get("state")),
                    country=_opt_str(club.get("country")),
                    address=_opt_str(club.get("address")),
                    num_holes=_as_int(course.get("numHoles") or course.get("num_holes")) or None,
                    has_gps=_truthy(course.get("hasGPS") or course.get("has_gps")),
                    distance=_as_float(club.get("distance")),
                    measure_unit=_opt_str(club.get("measureUnit")),
                    timestamp_updated=_as_int(course.get("timestampUpdated")) or None,
                )
            )
    return [hit for hit in hits if hit.course_id]


def parse_coordinates(payload: Any) -> list[dict[str, Any]]:
    """Normalize Golf API coordinate payloads into {hole, kind, lat, lon, raw}."""
    rows = _coordinate_rows(payload)
    points: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        lat = _as_float(row.get("latitude") or row.get("lat"))
        lon = _as_float(row.get("longitude") or row.get("lng") or row.get("lon"))
        if lat is None or lon is None:
            continue
        hole = _as_int(row.get("hole") or row.get("holeNumber") or row.get("hole_number") or row.get("no"))
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
            value = _as_float(tee.get(f"length{i}"))
            if value is not None:
                lengths.append(value)
        tees.append(
            {
                "tee_id": tee.get("teeID"),
                "name": tee.get("teeName"),
                "color": tee.get("teeColor"),
                "lengths": lengths,
                "rating_men": _as_float(tee.get("courseRatingMen")),
                "slope_men": _as_int(tee.get("slopeMen")) or None,
                "rating_women": _as_float(tee.get("courseRatingWomen")),
                "slope_women": _as_int(tee.get("slopeWomen")) or None,
            }
        )
    return {
        "pars_men": _int_list(payload.get("parsMen")),
        "indexes_men": _int_list(payload.get("indexesMen")),
        "pars_women": _int_list(payload.get("parsWomen")),
        "indexes_women": _int_list(payload.get("indexesWomen")),
        "measure": payload.get("measure"),
        "num_holes": _as_int(payload.get("numHoles")) or None,
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


def _cache_usable(entry: dict[str, Any], *, ttl_days: int | None, newer_than: int | None) -> bool:
    payload = entry.get("payload")
    if newer_than:
        cached_ts = 0
        if isinstance(payload, dict):
            cached_ts = _as_int(payload.get("timestampUpdated"))
        if cached_ts and cached_ts < int(newer_than):
            return False
    if ttl_days is None:
        return True
    cached_at = entry.get("cached_at")
    if not cached_at:
        return False
    try:
        when = datetime.fromisoformat(str(cached_at).replace("Z", "+00:00"))  # noqa: FURB162 — 3.11 rejects trailing Z
    except ValueError:
        return False
    age = datetime.now(UTC) - when.astimezone(UTC)
    return age.total_seconds() <= ttl_days * 86400


def _params_key(path: str, params: dict[str, Any]) -> str:
    blob = path + "?" + urlencode(sorted((str(k), str(v)) for k, v in params.items()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned[:80] or "unknown"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
    tmp.replace(path)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        n = _as_int(item)
        if n:
            out.append(n)
        elif item in {0, "0"}:
            out.append(0)
    return out


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
