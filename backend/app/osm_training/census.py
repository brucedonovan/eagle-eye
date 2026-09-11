"""Tiled Overpass census of OSM golf courses and golf=* priors."""

from __future__ import annotations

import asyncio
import csv
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from app.config import settings
from app.osm_training.scoring import CompletenessResult, FeatureCounts, completeness_score
from app.services.overpass import OverpassError, _endpoints, classify_osm

CENSUS_TIMEOUT_S = 90
HTTP_TIMEOUT_S = 120.0
MAX_ASSIGN_DEG = 0.03
HASH_DEG = 0.05
GOLF_TAG_RE = (
    "^(green|fairway|tee|bunker|sand_bunker|hole|cartpath|path|"
    "water_hazard|lateral_water_hazard|pin|rough|fringe|collar|"
    "driving_range|putting_green|practice)$"
)

CENSUS_QUERY = """
[out:json][timeout:{timeout}];
(
  nwr["leisure"="golf_course"]({s},{w},{n},{e});
  nwr["golf"~"{golf_re}"]({s},{w},{n},{e});
);
out tags center;
"""

# Land-biased regions where golf is actually mapped. `--global` unions these.
GOLF_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "lisbon": (-9.55, 38.60, -9.05, 38.90),
    "portugal": (-9.70, 36.90, -6.10, 42.20),
    "iberia": (-10.00, 35.50, 4.00, 44.00),
    "uk_ireland": (-11.00, 49.00, 2.20, 61.00),
    "france": (-5.50, 42.00, 8.50, 51.50),
    "central_europe": (4.00, 45.00, 25.00, 55.50),
    "nordics": (4.00, 54.50, 32.00, 71.00),
    "eastern_europe": (20.00, 41.00, 42.00, 56.00),
    "usa_west": (-125.00, 32.00, -102.00, 49.50),
    "usa_central": (-102.00, 25.50, -84.00, 49.50),
    "usa_east": (-84.00, 24.00, -66.50, 47.50),
    "canada": (-140.00, 42.00, -52.00, 60.00),
    "mexico_caribbean": (-118.00, 14.00, -60.00, 32.50),
    "hawaii": (-161.00, 18.50, -154.50, 22.50),
    "japan": (128.00, 30.00, 146.00, 46.00),
    "korea": (124.50, 33.00, 132.00, 39.00),
    "china_east": (110.00, 18.00, 123.00, 41.00),
    "se_asia": (99.00, -9.00, 121.00, 22.00),
    "australia": (113.00, -45.00, 154.00, -10.00),
    "new_zealand": (166.00, -47.50, 179.00, -34.00),
    "southern_africa": (16.00, -35.00, 33.00, -22.00),
    "south_america": (-78.00, -42.00, -34.00, 5.00),
    "india": (68.00, 8.00, 89.00, 35.00),
    "middle_east": (32.00, 22.00, 60.00, 38.00),
}

REGION_GROUPS: dict[str, tuple[str, ...]] = {
    "europe": (
        "iberia",
        "uk_ireland",
        "france",
        "central_europe",
        "nordics",
        "eastern_europe",
    ),
    "north_america": (
        "usa_west",
        "usa_central",
        "usa_east",
        "canada",
        "mexico_caribbean",
        "hawaii",
    ),
    "asia_pacific": (
        "japan",
        "korea",
        "china_east",
        "se_asia",
        "australia",
        "new_zealand",
        "india",
    ),
    "global": tuple(name for name in GOLF_REGIONS if name not in {"lisbon", "portugal"}),
}

CSV_FIELDS = [
    "rank",
    "score",
    "osm_type",
    "osm_id",
    "name",
    "lon",
    "lat",
    "west",
    "south",
    "east",
    "north",
    "expected_holes",
    "green",
    "fairway",
    "tee",
    "bunker",
    "hole",
    "cart_path",
    "water",
    "pin",
    "rough",
    "fringe",
    "driving_range",
    "putting_green",
    "flags",
    "holes_tag",
    "website",
]


@dataclass
class CourseRecord:
    osm_id: int
    osm_type: str
    name: str
    lon: float
    lat: float
    west: float | None = None
    south: float | None = None
    east: float | None = None
    north: float | None = None
    tags: dict[str, str] = field(default_factory=dict)
    counts: FeatureCounts = field(default_factory=FeatureCounts)
    score: float = 0.0
    expected_holes: int = 18
    flags: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int]:
        return self.osm_type, self.osm_id

    def apply_score(self, result: CompletenessResult) -> None:
        self.score = result.score
        self.expected_holes = result.expected_holes
        self.flags = result.flags
        self.components = result.components

    def to_csv_row(self, rank: int) -> dict[str, Any]:
        counts = self.counts.as_dict()
        return {
            "rank": rank,
            "score": f"{self.score:.3f}",
            "osm_type": self.osm_type,
            "osm_id": self.osm_id,
            "name": self.name,
            "lon": f"{self.lon:.6f}",
            "lat": f"{self.lat:.6f}",
            "west": "" if self.west is None else f"{self.west:.6f}",
            "south": "" if self.south is None else f"{self.south:.6f}",
            "east": "" if self.east is None else f"{self.east:.6f}",
            "north": "" if self.north is None else f"{self.north:.6f}",
            "expected_holes": self.expected_holes,
            "green": counts["green"],
            "fairway": counts["fairway"],
            "tee": counts["tee"],
            "bunker": counts["bunker"],
            "hole": counts["hole"],
            "cart_path": counts["cart_path"],
            "water": counts["water"],
            "pin": counts["pin"],
            "rough": counts["rough"],
            "fringe": counts["fringe"],
            "driving_range": counts["driving_range"],
            "putting_green": counts["putting_green"],
            "flags": "|".join(self.flags),
            "holes_tag": self.tags.get("holes") or self.tags.get("golf:holes") or "",
            "website": self.tags.get("website") or self.tags.get("contact:website") or "",
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "osm_id": self.osm_id,
            "osm_type": self.osm_type,
            "name": self.name,
            "lon": self.lon,
            "lat": self.lat,
            "bbox": [self.west, self.south, self.east, self.north],
            "tags": self.tags,
            "counts": self.counts.as_dict(),
            "score": self.score,
            "expected_holes": self.expected_holes,
            "flags": self.flags,
            "components": self.components,
        }


@dataclass
class GolfFeature:
    osm_id: int
    osm_type: str
    layer_id: str
    lon: float
    lat: float
    polygonal: bool = False


ProgressFn = Callable[[str], None]


def resolve_bboxes(
    *,
    bbox: tuple[float, float, float, float] | None = None,
    region: str | None = None,
    global_scan: bool = False,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    if bbox is not None:
        return [("bbox", bbox)]
    if global_scan:
        region = "global"
    if not region:
        raise ValueError("Provide --bbox, --region, or --global")
    key = region.strip().lower()
    if key in GOLF_REGIONS:
        return [(key, GOLF_REGIONS[key])]
    members = REGION_GROUPS.get(key)
    if not members:
        known = ", ".join([*sorted(GOLF_REGIONS), *sorted(REGION_GROUPS)])
        raise ValueError(f"Unknown region {region!r}. Known: {known}")
    return [(name, GOLF_REGIONS[name]) for name in members if name in GOLF_REGIONS]


def iter_tiles(
    west: float, south: float, east: float, north: float, step: float = 8.0
) -> list[tuple[float, float, float, float]]:
    tiles: list[tuple[float, float, float, float]] = []
    lat = south
    while lat < north - 1e-12:
        lon = west
        lat2 = min(lat + step, north)
        while lon < east - 1e-12:
            lon2 = min(lon + step, east)
            tiles.append((lon, lat, lon2, lat2))
            lon = lon2
        lat = lat2
    return tiles


async def rank_courses(
    *,
    bboxes: list[tuple[str, tuple[float, float, float, float]]],
    cache_dir: Path,
    step_deg: float = 8.0,
    delay_s: float = 1.0,
    progress: ProgressFn | None = None,
    query_fn: Callable[..., Any] | None = None,
    out_dir: Path | None = None,
) -> list[CourseRecord]:
    """Download tiled OSM golf features, join them to courses, and rank.

    When ``out_dir`` is set, ranking.csv / ranking.jsonl / progress.json are
    rewritten after every tile so a killed process still leaves a usable
    snapshot. The final pass re-assigns every feature against the full course
    set (more accurate at region boundaries).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    query = query_fn or query_overpass
    courses: dict[tuple[str, int], CourseRecord] = {}
    features: list[GolfFeature] = []
    tiles: list[tuple[float, float, float, float]] = []
    seen_tiles: set[tuple[float, float, float, float]] = set()
    for _name, bbox in bboxes:
        for tile in iter_tiles(*bbox, step=step_deg):
            key = tuple(round(v, 5) for v in tile)
            if key in seen_tiles:
                continue
            seen_tiles.add(key)
            tiles.append(tile)

    log = progress or (lambda _msg: None)
    log(f"Census: {len(tiles)} Overpass tiles (step={step_deg}°)")
    assigned = 0
    pending: list[GolfFeature] = []
    ranked: list[CourseRecord] = []
    dirty = False
    finished = False
    tile_index = 0

    def snapshot(*, partial: bool) -> list[CourseRecord]:
        nonlocal assigned, pending
        pending.extend(features[assigned:])
        assigned = len(features)
        if not partial:
            for course in courses.values():
                course.counts = FeatureCounts()
            assign_features(courses, features)
            pending = []
        elif pending:
            pending = _assign_returning_unmatched(courses, pending)
        rows = _scored_courses(courses)
        _write_rank_checkpoint(
            rows,
            out_dir,
            tile=tile_index,
            tiles=len(tiles),
            features=len(features),
            partial=partial,
            log=log,
        )
        return rows

    try:
        for tile_index, tile in enumerate(tiles, start=1):
            payload = await _fetch_tile_adaptive(
                tile, cache_dir=cache_dir, query=query, delay_s=delay_s, depth=0
            )
            _ingest_payload(payload, courses, features)
            dirty = True
            if out_dir is not None:
                ranked = snapshot(partial=True)
            if tile_index == 1 or tile_index == len(tiles) or tile_index % 5 == 0:
                log(
                    f"  tile {tile_index}/{len(tiles)} courses={len(courses)} "
                    f"features={len(features)}"
                )
            if delay_s and tile_index < len(tiles):
                await asyncio.sleep(delay_s)
        if out_dir is not None:
            ranked = snapshot(partial=False)
        else:
            assign_features(courses, features)
            ranked = _scored_courses(courses)
        dirty = False
        finished = True
        log(f"Ranked {len(ranked)} courses")
        return ranked
    finally:
        if dirty and not finished and courses and out_dir is not None:
            for course in courses.values():
                course.counts = FeatureCounts()
            assign_features(courses, features)
            _write_rank_checkpoint(
                _scored_courses(courses),
                out_dir,
                tile=tile_index,
                tiles=len(tiles),
                features=len(features),
                partial=True,
                log=log,
            )


def assign_features(
    courses: dict[tuple[str, int], CourseRecord], features: list[GolfFeature]
) -> None:
    """Join golf=* elements to the nearest containing / nearby course."""
    _assign_returning_unmatched(courses, features)


def write_ranking_csv(records: list[CourseRecord], path: Path) -> None:
    def _write(dest: Path) -> None:
        with dest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for rank, row in enumerate(records, start=1):
                writer.writerow(row.to_csv_row(rank))

    _replace_atomically(path, _write)


def write_ranking_jsonl(records: list[CourseRecord], path: Path) -> None:
    def _write(dest: Path) -> None:
        with dest.open("w", encoding="utf-8") as handle:
            for rank, row in enumerate(records, start=1):
                payload = row.to_json()
                payload["rank"] = rank
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    _replace_atomically(path, _write)


def _assign_returning_unmatched(
    courses: dict[tuple[str, int], CourseRecord], features: list[GolfFeature]
) -> list[GolfFeature]:
    if not courses:
        return list(features)
    buckets: dict[tuple[int, int], list[CourseRecord]] = {}
    for course in courses.values():
        for cell in _course_cells(course):
            buckets.setdefault(cell, []).append(course)

    unmatched: list[GolfFeature] = []
    for feat in features:
        course = _nearest_course(feat, buckets)
        if course is None:
            unmatched.append(feat)
            continue
        _bump(course.counts, feat.layer_id, polygonal=feat.polygonal)
    return unmatched


def _scored_courses(courses: dict[tuple[str, int], CourseRecord]) -> list[CourseRecord]:
    ranked = list(courses.values())
    for course in ranked:
        course.apply_score(completeness_score(course.counts, course.tags))
    ranked.sort(key=lambda row: (row.score, row.counts.green, row.name.lower()), reverse=True)
    return ranked


def _write_rank_checkpoint(
    records: list[CourseRecord],
    out_dir: Path | None,
    *,
    tile: int,
    tiles: int,
    features: int,
    partial: bool,
    log: ProgressFn,
) -> None:
    if out_dir is None:
        return
    try:
        write_ranking_csv(records, out_dir / "ranking.csv")
        write_ranking_jsonl(records, out_dir / "ranking.jsonl")
        perfect = sum(1 for row in records if row.score >= 100)
        progress = {
            "status": "partial" if partial else "complete",
            "tile": tile,
            "tiles": tiles,
            "courses": len(records),
            "features": features,
            "score_100": perfect,
            "top": [
                {
                    "rank": rank,
                    "score": row.score,
                    "name": row.name or f"{row.osm_type}/{row.osm_id}",
                }
                for rank, row in enumerate(records[:10], start=1)
            ],
        }

        def _write_progress(dest: Path) -> None:
            dest.write_text(json.dumps(progress, indent=2), encoding="utf-8")

        _replace_atomically(out_dir / "progress.json", _write_progress)
    except OSError as exc:
        log(f"  checkpoint write failed: {exc}")


def _replace_atomically(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        writer(tmp)
        with tmp.open("rb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def load_ranking_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


async def query_overpass(query: str, *, timeout: float = HTTP_TIMEOUT_S) -> dict[str, Any]:
    headers = {"User-Agent": settings.osm_user_agent}
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        for attempt, url in enumerate(_endpoints()):
            try:
                resp = await client.post(url, data={"data": query})
                if resp.status_code in {429, 502, 503, 504}:
                    last_error = OverpassError(f"Overpass {resp.status_code} from {url}")
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                if resp.status_code >= 400:
                    last_error = OverpassError(f"Overpass {resp.status_code} from {url}")
                    continue
                payload = resp.json()
                if not isinstance(payload, dict) or "elements" not in payload:
                    last_error = OverpassError(f"Unexpected Overpass payload from {url}")
                    continue
                remark = str(payload.get("remark") or "")
                if "timed out" in remark.lower() or "out of memory" in remark.lower():
                    raise OverpassError(remark)
                return payload
            except OverpassError:
                raise
            except (httpx.HTTPError, ValueError, TypeError, OSError) as exc:
                last_error = OverpassError(f"{url}: {exc}")
                await asyncio.sleep(0.8)
    raise last_error or OverpassError("All Overpass endpoints failed")


def _ingest_payload(
    payload: dict[str, Any],
    courses: dict[tuple[str, int], CourseRecord],
    features: list[GolfFeature],
) -> None:
    for el in payload.get("elements") or []:
        tags = {str(k): str(v) for k, v in (el.get("tags") or {}).items()}
        if not tags:
            continue
        lon, lat = _element_center(el)
        if lon is None or lat is None:
            continue
        osm_type = str(el.get("type") or "")
        osm_id = int(el.get("id") or 0)
        if tags.get("leisure") == "golf_course":
            key = (osm_type, osm_id)
            if key not in courses:
                west, south, east, north = _element_bounds(el)
                courses[key] = CourseRecord(
                    osm_id=osm_id,
                    osm_type=osm_type,
                    name=(tags.get("name") or "").strip(),
                    lon=lon,
                    lat=lat,
                    west=west,
                    south=south,
                    east=east,
                    north=north,
                    tags=tags,
                )
            continue
        layer_id = classify_osm(tags, osm_type)
        if not layer_id or layer_id == "boundary":
            continue
        features.append(
            GolfFeature(
                osm_id=osm_id,
                osm_type=osm_type,
                layer_id=layer_id,
                lon=lon,
                lat=lat,
                polygonal=osm_type in {"way", "relation"},
            )
        )


def _bump(counts: FeatureCounts, layer_id: str, *, polygonal: bool) -> None:
    mapping = {
        "green": "green",
        "fairway": "fairway",
        "tee": "tee",
        "bunker": "bunker",
        "hole_centerline": "hole",
        "cart_path": "cart_path",
        "water": "water",
        "lake": "water",
        "pin": "pin",
        "managed_rough": "rough",
        "natural_rough": "rough",
        "green_fringe": "fringe",
        "driving_range": "driving_range",
        "putting_green": "putting_green",
    }
    field_name = mapping.get(layer_id)
    if not field_name:
        return
    setattr(counts, field_name, getattr(counts, field_name) + 1)
    if polygonal and field_name in {"green", "fairway", "tee", "hole"}:
        way_field = {"green": "green_ways", "fairway": "fairway_ways", "tee": "tee_ways", "hole": "hole_ways"}[
            field_name
        ]
        setattr(counts, way_field, getattr(counts, way_field) + 1)


def _nearest_course(
    feat: GolfFeature, buckets: dict[tuple[int, int], list[CourseRecord]]
) -> CourseRecord | None:
    cx, cy = _cell(feat.lon, feat.lat)
    candidates: list[CourseRecord] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            candidates.extend(buckets.get((cx + dx, cy + dy), ()))
    if not candidates:
        return None
    contained = [c for c in candidates if _contains(c, feat.lon, feat.lat)]
    pool = contained or candidates
    best: CourseRecord | None = None
    best_d = MAX_ASSIGN_DEG**2
    if contained:
        best_d = 1e9
    for course in pool:
        d = (course.lon - feat.lon) ** 2 + (course.lat - feat.lat) ** 2
        limit = best_d if contained else MAX_ASSIGN_DEG**2
        if d <= limit and (best is None or d < best_d):
            best = course
            best_d = d
    return best


def _contains(course: CourseRecord, lon: float, lat: float) -> bool:
    if None in (course.west, course.south, course.east, course.north):
        return False
    return course.west <= lon <= course.east and course.south <= lat <= course.north


def _cell(lon: float, lat: float) -> tuple[int, int]:
    return int(lon / HASH_DEG), int(lat / HASH_DEG)


def _course_cells(course: CourseRecord) -> set[tuple[int, int]]:
    cells = {_cell(course.lon, course.lat)}
    if None in (course.west, course.south, course.east, course.north):
        return cells
    x0, y0 = _cell(course.west, course.south)
    x1, y1 = _cell(course.east, course.north)
    for x in range(min(x0, x1), max(x0, x1) + 1):
        for y in range(min(y0, y1), max(y0, y1) + 1):
            cells.add((x, y))
    return cells


def _element_center(el: dict[str, Any]) -> tuple[float | None, float | None]:
    if "lon" in el and "lat" in el:
        return float(el["lon"]), float(el["lat"])
    center = el.get("center") or {}
    if "lon" in center and "lat" in center:
        return float(center["lon"]), float(center["lat"])
    bounds = el.get("bounds") or {}
    if {"minlon", "maxlon", "minlat", "maxlat"} <= bounds.keys():
        lon = (float(bounds["minlon"]) + float(bounds["maxlon"])) / 2.0
        lat = (float(bounds["minlat"]) + float(bounds["maxlat"])) / 2.0
        return lon, lat
    return None, None


def _element_bounds(
    el: dict[str, Any],
) -> tuple[float | None, float | None, float | None, float | None]:
    bounds = el.get("bounds") or {}
    if {"minlon", "maxlon", "minlat", "maxlat"} <= bounds.keys():
        return (
            float(bounds["minlon"]),
            float(bounds["minlat"]),
            float(bounds["maxlon"]),
            float(bounds["maxlat"]),
        )
    return None, None, None, None


def _tile_cache_path(cache_dir: Path, tile: tuple[float, float, float, float]) -> Path:
    west, south, east, north = (f"{v:.4f}" for v in tile)
    return cache_dir / f"tile_{west}_{south}_{east}_{north}.json"


async def _fetch_tile_adaptive(
    tile: tuple[float, float, float, float],
    *,
    cache_dir: Path,
    query: Callable[..., Any],
    delay_s: float,
    depth: int,
) -> dict[str, Any]:
    cached = _tile_cache_path(cache_dir, tile)
    if cached.exists():
        try:
            payload = json.loads(cached.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "elements" in payload:
                return payload
        except json.JSONDecodeError:
            cached.unlink(missing_ok=True)

    west, south, east, north = tile
    ql = CENSUS_QUERY.format(
        timeout=CENSUS_TIMEOUT_S,
        s=south,
        w=west,
        n=north,
        e=east,
        golf_re=GOLF_TAG_RE,
    )
    try:
        payload = await query(ql)
        cached.write_text(json.dumps(payload), encoding="utf-8")
        return payload
    except OverpassError:
        if depth >= 3:
            raise
        mid_lon = (west + east) / 2.0
        mid_lat = (south + north) / 2.0
        parts = [
            (west, south, mid_lon, mid_lat),
            (mid_lon, south, east, mid_lat),
            (west, mid_lat, mid_lon, north),
            (mid_lon, mid_lat, east, north),
        ]
        merged: list[dict[str, Any]] = []
        for part in parts:
            if delay_s:
                await asyncio.sleep(delay_s)
            merged.append(
                await _fetch_tile_adaptive(
                    part, cache_dir=cache_dir, query=query, delay_s=delay_s, depth=depth + 1
                )
            )
        payload = {"elements": [el for part in merged for el in (part.get("elements") or [])]}
        cached.write_text(json.dumps(payload), encoding="utf-8")
        return payload
