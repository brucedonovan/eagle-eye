"""Overpass QL client for golf-course vector priors."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from app.catalog import OSM_GOLF_TO_LAYER
from app.config import settings
from app.services.geometry import as_geom, feature_collection, to_feature


COURSE_QUERY = """
[out:json][timeout:25];
nwr["leisure"="golf_course"]({s},{w},{n},{e});
out geom;
"""

GOLF_QUERY = """
[out:json][timeout:45];
(
  nwr["leisure"="golf_course"]({s},{w},{n},{e});
  nwr["golf"]({s},{w},{n},{e});
  nwr["natural"="water"]({s},{w},{n},{e});
  nwr["natural"="sand"]({s},{w},{n},{e});
  nwr["landuse"="forest"]({s},{w},{n},{e});
  nwr["natural"="wood"]({s},{w},{n},{e});
  nwr["natural"="tree"]({s},{w},{n},{e});
  nwr["amenity"="parking"]({s},{w},{n},{e});
);
out geom;
"""

SITE_QUERY = """
[out:json][timeout:30];
(
  nwr["building"]({s},{w},{n},{e});
  nwr["natural"="tree"]({s},{w},{n},{e});
  way["highway"="service"]({s},{w},{n},{e});
  way["barrier"="fence"]({s},{w},{n},{e});
  way["man_made"="bridge"]({s},{w},{n},{e});
);
out geom;
"""


class OverpassError(RuntimeError):
    pass


def _endpoints() -> list[str]:
    urls = [settings.overpass_url, *settings.overpass_fallbacks.split(",")]
    seen: list[str] = []
    for url in urls:
        url = url.strip()
        if url and url not in seen:
            seen.append(url)
    return seen


async def find_golf_courses(
    lon: float, lat: float, query_name: str, pad_deg: float = 0.03
) -> list[dict[str, Any]]:
    """Return nearby leisure=golf_course objects, best name match first."""
    west, south, east, north = lon - pad_deg, lat - pad_deg, lon + pad_deg, lat + pad_deg
    payload = await _query(COURSE_QUERY.format(s=south, w=west, n=north, e=east))
    found: list[dict[str, Any]] = []
    for el in payload.get("elements", []):
        tags = el.get("tags") or {}
        if tags.get("leisure") != "golf_course":
            continue
        geom = element_geometry(el, {}, {})
        bounds = el.get("bounds")
        if bounds:
            minx, miny, maxx, maxy = (
                float(bounds["minlon"]),
                float(bounds["minlat"]),
                float(bounds["maxlon"]),
                float(bounds["maxlat"]),
            )
        elif geom is not None and not geom.is_empty:
            minx, miny, maxx, maxy = geom.bounds
        else:
            continue
        if geom is None or geom.is_empty:
            from shapely.geometry import box

            geom = box(minx, miny, maxx, maxy)
        found.append(
            {
                "osm_id": el.get("id"),
                "osm_type": el.get("type"),
                "name": tags.get("name") or "",
                "tags": tags,
                "geometry": geom,
                "bbox": (minx, miny, maxx, maxy),
                "score": name_score(query_name, tags.get("name") or ""),
            }
        )
    found.sort(key=lambda row: (row["score"], row["geometry"].area), reverse=True)
    return found


async def find_courses_by_name(query_name: str) -> list[dict[str, Any]]:
    """Find leisure=golf_course objects by name tokens when Nominatim misses."""
    tokens = _distinctive_name_tokens(query_name)
    if not tokens:
        return []
    clauses = "".join(f'["name"~"{_overpass_escape(tok)}",i]' for tok in tokens[:2])
    payload = await _query(
        f'[out:json][timeout:20];\nnwr["leisure"="golf_course"]{clauses};\nout geom;\n'
    )
    found: list[dict[str, Any]] = []
    for el in payload.get("elements", []):
        tags = el.get("tags") or {}
        if tags.get("leisure") != "golf_course":
            continue
        geom = element_geometry(el, {}, {})
        bounds = el.get("bounds")
        if bounds:
            bbox = (
                float(bounds["minlon"]),
                float(bounds["minlat"]),
                float(bounds["maxlon"]),
                float(bounds["maxlat"]),
            )
        elif geom is not None and not geom.is_empty:
            bbox = geom.bounds
        else:
            continue
        if geom is None or geom.is_empty:
            from shapely.geometry import box

            geom = box(*bbox)
        found.append(
            {
                "osm_id": el.get("id"),
                "osm_type": el.get("type"),
                "name": tags.get("name") or "",
                "tags": tags,
                "geometry": geom,
                "bbox": bbox,
                "score": name_score(query_name, tags.get("name") or ""),
            }
        )
    found.sort(key=lambda row: (row["score"], row["geometry"].area), reverse=True)
    return found


def _distinctive_name_tokens(query_name: str) -> list[str]:
    stop = {
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
    raw = [t for t in _tokens(query_name) if t not in stop and len(t) >= 4]
    if not raw:
        return []
    if len(raw) == 1:
        return raw
    return [raw[-1], raw[-2]]


def _overpass_escape(token: str) -> str:
    return token.replace("\\", "\\\\").replace('"', '\\"')


async def fetch_aoi(west: float, south: float, east: float, north: float) -> dict[str, Any]:
    primary = await _query(GOLF_QUERY.format(s=south, w=west, n=north, e=east))
    span = max(east - west, north - south)
    if span <= 0.05:
        try:
            extra = await _query(SITE_QUERY.format(s=south, w=west, n=north, e=east))
            primary["elements"] = [
                *primary.get("elements", []),
                *extra.get("elements", []),
            ]
        except OverpassError:
            pass
    return primary


async def _query(query: str) -> dict[str, Any]:
    headers = {"User-Agent": settings.osm_user_agent}
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=40.0, headers=headers, follow_redirects=True) as client:
        for attempt, url in enumerate(_endpoints()):
            try:
                resp = await client.post(url, data={"data": query})
                if resp.status_code in {429, 502, 503, 504}:
                    last_error = OverpassError(f"Overpass {resp.status_code} from {url}")
                    await asyncio.sleep(1.2 * (attempt + 1))
                    continue
                if resp.status_code >= 400:
                    last_error = OverpassError(f"Overpass {resp.status_code} from {url}")
                    continue
                payload = resp.json()
                if not isinstance(payload, dict) or "elements" not in payload:
                    last_error = OverpassError(f"Unexpected Overpass payload from {url}")
                    continue
                return payload
            except Exception as exc:
                last_error = OverpassError(f"{url}: {exc}")
                await asyncio.sleep(0.8)
    raise last_error or OverpassError("All Overpass endpoints failed")


def elements_to_layers(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert an Overpass JSON response into catalog FeatureCollections."""
    nodes = {
        el["id"]: (el["lon"], el["lat"])
        for el in payload.get("elements", [])
        if el.get("type") == "node" and "lon" in el
    }
    ways = {el["id"]: el for el in payload.get("elements", []) if el.get("type") == "way"}
    buckets: dict[str, list[dict[str, Any]]] = {}

    for el in payload.get("elements", []):
        tags = el.get("tags") or {}
        if not tags:
            continue
        layer_id = classify_osm(tags, el.get("type", ""))
        if not layer_id:
            continue
        geom = element_geometry(el, nodes, ways)
        if geom is None or geom.is_empty:
            continue
        props = {
            "osm_id": el.get("id"),
            "osm_type": el.get("type"),
            "name": tags.get("name"),
            "ref": tags.get("ref") or tags.get("hole"),
            "golf": tags.get("golf"),
            "source": "openstreetmap",
            **{k: v for k, v in tags.items() if k not in {"name", "ref", "golf"}},
        }
        buckets.setdefault(layer_id, []).append(to_feature(geom, props))

    return {lid: feature_collection(feats) for lid, feats in buckets.items()}


def clip_layers_to_boundary(
    layers: dict[str, dict[str, Any]], boundary, pad_deg: float = 0.0004
) -> dict[str, dict[str, Any]]:
    if boundary is None or boundary.is_empty:
        return layers
    zone = make_valid(boundary).buffer(pad_deg)
    clipped: dict[str, dict[str, Any]] = {}
    for layer_id, fc in layers.items():
        if layer_id == "boundary":
            clipped[layer_id] = feature_collection(
                [to_feature(boundary, {"source": "openstreetmap", "role": "selected_course"})]
            )
            continue
        kept = []
        for feat in fc.get("features", []):
            try:
                geom = as_geom(feat)
            except Exception:
                continue
            if geom.intersects(zone):
                kept.append(feat)
        if kept:
            clipped[layer_id] = feature_collection(kept)
    return clipped


def classify_osm(tags: dict[str, str], osm_type: str) -> str | None:
    golf = tags.get("golf")
    if golf:
        for part in golf.split(";"):
            part = part.strip()
            if part in OSM_GOLF_TO_LAYER:
                return OSM_GOLF_TO_LAYER[part]
    if tags.get("leisure") == "golf_course":
        return "boundary"
    if tags.get("amenity") == "parking":
        return "parking"
    if tags.get("natural") == "water" or tags.get("water"):
        return "water"
    if tags.get("natural") == "sand" and not golf:
        return "bunker"
    if tags.get("waterway") in {"stream", "river", "ditch"}:
        return "stream"
    if tags.get("landuse") == "forest" or tags.get("natural") == "wood":
        return "woodland"
    if tags.get("natural") == "tree":
        return "tree"
    if tags.get("barrier") == "fence":
        return "fence"
    if tags.get("man_made") == "bridge" or tags.get("bridge") in {"yes", "boardwalk"}:
        return "bridge"
    if tags.get("building"):
        name = (tags.get("name") or "").lower()
        if "club" in name or tags.get("tourism") == "hotel":
            return "clubhouse"
        if "maintenance" in name or "shed" in name:
            return "maintenance"
        return "building"
    highway = tags.get("highway")
    if highway in {"path", "footway"}:
        return "walking_path"
    if highway in {"service", "track"} and golf:
        return "cart_path"
    if highway in {"residential", "unclassified", "tertiary", "secondary"}:
        return "road"
    return None


def element_geometry(el: dict[str, Any], nodes: dict[int, tuple[float, float]], ways: dict[int, dict]):
    etype = el.get("type")
    if etype == "node":
        if "lon" in el:
            return Point(el["lon"], el["lat"])
        return None
    if etype == "way":
        return _way_geometry(el, nodes)
    if etype == "relation":
        return _relation_geometry(el, nodes, ways)
    return None


def _way_geometry(el: dict[str, Any], nodes: dict[int, tuple[float, float]]):
    coords = _coords_from_geometry_field(el.get("geometry"))
    if len(coords) < 2:
        coords = [nodes[nid] for nid in el.get("nodes", []) if nid in nodes]
    if len(coords) < 2:
        return None
    closed = coords[0] == coords[-1] and len(coords) >= 4
    polygonal = closed or (el.get("tags") or {}).get("area") == "yes"
    if polygonal and len(coords) >= 3:
        if coords[0] != coords[-1]:
            coords = [*coords, coords[0]]
        try:
            return make_valid(Polygon(coords))
        except Exception:
            return LineString(coords)
    return LineString(coords)


def _relation_geometry(el: dict[str, Any], nodes: dict[int, tuple[float, float]], ways: dict[int, dict]):
    outer_lines: list[list[tuple[float, float]]] = []
    inner_lines: list[list[tuple[float, float]]] = []
    for member in el.get("members", []):
        if member.get("type") != "way":
            continue
        coords = _coords_from_geometry_field(member.get("geometry"))
        if len(coords) < 2:
            way = ways.get(member.get("ref"))
            if way:
                coords = _coords_from_geometry_field(way.get("geometry"))
                if len(coords) < 2:
                    coords = [nodes[nid] for nid in way.get("nodes", []) if nid in nodes]
        if len(coords) < 2:
            continue
        if member.get("role") == "inner":
            inner_lines.append(coords)
        else:
            outer_lines.append(coords)
    outers = [_ring_to_polygon(ring) for ring in _merge_rings(outer_lines)]
    inners = [_ring_to_polygon(ring) for ring in _merge_rings(inner_lines)]
    outers = [g for g in outers if g is not None]
    inners = [g for g in inners if g is not None]
    if not outers:
        return None
    try:
        outer = unary_union(outers)
        if inners:
            outer = outer.difference(unary_union(inners))
        return make_valid(outer)
    except Exception:
        return outers[0]


def _merge_rings(lines: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Join unclosed ways that share endpoints into closed rings."""
    unused = [list(line) for line in lines if len(line) >= 2]
    rings: list[list[tuple[float, float]]] = []
    while unused:
        ring = unused.pop(0)
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(unused):
                other = unused[i]
                if _pts_eq(ring[-1], other[0]):
                    ring.extend(other[1:])
                elif _pts_eq(ring[-1], other[-1]):
                    ring.extend(reversed(other[:-1]))
                elif _pts_eq(ring[0], other[-1]):
                    ring = other + ring[1:]
                elif _pts_eq(ring[0], other[0]):
                    ring = list(reversed(other[1:])) + ring
                else:
                    i += 1
                    continue
                unused.pop(i)
                changed = True
        if not _pts_eq(ring[0], ring[-1]):
            ring.append(ring[0])
        if len(ring) >= 4:
            rings.append(ring)
    return rings


def _ring_to_polygon(ring: list[tuple[float, float]]) -> Polygon | None:
    try:
        poly = Polygon(ring)
        return make_valid(poly) if not poly.is_empty else None
    except Exception:
        return None


def _pts_eq(a: tuple[float, float], b: tuple[float, float], tol: float = 1e-10) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _coords_from_geometry_field(geom: Any) -> list[tuple[float, float]]:
    if not geom:
        return []
    coords: list[tuple[float, float]] = []
    for pt in geom:
        if isinstance(pt, dict) and "lon" in pt:
            coords.append((float(pt["lon"]), float(pt["lat"])))
    return coords


def feature_bounds(layers: dict[str, dict[str, Any]]) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for fc in layers.values():
        if not fc:
            continue
        for feat in fc.get("features", []):
            geom = feat.get("geometry") or {}
            _collect_coords(geom.get("coordinates"), xs, ys)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _collect_coords(coords: Any, xs: list[float], ys: list[float]) -> None:
    if not coords:
        return
    if isinstance(coords[0], (int, float)) and len(coords) >= 2:
        xs.append(float(coords[0]))
        ys.append(float(coords[1]))
        return
    for item in coords:
        _collect_coords(item, xs, ys)


def name_score(query: str, candidate: str) -> float:
    stop = {"golf", "course", "links", "the", "club", "national", "of", "de", "do", "da"}
    q = {t for t in _tokens(query) if t not in stop}
    c = {t for t in _tokens(candidate) if t not in stop}
    if not q or not c:
        return 0.0
    return len(q & c) / len(q | c)


def _tokens(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [part for part in cleaned.split() if part]
