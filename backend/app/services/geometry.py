"""Geometry helpers used by vectorization, topology, and navigation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid

# ~0.25 m / ~0.12 m at mid-latitudes. Keep bunker lips and green edges.
POLYGON_SIMPLIFY_DEG = 0.0000025
LINE_SIMPLIFY_DEG = 0.0000012


def as_geom(obj: Any) -> BaseGeometry:
    if isinstance(obj, BaseGeometry):
        return obj
    if isinstance(obj, dict):
        geom = obj.get("geometry", obj)
        return shape(geom)
    raise TypeError(f"Cannot convert {type(obj)} to geometry")


def clean_polygon(geom: BaseGeometry, min_area_m2: float = 4.0) -> BaseGeometry | None:
    """Repair, simplify, and drop slivers. Area filter uses degrees² if unprojected."""
    if geom is None or geom.is_empty:
        return None
    repaired = make_valid(geom)
    if repaired.is_empty:
        return None
    simplified = repaired.simplify(POLYGON_SIMPLIFY_DEG, preserve_topology=True)
    if simplified.is_empty:
        return None
    # Geographic coords: ~1e-9 deg² ≈ 10 m² near mid-latitudes; keep tiny tees/bunkers.
    if simplified.area > 0 and simplified.area < 1e-12:
        return None
    if min_area_m2 and simplified.geom_type in {"Polygon", "MultiPolygon"}:
        # Caller may pass projected geometries; for WGS84 we only drop near-zero slivers.
        pass
    return simplified


def to_feature(geom: BaseGeometry, properties: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": properties or {},
    }


def feature_collection(features: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(features)}


def geoms_from_fc(fc: dict[str, Any] | None) -> list[BaseGeometry]:
    if not fc:
        return []
    out: list[BaseGeometry] = []
    for feat in fc.get("features", []):
        try:
            geom = as_geom(feat)
        except Exception:
            continue
        if not geom.is_empty:
            out.append(geom)
    return out


def union_layer(fc: dict[str, Any] | None) -> BaseGeometry | None:
    geoms = [g for g in geoms_from_fc(fc) if not g.is_empty]
    if not geoms:
        return None
    return unary_union(geoms)


def translate_meters(geom: BaseGeometry, east_m: float, north_m: float) -> BaseGeometry:
    """Shift a WGS84 geometry by metres (east, north)."""
    from shapely import affinity

    lat = float(geom.centroid.y)
    dlon = east_m / (111_320.0 * max(0.2, math.cos(math.radians(lat))))
    dlat = north_m / 111_320.0
    return affinity.translate(geom, xoff=dlon, yoff=dlat)


def buffer_meters(geom: BaseGeometry, meters: float) -> BaseGeometry:
    """Buffer a WGS84 geometry by metres via Web Mercator."""
    from pyproj import Transformer
    from shapely.ops import transform

    to_m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    return transform(to_ll, transform(to_m, geom).buffer(meters))


def distance_meters(a: BaseGeometry, b: BaseGeometry) -> float:
    return a.distance(b) * 111_320


def difference_safe(a: BaseGeometry, b: BaseGeometry) -> BaseGeometry:
    try:
        return make_valid(a.difference(b))
    except Exception:
        return a


def area_m2(geom: BaseGeometry) -> float:
    """Geodesic area in square metres (WGS84)."""
    if geom is None or geom.is_empty:
        return 0.0
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    total = 0.0
    geoms = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for part in geoms:
        if part.is_empty or part.geom_type not in {"Polygon", "MultiPolygon"}:
            continue
        try:
            area, _perim = geod.geometry_area_perimeter(part)
        except Exception:
            continue
        total += abs(area)
    return total


def iou(a: BaseGeometry, b: BaseGeometry) -> float:
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    try:
        inter = a.intersection(b).area
        union = a.union(b).area
    except Exception:
        return 0.0
    if union <= 0:
        return 0.0
    return float(inter / union)


def as_polygons(geom: BaseGeometry | None) -> list[BaseGeometry]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return [g for g in geom.geoms if not g.is_empty]
    if geom.geom_type == "GeometryCollection":
        out: list[BaseGeometry] = []
        for part in geom.geoms:
            out.extend(as_polygons(part))
        return out
    return []


def bbox_polygon(west: float, south: float, east: float, north: float) -> BaseGeometry:
    from shapely.geometry import box

    return box(west, south, east, north)


def expand_bbox(
    west: float, south: float, east: float, north: float, pad_deg: float = 0.002
) -> tuple[float, float, float, float]:
    return west - pad_deg, south - pad_deg, east + pad_deg, north + pad_deg


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    import math

    lat = min(max(lat, -85.05112878), 85.05112878)
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y
