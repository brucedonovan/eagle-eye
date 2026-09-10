"""CPU imagery analysis fused with OSM priors."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid

from app.services.geometry import (
    area_m2,
    as_geom,
    as_polygons,
    buffer_meters,
    clean_polygon,
    compactness as _compactness,
    difference_safe,
    distance_meters,
    feature_collection,
    geoms_from_fc,
    iou,
    to_feature,
    translate_meters,
    union_layer,
)

FUSION_LAYERS = ("green", "fairway", "bunker", "tee", "water")
PRIOR_LAYERS = ("green", "tee", "bunker", "fairway", "water", "cart_path")

GREEN_AREA_M2 = (150.0, 2500.0)
PIN_GREEN_AREA_M2 = (80.0, 2800.0)
BUNKER_AREA_M2 = (28.0, 1800.0)
FAIRWAY_AREA_M2 = (180.0, 90_000.0)
WATER_AREA_M2 = (250.0, 250_000.0)
TEE_AREA_M2 = (12.0, 450.0)

# Substantial overlap to treat OSM + imagery as the same object.
MATCH_IOU = 0.22
MATCH_OVERLAP = 0.30
TIGHT_IOU = 0.42
REFINE_IOU = 0.18
REFINE_MIN_OVERLAP = 0.30
REFINE_MAX_GROW = 1.10
REFINE_CENTROID_M = 16.0
REFINE_PIN_M = 8.0
FRINGE_WIDTH_M = 3.4
TREE_CROWN_M2 = (16.0, 320.0)
TREE_MIN_COMPACT = 0.32
TREE_DEDUP_M = 8.0
# Tees / greens / fairways are OSM or hole-graph derived. Imagery may only
# fill bunkers and water when both mosaics agree. Greens are refined separately.
IMAGERY_ADD_LAYERS = frozenset({"bunker", "water"})
GREEN_MIN_COMPACT = 0.28
BUNKER_MIN_COMPACT = 0.22
BUNKER_MAX_ELONGATION = 5.5
PLAY_NEAR_M = {"green": 42.0, "tee": 48.0, "bunker": 55.0, "fairway": 32.0, "water": 90.0}


@dataclass
class MosaicGeo:
    """Pixel ↔ lon/lat for an XYZ mosaic (Web Mercator) or a simple bbox affine."""

    width: int
    height: int
    west: float
    south: float
    east: float
    north: float
    zoom: int | None = None
    tile_x_min: int | None = None
    tile_y_min: int | None = None
    tile_size: int = 256

    @classmethod
    def from_imagery(cls, imagery: dict[str, Any], rgb: np.ndarray) -> MosaicGeo:
        h, w = rgb.shape[:2]
        zoom = imagery.get("zoom")
        tile_size = int(imagery.get("tile_size") or 256)
        bbox = imagery.get("mosaic_bbox") or imagery.get("bbox")
        if not bbox or len(bbox) != 4:
            raise ValueError("Imagery dict has no bbox")
        west, south, east, north = (float(v) for v in bbox)
        x_min = imagery.get("tile_x_min")
        y_min = imagery.get("tile_y_min")
        # Only snap to the XYZ tile grid when we actually fetched tiles.
        # Google Static / bbox crops keep their affine extent.
        return cls(
            width=int(imagery.get("width") or w),
            height=int(imagery.get("height") or h),
            west=west,
            south=south,
            east=east,
            north=north,
            zoom=int(zoom) if zoom is not None else None,
            tile_x_min=int(x_min) if x_min is not None else None,
            tile_y_min=int(y_min) if y_min is not None else None,
            tile_size=tile_size,
        )

    @classmethod
    def from_bbox(
        cls, width: int, height: int, bbox: tuple[float, float, float, float]
    ) -> MosaicGeo:
        west, south, east, north = bbox
        return cls(width=width, height=height, west=west, south=south, east=east, north=north)

    @property
    def meters_per_pixel(self) -> float:
        lat = (self.south + self.north) / 2.0
        if self.zoom is not None:
            return 156543.03392 * math.cos(math.radians(lat)) / (2 ** self.zoom)
        height_m = max(1e-9, (self.north - self.south) * 111_320.0)
        return height_m / max(self.height, 1)

    def pixel_to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        if self.zoom is not None and self.tile_x_min is not None and self.tile_y_min is not None:
            n = 2 ** self.zoom
            fx = (self.tile_x_min + x / self.tile_size) / n
            fy = (self.tile_y_min + y / self.tile_size) / n
            lon = fx * 360.0 - 180.0
            lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * fy))))
            return lon, lat
        lon = self.west + (x / max(self.width, 1)) * (self.east - self.west)
        lat = self.north - (y / max(self.height, 1)) * (self.north - self.south)
        return lon, lat

    def lonlat_to_pixel(self, lon: float, lat: float) -> tuple[float, float]:
        if self.zoom is not None and self.tile_x_min is not None and self.tile_y_min is not None:
            lat = min(max(lat, -85.05112878), 85.05112878)
            n = 2 ** self.zoom
            fx = (lon + 180.0) / 360.0 * n
            lat_rad = math.radians(lat)
            fy = (
                (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
            )
            return (fx - self.tile_x_min) * self.tile_size, (fy - self.tile_y_min) * self.tile_size
        x = (lon - self.west) / max(self.east - self.west, 1e-12) * self.width
        y = (self.north - lat) / max(self.north - self.south, 1e-12) * self.height
        return x, y

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "bbox": [self.west, self.south, self.east, self.north],
            "zoom": self.zoom,
            "tile_x_min": self.tile_x_min,
            "tile_y_min": self.tile_y_min,
            "tile_size": self.tile_size,
            "gsd_m": round(self.meters_per_pixel, 4),
        }


@dataclass
class ClassMasks:
    water: np.ndarray
    bunker: np.ndarray
    green: np.ndarray
    fairway: np.ndarray
    tee: np.ndarray
    soft_grass: np.ndarray
    veg: np.ndarray
    hsv: np.ndarray
    lab: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "water": self.water,
            "bunker": self.bunker,
            "green": self.green,
            "fairway": self.fairway,
            "tee": self.tee,
            "soft_grass": self.soft_grass,
        }


@dataclass
class AnalysisResult:
    polygons: dict[str, list[tuple[BaseGeometry, dict[str, Any]]]] = field(default_factory=dict)
    masks: ClassMasks | None = None
    geo: MosaicGeo | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    backend: str = "imagery_fusion"
    dual_source: bool = False
    fusion_stats: dict[str, Any] = field(default_factory=dict)
    pin_greens: list[dict[str, Any]] = field(default_factory=list)


def vegetation_index(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    return (g - r) / (g + r + 1e-6)


def _cv2():
    import cv2

    return cv2


def classify_pixels(
    rgb: np.ndarray,
    priors: dict[str, np.ndarray] | None = None,
    boundary: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
) -> ClassMasks:
    """Score pixels into golf classes. OSM priors boost; they do not replace the image."""
    cv2 = _cv2()
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    L, a_ch = lab[:, :, 0], lab[:, :, 1]
    veg = vegetation_index(rgb)
    g = rgb[:, :, 1].astype(np.int16)
    r = rgb[:, :, 0].astype(np.int16)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    tex = cv2.GaussianBlur((gray.astype(np.float32) - blur) ** 2, (7, 7), 0)

    greenness = (128.0 - a_ch.astype(np.float32)) / 128.0  # 0..~1, higher = greener
    priors = priors or {}

    water = (v < 20) & (tex < 70) & (L < 32)
    if "water" in priors:
        water |= priors["water"] & (v < 55) & (tex < 160)

    bunker = (v > 165) & (veg < 0.0) & (h >= 8) & (h < 36) & (s > 35) & (r > 150) & (L > 150) & ~water
    if "bunker" in priors:
        bunker |= priors["bunker"] & (v > 140) & (veg < 0.06) & ~water

    green = (
        (veg > 0.175)
        & (h >= 48)
        & (h <= 85)
        & (s > 65)
        & (v > 75)
        & (v < 190)
        & (g > r + 20)
        & (greenness > 0.04)
        & ~water
        & ~bunker
    )
    if "green" in priors:
        green |= priors["green"] & (veg > 0.08) & (g > r + 8) & ~water & ~bunker

    tee = green & (tex < np.percentile(tex, 55) if tex.size else True)
    if "tee" in priors:
        tee |= priors["tee"] & (veg > 0.06) & ~water & ~bunker

    soft_grass = (
        (v > 80)
        & (v < 205)
        & (h >= 20)
        & (h <= 90)
        & (g + 8 >= r)
        & ~water
        & ~bunker
    )
    fairway = soft_grass & ~green & ((veg > 0.03) | ((h >= 30) & (h <= 75) & (v > 100)))
    if "fairway" in priors:
        fairway |= priors["fairway"] & soft_grass

    if exclude is not None:
        water &= ~exclude
        bunker &= ~exclude
        green &= ~exclude
        fairway &= ~exclude
        tee &= ~exclude
        soft_grass &= ~exclude
    if boundary is not None:
        water &= boundary
        bunker &= boundary
        green &= boundary
        fairway &= boundary
        tee &= boundary
        soft_grass &= boundary

    # Cart paths look like thin tan strips — drop them from bunker.
    if "cart_path" in priors:
        bunker &= ~priors["cart_path"]

    return ClassMasks(
        water=_morph(water, open_k=3, close_k=9),
        bunker=_morph(bunker, open_k=3, close_k=5),
        green=_morph(green, open_k=3, close_k=7),
        fairway=_morph(fairway, open_k=5, close_k=11),
        tee=_morph(tee, open_k=3, close_k=5),
        soft_grass=_morph(soft_grass, open_k=3, close_k=7),
        veg=veg,
        hsv=hsv,
        lab=lab,
    )


def _morph(mask: np.ndarray, open_k: int = 0, close_k: int = 0) -> np.ndarray:
    cv2 = _cv2()
    out = mask.astype(np.uint8)
    if open_k and open_k >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_k and close_k >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out.astype(bool)


def rasterize_geoms(
    geoms: list[BaseGeometry], geo: MosaicGeo, shape: tuple[int, int], dilate_px: int = 0
) -> np.ndarray:
    cv2 = _cv2()
    mask = np.zeros(shape, np.uint8)
    for geom in geoms:
        for poly in as_polygons(geom):
            exterior = [_lonlat_xy(geo, x, y) for x, y in poly.exterior.coords]
            if len(exterior) < 3:
                continue
            cv2.fillPoly(mask, [np.array(exterior, np.int32)], 1)
            for interior in poly.interiors:
                hole = [_lonlat_xy(geo, x, y) for x, y in interior.coords]
                if len(hole) >= 3:
                    cv2.fillPoly(mask, [np.array(hole, np.int32)], 0)
        if geom.geom_type in {"LineString", "MultiLineString"}:
            lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
            for line in lines:
                pts = [_lonlat_xy(geo, x, y) for x, y in line.coords]
                if len(pts) >= 2:
                    cv2.polylines(mask, [np.array(pts, np.int32)], False, 1, 2)
    if dilate_px:
        k = max(3, dilate_px | 1)
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    return mask.astype(bool)


def _lonlat_xy(geo: MosaicGeo, lon: float, lat: float) -> list[int]:
    x, y = geo.lonlat_to_pixel(lon, lat)
    return [int(round(x)), int(round(y))]


def rasterize_layer(fc: dict | None, geo: MosaicGeo, shape: tuple[int, int], dilate_px: int = 0) -> np.ndarray:
    return rasterize_geoms(geoms_from_fc(fc), geo, shape, dilate_px=dilate_px)


def mask_to_polygons(
    mask: np.ndarray,
    geo: MosaicGeo,
    *,
    min_area_m2: float,
    max_area_m2: float,
    min_compactness: float = 0.0,
    simplify: float = 0.0000025,
    approx_frac: float = 0.004,
    min_eps: float = 1.2,
) -> list[Polygon]:
    cv2 = _cv2()
    u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out: list[Polygon] = []
    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = max(min_eps, approx_frac * peri)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        coords = [geo.pixel_to_lonlat(float(p[0][0]), float(p[0][1])) for p in approx]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            poly = make_valid(Polygon(coords))
        except Exception:
            continue
        for part in as_polygons(poly):
            cleaned = clean_polygon(part)
            if cleaned is None or cleaned.is_empty:
                continue
            cleaned = cleaned.simplify(simplify, preserve_topology=True)
            for piece in as_polygons(cleaned):
                a = area_m2(piece)
                if a < min_area_m2 or a > max_area_m2:
                    continue
                if min_compactness > 0 and _compactness(piece) < min_compactness:
                    continue
                out.append(piece)
    return out


def _elongation(geom: BaseGeometry) -> float:
    try:
        rect = geom.minimum_rotated_rectangle
        xs, ys = rect.exterior.coords.xy
        edges = [
            math.hypot(xs[i] - xs[i + 1], ys[i] - ys[i + 1])
            for i in range(4)
        ]
        length, width = max(edges), min(edges)
        return float(length / width) if width > 0 else 99.0
    except Exception:
        return 99.0


def _pixels_to_lonlat(geo: MosaicGeo, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if geo.zoom is not None and geo.tile_x_min is not None and geo.tile_y_min is not None:
        n = 2 ** geo.zoom
        fx = (geo.tile_x_min + xs / geo.tile_size) / n
        fy = (geo.tile_y_min + ys / geo.tile_size) / n
        lon = fx * 360.0 - 180.0
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * fy))))
        return lon, lat
    lon = geo.west + (xs / max(geo.width, 1)) * (geo.east - geo.west)
    lat = geo.north - (ys / max(geo.height, 1)) * (geo.north - geo.south)
    return lon, lat


def _lonlat_to_pixels(geo: MosaicGeo, lons: np.ndarray, lats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if geo.zoom is not None and geo.tile_x_min is not None and geo.tile_y_min is not None:
        lat = np.clip(lats, -85.05112878, 85.05112878)
        n = 2 ** geo.zoom
        fx = (lons + 180.0) / 360.0 * n
        lat_rad = np.radians(lat)
        fy = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * n
        return (fx - geo.tile_x_min) * geo.tile_size, (fy - geo.tile_y_min) * geo.tile_size
    x = (lons - geo.west) / max(geo.east - geo.west, 1e-12) * geo.width
    y = (geo.north - lats) / max(geo.north - geo.south, 1e-12) * geo.height
    return x, y


def remap_rgb(src_rgb: np.ndarray, src_geo: MosaicGeo, dst_geo: MosaicGeo) -> np.ndarray:
    """Warp `src_rgb` onto `dst_geo`'s pixel grid via lon/lat."""
    cv2 = _cv2()
    h, w = int(dst_geo.height), int(dst_geo.width)
    ys, xs = np.indices((h, w), dtype=np.float64)
    lons, lats = _pixels_to_lonlat(dst_geo, xs, ys)
    map_x, map_y = _lonlat_to_pixels(src_geo, lons, lats)
    return cv2.remap(
        src_rgb,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def and_class_masks(primary: ClassMasks, secondary: ClassMasks) -> ClassMasks:
    return ClassMasks(
        water=primary.water & secondary.water,
        bunker=primary.bunker & secondary.bunker,
        green=primary.green & secondary.green,
        fairway=primary.fairway & secondary.fairway,
        tee=primary.tee & secondary.tee,
        soft_grass=primary.soft_grass & secondary.soft_grass,
        veg=primary.veg,
        hsv=primary.hsv,
        lab=primary.lab,
    )


def extract_seeded(
    mask: np.ndarray,
    seeds: list[Point],
    geo: MosaicGeo,
    *,
    radius_m: float,
    min_area_m2: float,
    max_area_m2: float,
) -> list[Polygon]:
    """Connected component of `mask` nearest each seed (pin / hole end)."""
    cv2 = _cv2()
    if not seeds or not mask.any():
        return []
    h, w = mask.shape
    r = max(4, int(radius_m / max(geo.meters_per_pixel, 0.05)))
    found: list[Polygon] = []
    occupied = np.zeros_like(mask, dtype=bool)
    u8 = mask.astype(np.uint8)
    for seed in seeds:
        px, py = geo.lonlat_to_pixel(seed.x, seed.y)
        ix, iy = int(round(px)), int(round(py))
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        x0, x1 = max(0, ix - r), min(w, ix + r + 1)
        y0, y1 = max(0, iy - r), min(h, iy + r + 1)
        local = u8[y0:y1, x0:x1]
        if occupied[y0:y1, x0:x1].any():
            local = local.copy()
            local[occupied[y0:y1, x0:x1]] = 0
        if not local.any():
            continue
        num, labels, stats, cents = cv2.connectedComponentsWithStats(local, connectivity=8)
        best_i = None
        best_d = 1e18
        lx, ly = ix - x0, iy - y0
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < 8:
                continue
            cx, cy = cents[i]
            d = (cx - lx) ** 2 + (cy - ly) ** 2
            contains = labels[min(max(ly, 0), local.shape[0] - 1), min(max(lx, 0), local.shape[1] - 1)] == i
            if contains:
                d *= 0.05
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is None:
            continue
        blob = np.zeros_like(u8)
        blob[y0:y1, x0:x1][labels == best_i] = 1
        polys = mask_to_polygons(
            blob, geo, min_area_m2=min_area_m2, max_area_m2=max_area_m2, min_compactness=0.08
        )
        if not polys:
            continue
        poly = min(polys, key=lambda p: p.centroid.distance(seed))
        found.append(poly)
        occupied |= blob.astype(bool)
    return found


PIN_GREEN_RADIUS_M = 22.0
PIN_GREEN_AREA_RANGE = (90.0, 1800.0)
PIN_GREEN_COLOR_DELTA = 36.0
PIN_GREEN_MIN_COMPACT = 0.50
PIN_GREEN_FRINGE_M = 0.4
PIN_GREEN_SMOOTH_M = 0.6
PIN_GREEN_SIMPLIFY_DEG = 0.000004
PIN_GREEN_APPROX_FRAC = 0.002


def extract_pin_greens(
    rgb: np.ndarray,
    geo: MosaicGeo,
    pins: list[Point],
    *,
    second_rgb: np.ndarray | None = None,
    radius_m: float = PIN_GREEN_RADIUS_M,
) -> list[dict[str, Any]]:
    """Grow a compact putting surface from each pin on the high-res mosaic.

    Optional `second_rgb` must already be on `geo`'s grid (Clarity / alt date).
    Intersection is kept when it still looks like a green; otherwise the
    primary outline is kept if the second image still shows turf at the pin.
    """
    if not pins:
        return []
    primary = _grow_pin_green_mask(rgb, geo, pins, radius_m)
    confirmed: list[dict[str, Any]] = []
    second = None
    aligned_second = second_rgb is not None and second_rgb.shape[:2] == rgb.shape[:2]
    if aligned_second:
        second = _grow_pin_green_mask(second_rgb, geo, pins, radius_m)
    second_veg = vegetation_index(second_rgb) if aligned_second else None
    for idx, pin in enumerate(pins):
        poly = primary[idx] if idx < len(primary) else None
        alt = second[idx] if second is not None and idx < len(second) else None
        chosen = poly
        sources = ["primary"]
        if poly is not None and alt is not None:
            try:
                inter = poly.intersection(alt)
            except Exception:
                inter = None
            inter_ok = None
            if inter is not None and not inter.is_empty:
                pieces = [p for p in as_polygons(inter) if area_m2(p) >= PIN_GREEN_AREA_RANGE[0]]
                if pieces:
                    inter_ok = max(pieces, key=area_m2)
            if inter_ok is not None and (inter_ok.contains(pin) or distance_meters(inter_ok, pin) < 8):
                chosen = inter_ok
                sources = ["primary", "temporal"]
            else:
                sources = ["primary"]
        elif poly is None and alt is not None:
            chosen = alt
            sources = ["temporal"]
        if chosen is None:
            continue
        if second_veg is not None:
            px, py = geo.lonlat_to_pixel(pin.x, pin.y)
            ix, iy = int(round(px)), int(round(py))
            h, w = second_veg.shape
            if 0 <= ix < w and 0 <= iy < h and float(second_veg[iy, ix]) < 0.04:
                # Other date is not turf at the pin — still keep if the
                # primary blob is compact and contains the pin.
                if _compactness(chosen) < 0.20:
                    continue
        cleaned = _finish_pin_green(chosen, pin)
        if cleaned is None:
            continue
        confirmed.append(
            {
                "pin": [pin.x, pin.y],
                "geometry": cleaned,
                "area_m2": round(area_m2(cleaned), 1),
                "sources": sources,
            }
        )
    return confirmed


def extract_pin_greens_from_crops(crops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Grow a putting surface on each high-zoom pin crop (preferred over the course mosaic)."""
    found: list[dict[str, Any]] = []
    for crop in crops:
        if crop.get("status") != "ok" or not crop.get("path"):
            continue
        path = Path(crop["path"])
        if not path.exists():
            continue
        pin_xy = crop.get("pin") or []
        if len(pin_xy) < 2:
            continue
        try:
            rgb = load_rgb(path)
            geo = MosaicGeo.from_imagery(crop, rgb)
        except Exception:
            continue
        pin = Point(float(pin_xy[0]), float(pin_xy[1]))
        grown = extract_pin_greens(rgb, geo, [pin])
        for item in grown:
            sources = list(item.get("sources") or [])
            src = crop.get("source")
            if src and src not in sources:
                sources.append(src)
            item["sources"] = sources
            found.append(item)
    return found


def _prefer_crop_pin_greens(
    mosaic: list[dict[str, Any]], crops: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep the crop outline when both recover the same pin."""
    merged: list[dict[str, Any]] = []
    used_crop = [False] * len(crops)
    for item in mosaic:
        pin = item.get("pin") or []
        match = None
        match_i = -1
        if len(pin) >= 2:
            seed = Point(float(pin[0]), float(pin[1]))
            best_d = 14.0
            for i, crop in enumerate(crops):
                if used_crop[i]:
                    continue
                cpin = crop.get("pin") or []
                if len(cpin) < 2:
                    continue
                d = distance_meters(seed, Point(float(cpin[0]), float(cpin[1])))
                if d < best_d:
                    best_d = d
                    match = crop
                    match_i = i
        if match is not None:
            used_crop[match_i] = True
            merged.append(match)
        else:
            merged.append(item)
    for i, crop in enumerate(crops):
        if not used_crop[i]:
            merged.append(crop)
    return merged


def refine_osm_greens(
    osm_fc: dict | None,
    pin_greens: list[dict[str, Any]],
    pins: list[Point],
) -> tuple[list[dict], dict[str, int]]:
    """Replace an OSM green ring with a pin-grown outline when safe.

    Identity (osm_id) is kept. Shared greens (two pins inside) stay frozen.
    The new ring may sit slightly outside OSM so offset digitizing can snap
    to imagery; a far-away different green is rejected.
    """
    stats = {"osm_refined": 0, "osm_kept": 0, "refine_rejected_shared": 0, "refine_rejected_gate": 0}
    kept: list[dict] = []
    for feat in (osm_fc or {}).get("features", []):
        try:
            geom = as_geom(feat)
        except Exception:
            continue
        if geom.is_empty:
            continue
        props = dict(feat.get("properties") or {})
        props["source"] = props.get("source") or "openstreetmap"
        inside = [p for p in pins if _pin_hits_poly(p, geom, 6.0)]
        if len(inside) > 1:
            stats["refine_rejected_shared"] += 1
            stats["osm_kept"] += 1
            kept.append(to_feature(geom, props))
            continue
        seed = inside[0] if inside else _nearest_pin(geom, pins, 28.0)
        candidate = _pin_green_for(seed, pin_greens) if seed is not None else None
        refined = _accept_green_refine(geom, candidate, seed, pins)
        if refined is None:
            stats["refine_rejected_gate"] += 1 if candidate is not None else 0
            stats["osm_kept"] += 1
            kept.append(to_feature(geom, props))
            continue
        props["source"] = "osm_refined"
        props["osm_area_m2"] = round(area_m2(geom), 1)
        props["refined_area_m2"] = round(area_m2(refined), 1)
        stats["osm_refined"] += 1
        kept.append(to_feature(refined, props))
    return kept, stats


def _pin_hits_poly(pin: Point, geom: BaseGeometry, slop_m: float) -> bool:
    try:
        if geom.buffer(1e-7).contains(pin):
            return True
        return distance_meters(pin, geom) < slop_m
    except Exception:
        return False


def _nearest_pin(geom: BaseGeometry, pins: list[Point], max_m: float) -> Point | None:
    best = None
    best_d = max_m
    for pin in pins:
        d = distance_meters(pin, geom)
        if d < best_d:
            best_d = d
            best = pin
    return best


def _pin_green_for(pin: Point, pin_greens: list[dict[str, Any]]) -> BaseGeometry | None:
    best = None
    best_d = 20.0
    for item in pin_greens:
        geom = item.get("geometry")
        coords = item.get("pin") or []
        if geom is None or len(coords) < 2:
            continue
        d = distance_meters(pin, Point(float(coords[0]), float(coords[1])))
        if d < best_d:
            best_d = d
            best = geom
    return best


def _accept_green_refine(
    osm: BaseGeometry,
    candidate: BaseGeometry | None,
    pin: Point | None,
    pins: list[Point],
) -> Polygon | None:
    if candidate is None or pin is None:
        return None
    cleaned = clean_polygon(candidate) or candidate
    parts = [p for p in as_polygons(cleaned) if area_m2(p) >= PIN_GREEN_AREA_RANGE[0]]
    if not parts:
        return None
    refined = min(parts, key=lambda p: distance_meters(p, pin))
    if not _pin_hits_poly(pin, refined, REFINE_PIN_M):
        return None
    others = [p for p in pins if p is not pin and _pin_hits_poly(p, refined, 4.0)]
    if others:
        return None
    try:
        inter_a = refined.intersection(osm).area
        refined_a = refined.area or 0.0
        osm_a_deg = osm.area or 0.0
        overlap_new = inter_a / refined_a if refined_a else 0.0
        overlap_osm = inter_a / osm_a_deg if osm_a_deg else 0.0
        score = iou(osm, refined)
    except Exception:
        overlap_new = overlap_osm = score = 0.0
    centroid_d = distance_meters(refined.centroid, osm.centroid)
    same_object = (
        score >= REFINE_IOU
        or overlap_new >= REFINE_MIN_OVERLAP
        or overlap_osm >= REFINE_MIN_OVERLAP
        or (centroid_d < REFINE_CENTROID_M and _pin_hits_poly(pin, osm, 10.0))
    )
    if not same_object:
        return None
    osm_a = area_m2(osm)
    new_a = area_m2(refined)
    if new_a < PIN_GREEN_AREA_RANGE[0] or new_a > PIN_GREEN_AREA_RANGE[1]:
        return None
    if _compactness(refined) < PIN_GREEN_MIN_COMPACT:
        return None
    # Imagery outline when it is the same size or tighter. If it bled into
    # the collar, snap the OSM ring onto the imagery centroid instead.
    if osm_a > 0 and new_a <= osm_a * REFINE_MAX_GROW:
        return refined
    return _snap_osm_ring(osm, refined, pin)


def _snap_osm_ring(osm: BaseGeometry, imagery: BaseGeometry, pin: Point) -> Polygon | None:
    """Move a well-digitized OSM ring onto the imagery centroid; clip if still fat."""
    east = (imagery.centroid.x - osm.centroid.x) * 111_320.0 * math.cos(math.radians(osm.centroid.y))
    north = (imagery.centroid.y - osm.centroid.y) * 111_320.0
    if math.hypot(east, north) > REFINE_CENTROID_M:
        return None
    snapped = translate_meters(osm, east, north)
    if not _pin_hits_poly(pin, snapped, REFINE_PIN_M):
        snapped = osm
    clipped = snapped.intersection(buffer_meters(imagery, 3.0))
    parts = [p for p in as_polygons(clipped) if area_m2(p) >= PIN_GREEN_AREA_RANGE[0]]
    chosen = None
    if parts:
        best = min(parts, key=lambda p: distance_meters(p, pin))
        if _pin_hits_poly(pin, best, REFINE_PIN_M) and area_m2(best) >= area_m2(snapped) * 0.55:
            chosen = best
    if chosen is None:
        chosen = snapped if _pin_hits_poly(pin, snapped, REFINE_PIN_M) else None
    if chosen is None:
        return None
    finished = _smooth_green(chosen, pin)
    cleaned = clean_polygon(finished) or finished
    parts = [p for p in as_polygons(cleaned) if area_m2(p) >= PIN_GREEN_AREA_RANGE[0]]
    if not parts:
        return None
    out = min(parts, key=lambda p: distance_meters(p, pin))
    if _compactness(out) < PIN_GREEN_MIN_COMPACT:
        return None
    return out


def derive_green_fringes(
    greens_fc: dict | None,
    *,
    subtract: list[BaseGeometry] | None = None,
    boundary: BaseGeometry | None = None,
    width_m: float = FRINGE_WIDTH_M,
) -> list[dict]:
    """Collar ring around each putting surface (buffer − green), minus hazards."""
    mask = None
    extras = [g for g in (subtract or []) if g is not None and not g.is_empty]
    if extras:
        mask = unary_union(extras)
    green_geoms = [as_geom(f) for f in (greens_fc or {}).get("features", [])]
    green_geoms = [g for g in green_geoms if not g.is_empty]
    out: list[dict] = []
    for i, feat in enumerate((greens_fc or {}).get("features", [])):
        try:
            green = as_geom(feat)
        except Exception:
            continue
        if green.is_empty:
            continue
        ring = difference_safe(buffer_meters(green, width_m), green)
        others = [g for j, g in enumerate(green_geoms) if j != i]
        if others:
            ring = difference_safe(ring, unary_union(others))
        if mask is not None:
            ring = difference_safe(ring, mask)
        if boundary is not None:
            ring = ring.intersection(boundary)
        props = dict(feat.get("properties") or {})
        hole = props.get("hole")
        for part_i, part in enumerate(as_polygons(ring), start=1):
            cleaned = clean_polygon(part)
            if cleaned is None or cleaned.is_empty:
                continue
            if area_m2(cleaned) < 12:
                continue
            fringe_props = {
                "source": "imagery_fringe",
                "golf": "fringe",
                "instance_id": f"fringe-{props.get('instance_id') or props.get('osm_id') or i + 1}-{part_i}",
            }
            if hole is not None:
                fringe_props["hole"] = hole
            if props.get("osm_id") is not None:
                fringe_props["green_osm_id"] = props.get("osm_id")
            out.append(to_feature(cleaned, fringe_props))
    return out


def extract_play_trees(
    rgb: np.ndarray,
    geo: MosaicGeo,
    osm_layers: dict[str, dict],
) -> list[dict]:
    """Isolated canopy crowns in play (fairway / green-side), as obstacle points."""
    cv2 = _cv2()
    veg = vegetation_index(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    g = rgb[:, :, 1].astype(np.int16)
    r = rgb[:, :, 0].astype(np.int16)
    classified = classify_pixels(rgb)
    canopy = (
        (veg > 0.08)
        & (v < 105)
        & (h >= 30)
        & (h <= 95)
        & (s > 35)
        & (g + 4 >= r)
        & ~classified.water
        & ~classified.green
        & ~classified.bunker
    )
    canopy = _morph(canopy, open_k=3, close_k=5)
    polys = mask_to_polygons(
        canopy,
        geo,
        min_area_m2=TREE_CROWN_M2[0],
        max_area_m2=TREE_CROWN_M2[1],
        min_compactness=TREE_MIN_COMPACT,
    )
    play_parts = []
    for lid in ("fairway", "green", "tee", "hole_centerline"):
        play_parts.extend(geoms_from_fc(osm_layers.get(lid)))
    play = unary_union(play_parts) if play_parts else None
    greens = [g for g in geoms_from_fc(osm_layers.get("green")) if not g.is_empty]
    green_u = unary_union(greens) if greens else None
    woods = [g for g in geoms_from_fc(osm_layers.get("woodland")) if not g.is_empty]
    wood_u = unary_union(woods) if woods else None
    existing = _points_from_layer(osm_layers.get("tree"))
    found: list[dict] = []
    for i, poly in enumerate(polys):
        pt = poly.centroid
        if green_u is not None and (green_u.contains(pt) or distance_meters(pt, green_u) < 2.5):
            continue
        if play is not None and distance_meters(pt, play) > 55.0:
            continue
        if wood_u is not None and wood_u.contains(pt):
            try:
                if distance_meters(pt, wood_u.boundary) > 15:
                    continue
            except Exception:
                continue
        if any(distance_meters(pt, old) < TREE_DEDUP_M for old in existing):
            continue
        if any(distance_meters(pt, as_geom(f)) < TREE_DEDUP_M for f in found):
            continue
        found.append(
            to_feature(
                pt,
                {
                    "source": "imagery",
                    "natural": "tree",
                    "instance_id": f"imagery-tree-{i + 1}",
                    "crown_m2": round(area_m2(poly), 1),
                    "radius_m": round(math.sqrt(area_m2(poly) / math.pi), 1),
                },
            )
        )
    return found


def merge_tree_layer(osm_fc: dict | None, imagery_feats: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for feat in (osm_fc or {}).get("features", []):
        props = dict(feat.get("properties") or {})
        props["source"] = props.get("source") or "openstreetmap"
        kept.append({**feat, "properties": props})
    existing = _points_from_layer({"features": kept} if kept else None)
    for feat in imagery_feats:
        try:
            geom = as_geom(feat)
        except Exception:
            continue
        if geom.geom_type != "Point":
            geom = geom.centroid
        if any(distance_meters(geom, old) < TREE_DEDUP_M for old in existing):
            continue
        kept.append(feat)
        existing.append(Point(geom.x, geom.y))
    return kept


def _grabcut_pin_green(rgb: np.ndarray, geo: MosaicGeo, pin: Point) -> Polygon | None:
    """Follow the putting-surface / collar edge with GrabCut, seeded at the pin."""
    cv2 = _cv2()
    h, w = rgb.shape[:2]
    px, py = geo.lonlat_to_pixel(pin.x, pin.y)
    ix, iy = int(round(px)), int(round(py))
    if not (0 <= ix < w and 0 <= iy < h):
        return None
    mpp = max(geo.meters_per_pixel, 0.05)
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(xx - ix, yy - iy)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Trees further out are background. Shadows on the putting surface stay
    # in play so GrabCut can still claim them as foreground.
    trees = (hsv[:, :, 2] < 40) & (dist > 10.0 / mpp)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[dist < 16.0 / mpp] = cv2.GC_PR_FGD
    mask[dist < 5.0 / mpp] = cv2.GC_FGD
    mask[dist > 26.0 / mpp] = cv2.GC_BGD
    mask[trees] = cv2.GC_BGD
    try:
        cv2.grabCut(
            bgr,
            mask,
            None,
            np.zeros((1, 65), np.float64),
            np.zeros((1, 65), np.float64),
            5,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return None
    gc = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    gc = _fill_mask_holes(gc)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    gc = cv2.morphologyEx(gc, cv2.MORPH_CLOSE, close_k)
    num, labels = cv2.connectedComponents(gc)
    lid = int(labels[iy, ix]) if num else 0
    if lid == 0:
        return None
    blob = (labels == lid).astype(np.uint8)
    polys = mask_to_polygons(
        blob.astype(bool),
        geo,
        min_area_m2=PIN_GREEN_AREA_RANGE[0],
        max_area_m2=PIN_GREEN_AREA_RANGE[1],
        min_compactness=0.35,
        simplify=PIN_GREEN_SIMPLIFY_DEG,
        approx_frac=PIN_GREEN_APPROX_FRAC,
        min_eps=0.6,
    )
    if not polys:
        return None
    return min(polys, key=lambda p: p.centroid.distance(pin))


def _grow_pin_green_mask(
    rgb: np.ndarray, geo: MosaicGeo, pins: list[Point], radius_m: float
) -> list[Polygon | None]:
    cv2 = _cv2()
    veg = vegetation_index(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    g = rgb[:, :, 1].astype(np.int16)
    r = rgb[:, :, 0].astype(np.int16)
    classified = classify_pixels(rgb).green
    # Putting turf is vivid; collar/fairway is duller. Keep classified pixels
    # even when shade stripes fail a tight color delta.
    loose = (
        (veg > 0.11)
        & (h >= 32)
        & (h <= 90)
        & (g + 6 >= r)
        & (v > 55)
        & (v < 210)
        & (s > 30)
    )
    candidate = classified | loose
    out: list[Polygon | None] = []
    height, width = candidate.shape
    radius_px = max(6, int(radius_m / max(geo.meters_per_pixel, 0.05)))
    snap_px = max(3, int(8.0 / max(geo.meters_per_pixel, 0.05)))
    for pin in pins:
        grabbed = _grabcut_pin_green(rgb, geo, pin)
        if grabbed is not None:
            out.append(grabbed)
            continue
        px, py = geo.lonlat_to_pixel(pin.x, pin.y)
        ix, iy = int(round(px)), int(round(py))
        if not (0 <= ix < width and 0 <= iy < height):
            out.append(None)
            continue
        sx, sy = _snap_seed(candidate, veg, ix, iy, snap_px)
        x0, x1 = max(0, sx - radius_px), min(width, sx + radius_px + 1)
        y0, y1 = max(0, sy - radius_px), min(height, sy + radius_px + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.hypot(xx - sx, yy - sy)
        classified_p = classified[y0:y1, x0:x1]
        loose_p = loose[y0:y1, x0:x1]
        keep = classified_p.copy()
        seed_rgb = rgb[max(0, sy - 3) : sy + 4, max(0, sx - 3) : sx + 4].astype(np.float32)
        if seed_rgb.size:
            mean = seed_rgb.reshape(-1, 3).mean(axis=0)
            patch = rgb[y0:y1, x0:x1].astype(np.float32)
            delta = np.linalg.norm(patch - mean, axis=2)
            keep = keep | (loose_p & (delta <= PIN_GREEN_COLOR_DELTA))
        else:
            keep = classified_p | loose_p
        local = keep.astype(np.uint8)
        local[dist > radius_px] = 0
        if not local.any():
            out.append(None)
            continue
        num, labels = cv2.connectedComponents(local, connectivity=8)
        lx, ly = min(max(sx - x0, 0), local.shape[1] - 1), min(max(sy - y0, 0), local.shape[0] - 1)
        label = int(labels[ly, lx])
        if label == 0:
            # pick nearest positive component
            best_i, best_d = None, 1e18
            for i in range(1, num):
                ys, xs = np.where(labels == i)
                if not len(xs):
                    continue
                d = ((xs.astype(np.float32) - lx) ** 2 + (ys.astype(np.float32) - ly) ** 2).min()
                if d < best_d:
                    best_d = d
                    best_i = i
            label = best_i or 0
        if label == 0:
            out.append(None)
            continue
        blob = np.zeros(candidate.shape, np.uint8)
        blob[y0:y1, x0:x1][labels == label] = 1
        blob = _fill_mask_holes(blob)
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, open_k)
        blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, close_k)
        blob = _fill_mask_holes(blob)
        polys = mask_to_polygons(
            blob.astype(bool),
            geo,
            min_area_m2=PIN_GREEN_AREA_RANGE[0],
            max_area_m2=PIN_GREEN_AREA_RANGE[1],
            min_compactness=0.28,
            simplify=PIN_GREEN_SIMPLIFY_DEG,
            approx_frac=PIN_GREEN_APPROX_FRAC,
            min_eps=0.6,
        )
        if not polys:
            out.append(None)
            continue
        poly = min(polys, key=lambda p: p.centroid.distance(pin))
        out.append(poly)
    return out


def _snap_seed(mask: np.ndarray, veg: np.ndarray, ix: int, iy: int, snap_px: int) -> tuple[int, int]:
    h, w = mask.shape
    if mask[iy, ix]:
        return ix, iy
    y0, y1 = max(0, iy - snap_px), min(h, iy + snap_px + 1)
    x0, x1 = max(0, ix - snap_px), min(w, ix + snap_px + 1)
    local = mask[y0:y1, x0:x1]
    if not local.any():
        local = veg[y0:y1, x0:x1] > 0.12
        if not local.any():
            return ix, iy
    scored = veg[y0:y1, x0:x1].copy()
    scored[~local] = -1
    jy, jx = np.unravel_index(int(np.argmax(scored)), scored.shape)
    return x0 + int(jx), y0 + int(jy)


def _fill_mask_holes(mask_u8: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    h, w = mask_u8.shape
    if h < 3 or w < 3 or not mask_u8.any():
        return mask_u8
    flood = mask_u8.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 1)
    holes = flood == 0
    filled = mask_u8.copy()
    filled[holes] = 1
    return filled


def _peel_fringe(poly: BaseGeometry, pin: Point) -> BaseGeometry:
    """Light inset so a dull collar is not part of the putting surface."""
    inset = buffer_meters(poly, -PIN_GREEN_FRINGE_M)
    parts = [p for p in as_polygons(inset) if area_m2(p) >= PIN_GREEN_AREA_RANGE[0]]
    if not parts:
        return poly
    best = min(parts, key=lambda p: distance_meters(p, pin))
    if not (best.buffer(1e-7).contains(pin) or distance_meters(best, pin) < 4):
        return poly
    if area_m2(best) < area_m2(poly) * 0.72:
        return poly
    return best


def _smooth_green(poly: BaseGeometry, pin: Point) -> BaseGeometry:
    """Close jaggies in metres, then simplify. Must still hold the pin."""
    closed = buffer_meters(buffer_meters(poly, PIN_GREEN_SMOOTH_M), -PIN_GREEN_SMOOTH_M)
    parts = [p for p in as_polygons(closed) if area_m2(p) >= PIN_GREEN_AREA_RANGE[0]]
    if parts:
        candidate = min(parts, key=lambda p: distance_meters(p, pin))
        if candidate.buffer(1e-7).contains(pin) or distance_meters(candidate, pin) < 6:
            if area_m2(candidate) <= area_m2(poly) * 1.18:
                poly = candidate
    simplified = poly.simplify(PIN_GREEN_SIMPLIFY_DEG, preserve_topology=True)
    if simplified.is_empty:
        return poly
    if simplified.buffer(1e-7).contains(pin) or distance_meters(simplified, pin) < 6:
        return simplified
    return poly


def _finish_pin_green(geom: BaseGeometry, pin: Point) -> Polygon | None:
    pieces = [p for p in as_polygons(geom) if area_m2(p) >= PIN_GREEN_AREA_RANGE[0]]
    if not pieces:
        return None
    poly = min(pieces, key=lambda p: distance_meters(p, pin))
    poly = _smooth_green(poly, pin)
    poly = _peel_fringe(poly, pin)
    poly = _smooth_green(poly, pin)
    cleaned = clean_polygon(poly) or poly
    if cleaned is None or cleaned.is_empty:
        return None
    if area_m2(cleaned) < PIN_GREEN_AREA_RANGE[0] or area_m2(cleaned) > PIN_GREEN_AREA_RANGE[1]:
        return None
    if _compactness(cleaned) < PIN_GREEN_MIN_COMPACT:
        return None
    if not (cleaned.buffer(1e-7).contains(pin) or distance_meters(cleaned, pin) < 8):
        return None
    return cleaned


def grow_fairway_along_line(
    fairway_mask: np.ndarray,
    soft_mask: np.ndarray,
    line: LineString,
    geo: MosaicGeo,
    *,
    search_m: float = 38.0,
    seed_m: float = 8.0,
    subtract: BaseGeometry | None = None,
    boundary: BaseGeometry | None = None,
) -> list[Polygon]:
    """Grow fairway-class pixels from the hole centerline; no uniform sausage."""
    cv2 = _cv2()
    h, w = fairway_mask.shape
    search = rasterize_geoms([buffer_meters(line, search_m)], geo, (h, w))
    seed = rasterize_geoms([buffer_meters(line, seed_m)], geo, (h, w))
    core = search & fairway_mask
    soft = search & soft_mask
    # Prefer classified fairway grass; only loosen to soft turf if the core is sparse.
    if int(core.sum()) >= max(40, int(seed.sum() * 0.12)):
        allowed = core
    else:
        allowed = core | soft
    grown = (seed & allowed).astype(np.uint8)
    if grown.sum() < 12:
        grown = (allowed & search).astype(np.uint8)
        # Keep only components that touch a 4 m spine.
        spine = rasterize_geoms([buffer_meters(line, 4.0)], geo, (h, w))
        grown = _keep_touching(grown, spine)
    if grown.sum() < 12:
        return []
    radius_px = max(3, int(search_m / max(geo.meters_per_pixel, 0.05) / 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    allow_u8 = allowed.astype(np.uint8)
    for _ in range(min(40, radius_px)):
        nxt = cv2.dilate(grown, kernel) & allow_u8
        if np.array_equal(nxt, grown):
            break
        grown = nxt
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, close_k) & allow_u8
    polys = mask_to_polygons(
        grown.astype(bool), geo, min_area_m2=FAIRWAY_AREA_M2[0], max_area_m2=FAIRWAY_AREA_M2[1]
    )
    out: list[Polygon] = []
    for poly in polys:
        geom: BaseGeometry = poly
        if subtract is not None:
            geom = difference_safe(geom, subtract)
        if boundary is not None:
            geom = geom.intersection(boundary)
        for part in as_polygons(geom):
            cleaned = clean_polygon(part)
            if cleaned is None or cleaned.is_empty:
                continue
            if area_m2(cleaned) < FAIRWAY_AREA_M2[0]:
                continue
            if not cleaned.intersects(buffer_meters(line, seed_m + 4)):
                continue
            out.extend(as_polygons(cleaned))
    return out


def _keep_touching(mask_u8: np.ndarray, spine: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    num, labels = cv2.connectedComponents(mask_u8, connectivity=8)
    keep = np.zeros_like(mask_u8)
    spine_b = spine.astype(bool)
    for i in range(1, num):
        blob = labels == i
        if np.any(blob & spine_b):
            keep[blob] = 1
    return keep


def fuse_layer(
    osm_fc: dict | None,
    imagery: list[BaseGeometry],
    layer_id: str,
    *,
    contradiction: list[BaseGeometry] | None = None,
    agreed: list[bool] | None = None,
    stats: dict[str, int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Peg OSM geometry. Never swap an OSM ring for an imagery outline.

    Imagery may ADD a feature only when `layer_id` is bunker/water, `agreed[j]`
    is True, and the blob does not overlap existing OSM. Greens, tees, and
    fairways are never invented here.
    `contradiction` is ignored: imagery must never delete or downgrade OSM.
    Returns (kept_for_layer, unused_practice_bucket).
    """
    del contradiction
    osm_feats: list[dict] = []
    for feat in (osm_fc or {}).get("features", []):
        try:
            geom = as_geom(feat)
        except Exception:
            continue
        if geom.is_empty:
            continue
        osm_feats.append({"geom": geom, "properties": dict(feat.get("properties") or {}), "feat": feat})

    used_img = [False] * len(imagery)
    flags = list(agreed) if agreed is not None else [False] * len(imagery)
    if len(flags) < len(imagery):
        flags.extend([False] * (len(imagery) - len(flags)))
    kept: list[dict] = []
    counters = stats if stats is not None else {
        "osm_kept": 0,
        "imagery_added": 0,
        "imagery_rejected_single_source": 0,
        "imagery_rejected_shape": 0,
        "imagery_rejected_overlap": 0,
        "imagery_rejected_layer": 0,
        "osm_refined": 0,
    }

    for item in osm_feats:
        geom = item["geom"]
        props = dict(item["properties"])
        props["source"] = props.get("source") or "openstreetmap"
        best_j = -1
        best_score = 0.0
        for j, img in enumerate(imagery):
            if used_img[j]:
                continue
            overlap = 0.0
            try:
                inter = geom.intersection(img).area
                if geom.area > 0:
                    overlap = inter / geom.area
            except Exception:
                overlap = 0.0
            score = max(iou(geom, img), overlap)
            if score > best_score:
                best_score = score
                best_j = j
        if best_j >= 0 and best_score >= MATCH_IOU:
            used_img[best_j] = True
        kept.append(to_feature(geom, props))
        counters["osm_kept"] = counters.get("osm_kept", 0) + 1

    osm_union = unary_union([item["geom"] for item in osm_feats]) if osm_feats else None
    allow_add = layer_id in IMAGERY_ADD_LAYERS
    for j, img in enumerate(imagery):
        if used_img[j]:
            continue
        if not allow_add:
            counters["imagery_rejected_layer"] = counters.get("imagery_rejected_layer", 0) + 1
            continue
        if not flags[j]:
            counters["imagery_rejected_single_source"] = counters.get("imagery_rejected_single_source", 0) + 1
            continue
        if osm_union is not None and not osm_union.is_empty:
            try:
                overlap = img.intersection(osm_union).area / img.area if img.area else 0.0
            except Exception:
                overlap = 0.0
            if overlap >= 0.15 or iou(img, osm_union) >= MATCH_IOU:
                counters["imagery_rejected_overlap"] = counters.get("imagery_rejected_overlap", 0) + 1
                continue
        props = {
            "source": "imagery",
            "golf": layer_id,
            "instance_id": f"imagery-{layer_id}-{j + 1}",
            "dual_agreed": True,
        }
        kept.append(to_feature(img, props))
        counters["imagery_added"] = counters.get("imagery_added", 0) + 1
    if stats is None:
        pass
    return kept, []


def _points_from_layer(fc: dict | None) -> list[Point]:
    pts: list[Point] = []
    for geom in geoms_from_fc(fc):
        if geom.geom_type == "Point":
            pts.append(Point(geom.x, geom.y))
        else:
            pts.append(geom.centroid)
    return pts


def _hole_lines(fc: dict | None) -> list[LineString]:
    lines: list[LineString] = []
    for geom in geoms_from_fc(fc):
        if geom.geom_type == "LineString" and geom.length > 0:
            lines.append(geom)
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                if part.length > 0:
                    lines.append(part)
    return lines


def _hole_ends(lines: list[LineString]) -> list[Point]:
    ends: list[Point] = []
    for line in lines:
        ends.append(Point(line.coords[0]))
        ends.append(Point(line.coords[-1]))
    return ends


def analyze_mosaic(
    rgb: np.ndarray,
    geo: MosaicGeo,
    osm_layers: dict[str, dict],
    *,
    hole_lines: list[LineString] | None = None,
    pins: list[Point] | None = None,
    second_rgb: np.ndarray | None = None,
    second_geo: MosaicGeo | None = None,
) -> AnalysisResult:
    shape = rgb.shape[:2]
    priors = {
        lid: rasterize_layer(
            osm_layers.get(lid),
            geo,
            shape,
            dilate_px=5 if lid == "cart_path" else 0,
        )
        for lid in PRIOR_LAYERS
        if osm_layers.get(lid)
    }
    boundary = None
    if osm_layers.get("boundary"):
        boundary = rasterize_layer(osm_layers.get("boundary"), geo, shape, dilate_px=2)
    exclude_parts = []
    for lid in ("building", "parking", "driving_range", "woodland"):
        if osm_layers.get(lid):
            exclude_parts.append(rasterize_layer(osm_layers.get(lid), geo, shape, dilate_px=1))
    exclude = None
    if exclude_parts:
        exclude = np.logical_or.reduce(exclude_parts)

    primary_masks = classify_pixels(rgb, priors=priors, boundary=boundary, exclude=exclude)
    dual_source = False
    masks = primary_masks
    if second_rgb is not None:
        aligned = second_rgb
        if second_geo is not None and (
            aligned.shape[:2] != (geo.height, geo.width)
            or second_geo.west != geo.west
            or second_geo.zoom != geo.zoom
        ):
            aligned = remap_rgb(second_rgb, second_geo, geo)
        elif aligned.shape[:2] != (int(geo.height), int(geo.width)):
            aligned = remap_rgb(second_rgb, second_geo or geo, geo)
        second_masks = classify_pixels(aligned, priors=priors, boundary=boundary, exclude=exclude)
        masks = and_class_masks(primary_masks, second_masks)
        dual_source = True

    hole_lines = hole_lines or _hole_lines(osm_layers.get("hole_centerline"))
    pins = pins or _points_from_layer(osm_layers.get("pin"))
    ends = _hole_ends(hole_lines)
    seeds = list(pins) if pins else ends
    fusion_stats = {
        "osm_kept": 0,
        "imagery_added": 0,
        "imagery_rejected_single_source": 0,
        "imagery_rejected_shape": 0,
        "imagery_rejected_overlap": 0,
        "imagery_rejected_layer": 0,
        "osm_refined": 0,
        "dual_source": dual_source,
    }

    green_polys = mask_to_polygons(
        masks.green, geo, min_area_m2=GREEN_AREA_M2[0], max_area_m2=GREEN_AREA_M2[1], min_compactness=GREEN_MIN_COMPACT
    )
    seeded = extract_seeded(
        masks.green,
        seeds,
        geo,
        radius_m=28.0,
        min_area_m2=PIN_GREEN_AREA_M2[0],
        max_area_m2=PIN_GREEN_AREA_M2[1],
    )
    green_polys = _merge_new(green_polys, seeded)
    green_polys, n_shape = _shape_filter(green_polys, "green", hole_lines, pins or seeds)
    fusion_stats["imagery_rejected_shape"] += n_shape

    bunker_polys = mask_to_polygons(
        masks.bunker, geo, min_area_m2=BUNKER_AREA_M2[0], max_area_m2=BUNKER_AREA_M2[1], min_compactness=BUNKER_MIN_COMPACT
    )
    bunker_polys, n_shape = _shape_filter(bunker_polys, "bunker", hole_lines, pins or seeds)
    fusion_stats["imagery_rejected_shape"] += n_shape
    water_polys = mask_to_polygons(
        masks.water, geo, min_area_m2=WATER_AREA_M2[0], max_area_m2=WATER_AREA_M2[1], min_compactness=0.12
    )

    tee_polys = _extract_tees(masks.tee, geo, osm_layers, hole_lines, pins or seeds, green_polys)
    tee_polys, n_shape = _shape_filter(tee_polys, "tee", hole_lines, pins or seeds)
    fusion_stats["imagery_rejected_shape"] += n_shape

    if dual_source:
        primary_bunkers = mask_to_polygons(
            primary_masks.bunker,
            geo,
            min_area_m2=BUNKER_AREA_M2[0],
            max_area_m2=BUNKER_AREA_M2[1],
            min_compactness=0.12,
        )
        fusion_stats["imagery_rejected_single_source"] += max(0, len(primary_bunkers) - len(bunker_polys))

    subtract_parts = [p for p in green_polys + bunker_polys + water_polys + tee_polys]
    for lid in ("green", "tee", "bunker", "water", "putting_green", "building", "driving_range"):
        subtract_parts.extend(geoms_from_fc(osm_layers.get(lid)))
    subtract = unary_union(subtract_parts) if subtract_parts else None
    boundary_g = union_layer(osm_layers.get("boundary"))
    osm_fw = geoms_from_fc(osm_layers.get("fairway"))

    fairway_polys: list[Polygon] = []
    fairway_by_hole: list[tuple[int | None, Polygon]] = []
    if hole_lines:
        for idx, line in enumerate(hole_lines, start=1):
            if any(fw.intersects(buffer_meters(line, 28.0)) for fw in osm_fw):
                continue
            grown = grow_fairway_along_line(
                masks.fairway,
                masks.soft_grass,
                line,
                geo,
                subtract=subtract,
                boundary=boundary_g,
            )
            ref = _line_ref(osm_layers.get("hole_centerline"), idx - 1)
            if grown:
                merged = as_polygons(unary_union(grown))
            else:
                merged = []
            # One corridor per hole: keep the largest piece.
            merged = sorted(merged, key=area_m2, reverse=True)[:1]
            for poly in merged:
                if area_m2(poly) < FAIRWAY_AREA_M2[0]:
                    continue
                fairway_polys.append(poly)
                fairway_by_hole.append((ref, poly))
    elif dual_source:
        fairway_polys = mask_to_polygons(
            masks.fairway, geo, min_area_m2=FAIRWAY_AREA_M2[0], max_area_m2=FAIRWAY_AREA_M2[1]
        )

    polygons: dict[str, list[tuple[BaseGeometry, dict[str, Any]]]] = {
        "green": [(g, {"source": "imagery", "golf": "green"}) for g in green_polys],
        "bunker": [(g, {"source": "imagery", "golf": "bunker"}) for g in bunker_polys],
        "water": [(g, {"source": "imagery", "golf": "water"}) for g in water_polys],
        "tee": [(g, {"source": "imagery", "golf": "tee"}) for g in tee_polys],
        "fairway": [
            (g, {"source": "imagery", "golf": "fairway", "hole": hole})
            for hole, g in (fairway_by_hole or [(None, p) for p in fairway_polys])
        ],
    }
    stats = {
        "pixels": {k: int(v.sum()) for k, v in masks.as_dict().items()},
        "polygons": {k: len(v) for k, v in polygons.items()},
        "gsd_m": round(geo.meters_per_pixel, 4),
        "dual_source": dual_source,
        "fusion": fusion_stats,
    }
    backend = "dual_imagery_fusion" if dual_source else "imagery_fusion"
    return AnalysisResult(
        polygons=polygons,
        masks=masks,
        geo=geo,
        stats=stats,
        backend=backend,
        dual_source=dual_source,
        fusion_stats=fusion_stats,
    )


def _line_ref(fc: dict | None, index: int) -> int | None:
    feats = (fc or {}).get("features") or []
    if index < 0 or index >= len(feats):
        return None
    ref = (feats[index].get("properties") or {}).get("ref")
    if ref is None:
        return None
    digits = "".join(ch for ch in str(ref) if ch.isdigit())
    return int(digits) if digits else None


def _merge_new(existing: list[Polygon], extra: list[Polygon]) -> list[Polygon]:
    kept = list(existing)
    for poly in extra:
        if any(iou(poly, e) >= 0.2 or poly.intersects(e) and iou(poly, e) >= 0.08 for e in kept):
            continue
        kept.append(poly)
    return kept


def _near_play(
    geom: BaseGeometry,
    layer_id: str,
    hole_lines: list[LineString],
    pins: list[Point],
) -> bool:
    if not hole_lines and not pins:
        return True
    limit = PLAY_NEAR_M.get(layer_id, 50.0)
    c = geom.centroid
    for pin in pins:
        if distance_meters(c, pin) <= limit:
            return True
    for line in hole_lines:
        if distance_meters(c, line) <= limit:
            return True
        if distance_meters(c, Point(line.coords[0])) <= limit or distance_meters(c, Point(line.coords[-1])) <= limit:
            return True
    return False


def _shape_ok(geom: BaseGeometry, layer_id: str) -> BaseGeometry | None:
    if layer_id == "green":
        hull = geom.convex_hull
        if hull.is_empty or hull.geom_type not in {"Polygon"}:
            return None
        if area_m2(hull) < PIN_GREEN_AREA_M2[0] or area_m2(hull) > PIN_GREEN_AREA_M2[1]:
            return None
        if _compactness(hull) < GREEN_MIN_COMPACT:
            return None
        return hull
    if layer_id == "bunker":
        if _compactness(geom) < BUNKER_MIN_COMPACT:
            return None
        if _elongation(geom) > BUNKER_MAX_ELONGATION:
            return None
        if area_m2(geom) < BUNKER_AREA_M2[0] or area_m2(geom) > BUNKER_AREA_M2[1]:
            return None
        return geom
    if layer_id == "tee":
        if _compactness(geom) < 0.18:
            return None
        if area_m2(geom) < TEE_AREA_M2[0] or area_m2(geom) > TEE_AREA_M2[1]:
            return None
        return geom
    return geom


def _shape_filter(
    polys: list[Polygon],
    layer_id: str,
    hole_lines: list[LineString],
    pins: list[Point],
) -> tuple[list[Polygon], int]:
    kept: list[Polygon] = []
    rejected = 0
    for poly in polys:
        shaped = _shape_ok(poly, layer_id)
        if shaped is None:
            rejected += 1
            continue
        if not _near_play(shaped, layer_id, hole_lines, pins):
            rejected += 1
            continue
        kept.append(shaped)
    return kept, rejected


def _extract_tees(
    tee_mask: np.ndarray,
    geo: MosaicGeo,
    osm_layers: dict,
    hole_lines: list[LineString],
    pins: list[Point],
    greens: list[Polygon],
) -> list[Polygon]:
    polys = mask_to_polygons(
        tee_mask, geo, min_area_m2=TEE_AREA_M2[0], max_area_m2=TEE_AREA_M2[1], min_compactness=0.18
    )
    osm_tees = geoms_from_fc(osm_layers.get("tee"))
    starts: list[Point] = []
    for line in hole_lines:
        starts.append(Point(line.coords[0]))
        starts.append(Point(line.coords[-1]))
    for t in osm_tees:
        starts.append(t.centroid)
    pin_pts = pins
    green_u = unary_union(greens) if greens else None
    out: list[Polygon] = []
    for poly in polys:
        c = poly.centroid
        if green_u is not None and (green_u.contains(c) or distance_meters(poly, green_u) < 8):
            continue
        near_pin = any(distance_meters(c, p) < 22 for p in pin_pts)
        if near_pin:
            continue
        near_start = any(distance_meters(c, s) < 48 for s in starts) if starts else True
        if not near_start:
            continue
        out.append(poly)
    return out


def try_onnx_sam(rgb: np.ndarray, weights: str | None) -> dict[str, Any]:
    """Optional SAM/ONNX hook. Default path never depends on this."""
    if not weights:
        return {"status": "skipped", "reason": "no_weights"}
    path = Path(weights)
    if not path.exists():
        return {"status": "skipped", "reason": "weights_missing", "path": str(path)}
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return {"status": "skipped", "reason": "onnxruntime_not_installed", "path": str(path)}
    return {
        "status": "weights_present_not_applied",
        "path": str(path),
        "note": "Color+texture fusion is the default CPU path; SAM/ONNX refine is a future hook.",
        "rgb_shape": list(rgb.shape),
    }


def mosaic_path(imagery: dict[str, Any]) -> Path | None:
    for key in ("enhanced_path", "path"):
        val = imagery.get(key)
        if val and Path(val).exists():
            return Path(val)
    return None


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def save_class_preview(masks: ClassMasks, dest: Path) -> None:
    h, w = masks.green.shape
    preview = np.zeros((h, w, 3), np.uint8)
    preview[masks.soft_grass] = (40, 90, 40)
    preview[masks.fairway] = (61, 139, 64)
    preview[masks.green] = (124, 252, 0)
    preview[masks.tee] = (152, 251, 152)
    preview[masks.bunker] = (237, 201, 175)
    preview[masks.water] = (30, 144, 255)
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview).save(dest)


def analyze_and_fuse(ctx: Any) -> AnalysisResult:
    """Run mosaic analysis and merge polygons into ctx.layers (OSM pegged)."""
    path = mosaic_path(ctx.imagery)
    if path is None:
        ctx.masks = {
            "backend": "osm_prior",
            "fusion": "osm_pegged",
            "status": "osm_prior",
            "dual_source": False,
            "note": "No mosaic on disk; OSM prior only.",
        }
        ctx.quality["segmentation_backend"] = "osm_prior"
        ctx.log("Segmentation: backend=osm_prior (no mosaic)")
        return AnalysisResult(backend="osm_prior", stats={"reason": "no_mosaic"})

    rgb = load_rgb(path)
    geo = MosaicGeo.from_imagery(ctx.imagery, rgb)
    osm_layers = ctx.layers
    osm_ids = {
        lid: {
            (feat.get("properties") or {}).get("osm_id")
            for feat in (osm_layers.get(lid) or {}).get("features", [])
            if (feat.get("properties") or {}).get("osm_id") is not None
        }
        for lid in FUSION_LAYERS
    }
    osm_counts = {lid: len((osm_layers.get(lid) or {}).get("features") or []) for lid in FUSION_LAYERS}

    second_info = ctx.imagery.get("second") or {}
    second_rgb = None
    second_geo = None
    second_path = second_info.get("path")
    if second_info.get("status") == "ok" and second_path and Path(second_path).exists():
        try:
            second_rgb = load_rgb(Path(second_path))
            second_geo = MosaicGeo.from_imagery(second_info, second_rgb)
        except Exception as exc:
            ctx.log(f"Second mosaic unreadable ({exc}); imagery adds disabled")
            second_rgb = None
            second_geo = None

    has_dual = second_rgb is not None
    pins = _points_from_layer(osm_layers.get("pin"))
    temporal_rgb = None
    temporal_info = ctx.imagery.get("temporal") or {}
    temporal_path = temporal_info.get("path")
    if temporal_info.get("status") == "ok" and temporal_path and Path(temporal_path).exists():
        try:
            t_rgb = load_rgb(Path(temporal_path))
            t_geo = MosaicGeo.from_imagery(temporal_info, t_rgb)
            if t_rgb.shape[:2] == (geo.height, geo.width) and t_geo.west == geo.west:
                temporal_rgb = t_rgb
            else:
                temporal_rgb = remap_rgb(t_rgb, t_geo, geo)
        except Exception as exc:
            ctx.log(f"Temporal mosaic unreadable ({exc}); pin greens use primary only")
            temporal_rgb = None
    pin_greens = extract_pin_greens(rgb, geo, pins, second_rgb=temporal_rgb)
    crop_greens = extract_pin_greens_from_crops(
        (ctx.imagery.get("pin_crops") or {}).get("crops") or []
    )
    pin_greens = _prefer_crop_pin_greens(pin_greens, crop_greens)
    result = analyze_mosaic(
        rgb,
        geo,
        osm_layers,
        hole_lines=_hole_lines(osm_layers.get("hole_centerline")),
        pins=pins,
        second_rgb=second_rgb,
        second_geo=second_geo,
    )
    result.pin_greens = pin_greens

    fusion_stats = dict(result.fusion_stats or result.stats.get("fusion") or {})
    fusion_stats.setdefault("osm_kept", 0)
    fusion_stats.setdefault("imagery_added", 0)
    fusion_stats.setdefault("imagery_rejected_single_source", 0)
    fusion_stats.setdefault("imagery_rejected_shape", 0)
    fusion_stats.setdefault("imagery_rejected_overlap", 0)
    fusion_stats.setdefault("imagery_rejected_layer", 0)
    fusion_stats.setdefault("osm_refined", 0)

    for layer_id in FUSION_LAYERS:
        imagery_geoms = [g for g, _p in result.polygons.get(layer_id, [])]
        agreed = [has_dual] * len(imagery_geoms)
        if not has_dual:
            # Single mosaic is never enough to ADD features.
            agreed = [False] * len(imagery_geoms)
        kept, _practice = fuse_layer(
            ctx.layers.get(layer_id),
            imagery_geoms,
            layer_id,
            agreed=agreed,
            stats=fusion_stats,
        )
        ctx.layers[layer_id] = feature_collection(kept)
        after_ids = {
            (feat.get("properties") or {}).get("osm_id")
            for feat in kept
            if (feat.get("properties") or {}).get("osm_id") is not None
        }
        missing = osm_ids.get(layer_id, set()) - after_ids
        if missing:
            ctx.log(f"WARNING: OSM {layer_id} ids dropped: {sorted(missing)[:8]}")

    refined, refine_stats = refine_osm_greens(ctx.layers.get("green"), pin_greens, pins)
    if refined:
        ctx.layers["green"] = feature_collection(refined)
    fusion_stats["osm_refined"] = refine_stats.get("osm_refined", 0)
    fusion_stats["refine_rejected_shared"] = refine_stats.get("refine_rejected_shared", 0)
    fusion_stats["refine_rejected_gate"] = refine_stats.get("refine_rejected_gate", 0)

    tree_feats = extract_play_trees(rgb, geo, ctx.layers)
    merged_trees = merge_tree_layer(ctx.layers.get("tree"), tree_feats)
    if merged_trees:
        ctx.layers["tree"] = feature_collection(merged_trees)
    fusion_stats["trees_imagery"] = len(tree_feats)
    fusion_stats["trees_total"] = len(merged_trees)

    preview = ctx.work_dir / "imagery" / "class_preview.png"
    if result.masks is not None:
        try:
            save_class_preview(result.masks, preview)
        except Exception:
            preview = None

    from app.config import settings

    onnx = try_onnx_sam(rgb, settings.segmentation_weights or None)
    backend = "dual_imagery_fusion" if has_dual else "osm_priority"
    ctx.masks = {
        "backend": backend,
        "fusion": "osm_refine_dual_agree",
        "status": backend,
        "dual_source": has_dual,
        "second_source": (second_info.get("source") if has_dual else None),
        "model_weights": settings.segmentation_weights or None,
        "onnx": onnx,
        "geo": geo.to_dict(),
        "counts": result.stats.get("polygons", {}),
        "pixel_counts": result.stats.get("pixels", {}),
        "preview": str(preview) if preview else None,
        "note": (
            "OSM play features keep their identity. Greens may shrink to a "
            "pin-seeded imagery outline. Imagery adds require "
            f"{'Sentinel-2/Clarity dual agreement' if has_dual else 'a second mosaic (none loaded)'}"
        ),
        "class_masks": result.masks.as_dict() if result.masks is not None else {},
        "fusion_stats": fusion_stats,
        "pin_greens": [
            {
                "pin": item["pin"],
                "coordinates": [list(c) for c in item["geometry"].exterior.coords],
                "area_m2": item["area_m2"],
                "sources": item.get("sources") or [],
            }
            for item in pin_greens
            if getattr(item.get("geometry"), "exterior", None) is not None
        ],
        "temporal_source": temporal_info.get("source"),
    }
    ctx.quality["segmentation_backend"] = backend
    ctx.quality["imagery_polygons"] = result.stats.get("polygons", {})
    ctx.quality["osm_counts_before_fusion"] = osm_counts
    ctx.quality["fusion_stats"] = {
        k: v for k, v in fusion_stats.items() if k != "class_masks" and not hasattr(v, "shape")
    }
    ctx.quality["second_source"] = second_info.get("source") if has_dual else None
    ctx.quality["temporal_source"] = temporal_info.get("source")
    ctx.quality["pin_crop_source"] = (ctx.imagery.get("pin_crops") or {}).get("source")
    ctx.quality["pin_greens_recovered"] = len(pin_greens)
    n = result.stats.get("polygons", {})
    ctx.log(
        f"Segmentation: backend={backend} dual={has_dual} "
        f"second={second_info.get('source') or 'none'} "
        f"green={n.get('green', 0)} fairway={n.get('fairway', 0)} "
        f"bunker={n.get('bunker', 0)} tee={n.get('tee', 0)} water={n.get('water', 0)} "
        f"osm_kept={fusion_stats.get('osm_kept', 0)} "
        f"osm_refined={fusion_stats.get('osm_refined', 0)} "
        f"imagery_added={fusion_stats.get('imagery_added', 0)} "
        f"rejected_single={fusion_stats.get('imagery_rejected_single_source', 0)} "
        f"rejected_shape={fusion_stats.get('imagery_rejected_shape', 0)} "
        f"pin_greens={len(pin_greens)} "
        f"trees={fusion_stats.get('trees_total', 0)}"
    )
    return result
