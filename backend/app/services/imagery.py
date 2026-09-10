"""Imagery source registry and mosaic download."""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from app.config import settings
from app.services.geometry import lonlat_to_tile, tile_y_to_lat


@dataclass(frozen=True)
class ImagerySource:
    id: str
    title: str
    kind: str  # xyz | stac | api
    gsd_m: float
    url_template: str | None = None
    requires_key: str | None = None
    tile_size: int = 256

    @property
    def available(self) -> bool:
        if not self.requires_key:
            return True
        return bool(getattr(settings, self.requires_key, ""))


SOURCES: tuple[ImagerySource, ...] = (
    ImagerySource("nearmap", "Nearmap", "api", 0.08, requires_key="nearmap_api_key"),
    ImagerySource("maxar", "Maxar", "api", 0.30, requires_key="maxar_api_key"),
    ImagerySource("planet", "Planet", "api", 0.50, requires_key="planet_api_key"),
    ImagerySource("google", "Google Satellite", "xyz", 0.30, requires_key="google_maps_key"),
    ImagerySource("mapbox", "Mapbox Satellite", "xyz", 0.50, requires_key="mapbox_token"),
    ImagerySource(
        "esri",
        "ESRI World Imagery",
        "xyz",
        0.30,
        url_template="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    ),
    ImagerySource("sentinel2", "Sentinel-2 L2A", "stac", 10.0),
    ImagerySource("usgs", "USGS NAIP / 3DEP", "stac", 1.0),
)

# Independent public mosaics used only as the dual-fusion second look.
# EOX s2cloudless is Sentinel-2 (no key). ESRI Clarity is a different date/process.
SENTINEL2_CLOUDLESS = ImagerySource(
    "sentinel2_cloudless",
    "Sentinel-2 cloudless (EOX)",
    "xyz",
    10.0,
    url_template="https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg",
)
ESRI_CLARITY = ImagerySource(
    "esri_clarity",
    "ESRI World Imagery Clarity",
    "xyz",
    0.50,
    url_template="https://clarity.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
)
# Older yearly composite — used only to confirm turf at a pin, not for outlines.
SENTINEL2_CLOUDLESS_2020 = ImagerySource(
    "sentinel2_cloudless_2020",
    "Sentinel-2 cloudless 2020 (EOX)",
    "xyz",
    10.0,
    url_template="https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg",
)


def available_sources() -> list[ImagerySource]:
    return [s for s in SOURCES if s.available]


def mapbox_xyz() -> ImagerySource | None:
    token = settings.mapbox_token
    if not token:
        return None
    return ImagerySource(
        "mapbox",
        "Mapbox Satellite",
        "xyz",
        0.30,
        url_template=(
            "https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.jpg90"
            f"?access_token={token}"
        ),
        requires_key="mapbox_token",
        tile_size=512,
    )


def esri_xyz() -> ImagerySource:
    return next(s for s in SOURCES if s.id == "esri")


def pin_crop_xyz() -> ImagerySource:
    """Finest downloadable XYZ for a ~100 m pin crop."""
    keyed = mapbox_xyz()
    return keyed or esri_xyz()


def _fit_zoom(west: float, south: float, east: float, north: float, max_zoom: int) -> int:
    for zoom in range(max_zoom, 11, -1):
        x0, y1 = lonlat_to_tile(west, south, zoom)
        x1, y0 = lonlat_to_tile(east, north, zoom)
        tiles = (abs(x1 - x0) + 1) * (abs(y1 - y0) + 1)
        if tiles <= settings.imagery_max_tiles:
            return zoom
    return 12


def _mosaic_tile_bbox(
    x_min: int, y_min: int, x_max: int, y_max: int, zoom: int
) -> tuple[float, float, float, float]:
    n = 2**zoom
    west = x_min / n * 360.0 - 180.0
    east = (x_max + 1) / n * 360.0 - 180.0
    north = tile_y_to_lat(y_min, zoom)
    south = tile_y_to_lat(y_max + 1, zoom)
    return west, south, east, north


def best_source() -> ImagerySource:
    ranked = sorted(available_sources(), key=lambda s: s.gsd_m)
    return ranked[0]


async def download_xyz_mosaic(
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    dest: Path,
    source: ImagerySource | None = None,
    zoom: int | None = None,
) -> dict[str, Any]:
    source = source or next((s for s in available_sources() if s.kind == "xyz" and s.url_template), None)
    if source is None or not source.url_template:
        return {"status": "skipped", "reason": "no_public_xyz_source"}

    zoom = zoom or _fit_zoom(west, south, east, north, settings.imagery_max_zoom)
    x0, y1 = lonlat_to_tile(west, south, zoom)
    x1, y0 = lonlat_to_tile(east, north, zoom)
    x_min, x_max = min(x0, x1), max(x0, x1)
    y_min, y_max = min(y0, y1), max(y0, y1)
    nx, ny = x_max - x_min + 1, y_max - y_min + 1
    tiles = nx * ny
    if tiles > settings.imagery_max_tiles:
        return {
            "status": "skipped",
            "reason": "aoi_too_large",
            "tiles": tiles,
            "limit": settings.imagery_max_tiles,
            "source": source.id,
            "zoom": zoom,
        }

    dest.parent.mkdir(parents=True, exist_ok=True)
    mosaic = Image.new("RGB", (nx * source.tile_size, ny * source.tile_size))
    headers = {"User-Agent": settings.osm_user_agent}
    fetched = 0
    async with httpx.AsyncClient(timeout=30.0, headers=headers, follow_redirects=True) as client:
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                url = source.url_template.format(z=zoom, x=x, y=y)
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    tile = Image.open(BytesIO(resp.content)).convert("RGB")
                    mosaic.paste(tile, ((x - x_min) * source.tile_size, (y - y_min) * source.tile_size))
                    fetched += 1
                except Exception:
                    continue

    mosaic.save(dest, format="JPEG", quality=88)
    mosaic_bbox = _mosaic_tile_bbox(x_min, y_min, x_max, y_max, zoom)
    return {
        "status": "ok" if fetched else "empty",
        "source": source.id,
        "title": source.title,
        "gsd_m": source.gsd_m,
        "zoom": zoom,
        "tiles_fetched": fetched,
        "tiles_expected": tiles,
        "path": str(dest),
        "width": mosaic.size[0],
        "height": mosaic.size[1],
        "bbox": [west, south, east, north],
        "mosaic_bbox": list(mosaic_bbox),
        "tile_x_min": x_min,
        "tile_y_min": y_min,
        "tile_x_max": x_max,
        "tile_y_max": y_max,
        "tile_size": source.tile_size,
    }


async def download_second_mosaic(
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    dest: Path,
    primary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Second look for conservative fusion. Never the same tiles as the primary mosaic.

    Preference: Sentinel-2 cloudless (EOX public XYZ) at ~10 m, then ESRI Clarity.
    Features may be *added* only when this mosaic agrees with the primary.
    """
    primary = primary or {}
    attempts: list[dict[str, Any]] = []
    candidates: list[tuple[ImagerySource, int]] = [
        (SENTINEL2_CLOUDLESS, 15),
        (ESRI_CLARITY, max(12, int(primary.get("zoom") or 18) - 1)),
    ]
    for source, max_zoom in candidates:
        result = await download_xyz_mosaic(
            west=west,
            south=south,
            east=east,
            north=north,
            dest=dest,
            source=source,
            zoom=_fit_zoom(west, south, east, north, max_zoom),
        )
        attempts.append(
            {
                "source": source.id,
                "status": result.get("status"),
                "reason": result.get("reason"),
                "tiles_fetched": result.get("tiles_fetched"),
            }
        )
        if result.get("status") == "ok" and int(result.get("tiles_fetched") or 0) > 0:
            result["role"] = "dual_fusion_second"
            result["attempts"] = attempts
            result["note"] = (
                f"Second source={source.id} ({source.title}). "
                "Resampled onto the primary grid for pixel-wise class agreement."
            )
            return result
    return {
        "status": "skipped",
        "reason": "no_second_source",
        "role": "dual_fusion_second",
        "attempts": attempts,
        "note": "No second mosaic; imagery-only adds are disabled (OSM stays).",
    }


async def download_temporal_mosaic(
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    dest: Path,
    primary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """High-res alternate date for pin-green outlines (ESRI Clarity).

    Sentinel-2 is too coarse to draw a green. Clarity is a different ESRI
    mosaic/date at roughly the same GSD as World Imagery.
    """
    primary = primary or {}
    primary_zoom = int(primary.get("zoom") or 18)
    zoom = _fit_zoom(west, south, east, north, min(primary_zoom, 18))
    result = await download_xyz_mosaic(
        west=west,
        south=south,
        east=east,
        north=north,
        dest=dest,
        source=ESRI_CLARITY,
        zoom=zoom,
    )
    result["role"] = "temporal_pin_green"
    if result.get("status") == "ok" and int(result.get("tiles_fetched") or 0) > 0:
        result["note"] = "ESRI Clarity (alternate date) for pin-seeded green outlines."
        return result
    older = dest.with_name("mosaic_s2_2020.jpg")
    fallback = await download_xyz_mosaic(
        west=west,
        south=south,
        east=east,
        north=north,
        dest=older,
        source=SENTINEL2_CLOUDLESS_2020,
        zoom=_fit_zoom(west, south, east, north, 15),
    )
    fallback["role"] = "temporal_pin_green"
    fallback["clarity"] = {"status": result.get("status"), "reason": result.get("reason")}
    fallback["note"] = "Clarity unavailable; older Sentinel-2 2020 used only as veg confirm."
    return fallback


def pin_bbox(lon: float, lat: float, radius_m: float) -> tuple[float, float, float, float]:
    dlat = radius_m / 111_320.0
    cos_lat = max(0.15, math.cos(math.radians(lat)))
    dlon = radius_m / (111_320.0 * cos_lat)
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def static_map_bbox(
    lon: float, lat: float, zoom: int, size_px: int
) -> tuple[float, float, float, float]:
    """WGS84 extent of a square static map centered on lon/lat (Web Mercator)."""
    n = 2**zoom
    fx = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(min(max(lat, -85.05112878), 85.05112878))
    fy = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    half = (size_px / 2.0) / 256.0
    west = (fx - half) / n * 360.0 - 180.0
    east = (fx + half) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (fy - half) / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (fy + half) / n))))
    return west, south, east, north


def _pin_inside_green(pin: Point, greens: list[BaseGeometry]) -> bool:
    for green in greens:
        try:
            if green.buffer(1e-7).contains(pin) or pin.distance(green) * 111_320 < 6:
                return True
        except Exception:
            continue
    return False


async def download_google_static_crop(
    *, lon: float, lat: float, dest: Path, zoom: int, size: int = 640
) -> dict[str, Any]:
    key = settings.google_maps_key
    if not key:
        return {"status": "skipped", "reason": "no_google_maps_key"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://maps.googleapis.com/maps/api/staticmap"
        f"?center={lat},{lon}&zoom={zoom}&size={size}x{size}"
        f"&maptype=satellite&scale=2&key={key}"
    )
    headers = {"User-Agent": settings.osm_user_agent}
    try:
        async with httpx.AsyncClient(timeout=30.0, headers=headers, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            if "image" not in (resp.headers.get("content-type") or ""):
                return {"status": "empty", "reason": "not_an_image", "source": "google_static"}
            dest.write_bytes(resp.content)
    except Exception as exc:
        return {"status": "empty", "reason": str(exc)[:160], "source": "google_static"}
    bbox = static_map_bbox(lon, lat, zoom, size)
    width, height = size * 2, size * 2
    try:
        with Image.open(dest) as im:
            width, height = im.size
    except Exception:
        pass
    return {
        "status": "ok",
        "source": "google_static",
        "title": "Google Satellite (Static)",
        "gsd_m": 156543.03392 * math.cos(math.radians(lat)) / (2**zoom) / 2.0,
        "zoom": zoom,
        "path": str(dest),
        "width": width,
        "height": height,
        "bbox": list(bbox),
        "mosaic_bbox": list(bbox),
        "pin": [lon, lat],
    }


async def download_pin_crops(
    *,
    pins: list[Point],
    osm_greens: list[BaseGeometry],
    dest_dir: Path,
) -> dict[str, Any]:
    """High-zoom crops for every pin — refine existing OSM greens and fill gaps."""
    if not pins:
        return {"status": "skipped", "reason": "no_pins", "crops": []}

    dest_dir.mkdir(parents=True, exist_ok=True)
    zoom = int(settings.imagery_pin_zoom)
    radius = float(settings.imagery_pin_radius_m)
    crops: list[dict[str, Any]] = []
    preferred = "google_static" if settings.google_maps_key else pin_crop_xyz().id

    for i, pin in enumerate(pins):
        dest = dest_dir / f"pin_{i}.jpg"
        result: dict[str, Any] | None = None
        if settings.google_maps_key:
            result = await download_google_static_crop(
                lon=pin.x, lat=pin.y, dest=dest, zoom=zoom
            )
            if result.get("status") != "ok":
                result = None
        if result is None:
            west, south, east, north = pin_bbox(pin.x, pin.y, radius)
            result = await download_xyz_mosaic(
                west=west,
                south=south,
                east=east,
                north=north,
                dest=dest,
                source=pin_crop_xyz(),
                zoom=zoom,
            )
        result["pin"] = [pin.x, pin.y]
        result["has_osm_green"] = _pin_inside_green(pin, osm_greens)
        result["role"] = "pin_green_crop"
        crops.append(result)

    ok = sum(1 for c in crops if c.get("status") == "ok")
    return {
        "status": "ok" if ok else "empty",
        "source": preferred,
        "zoom": zoom,
        "crops": crops,
        "pins_needed": len(pins),
        "crops_ok": ok,
        "note": (
            f"Pin crops via {preferred} at z{zoom} "
            f"({ok}/{len(pins)} pins for green refine/fill)."
        ),
    }
