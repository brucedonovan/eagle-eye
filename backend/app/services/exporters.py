"""Export FeatureCollections to GIS / robotics formats."""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from shapely.geometry import mapping

from app.catalog import LAYER_BY_ID
from app.services.geometry import as_geom


def write_geojson(layers: dict[str, dict[str, Any]], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "name": "eagle-eye-course",
        "features": _flatten(layers),
    }
    dest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return dest


def write_layer_geojsons(layers: dict[str, dict[str, Any]], dest_dir: Path) -> dict[str, str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for layer_id, fc in layers.items():
        path = dest_dir / f"{layer_id}.geojson"
        path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
        paths[layer_id] = str(path)
    return paths


def write_wkt(layers: dict[str, dict[str, Any]], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for feat in _flatten(layers):
        geom = as_geom(feat)
        props = feat.get("properties") or {}
        lines.append(f"{props.get('layer')}\t{props.get('hole', '')}\t{geom.wkt}")
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def write_kml(layers: dict[str, dict[str, Any]], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    folders = []
    for layer_id, fc in layers.items():
        if not fc:
            continue
        spec = LAYER_BY_ID.get(layer_id)
        title = spec.title if spec else layer_id
        color = _kml_color(spec.color if spec else "#ffffff")
        placemarks = []
        for feat in fc.get("features", []):
            geom = as_geom(feat)
            name = escape(str((feat.get("properties") or {}).get("name") or layer_id))
            coords = _kml_coords(mapping(geom))
            gtype = geom.geom_type
            if gtype in {"Polygon", "MultiPolygon"}:
                inner = f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon>"
            elif gtype in {"LineString", "MultiLineString"}:
                inner = f"<LineString><coordinates>{coords}</coordinates></LineString>"
            else:
                inner = f"<Point><coordinates>{coords}</coordinates></Point>"
            placemarks.append(f"<Placemark><name>{name}</name>{inner}</Placemark>")
        folders.append(
            f"<Folder><name>{escape(title)}</name><Style><LineStyle><color>{color}</color></LineStyle>"
            f"<PolyStyle><color>{color}</color></PolyStyle></Style>{''.join(placemarks)}</Folder>"
        )
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f"{''.join(folders)}</Document></kml>"
    )
    dest.write_text(kml, encoding="utf-8")
    return dest


def write_gpx(layers: dict[str, dict[str, Any]], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    trks = []
    for layer_id in ("hole_centerline", "cart_path", "waypoint_graph"):
        for feat in (layers.get(layer_id) or {}).get("features", []):
            geom = as_geom(feat)
            if geom.geom_type != "LineString":
                continue
            name = escape(str((feat.get("properties") or {}).get("name") or layer_id))
            pts = "".join(f'<trkpt lon="{x}" lat="{y}"></trkpt>' for x, y in geom.coords)
            trks.append(f"<trk><name>{name}</name><trkseg>{pts}</trkseg></trk>")
    gpx = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<gpx version="1.1" creator="Eagle Eye">'
        f"{''.join(trks)}</gpx>"
    )
    dest.write_text(gpx, encoding="utf-8")
    return dest


def write_dxf(layers: dict[str, dict[str, Any]], dest: Path) -> Path:
    """Minimal ASCII DXF (LWPOLYLINE / LINE) — no ezdxf dependency."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    ents: list[str] = []
    for layer_id, fc in layers.items():
        if not fc:
            continue
        for feat in fc.get("features", []):
            geom = as_geom(feat)
            if geom.geom_type in {"Polygon"}:
                ents.extend(_dxf_polyline(layer_id, list(geom.exterior.coords), closed=True))
            elif geom.geom_type == "LineString":
                ents.extend(_dxf_polyline(layer_id, list(geom.coords), closed=False))
            elif geom.geom_type == "Point":
                ents.extend(
                    [
                        "0", "POINT", "8", layer_id,
                        "10", str(geom.x), "20", str(geom.y), "30", "0.0",
                    ]
                )
    body = [
        "0", "SECTION", "2", "HEADER", "0", "ENDSEC",
        "0", "SECTION", "2", "TABLES", "0", "ENDSEC",
        "0", "SECTION", "2", "ENTITIES",
        *ents,
        "0", "ENDSEC", "0", "EOF",
    ]
    dest.write_text("\n".join(body) + "\n", encoding="utf-8")
    return dest


def write_ros_grid(layers: dict[str, dict[str, Any]], dest_dir: Path, resolution: float = 1.0) -> Path:
    """Write a coarse occupancy grid (PGM + YAML) in WGS84-projected meters approx."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    nav = layers.get("nav_mesh") or layers.get("boundary")
    no_go = layers.get("no_go")
    if not nav:
        raise RuntimeError("No navigation mesh or boundary to rasterize")

    from shapely.ops import unary_union

    traversable = unary_union([as_geom(f) for f in nav.get("features", [])])
    blocked = None
    if no_go:
        blocked = unary_union([as_geom(f) for f in no_go.get("features", [])])
    minx, miny, maxx, maxy = traversable.bounds
    # ~111_320 m per degree at equator; scale lon by cos(lat)
    lat0 = (miny + maxy) / 2
    mx = 111_320 * math.cos(math.radians(lat0))
    my = 110_540
    width = max(8, min(512, int((maxx - minx) * mx / resolution)))
    height = max(8, min(512, int((maxy - miny) * my / resolution)))
    from PIL import Image

    img = Image.new("L", (width, height), 205)  # unknown
    pixels = img.load()
    for row in range(height):
        y = maxy - (row + 0.5) * (maxy - miny) / height
        for col in range(width):
            x = minx + (col + 0.5) * (maxx - minx) / width
            from shapely.geometry import Point

            pt = Point(x, y)
            if blocked is not None and blocked.contains(pt):
                pixels[col, row] = 0
            elif traversable.contains(pt):
                pixels[col, row] = 254
    pgm = dest_dir / "map.pgm"
    yaml = dest_dir / "map.yaml"
    img.save(pgm)
    yaml.write_text(
        "\n".join(
            [
                f"image: {pgm.name}",
                f"resolution: {resolution}",
                f"origin: [{minx}, {miny}, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.25",
                "mode: trinary",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    zip_path = dest_dir / "ros_occupancy_grid.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(pgm, pgm.name)
        zf.write(yaml, yaml.name)
    return zip_path


def write_geopackage(layers: dict[str, dict[str, Any]], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import geopandas as gpd
        from shapely.geometry import shape
    except ImportError as exc:
        raise RuntimeError("geopandas is required for GeoPackage export") from exc

    frames = []
    for layer_id, fc in layers.items():
        if not fc or not fc.get("features"):
            continue
        gdf = gpd.GeoDataFrame.from_features(fc, crs="EPSG:4326")
        if "geometry" not in gdf.columns:
            gdf["geometry"] = [shape(f["geometry"]) for f in fc["features"]]
        gdf["layer"] = layer_id
        frames.append(gdf)
    if not frames:
        raise RuntimeError("No features to write")
    import pandas as pd

    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    merged.to_file(dest, driver="GPKG")
    return dest


def write_shapefile_zip(layers: dict[str, dict[str, Any]], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required for Shapefile export") from exc

    work = dest.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    written = []
    for layer_id, fc in layers.items():
        if not fc or not fc.get("features"):
            continue
        gdf = gpd.GeoDataFrame.from_features(fc, crs="EPSG:4326")
        shp = work / f"{layer_id}.shp"
        gdf.to_file(shp)
        written.append(layer_id)
    zip_path = dest
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in work.rglob("*"):
            zf.write(path, path.relative_to(work))
    return zip_path


def _flatten(layers: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    features = []
    for layer_id, fc in layers.items():
        if not fc:
            continue
        spec = LAYER_BY_ID.get(layer_id)
        for feat in fc.get("features", []):
            props = dict(feat.get("properties") or {})
            props["layer"] = layer_id
            props["layer_title"] = spec.title if spec else layer_id
            features.append({"type": "Feature", "geometry": feat.get("geometry"), "properties": props})
    return features


def _kml_color(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "ffffffff"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"a0{b}{g}{r}"


def _kml_coords(geom: dict[str, Any]) -> str:
    coords = geom.get("coordinates")
    flat: list[str] = []

    def walk(node: Any) -> None:
        if not node:
            return
        if isinstance(node[0], (int, float)):
            lon, lat = node[0], node[1]
            flat.append(f"{lon},{lat},0")
            return
        for child in node:
            walk(child)

    walk(coords)
    return " ".join(flat)


def _dxf_polyline(layer: str, coords: list[tuple], closed: bool) -> list[str]:
    ents = ["0", "LWPOLYLINE", "8", layer, "90", str(len(coords)), "70", "1" if closed else "0"]
    for x, y, *_rest in coords:
        ents.extend(["10", str(x), "20", str(y)])
    return ents
