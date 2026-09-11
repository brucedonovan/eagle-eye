"""Build a SegFormer-ready dataset from top-ranked OSM courses.

Downloads ESRI World Imagery plus ESRI Clarity, rasterizes OSM vectors onto
each mosaic, and writes 512px chips: images/, masks/, mosaics/, preview/,
manifest.csv, label_map.json.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from shapely.geometry import box

from app.catalog import LAYER_BY_ID, SEMANTIC_CLASSES
from app.osm_training.census import CourseRecord, load_ranking_csv
from app.osm_training.scoring import completeness_score, counts_from_layers
from app.services.geometry import as_geom, buffer_meters, expand_bbox, geoms_from_fc
from app.services.imagery import ESRI_CLARITY, download_xyz_mosaic, esri_xyz
from app.services.imagery_segment import MosaicGeo, rasterize_geoms
from app.services.overpass import clip_layers_to_boundary, elements_to_layers, fetch_aoi

CLASS_ID = {name: index for index, name in enumerate(SEMANTIC_CLASSES)}

LAYER_TO_CLASS = {
    "fairway": "fairway",
    "green": "green",
    "green_fringe": "first_cut",
    "tee": "tee",
    "bunker": "bunker",
    "water": "water",
    "lake": "water",
    "cart_path": "cart_path",
    "walking_path": "walking_path",
    "tree": "tree",
    "woodland": "woodland",
    "building": "building",
    "clubhouse": "building",
    "maintenance": "building",
    "parking": "parking",
    "road": "road",
    "bridge": "bridge",
    "driving_range": "driving_range",
    "putting_green": "putting_green",
    "practice_bunker": "bunker",
    "natural_rough": "natural_rough",
    "managed_rough": "managed_rough",
}

# Later entries overwrite earlier ones so greens sit on top of fairways.
PAINT_ORDER = (
    "woodland",
    "natural_rough",
    "managed_rough",
    "driving_range",
    "putting_green",
    "practice",
    "fairway",
    "first_cut",
    "water",
    "bunker",
    "sand",
    "tee",
    "green",
    "cart_path",
    "walking_path",
    "road",
    "parking",
    "building",
    "bridge",
    "tree",
)

LINE_CLASSES = {"cart_path", "walking_path", "road"}
POINT_CLASSES = {"tree"}

CHIP_SIZE = 512
CHIP_STRIDE = 384
MIN_LABEL_FRAC = 0.001
EMPTY_KEEP_FRAC = 0.05

TRAIN_SOURCES = (
    ("esri", esri_xyz),
    ("clarity", lambda: ESRI_CLARITY),
)

MANIFEST_FIELDS = [
    "osm_type",
    "osm_id",
    "name",
    "score",
    "split",
    "image",
    "mask",
    "chip",
    "width",
    "height",
    "west",
    "south",
    "east",
    "north",
    "source",
    "zoom",
    "green",
    "fairway",
    "tee",
    "bunker",
    "hole",
    "status",
]


ProgressFn = Callable[[str], None]


def rows_from_ranking(
    ranking: Path | list[dict[str, Any]] | list[CourseRecord],
    *,
    top: int = 100,
    min_score: float = 100.0,
) -> list[dict[str, Any]]:
    if isinstance(ranking, Path):
        rows = load_ranking_csv(ranking)
    elif ranking and isinstance(ranking[0], CourseRecord):
        rows = [row.to_csv_row(i) for i, row in enumerate(ranking, start=1)]  # type: ignore[arg-type]
    else:
        rows = list(ranking)  # type: ignore[arg-type]
    kept: list[dict[str, Any]] = []
    for row in rows:
        flags = str(row.get("flags") or "")
        if "miniature" in flags or "driving_range_only" in flags:
            continue
        try:
            score = float(row.get("score") or 0)
        except ValueError:
            continue
        if score < min_score:
            continue
        kept.append(row)
        if len(kept) >= top:
            break
    return kept


def build_label_mask(
    layers: dict[str, Any],
    geo: MosaicGeo,
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize catalog layers into a uint8 semantic mask (0 = background)."""
    h, w = shape
    label = np.zeros((h, w), np.uint8)
    by_class: dict[str, list] = {name: [] for name in SEMANTIC_CLASSES}
    for layer_id, class_name in LAYER_TO_CLASS.items():
        fc = layers.get(layer_id)
        if not fc:
            continue
        by_class.setdefault(class_name, [])
        for geom in geoms_from_fc(fc):
            if geom.geom_type == "Point":
                by_class[class_name].append(buffer_meters(geom, 2.5))
            elif geom.geom_type == "MultiPoint":
                for pt in geom.geoms:
                    by_class[class_name].append(buffer_meters(pt, 2.5))
            else:
                by_class[class_name].append(geom)

    for class_name in PAINT_ORDER:
        class_id = CLASS_ID.get(class_name)
        if class_id is None:
            continue
        geoms = by_class.get(class_name) or []
        if not geoms:
            continue
        dilate = 2 if class_name in LINE_CLASSES else 0
        mask = rasterize_geoms(geoms, geo, shape, dilate_px=dilate)
        label[mask] = class_id
    return label


def colorize_mask(label: np.ndarray) -> np.ndarray:
    palette = _class_palette()
    h, w = label.shape
    rgb = np.zeros((h, w, 3), np.uint8)
    for class_id, color in palette.items():
        rgb[label == class_id] = color
    return rgb


def iter_chip_windows(
    height: int,
    width: int,
    *,
    size: int = CHIP_SIZE,
    stride: int = CHIP_STRIDE,
) -> list[tuple[int, int]]:
    if height <= 0 or width <= 0:
        return []
    if height <= size and width <= size:
        return [(0, 0)]
    last_y = max(height - size, 0)
    last_x = max(width - size, 0)
    ys = list(range(0, last_y + 1, stride))
    xs = list(range(0, last_x + 1, stride))
    if ys[-1] != last_y:
        ys.append(last_y)
    if xs[-1] != last_x:
        xs.append(last_x)
    return [(y, x) for y in ys for x in xs]


def chip_image_and_mask(
    rgb: np.ndarray,
    label: np.ndarray,
    *,
    size: int = CHIP_SIZE,
    stride: int = CHIP_STRIDE,
    min_label_frac: float = MIN_LABEL_FRAC,
    empty_keep_frac: float = EMPTY_KEEP_FRAC,
    seed_key: str = "",
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Cut a mosaic into padded `size` chips, dropping empty tiles."""
    height, width = label.shape
    windows = iter_chip_windows(height, width, size=size, stride=stride)
    chips: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    keep_all = len(windows) == 1
    for y, x in windows:
        tile_rgb, tile_lab = _extract_chip(rgb, label, y, x, size)
        labeled = float(np.count_nonzero(tile_lab)) / float(size * size)
        if keep_all or labeled >= min_label_frac:
            chips.append((y, x, tile_rgb, tile_lab))
            continue
        digest = sha1(f"{seed_key}:{y}:{x}".encode()).hexdigest()
        if int(digest[:8], 16) / 0xFFFFFFFF < empty_keep_frac:
            chips.append((y, x, tile_rgb, tile_lab))
    return chips


def _extract_chip(
    rgb: np.ndarray, label: np.ndarray, y: int, x: int, size: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = label.shape
    y2, x2 = min(y + size, height), min(x + size, width)
    tile_rgb = np.zeros((size, size, 3), np.uint8)
    tile_lab = np.zeros((size, size), np.uint8)
    patch_h, patch_w = y2 - y, x2 - x
    tile_rgb[:patch_h, :patch_w] = rgb[y:y2, x:x2]
    tile_lab[:patch_h, :patch_w] = label[y:y2, x:x2]
    return tile_rgb, tile_lab


def choose_split(osm_type: str, osm_id: int, val_frac: float = 0.15) -> str:
    digest = sha1(f"{osm_type}:{osm_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_frac else "train"


async def build_dataset(
    *,
    ranking: Path,
    out_dir: Path,
    top: int = 100,
    min_score: float = 100.0,
    val_frac: float = 0.15,
    chip_size: int = CHIP_SIZE,
    chip_stride: int = CHIP_STRIDE,
    progress: ProgressFn | None = None,
    fetch_layers=None,
    download_mosaic=None,
    imagery_sources: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    log = progress or (lambda _msg: None)
    rows = rows_from_ranking(ranking, top=top, min_score=min_score)
    if not rows:
        raise RuntimeError("No ranking rows passed the score / flag filter")

    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    preview_dir = out_dir / "preview"
    mosaics_dir = out_dir / "mosaics"
    for folder in (images_dir, masks_dir, preview_dir, mosaics_dir):
        folder.mkdir(parents=True, exist_ok=True)

    fetch_layers = fetch_layers or _fetch_course_layers
    download_mosaic = download_mosaic or _download_course_mosaic
    wanted = imagery_sources or tuple(source_id for source_id, _factory in TRAIN_SOURCES)
    sources = [
        (source_id, factory())
        for source_id, factory in TRAIN_SOURCES
        if source_id in wanted
    ]
    if not sources:
        raise RuntimeError(f"No training imagery sources from {wanted}")

    manifest: list[dict[str, Any]] = []
    label_map = {str(i): name for i, name in enumerate(SEMANTIC_CLASSES)}
    (out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    def flush(*, partial: bool) -> dict[str, Any]:
        _write_manifest(out_dir / "manifest.csv", manifest)
        meta = {
            "version": 2,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "partial" if partial else "complete",
            "classes": list(SEMANTIC_CLASSES),
            "imagery_sources": [source_id for source_id, _src in sources],
            "chip_size": chip_size,
            "chip_stride": chip_stride,
            "top": top,
            "min_score": min_score,
            "samples_ok": sum(1 for row in manifest if row.get("status") == "ok"),
            "samples_attempted": len(manifest),
            "courses_total": len(rows),
            "note": (
                "512px chips from ESRI World Imagery plus ESRI Clarity, "
                "with OSM vectors rasterized onto each mosaic. "
                "Only train on imagery you have rights to use."
            ),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        splits = {
            "train": [row["image"] for row in manifest if row.get("split") == "train"],
            "val": [row["image"] for row in manifest if row.get("split") == "val"],
        }
        (out_dir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
        return meta

    finished = False
    try:
        for index, row in enumerate(rows, start=1):
            osm_type = str(row.get("osm_type") or "way")
            osm_id = str(row.get("osm_id") or index)
            name = str(row.get("name") or f"{osm_type}/{osm_id}")
            stem = f"{osm_type}_{osm_id}"
            log(f"[{index}/{len(rows)}] {name} ({stem})")
            try:
                added = await _process_course(
                    row,
                    stem=stem,
                    osm_type=osm_type,
                    osm_id=osm_id,
                    name=name,
                    out_dir=out_dir,
                    sources=sources,
                    val_frac=val_frac,
                    chip_size=chip_size,
                    chip_stride=chip_stride,
                    fetch_layers=fetch_layers,
                    download_mosaic=download_mosaic,
                    log=log,
                )
                manifest.extend(added)
                if not any(item.get("status") == "ok" for item in added):
                    manifest.append(_manifest_row(row, split="", status="no_chips"))
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                log(f"  skipped: {exc}")
                manifest.append(_manifest_row(row, split="", status=str(exc).split("\n", 1)[0][:180]))
            flush(partial=True)
        finished = True
    finally:
        meta = flush(partial=not finished)

    log(f"Dataset {meta['samples_ok']}/{meta['samples_attempted']} chips → {out_dir}")
    return meta


async def _process_course(
    row: dict[str, Any],
    *,
    stem: str,
    osm_type: str,
    osm_id: str,
    name: str,
    out_dir: Path,
    sources: list[tuple[str, Any]],
    val_frac: float,
    chip_size: int,
    chip_stride: int,
    fetch_layers,
    download_mosaic,
    log: ProgressFn,
) -> list[dict[str, Any]]:
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    preview_dir = out_dir / "preview"
    mosaics_dir = out_dir / "mosaics"
    _migrate_legacy_full_mosaic(stem, images_dir, masks_dir, mosaics_dir)

    split = choose_split(osm_type, int(osm_id), val_frac)
    layers_box: tuple[tuple[float, float, float, float], dict[str, Any]] | None = None
    counts = None
    refined_score: float | None = None
    samples: list[dict[str, Any]] = []

    async def ensure_layers():
        nonlocal layers_box, counts, refined_score
        if layers_box is None:
            layers_box = await fetch_layers(row)
            counts = counts_from_layers(layers_box[1])
            refined_score = completeness_score(
                counts, {"name": name, "holes": str(row.get("holes_tag") or "")}
            ).score
        return layers_box

    for source_id, source_obj in sources:
        mosaic_dir = mosaics_dir / source_id
        mosaic_dir.mkdir(parents=True, exist_ok=True)
        mosaic_path = mosaic_dir / f"{stem}.jpg"
        mask_path = mosaic_dir / f"{stem}_mask.png"
        existing_chips = sorted(images_dir.glob(f"{stem}_{source_id}_r*_c*.jpg"))
        if existing_chips and all(
            (masks_dir / f"{chip.stem}.png").exists() for chip in existing_chips
        ):
            bbox = _row_bbox(row)
            for chip in existing_chips:
                rgb = np.array(Image.open(chip).convert("RGB"))
                samples.append(
                    _ok_sample_row(
                        row,
                        osm_type=osm_type,
                        osm_id=osm_id,
                        name=name,
                        stem=chip.stem,
                        split=split,
                        width=rgb.shape[1],
                        height=rgb.shape[0],
                        bbox=bbox,
                        source=source_id,
                        zoom="",
                        counts=counts,
                        score=refined_score,
                        chip=_chip_token(chip.stem),
                    )
                )
            log(f"  resume {source_id} chips={len(existing_chips)}")
            continue
        try:
            imagery = await _ensure_mosaic(
                mosaic_path, row, source_obj, download_mosaic, ensure_layers
            )
            if imagery.get("status") != "ok":
                log(f"  {source_id}: {imagery.get('reason') or 'imagery_failed'}")
                continue
            rgb = np.array(Image.open(mosaic_path).convert("RGB"))
            if mask_path.exists():
                label = np.array(Image.open(mask_path))
            else:
                bbox, layers = await ensure_layers()
                geo = MosaicGeo.from_imagery(imagery, rgb)
                label = build_label_mask(layers, geo, rgb.shape[:2])
                Image.fromarray(label, mode="L").save(mask_path)
                Image.fromarray(colorize_mask(label), mode="RGB").save(
                    preview_dir / f"{stem}_{source_id}.png"
                )
            chips = chip_image_and_mask(
                rgb,
                label,
                size=chip_size,
                stride=chip_stride,
                seed_key=f"{stem}:{source_id}",
            )
            west, south, east, north = _bbox_from_imagery(imagery, row)
            for y, x, tile_rgb, tile_lab in chips:
                chip_stem = f"{stem}_{source_id}_r{y}_c{x}"
                Image.fromarray(tile_rgb, mode="RGB").save(
                    images_dir / f"{chip_stem}.jpg", format="JPEG", quality=90
                )
                Image.fromarray(tile_lab, mode="L").save(masks_dir / f"{chip_stem}.png")
                samples.append(
                    _ok_sample_row(
                        row,
                        osm_type=osm_type,
                        osm_id=osm_id,
                        name=name,
                        stem=chip_stem,
                        split=split,
                        width=tile_rgb.shape[1],
                        height=tile_rgb.shape[0],
                        bbox=(west, south, east, north),
                        source=source_id,
                        zoom=imagery.get("zoom") or "",
                        counts=counts,
                        score=refined_score,
                        chip=f"r{y}_c{x}",
                    )
                )
            log(f"  {source_id} chips={len(chips)}")
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            log(f"  {source_id} skipped: {exc}")
    return samples


async def _ensure_mosaic(
    mosaic_path: Path,
    row: dict[str, Any],
    source_obj: Any,
    download_mosaic,
    ensure_layers,
) -> dict[str, Any]:
    sidecar = mosaic_path.with_suffix(".json")
    if mosaic_path.exists():
        rgb = np.array(Image.open(mosaic_path).convert("RGB"))
        if sidecar.exists():
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    payload.setdefault("path", str(mosaic_path))
                    payload.setdefault("status", "ok")
                    return payload
            except json.JSONDecodeError:
                pass
        west, south, east, north = _row_bbox(row)
        return {
            "status": "ok",
            "path": str(mosaic_path),
            "width": rgb.shape[1],
            "height": rgb.shape[0],
            "bbox": [west, south, east, north],
            "mosaic_bbox": [west, south, east, north],
        }
    bbox, _layers = await ensure_layers()
    imagery = await _call_download(download_mosaic, bbox, mosaic_path, source_obj)
    if imagery.get("status") == "ok":
        _write_mosaic_sidecar(sidecar, imagery)
    return imagery


async def _call_download(download_mosaic, bbox, dest: Path, source_obj: Any) -> dict[str, Any]:
    try:
        return await download_mosaic(bbox, dest, source=source_obj)
    except TypeError:
        return await download_mosaic(bbox, dest)


def _write_mosaic_sidecar(path: Path, imagery: dict[str, Any]) -> None:
    keep = {
        key: imagery.get(key)
        for key in (
            "status",
            "source",
            "zoom",
            "width",
            "height",
            "bbox",
            "mosaic_bbox",
            "tile_x_min",
            "tile_y_min",
            "tile_x_max",
            "tile_y_max",
            "tile_size",
        )
        if key in imagery
    }
    keep["path"] = imagery.get("path")
    path.write_text(json.dumps(keep), encoding="utf-8")


def _bbox_from_imagery(
    imagery: dict[str, Any], row: dict[str, Any]
) -> tuple[float, float, float, float]:
    bbox = imagery.get("mosaic_bbox") or imagery.get("bbox")
    if bbox and len(bbox) == 4:
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return _row_bbox(row)


def _chip_token(stem: str) -> str:
    parts = stem.rsplit("_", 2)
    if len(parts) >= 3 and parts[-2].startswith("r") and parts[-1].startswith("c"):
        return f"{parts[-2]}_{parts[-1]}"
    return ""


def _migrate_legacy_full_mosaic(
    stem: str, images_dir: Path, masks_dir: Path, mosaics_dir: Path
) -> None:
    legacy = images_dir / f"{stem}.jpg"
    dest_dir = mosaics_dir / "esri"
    dest = dest_dir / f"{stem}.jpg"
    if not legacy.exists() or dest.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy), str(dest))
    legacy_mask = masks_dir / f"{stem}.png"
    if legacy_mask.exists():
        shutil.move(str(legacy_mask), str(dest_dir / f"{stem}_mask.png"))


async def _fetch_course_layers(row: dict[str, Any]) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
    bbox = _row_bbox(row)
    west, south, east, north = expand_bbox(*bbox, pad_deg=0.0015)
    payload = await fetch_aoi(west, south, east, north)
    layers = elements_to_layers(payload)
    boundary = _select_boundary(layers, row)
    if boundary is not None:
        layers = clip_layers_to_boundary(layers, boundary)
        west, south, east, north = expand_bbox(*boundary.bounds, pad_deg=0.0008)
    return (west, south, east, north), layers


async def _download_course_mosaic(
    bbox: tuple[float, float, float, float],
    dest: Path,
    source=None,
) -> dict[str, Any]:
    west, south, east, north = bbox
    return await download_xyz_mosaic(
        west=west, south=south, east=east, north=north, dest=dest, source=source
    )


def _select_boundary(layers: dict[str, Any], row: dict[str, Any]):
    fc = layers.get("boundary")
    if not fc or not fc.get("features"):
        west, south, east, north = _row_bbox(row)
        return box(west, south, east, north)
    osm_id = str(row.get("osm_id") or "")
    for feat in fc["features"]:
        props = feat.get("properties") or {}
        if str(props.get("osm_id") or "") == osm_id:
            try:
                return as_geom(feat)
            except (TypeError, ValueError):
                continue
    geoms = geoms_from_fc(fc)
    if not geoms:
        return None
    return max(geoms, key=lambda g: g.area)


def _row_bbox(row: dict[str, Any]) -> tuple[float, float, float, float]:
    try:
        west = float(row["west"])
        south = float(row["south"])
        east = float(row["east"])
        north = float(row["north"])
        if east > west and north > south:
            return west, south, east, north
    except (KeyError, TypeError, ValueError):
        pass
    lon = float(row.get("lon") or 0)
    lat = float(row.get("lat") or 0)
    pad = 0.012
    return lon - pad, lat - pad, lon + pad, lat + pad


def _ok_sample_row(
    row: dict[str, Any],
    *,
    osm_type: str,
    osm_id: str,
    name: str,
    stem: str,
    split: str,
    width: int,
    height: int,
    bbox: tuple[float, float, float, float],
    source: str,
    zoom: Any,
    counts: Any,
    score: float | None = None,
    chip: str = "",
) -> dict[str, Any]:
    west, south, east, north = bbox
    green = counts.green if counts is not None else row.get("green") or ""
    fairway = counts.fairway if counts is not None else row.get("fairway") or ""
    tee = counts.tee if counts is not None else row.get("tee") or ""
    bunker = counts.bunker if counts is not None else row.get("bunker") or ""
    hole = counts.hole if counts is not None else row.get("hole") or ""
    return {
        "osm_type": osm_type,
        "osm_id": osm_id,
        "name": name,
        "score": f"{score:.3f}" if score is not None else (row.get("score") or ""),
        "split": split,
        "image": f"images/{stem}.jpg",
        "mask": f"masks/{stem}.png",
        "chip": chip,
        "width": width,
        "height": height,
        "west": f"{west:.6f}",
        "south": f"{south:.6f}",
        "east": f"{east:.6f}",
        "north": f"{north:.6f}",
        "source": source,
        "zoom": zoom,
        "green": green,
        "fairway": fairway,
        "tee": tee,
        "bunker": bunker,
        "hole": hole,
        "status": "ok",
    }


def _manifest_row(row: dict[str, Any], *, split: str, status: str) -> dict[str, Any]:
    return {
        "osm_type": row.get("osm_type") or "",
        "osm_id": row.get("osm_id") or "",
        "name": row.get("name") or "",
        "score": row.get("score") or "",
        "split": split,
        "image": "",
        "mask": "",
        "chip": "",
        "width": "",
        "height": "",
        "west": row.get("west") or "",
        "south": row.get("south") or "",
        "east": row.get("east") or "",
        "north": row.get("north") or "",
        "source": "",
        "zoom": "",
        "green": row.get("green") or "",
        "fairway": row.get("fairway") or "",
        "tee": row.get("tee") or "",
        "bunker": row.get("bunker") or "",
        "hole": row.get("hole") or "",
        "status": status,
    }


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in MANIFEST_FIELDS})


def _class_palette() -> dict[int, tuple[int, int, int]]:
    palette: dict[int, tuple[int, int, int]] = {0: (0, 0, 0)}
    fallback = {
        "sand": "bunker",
        "practice": "putting_green",
        "first_cut": "green_fringe",
    }
    for index, name in enumerate(SEMANTIC_CLASSES):
        if index == 0:
            continue
        spec = LAYER_BY_ID.get(name) or LAYER_BY_ID.get(fallback.get(name, ""))
        hex_color = spec.color if spec else "#888888"
        palette[index] = _hex_rgb(hex_color)
    return palette


def _hex_rgb(value: str) -> tuple[int, int, int]:
    raw = value.lstrip("#")
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
