import pytest
from shapely.geometry import Point, box

from app.services.imagery import download_pin_crops, esri_xyz, pin_bbox, pin_crop_xyz, static_map_bbox
from app.services.imagery_segment import _prefer_crop_pin_greens, extract_pin_greens_from_crops


def test_pin_bbox_is_about_requested_radius():
    west, south, east, north = pin_bbox(-9.252, 38.713, 48.0)
    assert west < -9.252 < east
    assert south < 38.713 < north
    width_m = (east - west) * 111_320 * 0.78
    height_m = (north - south) * 111_320
    assert 70 < width_m < 140
    assert 85 < height_m < 110


def test_static_map_bbox_is_centered():
    west, south, east, north = static_map_bbox(-9.252, 38.713, 19, 640)
    assert abs((west + east) / 2 + 9.252) < 1e-6
    assert abs((south + north) / 2 - 38.713) < 2e-5
    assert east > west and north > south


def test_pin_crop_xyz_defaults_to_esri_without_keys(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "mapbox_token", "")
    monkeypatch.setattr(settings, "google_maps_key", "")
    src = pin_crop_xyz()
    assert src.id == esri_xyz().id
    assert src.url_template


def test_prefer_crop_outline_over_mosaic():
    mosaic = [
        {
            "pin": [-9.25, 38.71],
            "geometry": box(-9.2502, 38.7098, -9.2498, 38.7102),
            "area_m2": 900,
            "sources": ["primary"],
        }
    ]
    crops = [
        {
            "pin": [-9.25, 38.71],
            "geometry": box(-9.25015, 38.70985, -9.24985, 38.71015),
            "area_m2": 650,
            "sources": ["google_static"],
        }
    ]
    merged = _prefer_crop_pin_greens(mosaic, crops)
    assert len(merged) == 1
    assert merged[0]["sources"] == ["google_static"]
    assert merged[0]["area_m2"] == 650


def test_extract_pin_greens_from_crops_skips_bad_entries(tmp_path):
    found = extract_pin_greens_from_crops(
        [
            {"status": "empty", "path": str(tmp_path / "missing.jpg"), "pin": [0, 0]},
            {"status": "ok", "path": str(tmp_path / "missing.jpg"), "pin": [0, 0]},
        ]
    )
    assert found == []


@pytest.mark.asyncio
async def test_pin_crops_include_pins_that_already_have_osm_greens(monkeypatch, tmp_path):
    from app.config import settings

    async def fake_xyz(**kwargs):
        dest = kwargs["dest"]
        dest.write_bytes(b"x")
        return {
            "status": "ok",
            "source": "esri",
            "path": str(dest),
            "bbox": [0, 0, 1, 1],
            "mosaic_bbox": [0, 0, 1, 1],
            "width": 10,
            "height": 10,
        }

    monkeypatch.setattr(settings, "google_maps_key", "")
    monkeypatch.setattr("app.services.imagery.download_xyz_mosaic", fake_xyz)
    pin = Point(-9.25, 38.71)
    green = box(-9.2502, 38.7098, -9.2498, 38.7102)
    result = await download_pin_crops(pins=[pin], osm_greens=[green], dest_dir=tmp_path)
    assert result["status"] == "ok"
    assert result["pins_needed"] == 1
    assert result["crops"][0]["has_osm_green"] is True
