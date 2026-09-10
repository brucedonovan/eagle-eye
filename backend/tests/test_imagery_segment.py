import numpy as np
from shapely.geometry import Point, box

from app.services.geometry import area_m2, as_geom, compactness, feature_collection, iou, to_feature
from app.services.imagery_segment import (
    MosaicGeo,
    analyze_mosaic,
    derive_green_fringes,
    extract_pin_greens,
    extract_pin_greens_from_crops,
    extract_play_trees,
    fuse_layer,
    refine_osm_greens,
    vegetation_index,
)


def _synthetic_mosaic():
    """Tiny RGB with a vivid green patch and a tan sand patch."""
    rgb = np.zeros((80, 80, 3), np.uint8)
    rgb[:] = (42, 48, 36)  # dark rough
    rgb[12:38, 10:42] = (88, 145, 95)  # putting-green turf
    rgb[50:66, 48:70] = (210, 195, 160)  # bunker sand
    # ~89 m × 89 m so the patches fall in catalog area ranges
    geo = MosaicGeo.from_bbox(80, 80, (0.0, 0.0, 0.0008, 0.0008))
    return rgb, geo


def test_vegetation_index_positive_on_green():
    rgb, _geo = _synthetic_mosaic()
    veg = vegetation_index(rgb)
    assert float(veg[20, 20]) > 0.15
    assert float(veg[58, 58]) < 0.05


def test_from_imagery_bbox_crop_keeps_affine():
    rgb, geo = _synthetic_mosaic()
    crop = {
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "bbox": [geo.west, geo.south, geo.east, geo.north],
        "mosaic_bbox": [geo.west, geo.south, geo.east, geo.north],
        "zoom": 19,
    }
    parsed = MosaicGeo.from_imagery(crop, rgb)
    assert parsed.tile_x_min is None
    assert abs(parsed.west - geo.west) < 1e-12
    assert abs(parsed.east - geo.east) < 1e-12


def test_geo_affine_roundtrip():
    geo = MosaicGeo.from_bbox(100, 50, (-9.26, 38.70, -9.25, 38.71))
    lon, lat = geo.pixel_to_lonlat(0, 0)
    assert abs(lon + 9.26) < 1e-9
    assert abs(lat - 38.71) < 1e-9
    x, y = geo.lonlat_to_pixel(lon, lat)
    assert abs(x) < 1e-6 and abs(y) < 1e-6
    lon2, lat2 = geo.pixel_to_lonlat(100, 50)
    assert abs(lon2 + 9.25) < 1e-9
    assert abs(lat2 - 38.70) < 1e-9


def test_synthetic_image_yields_green_and_bunker_polygons():
    rgb, geo = _synthetic_mosaic()
    result = analyze_mosaic(rgb, geo, {})
    greens = [g for g, _p in result.polygons.get("green", [])]
    bunkers = [g for g, _p in result.polygons.get("bunker", [])]
    assert greens, "expected at least one green-like polygon from the mosaic"
    assert bunkers, "expected at least one sand-like polygon from the mosaic"
    assert all(p.get("source") == "imagery" for _g, p in result.polygons["green"])
    assert GREEN_OK(greens[0])
    assert BUNKER_OK(bunkers[0])
    assert result.backend == "imagery_fusion"


def GREEN_OK(geom) -> bool:
    a = area_m2(geom)
    return 150 <= a <= 2500


def BUNKER_OK(geom) -> bool:
    a = area_m2(geom)
    return 15 <= a <= 2500


def test_fusion_adds_imagery_only_when_dual_agrees():
    rgb, geo = _synthetic_mosaic()
    result = analyze_mosaic(rgb, geo, {})
    imagery = [g for g, _p in result.polygons["green"]]
    kept, practice = fuse_layer(None, imagery, "green", agreed=[False] * len(imagery))
    assert practice == []
    assert kept == []
    kept, practice = fuse_layer(None, imagery, "green", agreed=[True] * len(imagery))
    assert practice == []
    assert kept == []
    bunkers = [g for g, _p in result.polygons["bunker"]]
    assert bunkers
    kept, _ = fuse_layer(None, bunkers, "bunker", agreed=[False] * len(bunkers))
    assert kept == []
    kept, _ = fuse_layer(None, bunkers, "bunker", agreed=[True] * len(bunkers))
    assert len(kept) == len(bunkers)
    assert kept[0]["properties"]["source"] == "imagery"


def test_fusion_keeps_osm_geometry_frozen():
    osm_green = box(0.00010, 0.00048, 0.00042, 0.00072)
    osm_fc = feature_collection([to_feature(osm_green, {"source": "openstreetmap", "osm_id": 1, "golf": "green"})])
    rgb, geo = _synthetic_mosaic()
    result = analyze_mosaic(rgb, geo, {"green": osm_fc})
    imagery = [g for g, _p in result.polygons["green"]]
    assert imagery
    kept, _practice = fuse_layer(osm_fc, imagery, "green", agreed=[True] * len(imagery))
    assert all(f["properties"]["source"] == "openstreetmap" for f in kept)
    assert any((f["properties"] or {}).get("osm_id") == 1 for f in kept)
    assert sum(1 for f in kept if iou(as_geom(f), osm_green) > 0.2) == 1
    osm_kept = next(f for f in kept if (f["properties"] or {}).get("osm_id") == 1)
    assert iou(as_geom(osm_kept), osm_green) > 0.99
    assert not any(f["properties"]["source"] == "imagery" for f in kept)


def test_osm_green_survives_sand_contradiction():
    sand = box(0.0, 0.0, 0.0004, 0.0004)
    osm_fc = feature_collection([to_feature(sand, {"source": "openstreetmap", "golf": "green", "osm_id": 9})])
    kept, practice = fuse_layer(osm_fc, [], "green", contradiction=[sand])
    assert practice == []
    assert len(kept) == 1
    assert kept[0]["properties"].get("osm_id") == 9
    assert kept[0]["properties"].get("practice") is not True


def test_single_image_false_bunker_is_dropped():
    rgb, geo = _synthetic_mosaic()
    result = analyze_mosaic(rgb, geo, {})
    bunkers = [g for g, _p in result.polygons["bunker"]]
    assert bunkers, "classifier still sees sand on the primary mosaic"
    kept, _ = fuse_layer(None, bunkers, "bunker", agreed=[False] * len(bunkers))
    assert kept == []


def test_dual_agree_and_keeps_overlapping_class():
    rgb, geo = _synthetic_mosaic()
    second = rgb.copy()
    result = analyze_mosaic(rgb, geo, {}, second_rgb=second, second_geo=geo)
    assert result.dual_source is True
    assert result.backend == "dual_imagery_fusion"
    assert result.polygons.get("green")
    assert result.polygons.get("bunker")


def test_pin_seed_recovers_green():
    rgb, geo = _synthetic_mosaic()
    pin = Point(0.00026, 0.00055)  # inside the painted green
    result = analyze_mosaic(rgb, geo, {"pin": feature_collection([to_feature(pin, {"golf": "pin"})])})
    greens = [g for g, _p in result.polygons["green"]]
    assert greens
    assert any(g.buffer(1e-7).contains(pin) or g.distance(pin) < 1e-4 for g in greens)


def test_extract_pin_green_is_compact_and_not_a_circle():
    rgb, geo = _synthetic_mosaic()
    pin = Point(0.00026, 0.00055)
    found = extract_pin_greens(rgb, geo, [pin])
    assert found
    geom = found[0]["geometry"]
    assert geom.buffer(1e-7).contains(pin) or geom.distance(pin) < 1e-4
    assert 150 <= area_m2(geom) <= 1800
    assert compactness(geom) >= 0.50
    # A painted rectangle must not collapse to the 11 m fallback circle.
    circle = pin.buffer(0.0001)
    assert iou(geom, circle) < 0.85


def test_peel_fringe_keeps_pin_and_shrinks():
    from shapely.geometry import box
    from app.services.imagery_segment import _peel_fringe

    pin = Point(-9.25150, 38.71411)
    fat = box(-9.25172, 38.71393, -9.25128, 38.71429)
    peeled = _peel_fringe(fat, pin)
    assert peeled.buffer(1e-7).contains(pin)
    assert area_m2(peeled) < area_m2(fat)
    assert area_m2(peeled) >= area_m2(fat) * 0.72
    assert area_m2(peeled) >= 90


def test_extract_pin_greens_from_crop_file(tmp_path):
    from PIL import Image

    rgb, geo = _synthetic_mosaic()
    path = tmp_path / "pin_crop.jpg"
    Image.fromarray(rgb).save(path)
    pin = Point(0.00026, 0.00055)
    found = extract_pin_greens_from_crops(
        [
            {
                "status": "ok",
                "path": str(path),
                "pin": [pin.x, pin.y],
                "source": "esri",
                "width": rgb.shape[1],
                "height": rgb.shape[0],
                "bbox": [geo.west, geo.south, geo.east, geo.north],
                "mosaic_bbox": [geo.west, geo.south, geo.east, geo.north],
            }
        ]
    )
    assert found
    assert found[0]["geometry"].buffer(1e-7).contains(pin) or found[0]["geometry"].distance(pin) < 1e-4
    assert "esri" in found[0]["sources"]


def test_refine_osm_green_shrinks_to_pin_outline():
    rgb, geo = _synthetic_mosaic()
    pin = Point(0.00026, 0.00055)
    found = extract_pin_greens(rgb, geo, [pin])
    assert found
    tight = found[0]["geometry"]
    fat = box(0.00005, 0.00035, 0.00055, 0.00075)
    assert area_m2(fat) > area_m2(tight) * 1.15
    osm_fc = feature_collection(
        [to_feature(fat, {"source": "openstreetmap", "osm_id": 7, "golf": "green"})]
    )
    kept, stats = refine_osm_greens(osm_fc, found, [pin])
    assert stats["osm_refined"] == 1
    assert kept[0]["properties"]["source"] == "osm_refined"
    assert kept[0]["properties"]["osm_id"] == 7
    assert iou(as_geom(kept[0]), fat) < 0.99
    assert area_m2(as_geom(kept[0])) < area_m2(fat)
    assert as_geom(kept[0]).buffer(1e-7).contains(pin)


def test_refine_skips_shared_osm_green():
    pin_a = Point(-9.2510, 38.7140)
    pin_b = Point(-9.2507, 38.7140)
    shared = box(-9.2512, 38.7138, -9.2505, 38.7142)
    pin_green = {
        "pin": [pin_a.x, pin_a.y],
        "geometry": box(-9.25115, 38.71385, -9.25085, 38.71415),
        "area_m2": 600,
        "sources": ["primary"],
    }
    osm_fc = feature_collection(
        [to_feature(shared, {"source": "openstreetmap", "osm_id": 3, "golf": "green"})]
    )
    kept, stats = refine_osm_greens(osm_fc, [pin_green], [pin_a, pin_b])
    assert stats["refine_rejected_shared"] == 1
    assert stats["osm_refined"] == 0
    assert iou(as_geom(kept[0]), shared) > 0.99
    assert kept[0]["properties"]["source"] == "openstreetmap"


def test_refine_rejects_outline_outside_osm():
    osm = box(-9.2512, 38.7138, -9.2508, 38.7141)
    pin = Point(-9.2510, 38.71395)
    elsewhere = box(-9.2496, 38.7124, -9.2492, 38.7127)
    pin_green = {"pin": [pin.x, pin.y], "geometry": elsewhere, "area_m2": 500, "sources": ["primary"]}
    osm_fc = feature_collection(
        [to_feature(osm, {"source": "openstreetmap", "osm_id": 4, "golf": "green"})]
    )
    kept, stats = refine_osm_greens(osm_fc, [pin_green], [pin])
    assert stats["osm_refined"] == 0
    assert iou(as_geom(kept[0]), osm) > 0.99


def test_refine_accepts_offset_overlapping_outline():
    """Hole 4 style: OSM is the same green, shifted a few metres."""
    osm = box(-9.25130, 38.71495, -9.25095, 38.71528)
    pin = Point(-9.25120, 38.71512)
    shifted = box(-9.25122, 38.71490, -9.25086, 38.71524)
    pin_green = {"pin": [pin.x, pin.y], "geometry": shifted, "area_m2": 700, "sources": ["primary"]}
    osm_fc = feature_collection(
        [to_feature(osm, {"source": "openstreetmap", "osm_id": 1046144887, "golf": "green"})]
    )
    kept, stats = refine_osm_greens(osm_fc, [pin_green], [pin])
    assert stats["osm_refined"] == 1
    assert kept[0]["properties"]["source"] == "osm_refined"
    from app.services.geometry import distance_meters

    snapped = as_geom(kept[0])
    assert distance_meters(snapped.centroid, shifted.centroid) < distance_meters(osm.centroid, shifted.centroid)
    assert snapped.buffer(1e-7).contains(pin)


def test_derive_green_fringe_is_collar_not_green():
    green = box(-9.2512, 38.7139, -9.2508, 38.7142)
    pin_pt = Point(-9.2510, 38.71405)
    fc = feature_collection(
        [to_feature(green, {"instance_id": "green-1", "osm_id": 11, "hole": 1, "golf": "green"})]
    )
    fringes = derive_green_fringes(fc)
    assert fringes
    ring = as_geom(fringes[0])
    assert not ring.contains(pin_pt)
    assert area_m2(ring) > 20
    assert fringes[0]["properties"]["golf"] == "fringe"
    assert fringes[0]["properties"]["hole"] == 1


def test_extract_play_trees_from_dark_canopy():
    rgb, geo = _synthetic_mosaic()
    rgb[18:30, 58:72] = (22, 62, 28)
    fairway = box(0.00045, 0.00040, 0.00075, 0.00070)
    found = extract_play_trees(
        rgb,
        geo,
        {
            "fairway": feature_collection([to_feature(fairway, {"golf": "fairway"})]),
            "green": feature_collection([]),
            "tree": feature_collection([]),
        },
    )
    assert found
    assert found[0]["properties"]["source"] == "imagery"
    assert found[0]["geometry"]["type"] == "Point"
