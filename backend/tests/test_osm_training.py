import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

from app.catalog import SEMANTIC_CLASSES
from app.osm_training.census import (
    CourseRecord,
    GolfFeature,
    assign_features,
    iter_tiles,
    rank_courses,
    resolve_bboxes,
)
from app.osm_training.cli import main
from app.osm_training.dataset import (
    CLASS_ID,
    build_label_mask,
    chip_image_and_mask,
    choose_split,
    colorize_mask,
    iter_chip_windows,
    rows_from_ranking,
)
from app.osm_training.scoring import FeatureCounts, completeness_score, counts_from_layers
from app.services.geometry import feature_collection, to_feature
from app.services.imagery_segment import MosaicGeo


def _complete_18() -> FeatureCounts:
    return FeatureCounts(
        green=18,
        fairway=18,
        tee=18,
        bunker=40,
        hole=18,
        cart_path=12,
        water=4,
        pin=18,
        rough=8,
        green_ways=18,
        fairway_ways=18,
        tee_ways=18,
        hole_ways=18,
    )


def test_complete_18_hole_scores_near_100():
    result = completeness_score(_complete_18(), {"name": "Pebble Beach Golf Links", "holes": "18"})
    assert result.expected_holes == 18
    assert result.score >= 95
    assert "incomplete_core" not in result.flags


def test_complete_9_hole_is_not_punished_for_missing_the_back_nine():
    counts = FeatureCounts(
        green=9,
        fairway=9,
        tee=9,
        bunker=12,
        hole=9,
        cart_path=4,
        green_ways=9,
        fairway_ways=9,
        hole_ways=9,
    )
    result = completeness_score(counts, {"name": "Nine Hole Club", "holes": "9"})
    assert result.expected_holes == 9
    assert result.score >= 85


def test_boundary_only_course_ranks_near_zero():
    result = completeness_score(FeatureCounts(), {"name": "Empty Municipal"})
    assert result.score < 10
    assert "boundary_only" in result.flags
    assert "incomplete_core" in result.flags


def test_unnamed_empty_is_lowest():
    named = completeness_score(FeatureCounts(), {"name": "Named"})
    unnamed = completeness_score(FeatureCounts(), {})
    assert unnamed.score < named.score
    assert "unnamed" in unnamed.flags


def test_miniature_golf_is_downweighted():
    full = completeness_score(_complete_18(), {"name": "Real Course"})
    mini = completeness_score(_complete_18(), {"name": "Boardwalk Mini Golf", "golf": "miniature"})
    assert "miniature" in mini.flags
    assert mini.score < full.score * 0.5


def test_missing_water_does_not_zero_a_complete_course():
    dry = _complete_18()
    dry.water = 0
    result = completeness_score(dry, {"name": "Desert Dunes"})
    assert result.score >= 90


def test_counts_from_layers_matches_catalog_ids(fake_course):
    counts = counts_from_layers(fake_course.layers)
    assert counts.green == 2
    assert counts.fairway == 1
    assert counts.bunker == 1
    assert counts.cart_path == 1
    assert counts.green_ways == 2


def test_rank_prefers_complete_mapping():
    complete = completeness_score(_complete_18(), {"name": "A"})
    partial = completeness_score(
        FeatureCounts(green=5, fairway=4, tee=2, bunker=1, green_ways=5),
        {"name": "B"},
    )
    empty = completeness_score(FeatureCounts(), {"name": "C"})
    assert complete.score > partial.score > empty.score


@pytest.mark.asyncio
async def test_rank_courses_joins_features_and_writes_csv(tmp_path: Path):
    payload = {
        "elements": [
            {
                "type": "way",
                "id": 1,
                "center": {"lon": -9.25, "lat": 38.71},
                "bounds": {"minlon": -9.26, "minlat": 38.70, "maxlon": -9.24, "maxlat": 38.72},
                "tags": {"leisure": "golf_course", "name": "Jamor", "holes": "18"},
            },
            {
                "type": "way",
                "id": 2,
                "center": {"lon": -9.251, "lat": 38.711},
                "tags": {"golf": "green"},
            },
            {
                "type": "way",
                "id": 3,
                "center": {"lon": -9.252, "lat": 38.712},
                "tags": {"golf": "fairway"},
            },
            {
                "type": "way",
                "id": 4,
                "center": {"lon": -8.0, "lat": 38.0},
                "center_far": True,
                "tags": {"golf": "green"},
            },
        ]
    }

    async def fake_query(_ql: str) -> dict:
        return payload

    ranked = await rank_courses(
        bboxes=[("test", (-9.3, 38.6, -9.2, 38.8))],
        cache_dir=tmp_path / "cache",
        step_deg=1.0,
        delay_s=0.0,
        query_fn=fake_query,
        out_dir=tmp_path,
    )
    assert len(ranked) == 1
    course = ranked[0]
    assert course.name == "Jamor"
    assert course.counts.green == 1
    assert course.counts.fairway == 1
    assert course.counts.green_ways == 1
    assert course.score > 0

    csv_path = tmp_path / "ranking.csv"
    text = csv_path.read_text(encoding="utf-8")
    assert "Jamor" in text
    assert text.splitlines()[0].startswith("rank,score,osm_type,osm_id,name")
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "complete"
    assert progress["courses"] == 1


@pytest.mark.asyncio
async def test_rank_courses_writes_checkpoint_if_later_tile_fails(tmp_path: Path):
    payload = {
        "elements": [
            {
                "type": "way",
                "id": 1,
                "center": {"lon": -9.25, "lat": 38.71},
                "tags": {"leisure": "golf_course", "name": "Jamor", "holes": "18"},
            },
            {
                "type": "way",
                "id": 2,
                "center": {"lon": -9.251, "lat": 38.711},
                "tags": {"golf": "green"},
            },
        ]
    }
    calls = {"n": 0}

    async def flaky_query(_ql: str) -> dict:
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("overpass down")
        return payload

    with pytest.raises(RuntimeError, match="overpass down"):
        await rank_courses(
            bboxes=[("test", (-10.0, 38.0, -8.0, 39.0))],
            cache_dir=tmp_path / "cache",
            step_deg=1.0,
            delay_s=0.0,
            query_fn=flaky_query,
            out_dir=tmp_path,
        )

    csv_text = (tmp_path / "ranking.csv").read_text(encoding="utf-8")
    assert "Jamor" in csv_text
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "partial"
    assert progress["courses"] == 1
    assert progress["tile"] >= 1


def test_assign_features_prefers_containing_bounds():
    near = CourseRecord(
        osm_id=1,
        osm_type="way",
        name="Near",
        lon=-9.20,
        lat=38.70,
        west=-9.21,
        south=38.69,
        east=-9.19,
        north=38.71,
    )
    far_but_contains = CourseRecord(
        osm_id=2,
        osm_type="way",
        name="Contains",
        lon=-9.30,
        lat=38.70,
        west=-9.26,
        south=38.69,
        east=-9.22,
        north=38.71,
    )
    courses = {near.key: near, far_but_contains.key: far_but_contains}
    assign_features(
        courses,
        [GolfFeature(10, "way", "green", lon=-9.24, lat=38.70, polygonal=True)],
    )
    assert far_but_contains.counts.green == 1
    assert near.counts.green == 0


def test_assign_features_ignores_distant_points():
    course = CourseRecord(osm_id=1, osm_type="way", name="A", lon=-9.2, lat=38.7)
    assign_features(
        {course.key: course},
        [GolfFeature(10, "way", "green", lon=0.0, lat=0.0, polygonal=True)],
    )
    assert course.counts.green == 0


def test_iter_tiles_covers_bbox_without_gaps():
    tiles = iter_tiles(-10.0, 38.0, -8.0, 40.0, step=1.0)
    assert tiles[0] == (-10.0, 38.0, -9.0, 39.0)
    assert tiles[-1][2] == -8.0
    assert tiles[-1][3] == 40.0
    assert len(tiles) == 4


def test_resolve_bboxes_region_and_bbox():
    named = resolve_bboxes(region="lisbon")
    assert named[0][0] == "lisbon"
    custom = resolve_bboxes(bbox=(-9.5, 38.6, -9.1, 38.9))
    assert custom[0][1][0] == -9.5
    with pytest.raises(ValueError):
        resolve_bboxes()


def test_build_label_mask_paints_green_over_fairway():
    geo = MosaicGeo.from_bbox(64, 64, (-9.21, 38.69, -9.19, 38.71))
    fairway = box(-9.209, 38.691, -9.191, 38.709)
    green = box(-9.202, 38.698, -9.198, 38.702)
    layers = {
        "fairway": feature_collection([to_feature(fairway, {})]),
        "green": feature_collection([to_feature(green, {})]),
    }
    label = build_label_mask(layers, geo, (64, 64))
    assert CLASS_ID["fairway"] in set(label.ravel())
    assert CLASS_ID["green"] in set(label.ravel())
    # Center pixel of the green should not remain fairway.
    cx, cy = geo.lonlat_to_pixel(-9.200, 38.700)
    assert label[round(cy), round(cx)] == CLASS_ID["green"]
    preview = colorize_mask(label)
    assert preview.shape == (64, 64, 3)
    assert preview.dtype == np.uint8


def test_rows_from_ranking_drops_low_score_and_miniature(tmp_path: Path):
    csv_path = tmp_path / "ranking.csv"
    csv_path.write_text(
        "rank,score,osm_type,osm_id,name,flags\n"
        "1,92.0,way,1,Good,\n"
        "2,40.0,way,2,Sparse,\n"
        "3,99.0,way,3,Mini,miniature\n"
        "4,80.0,way,4,AlsoGood,\n",
        encoding="utf-8",
    )
    rows = rows_from_ranking(csv_path, top=10, min_score=65)
    names = [row["name"] for row in rows]
    assert names == ["Good", "AlsoGood"]


def test_rows_from_ranking_defaults_to_score_100_and_honors_limit(tmp_path: Path):
    csv_path = tmp_path / "ranking.csv"
    csv_path.write_text(
        "rank,score,osm_type,osm_id,name,flags\n"
        "1,100.0,way,1,Perfect A,\n"
        "2,100.0,way,2,Perfect B,\n"
        "3,100.0,way,3,Perfect C,\n"
        "4,99.0,way,4,Almost,\n"
        "5,98.0,way,5,Cutoff,\n",
        encoding="utf-8",
    )
    hundred = rows_from_ranking(csv_path, top=2)
    assert [row["name"] for row in hundred] == ["Perfect A", "Perfect B"]
    near = rows_from_ranking(csv_path, top=10, min_score=98)
    assert [row["name"] for row in near] == ["Perfect A", "Perfect B", "Perfect C", "Almost", "Cutoff"]


def test_choose_split_is_deterministic():
    assert choose_split("way", 42) == choose_split("way", 42)
    seen = {choose_split("way", i) for i in range(200)}
    assert seen == {"train", "val"}


def test_cli_regions_and_help(capsys):
    assert main(["regions"]) == 0
    out = capsys.readouterr().out
    assert "lisbon" in out
    assert "europe" in out
    with pytest.raises(SystemExit):
        main(["rank", "--help"])


@pytest.mark.asyncio
async def test_build_dataset_writes_masks(tmp_path: Path, fake_course):
    from PIL import Image

    from app.osm_training.dataset import build_dataset

    ranking = tmp_path / "ranking.csv"
    ranking.write_text(
        "rank,score,osm_type,osm_id,name,lon,lat,west,south,east,north,flags\n"
        "1,90.0,way,99,Test Links,-9.2,38.7,-9.21,38.69,-9.19,38.71,\n",
        encoding="utf-8",
    )
    rgb = Image.fromarray(np.zeros((32, 32, 3), np.uint8), mode="RGB")

    async def fake_layers(_row):
        return fake_course.bbox, fake_course.layers

    async def fake_mosaic(bbox, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(dest, format="JPEG")
        west, south, east, north = bbox
        return {
            "status": "ok",
            "source": "esri",
            "path": str(dest),
            "zoom": 18,
            "width": 32,
            "height": 32,
            "bbox": [west, south, east, north],
            "mosaic_bbox": [west, south, east, north],
        }

    out = tmp_path / "dataset"
    meta = await build_dataset(
        ranking=ranking,
        out_dir=out,
        top=1,
        min_score=50,
        fetch_layers=fake_layers,
        download_mosaic=fake_mosaic,
    )
    assert meta["samples_ok"] == 2
    assert (out / "label_map.json").exists()
    chips = list((out / "images").glob("way_99_*.jpg"))
    assert len(chips) == 2
    mask = np.array(Image.open(next((out / "masks").glob("*.png"))))
    assert mask.shape == (512, 512)
    assert mask.max() > 0
    assert meta["chip_size"] == 512
    assert meta["imagery_sources"] == ["esri", "clarity"]


@pytest.mark.asyncio
async def test_build_dataset_resumes_existing_image_and_mask(tmp_path: Path):
    from PIL import Image

    from app.osm_training.dataset import build_dataset

    ranking = tmp_path / "ranking.csv"
    ranking.write_text(
        "rank,score,osm_type,osm_id,name,lon,lat,flags\n"
        "1,99.0,way,99,Resume Links,-9.2,38.7,\n",
        encoding="utf-8",
    )
    out = tmp_path / "dataset"
    (out / "images").mkdir(parents=True)
    (out / "masks").mkdir(parents=True)
    Image.fromarray(np.zeros((32, 32, 3), np.uint8), mode="RGB").save(
        out / "images" / "way_99.jpg", format="JPEG"
    )
    Image.fromarray(np.zeros((32, 32), np.uint8), mode="L").save(out / "masks" / "way_99.png")

    async def boom(*_args, **_kwargs):
        raise RuntimeError("should not fetch on resume")

    meta = await build_dataset(
        ranking=ranking,
        out_dir=out,
        top=1,
        min_score=98.001,
        fetch_layers=boom,
        download_mosaic=boom,
    )
    assert meta["samples_ok"] == 1
    assert meta["status"] == "complete"
    assert list((out / "images").glob("way_99_esri_*.jpg"))
    assert not (out / "images" / "way_99.jpg").exists()


def test_chip_windows_cover_mosaic_and_drop_empty_tiles():
    rgb = np.zeros((600, 600, 3), np.uint8)
    label = np.zeros((600, 600), np.uint8)
    label[10:80, 10:80] = CLASS_ID["green"]
    windows = iter_chip_windows(600, 600, size=512, stride=384)
    assert windows[0] == (0, 0)
    assert (88, 88) in windows
    chips = chip_image_and_mask(rgb, label, size=512, stride=384, empty_keep_frac=0.0, seed_key="test")
    assert any(y == 0 and x == 0 for y, x, _rgb, _lab in chips)
    assert all(tile.shape == (512, 512, 3) for _y, _x, tile, _lab in chips)
    labeled = [lab for _y, _x, _rgb, lab in chips if lab.max() > 0]
    assert labeled
    empty = [lab for _y, _x, _rgb, lab in chips if lab.max() == 0]
    assert not empty


def test_semantic_class_ids_are_stable():
    assert SEMANTIC_CLASSES[0] == "background"
    assert CLASS_ID["green"] == SEMANTIC_CLASSES.index("green")
    assert CLASS_ID["fairway"] == SEMANTIC_CLASSES.index("fairway")
