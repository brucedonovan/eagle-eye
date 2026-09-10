from pathlib import Path

import pytest
from shapely.geometry import LineString, Point, box

from app.pipeline.context import PipelineContext
from app.services.geometry import feature_collection, to_feature


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def fake_course(work_dir: Path) -> PipelineContext:
    ctx = PipelineContext(
        job_id="test-job",
        work_dir=work_dir,
        request={"name": "Test Links", "include_navigation": True, "include_imagery": False},
        course={"name": "Test Links", "lat": 38.7, "lon": -9.2},
        bbox=(-9.21, 38.69, -9.19, 38.71),
    )
    green_a = box(-9.2002, 38.7008, -9.1998, 38.7012)
    green_b = box(-9.2042, 38.7048, -9.2038, 38.7052)
    tee_a = box(-9.1982, 38.6988, -9.1978, 38.6992)
    tee_b = box(-9.2022, 38.7028, -9.2018, 38.7032)
    bunker = box(-9.20015, 38.7009, -9.20005, 38.7010)  # overlaps green_a
    water = box(-9.206, 38.699, -9.205, 38.700)
    fairway = box(-9.201, 38.699, -9.198, 38.7015)
    building = box(-9.197, 38.697, -9.196, 38.698)
    ctx.layers = {
        "boundary": feature_collection([to_feature(box(-9.21, 38.69, -9.19, 38.71), {"name": "Test Links"})]),
        "green": feature_collection(
            [
                to_feature(green_a, {"instance_id": "green-1", "name": "G1"}),
                to_feature(green_b, {"instance_id": "green-2", "name": "G2"}),
            ]
        ),
        "tee": feature_collection(
            [
                to_feature(tee_a, {"instance_id": "tee-1"}),
                to_feature(tee_b, {"instance_id": "tee-2"}),
            ]
        ),
        "bunker": feature_collection([to_feature(bunker, {"instance_id": "bunker-1"})]),
        "water": feature_collection([to_feature(water, {"instance_id": "water-1"})]),
        "fairway": feature_collection([to_feature(fairway, {"instance_id": "fairway-1"})]),
        "building": feature_collection([to_feature(building, {"instance_id": "bldg-1"})]),
        "cart_path": feature_collection(
            [to_feature(LineString([(-9.198, 38.699), (-9.200, 38.701)]), {"highway": "service"})]
        ),
        "tree": feature_collection([to_feature(Point(-9.1975, 38.6985), {})]),
    }
    return ctx
