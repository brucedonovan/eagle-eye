import pytest

from app.pipeline.stages import holes, topology, vectorize
from app.services.geometry import as_geom


@pytest.mark.asyncio
async def test_topology_subtracts_bunker_from_green(fake_course):
    await holes.run(fake_course)
    await vectorize.run(fake_course)
    await topology.run(fake_course)

    bunker = as_geom(fake_course.layers["bunker"]["features"][0])
    for feat in fake_course.layers["green"]["features"]:
        green = as_geom(feat)
        assert green.intersection(bunker).area < 1e-16
    assert fake_course.quality["polygon_validity"] == 1.0
    assert fake_course.layers.get("green_fringe", {}).get("features")
    assert fake_course.quality.get("green_fringes") >= 1


@pytest.mark.asyncio
async def test_holes_are_numbered(fake_course):
    await holes.run(fake_course)
    numbers = [f["properties"]["hole"] for f in fake_course.layers["hole_centerline"]["features"]]
    assert sorted(numbers) == [1, 2]
    stamped = [f["properties"].get("hole") for f in fake_course.layers["green"]["features"]]
    assert set(stamped) == {1, 2}
