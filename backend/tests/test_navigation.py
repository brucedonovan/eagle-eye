import pytest

from app.pipeline.stages import holes, navigation, topology, vectorize


@pytest.mark.asyncio
async def test_navigation_layers(fake_course):
    await holes.run(fake_course)
    await vectorize.run(fake_course)
    await topology.run(fake_course)
    await navigation.run(fake_course)
    assert "nav_mesh" in fake_course.layers
    assert "no_go" in fake_course.layers
    assert "coverage" in fake_course.layers
    assert "waypoint_graph" in fake_course.layers
    assert fake_course.layers["waypoint_graph"]["features"]
