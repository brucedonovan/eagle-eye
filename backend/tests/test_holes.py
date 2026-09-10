import pytest
from shapely.geometry import LineString, Point, box

from app.pipeline.stages import holes, topology
from app.services.geometry import as_geom, feature_collection, to_feature


def _ctx(work_dir, layers: dict):
    from app.pipeline.context import PipelineContext

    ctx = PipelineContext(
        job_id="jamor-fix",
        work_dir=work_dir,
        request={"name": "Jamor", "include_navigation": False, "include_imagery": False},
        course={"name": "Jamor", "lat": 38.713, "lon": -9.252},
        bbox=(-9.256, 38.710, -9.248, 38.716),
    )
    ctx.layers = layers
    return ctx


@pytest.mark.asyncio
async def test_practice_green_does_not_steal_hole_and_fairways_are_derived(work_dir):
    # Hole 1 pin has no mapped green; hole 4 pin sits on the real green.
    # A small chipping green far from both pins must not take a hole slot.
    pin1 = Point(-9.2515053, 38.71411)
    pin4 = Point(-9.2502, 38.7132)
    tee1 = box(-9.2540, 38.7122, -9.2537, 38.71245)
    tee4 = box(-9.2530, 38.7116, -9.2527, 38.71185)
    real_green4 = box(-9.2504, 38.71305, -9.2500, 38.71335)  # contains pin4
    practice = box(-9.2490, 38.7110, -9.24885, 38.71112)  # ~150 m from pins, tiny
    hole1 = LineString([(-9.2538563, 38.7123324), (-9.2515053, 38.71411)])
    hole4 = LineString([(-9.25285, 38.71172), (-9.2502, 38.7132)])
    boundary = box(-9.256, 38.710, -9.248, 38.716)

    ctx = _ctx(
        work_dir,
        {
            "boundary": feature_collection([to_feature(boundary, {"name": "Jamor"})]),
            "green": feature_collection(
                [
                    to_feature(real_green4, {"instance_id": "green-4", "osm_id": 1046144887}),
                    to_feature(practice, {"instance_id": "green-chip", "osm_id": 1046144889}),
                ]
            ),
            "tee": feature_collection(
                [
                    to_feature(tee1, {"instance_id": "tee-1"}),
                    to_feature(tee4, {"instance_id": "tee-4"}),
                ]
            ),
            "pin": feature_collection(
                [
                    to_feature(pin1, {"osm_id": 1, "golf": "pin"}),
                    to_feature(pin4, {"osm_id": 4, "golf": "pin"}),
                ]
            ),
            "hole_centerline": feature_collection(
                [
                    to_feature(hole1, {"ref": "1", "golf": "hole"}),
                    to_feature(hole4, {"ref": "4", "golf": "hole"}),
                ]
            ),
        },
    )

    await holes.run(ctx)

    lines = ctx.layers["hole_centerline"]["features"]
    assert sorted(f["properties"]["hole"] for f in lines) == [1, 4]
    by_id = {f["properties"].get("instance_id"): f for f in ctx.layers["green"]["features"]}
    assert "green-4" in by_id
    assert "green-chip" in by_id
    assert not any(f["properties"].get("source") in {"pin", "imagery_pin"} for f in ctx.layers["green"]["features"])
    hole_ids = {f["properties"].get("green_id") for f in lines}
    assert "green-4" in hole_ids
    assert None in hole_ids or "green-chip" not in hole_ids
    putting = ctx.layers.get("putting_green", {}).get("features", [])
    assert not any(f["properties"].get("instance_id") == "green-chip" for f in putting)

    line1 = next(as_geom(f) for f in lines if f["properties"]["hole"] == 1)
    assert list(line1.coords) == list(hole1.coords) or list(line1.coords) == list(hole1.coords)[::-1]

    fairways = ctx.layers.get("fairway", {}).get("features", [])
    assert len(fairways) >= 2
    holes_with_fw = {f["properties"]["hole"] for f in fairways}
    assert holes_with_fw == {1, 4}
    assert all(f["properties"]["source"] == "centerline_buffer" for f in fairways)


def _pin_ring(west, south, east, north):
    return [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
    ]


@pytest.mark.asyncio
async def test_missing_green_is_filled_from_pin_imagery(work_dir):
    pin1 = Point(-9.2515053, 38.71411)
    tee1 = box(-9.2540, 38.7122, -9.2537, 38.71245)
    hole1 = LineString([(-9.2538563, 38.7123324), (-9.2515053, 38.71411)])
    ctx = _ctx(
        work_dir,
        {
            "boundary": feature_collection([to_feature(box(-9.256, 38.710, -9.248, 38.716), {})]),
            "green": feature_collection([]),
            "tee": feature_collection([to_feature(tee1, {"instance_id": "tee-1"})]),
            "pin": feature_collection([to_feature(pin1, {"golf": "pin"})]),
            "hole_centerline": feature_collection([to_feature(hole1, {"ref": "1", "golf": "hole"})]),
        },
    )
    ctx.masks = {
        "pin_greens": [
            {
                "pin": [-9.2515053, 38.71411],
                "coordinates": _pin_ring(-9.25170, 38.71395, -9.25130, 38.71427),
                "area_m2": 400,
                "sources": ["primary", "google_static"],
            }
        ]
    }
    await holes.run(ctx)
    greens = ctx.layers["green"]["features"]
    assert len(greens) == 1
    assert greens[0]["properties"]["source"] == "imagery_pin"
    assert greens[0]["properties"]["hole"] == 1
    assert ctx.layers["hole_centerline"]["features"][0]["properties"]["green_id"] == "imagery-green-1"
    assert ctx.quality["greens_imagery_filled"] == [1]
    assert ctx.quality["holes_missing_green"] == []


@pytest.mark.asyncio
async def test_missing_green_stays_empty_without_pin_imagery(work_dir):
    pin1 = Point(-9.2515053, 38.71411)
    hole1 = LineString([(-9.2538563, 38.7123324), (-9.2515053, 38.71411)])
    ctx = _ctx(
        work_dir,
        {
            "boundary": feature_collection([to_feature(box(-9.256, 38.710, -9.248, 38.716), {})]),
            "green": feature_collection([]),
            "tee": feature_collection([]),
            "pin": feature_collection([to_feature(pin1, {"golf": "pin"})]),
            "hole_centerline": feature_collection([to_feature(hole1, {"ref": "1", "golf": "hole"})]),
        },
    )
    await holes.run(ctx)
    assert ctx.layers.get("green", {}).get("features", []) == []
    assert ctx.layers["hole_centerline"]["features"][0]["properties"]["green_id"] is None
    assert ctx.quality["holes_missing_green"] == [1]
    assert ctx.quality["holes_need_review"] is True


@pytest.mark.asyncio
async def test_pin_green_rejected_when_it_overlaps_osm_green(work_dir):
    """Hole 8 cannot stack an imagery blob on hole 9's OSM green."""
    pin8 = Point(-9.25600, 38.71100)
    pin9 = Point(-9.25570, 38.71100)
    osm = box(-9.25590, 38.71085, -9.25550, 38.71115)
    hole8 = LineString([(-9.25720, 38.71040), (-9.25600, 38.71100)])
    hole9 = LineString([(-9.25640, 38.71040), (-9.25570, 38.71100)])
    ctx = _ctx(
        work_dir,
        {
            "boundary": feature_collection([to_feature(box(-9.258, 38.710, -9.254, 38.712), {})]),
            "green": feature_collection(
                [to_feature(osm, {"instance_id": "green-9", "osm_id": 99, "source": "openstreetmap"})]
            ),
            "tee": feature_collection([]),
            "pin": feature_collection(
                [to_feature(pin8, {"golf": "pin"}), to_feature(pin9, {"golf": "pin"})]
            ),
            "hole_centerline": feature_collection(
                [
                    to_feature(hole8, {"ref": "8", "golf": "hole"}),
                    to_feature(hole9, {"ref": "9", "golf": "hole"}),
                ]
            ),
        },
    )
    ctx.masks = {
        "pin_greens": [
            {
                "pin": [-9.25600, 38.71100],
                "coordinates": _pin_ring(-9.25616, 38.71088, -9.25580, 38.71112),
                "area_m2": 800,
                "sources": ["primary"],
            }
        ]
    }
    await holes.run(ctx)
    lines = {f["properties"]["hole"]: f["properties"] for f in ctx.layers["hole_centerline"]["features"]}
    assert lines[9]["green_id"] == "green-9"
    assert lines[8]["green_id"] is None
    sources = {f["properties"].get("source") for f in ctx.layers["green"]["features"]}
    assert sources == {"openstreetmap"}
    assert 8 in ctx.quality["holes_missing_green"]


@pytest.mark.asyncio
async def test_pin_green_fills_only_the_hole_without_osm(work_dir):
    pin8 = Point(-9.25685, 38.71078)
    pin9 = Point(-9.25505, 38.71275)
    shared = box(-9.25520, 38.71260, -9.25490, 38.71290)
    hole8 = LineString([(-9.25740, 38.71020), (-9.25685, 38.71078)])
    hole9 = LineString([(-9.25732, 38.71073), (-9.25505, 38.71275)])
    ctx = _ctx(
        work_dir,
        {
            "boundary": feature_collection([to_feature(box(-9.258, 38.710, -9.254, 38.714), {})]),
            "green": feature_collection(
                [to_feature(shared, {"instance_id": "green-shared", "osm_id": 820920260, "source": "openstreetmap"})]
            ),
            "tee": feature_collection([]),
            "pin": feature_collection(
                [to_feature(pin8, {"golf": "pin"}), to_feature(pin9, {"golf": "pin"})]
            ),
            "hole_centerline": feature_collection(
                [
                    to_feature(hole8, {"ref": "8", "golf": "hole"}),
                    to_feature(hole9, {"ref": "9", "golf": "hole"}),
                ]
            ),
        },
    )
    ctx.masks = {
        "pin_greens": [
            {
                "pin": [-9.25685, 38.71078],
                "coordinates": _pin_ring(-9.25700, 38.71066, -9.25670, 38.71090),
                "area_m2": 500,
                "sources": ["esri"],
            }
        ]
    }
    await holes.run(ctx)
    lines = {f["properties"]["hole"]: f["properties"] for f in ctx.layers["hole_centerline"]["features"]}
    assert lines[9]["green_id"] == "green-shared"
    assert lines[8]["green_id"] == "imagery-green-8"
    sources = {f["properties"].get("source") for f in ctx.layers["green"]["features"]}
    assert sources == {"openstreetmap", "imagery_pin"}
    assert ctx.quality["greens_imagery_filled"] == [8]


@pytest.mark.asyncio
async def test_pin_at_hole_start_claims_its_osm_green(work_dir):
    """Jamor 8: OSM way has the pin at the start; that pin sits on an unused green."""
    pin8 = Point(-9.25685, 38.71078)
    pin9 = Point(-9.25505, 38.71275)
    green8 = box(-9.25705, 38.71064, -9.25665, 38.71092)
    green9 = box(-9.25520, 38.71260, -9.25490, 38.71290)
    hole8 = LineString([(-9.25685, 38.71078), (-9.25499, 38.71223)])
    hole9 = LineString([(-9.25732, 38.71073), (-9.25505, 38.71275)])
    ctx = _ctx(
        work_dir,
        {
            "boundary": feature_collection([to_feature(box(-9.258, 38.710, -9.254, 38.714), {})]),
            "green": feature_collection(
                [
                    to_feature(green8, {"instance_id": "green-6-1", "osm_id": 6, "source": "openstreetmap"}),
                    to_feature(green9, {"instance_id": "green-1-1", "osm_id": 1, "source": "openstreetmap"}),
                ]
            ),
            "tee": feature_collection(
                [to_feature(box(-9.25700, 38.71070, -9.25680, 38.71085), {"instance_id": "tee-near-8"})]
            ),
            "pin": feature_collection(
                [to_feature(pin8, {"golf": "pin"}), to_feature(pin9, {"golf": "pin"})]
            ),
            "hole_centerline": feature_collection(
                [
                    to_feature(hole8, {"ref": "8", "golf": "hole"}),
                    to_feature(hole9, {"ref": "9", "golf": "hole"}),
                ]
            ),
        },
    )
    await holes.run(ctx)
    lines = {f["properties"]["hole"]: f for f in ctx.layers["hole_centerline"]["features"]}
    assert lines[8]["properties"]["green_id"] == "green-6-1"
    assert lines[9]["properties"]["green_id"] == "green-1-1"
    end8 = as_geom(lines[8])
    assert list(end8.coords)[-1][0] == pytest.approx(-9.25685, abs=1e-5)
    pin_holes = {f["properties"].get("hole") for f in ctx.layers["pin"]["features"]}
    assert pin_holes == {8, 9}


@pytest.mark.asyncio
async def test_pin_inside_green_is_not_stolen_by_earlier_hole(work_dir):
    """Hole 8 used to claim hole 9's OSM green via a loose fallback, then a
    generated green was stacked on the same surface."""
    shared = box(-9.25520, 38.71260, -9.25490, 38.71290)
    pin8 = Point(-9.25685, 38.71078)
    pin9 = Point(-9.25505, 38.71275)  # inside shared green
    hole8 = LineString([(-9.25685, 38.71078), (-9.25499, 38.71223)])
    hole9 = LineString([(-9.25732, 38.71073), (-9.25505, 38.71275)])
    ctx = _ctx(
        work_dir,
        {
            "boundary": feature_collection([to_feature(box(-9.258, 38.710, -9.254, 38.714), {})]),
            "green": feature_collection(
                [to_feature(shared, {"instance_id": "green-shared", "osm_id": 820920260, "source": "openstreetmap"})]
            ),
            "tee": feature_collection([]),
            "pin": feature_collection(
                [
                    to_feature(pin8, {"golf": "pin"}),
                    to_feature(pin9, {"golf": "pin"}),
                ]
            ),
            "hole_centerline": feature_collection(
                [
                    to_feature(hole8, {"ref": "8", "golf": "hole"}),
                    to_feature(hole9, {"ref": "9", "golf": "hole"}),
                ]
            ),
        },
    )
    await holes.run(ctx)
    lines = {f["properties"]["hole"]: f["properties"] for f in ctx.layers["hole_centerline"]["features"]}
    assert lines[9]["green_id"] == "green-shared"
    assert lines[8]["green_id"] != "green-shared"
    sources = {f["properties"].get("source") for f in ctx.layers["green"]["features"]}
    assert sources == {"openstreetmap"}
    assert len(ctx.layers["green"]["features"]) == 1


@pytest.mark.asyncio
async def test_existing_osm_fairways_are_kept(fake_course):
    await holes.run(fake_course)
    ids = {f["properties"]["instance_id"] for f in fake_course.layers["fairway"]["features"]}
    assert "fairway-1" in ids


@pytest.mark.asyncio
async def test_tees_are_exclusive_and_practice_tag_moves(work_dir):
    tee_shared = box(-9.25305, 38.71200, -9.25285, 38.71215)
    tee_only4 = box(-9.25240, 38.71150, -9.25220, 38.71165)
    pin1 = Point(-9.25150, 38.71411)
    pin4 = Point(-9.25020, 38.71320)
    green1 = box(-9.2517, 38.71395, -9.2513, 38.71425)
    green4 = box(-9.2504, 38.71305, -9.2500, 38.71335)
    practice = box(-9.2490, 38.7110, -9.24885, 38.71112)
    hole1 = LineString([(-9.25300, 38.71208), (-9.25150, 38.71411)])
    hole4 = LineString([(-9.25230, 38.71158), (-9.25020, 38.71320)])

    ctx = _ctx(
        work_dir,
        {
            "boundary": feature_collection([to_feature(box(-9.256, 38.710, -9.248, 38.716), {})]),
            "green": feature_collection(
                [
                    to_feature(green1, {"instance_id": "green-1", "osm_id": 1}),
                    to_feature(green4, {"instance_id": "green-4", "osm_id": 4}),
                    to_feature(practice, {"instance_id": "green-chip", "osm_id": 9, "golf": "practice"}),
                ]
            ),
            "tee": feature_collection(
                [
                    to_feature(tee_shared, {"instance_id": "tee-shared"}),
                    to_feature(tee_only4, {"instance_id": "tee-4"}),
                ]
            ),
            "pin": feature_collection(
                [
                    to_feature(pin1, {"golf": "pin"}),
                    to_feature(pin4, {"golf": "pin"}),
                ]
            ),
            "hole_centerline": feature_collection(
                [
                    to_feature(hole1, {"ref": "1", "golf": "hole"}),
                    to_feature(hole4, {"ref": "4", "golf": "hole"}),
                ]
            ),
        },
    )
    await holes.run(ctx)
    await topology.run(ctx)
    lines = {f["properties"]["hole"]: f["properties"] for f in ctx.layers["hole_centerline"]["features"]}
    assert set(lines[1]["tee_ids"]).isdisjoint(set(lines[4]["tee_ids"]))
    putting_ids = {f["properties"].get("instance_id") for f in ctx.layers.get("putting_green", {}).get("features", [])}
    assert "green-chip" in putting_ids
    green_ids = {f["properties"].get("instance_id") for f in ctx.layers["green"]["features"]}
    assert "green-chip" not in green_ids
    assert not ctx.quality.get("hole_centerlines_cross")
    fringes = ctx.layers.get("green_fringe", {}).get("features", [])
    assert fringes
    assert ctx.quality.get("green_fringes") == len(fringes)
    assert {f["properties"].get("hole") for f in fringes} <= {1, 4}
