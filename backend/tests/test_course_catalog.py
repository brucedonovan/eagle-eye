from typing import ClassVar

import pytest
from fastapi import HTTPException
from shapely.geometry import LineString, Point, box

from app.api import _chips_from_catalog, _items_from_completed_jobs, search_courses
from app.models import Course, Job, LayerArtifact
from app.config import settings
from app.pipeline.context import PipelineContext
from app.pipeline.stages import discovery
from app.schemas import CourseCreate
from app.services.course_catalog import PROVIDERS, get_provider, register_provider
from app.services.course_catalog.layers import fuse_layers, prefer_clip
from app.services.course_catalog.provider import (
    CourseCatalogProvider,
    CourseHit,
    CourseRecord,
    SearchResult,
)
from app.services.course_catalog.providers.golfapi import (
    GolfApiClient,
    flatten_club_search,
    parse_coordinates,
)
from app.services.geometry import as_geom, feature_collection, to_feature

PEBBLE_CLUBS = {
    "apiRequestsLeft": "777",
    "numClubs": 1,
    "clubs": [
        {
            "clubID": "141520610397251566",
            "clubName": "Pebble Beach Golf Links",
            "city": "Pebble Beach",
            "state": "CA",
            "country": "USA",
            "address": "1700 17 Mile Drive",
            "latitude": 36.568,
            "longitude": -121.950,
            "courses": [
                {
                    "courseID": "012141520658891108829",
                    "courseName": "Pebble Beach",
                    "numHoles": 18,
                    "timestampUpdated": "1704549296",
                    "hasGPS": 1,
                },
                {
                    "courseID": "011141520658921417047",
                    "courseName": "The Hay",
                    "numHoles": 9,
                    "timestampUpdated": "1672862025",
                    "hasGPS": 0,
                },
            ],
        }
    ],
}


def test_clubs_own_courses():
    hits = flatten_club_search(PEBBLE_CLUBS)
    assert len(hits) == 2
    assert {hit.course_name for hit in hits} == {"Pebble Beach", "The Hay"}
    assert all(hit.club_name == "Pebble Beach Golf Links" for hit in hits)
    pebble = next(hit for hit in hits if hit.course_name == "Pebble Beach")
    hay = next(hit for hit in hits if hit.course_name == "The Hay")
    assert pebble.has_gps
    assert pebble.num_holes == 18
    assert pebble.lat == 36.568
    assert pebble.lon == -121.950
    assert pebble.display_name == "Pebble Beach Golf Links"
    assert "The Hay" in hay.display_name


def test_parse_coordinates_variants():
    a = parse_coordinates(
        {"coordinates": [{"hole": 1, "poi": "green", "latitude": 36.57, "longitude": -121.95}]}
    )
    b = parse_coordinates(
        [{"holeNumber": "2", "poiType": "Tee Box", "lat": 36.56, "lng": -121.94}]
    )
    c = parse_coordinates({"pois": [{"type": "Pin", "lat": 1.0, "lon": 2.0, "hole_number": 3}]})
    assert a[0]["kind"] == "green" and a[0]["hole"] == 1
    assert b[0]["kind"] == "tee" and b[0]["hole"] == 2
    assert c[0]["kind"] == "pin" and c[0]["hole"] == 3


def test_parse_golfapi_numeric_poi_codes():
    payload = {
        "coordinates": [
            {"poi": 1, "location": 2, "hole": 1, "latitude": 38.7141, "longitude": -9.2514},
            {"poi": 1, "location": 1, "hole": 1, "latitude": 38.7140, "longitude": -9.2514},
            {"poi": 12, "location": 2, "hole": 1, "latitude": 38.7123, "longitude": -9.2538},
            {"poi": 1, "location": 2, "hole": 10, "latitude": 38.7141, "longitude": -9.2514},
            {"poi": 2, "location": 1, "hole": 1, "latitude": 38.713, "longitude": -9.252},
            {"poi": 11, "location": 2, "hole": 1, "latitude": 38.7132, "longitude": -9.2525},
            {"poi": 9, "location": 2, "hole": 1, "latitude": 38.7134, "longitude": -9.2528},
            {"poi": 6, "location": 2, "hole": 1, "latitude": 38.7135, "longitude": -9.2529},
        ]
    }
    pts = parse_coordinates(payload, num_holes=9)
    by_kind = {(p["kind"], p["hole"]) for p in pts}
    assert ("pin", 1) in by_kind
    assert ("green", 1) in by_kind
    assert ("tee", 1) in by_kind
    assert ("bunker", 1) in by_kind
    assert ("fairway", 1) in by_kind
    assert ("tree", 1) in by_kind
    assert sum(1 for p in pts if p["kind"] == "tree") == 2
    assert all(p["hole"] != 10 for p in pts)
    assert sum(1 for p in pts if p["kind"] == "pin") == 1


def test_course_create_maps_legacy_golfapi_id():
    payload = CourseCreate(golfapi_course_id="012141520658891108829")
    assert payload.catalog_course_id == "012141520658891108829"
    assert payload.catalog_provider == "golfapi"


def test_course_create_accepts_catalog_id():
    payload = CourseCreate(catalog_provider="fake", catalog_course_id="course-1")
    assert payload.catalog_course_id == "course-1"
    assert payload.catalog_provider == "fake"


async def test_course_detail_is_cached_forever(tmp_path):
    client = GolfApiClient(cache_dir=tmp_path, api_key="test-key")
    calls = {"n": 0}

    async def fake_http(path, params):
        calls["n"] += 1
        return {"courseID": "abc", "timestampUpdated": "10", "hasGPS": "0", "courseName": "X"}

    client._http_get = fake_http  # type: ignore[method-assign]
    first = await client.get_course("abc")
    second = await client.get_course("abc")
    assert first.cached is False
    assert second.cached is True
    assert calls["n"] == 1

    stale = await client.get_course("abc", timestamp_updated=20)
    assert stale.cached is True
    assert calls["n"] == 1
    cached_again = await client.get_course("abc", timestamp_updated=99)
    assert cached_again.cached is True
    assert calls["n"] == 1


async def test_vectorize_reuses_cached_course_and_gps(tmp_path):
    client = GolfApiClient(cache_dir=tmp_path, api_key="test-key")
    calls = {"n": 0}

    async def fake_http(path, params):
        calls["n"] += 1
        if str(path).startswith("coordinates/"):
            return {"coordinates": [{"hole": 1, "poi": "green", "latitude": 36.57, "longitude": -121.95}]}
        return {
            "courseID": "abc",
            "clubName": "Pebble",
            "courseName": "Links",
            "latitude": 36.568,
            "longitude": -121.95,
            "hasGPS": 1,
            "timestampUpdated": "10",
        }

    client._http_get = fake_http  # type: ignore[method-assign]
    first = await client.load_course_bundle("abc", timestamp_updated=10)
    second = await client.load_course_bundle("abc", timestamp_updated=99)
    third = await client.load_course_bundle("abc")
    assert first["course_cached"] is False
    assert second["course_cached"] is True
    assert third["course_cached"] is True
    assert second["coordinates_cached"] is True
    assert calls["n"] == 2
    assert len(third["points"]) == 1


async def test_search_cache_avoids_http(tmp_path):
    client = GolfApiClient(cache_dir=tmp_path, api_key="test-key", search_ttl_days=30)
    calls = {"n": 0}

    async def fake_http(path, params):
        calls["n"] += 1
        return PEBBLE_CLUBS

    client._http_get = fake_http  # type: ignore[method-assign]
    hits, first = await client.search_course_hits("pebble beach")
    hits2, second = await client.search_course_hits("pebble beach")
    assert len(hits) == 2
    assert first.cached is False
    assert second.cached is True
    assert calls["n"] == 1
    _, cased = await client.search_course_hits("Pebble Beach")
    assert cased.cached is True
    assert calls["n"] == 1
    _, similar = await client.search_course_hits("Pebble Beach Golf Links")
    assert similar.cached is True
    assert calls["n"] == 1
    assert hits2[0].club_id == hits[0].club_id


async def test_list_cached_hits_skips_http(tmp_path):
    client = GolfApiClient(cache_dir=tmp_path, api_key="test-key", search_ttl_days=30)
    calls = {"n": 0}

    async def fake_http(path, params):
        calls["n"] += 1
        if str(path).startswith("courses/"):
            return {
                "courseID": "012141520658891108829",
                "clubID": "141520610397251566",
                "clubName": "Pebble Beach Golf Links",
                "courseName": "Pebble Beach",
                "latitude": 36.568,
                "longitude": -121.95,
                "numHoles": 18,
                "hasGPS": 1,
                "timestampUpdated": "1704549296",
            }
        return PEBBLE_CLUBS

    client._http_get = fake_http  # type: ignore[method-assign]
    await client.search_course_hits("pebble beach")
    assert client.list_cached_hits() == []
    await client.get_course("012141520658891108829")
    cached = client.list_cached_hits()
    assert {hit.course_name for hit in cached} == {"Pebble Beach"}
    pebble = next(hit for hit in cached if hit.course_name == "Pebble Beach")
    assert pebble.lat == 36.568
    assert pebble.has_gps
    assert calls["n"] == 2
    again = client.list_cached_hits()
    assert len(again) == 1
    assert calls["n"] == 2


def test_fuse_uses_provider_source_not_golfapi():
    osm_pin = to_feature(Point(-9.2515, 38.7141), {"source": "openstreetmap", "osm_id": 1})
    far_pin = to_feature(Point(-9.26, 38.72), {"source": "openstreetmap", "osm_id": 2})
    layers = {
        "pin": feature_collection([osm_pin, far_pin]),
        "green": feature_collection([to_feature(box(-9.2517, 38.7139, -9.2513, 38.7143), {"osm_id": 9})]),
    }
    points = [
        {"hole": 1, "kind": "green", "lat": 38.7141, "lon": -9.2515},
        {"hole": 1, "kind": "tee", "lat": 38.7123, "lon": -9.2538},
    ]
    stats = fuse_layers(layers, points, scorecard={"pars_men": [4], "indexes_men": [6]}, source="fake")
    pins = layers["pin"]["features"]
    sources = [f["properties"]["source"] for f in pins]
    assert sources == ["fake"]
    assert stats["pins_osm_kept"] == 0
    assert stats["pins_catalog"] == 1
    assert layers["green"]["features"][0]["properties"]["hole"] == 1
    assert layers["hole_centerline"]["features"][0]["properties"]["par"] == 4
    assert layers["hole_centerline"]["features"][0]["properties"]["source"] == "fake"


def test_fuse_stamps_catalog_source_on_matched_tees():
    osm_tee = to_feature(
        box(-9.2540, 38.7121, -9.2536, 38.7125),
        {"source": "openstreetmap", "osm_id": 7},
    )
    layers = {"tee": feature_collection([osm_tee])}
    fuse_layers(
        layers,
        [{"hole": 1, "kind": "tee", "lat": 38.7123, "lon": -9.2538}],
        source="golfapi",
    )
    props = layers["tee"]["features"][0]["properties"]
    assert props["source"] == "golfapi"
    assert props["geom_source"] == "openstreetmap"
    assert props["hole"] == 1


def test_gps_centerlines_replace_osm_ways():
    osm_line = to_feature(
        LineString([(-121.951, 36.560), (-121.949, 36.564)]),
        {"source": "openstreetmap", "ref": "9"},
    )
    layers = {"hole_centerline": feature_collection([osm_line])}
    points = [
        {"hole": 1, "kind": "tee", "lat": 36.560, "lon": -121.950},
        {"hole": 1, "kind": "green", "lat": 36.564, "lon": -121.950},
    ]
    stats = fuse_layers(layers, points, source="fake")
    assert stats["centerlines_added"] == 1
    assert layers["hole_centerline"]["features"][0]["properties"]["hole"] == 1
    assert layers["hole_centerline"]["features"][0]["properties"]["source"] == "fake"


def test_prefer_clip_uses_gps_hull():
    osm = box(-9.5, 38.6, -9.3, 38.8)
    gps = box(-9.40, 38.71, -9.39, 38.72)
    chosen = prefer_clip(osm, gps)
    assert chosen is gps
    assert prefer_clip(osm, None) is osm


def test_fuse_dogleg_bends_centerline():
    layers: dict = {}
    points = [
        {"hole": 1, "kind": "tee", "lat": 36.560, "lon": -121.950},
        {"hole": 1, "kind": "dogleg", "lat": 36.562, "lon": -121.947},
        {"hole": 1, "kind": "green", "lat": 36.564, "lon": -121.950},
        {"hole": 1, "kind": "bunker", "lat": 36.563, "lon": -121.949},
    ]
    stats = fuse_layers(layers, points, source="fake")
    assert stats["dogleg_seeds"] == 1
    assert stats["bunker_seeds"] == 1
    geom = as_geom(layers["hole_centerline"]["features"][0])
    assert len(list(geom.coords)) == 3


class FakeProvider(CourseCatalogProvider):
    id = "fake"
    title = "Fake Catalog"
    last_search: ClassVar[dict[str, object]] = {}

    @property
    def configured(self) -> bool:
        return True

    async def search(self, query: str, *, lat: float | None = None, lon: float | None = None) -> SearchResult:
        FakeProvider.last_search = {"query": query, "lat": lat, "lon": lon}
        return SearchResult(
            provider=self.id,
            title=self.title,
            cached=True,
            hits=[
                CourseHit(
                    provider=self.id,
                    club_id="club-1",
                    club_name="Fake Club",
                    course_id="course-1",
                    course_name="Links",
                )
            ],
        )

    async def load_course(
        self,
        course_id: str,
        *,
        club_id: str | None = None,
        timestamp_updated: int | None = None,
    ) -> CourseRecord:
        return CourseRecord(
            provider=self.id,
            course_id=course_id,
            club_id=club_id or "club-1",
            club_name="Fake Club",
            course_name="Links",
            lat=36.569,
            lon=-121.949,
            country="USA",
            city="Pebble Beach",
            num_holes=18,
            has_gps=True,
            scorecard={"pars_men": [4] * 18},
            points=[
                {"hole": 1, "kind": "green", "lat": 36.57, "lon": -121.95},
                {"hole": 1, "kind": "tee", "lat": 36.568, "lon": -121.948},
            ],
            cached=True,
        )

    def list_cached(self) -> list[CourseHit]:
        return [
            CourseHit(
                provider=self.id,
                club_id="club-1",
                club_name="Fake Club",
                course_id="course-1",
                course_name="Links",
            )
        ]


def test_drop_in_provider_is_selectable(monkeypatch):
    register_provider("fake", FakeProvider)
    monkeypatch.setattr(settings, "course_catalog_provider", "fake")
    try:
        provider = get_provider()
        assert provider is not None
        assert provider.id == "fake"
        assert provider.title == "Fake Catalog"
    finally:
        PROVIDERS.pop("fake", None)


async def test_discovery_uses_drop_in_catalog(work_dir, monkeypatch):
    register_provider("fake", FakeProvider)
    monkeypatch.setattr(settings, "course_catalog_provider", "fake")

    async def no_osm(lon, lat, name, pad_deg=0.03):
        return []

    monkeypatch.setattr("app.services.overpass.find_golf_courses", no_osm)
    try:
        ctx = PipelineContext(
            job_id="catalog-job",
            work_dir=work_dir,
            request={"catalog_provider": "fake", "catalog_course_id": "course-1"},
        )
        await discovery.run(ctx)
        assert ctx.course["source"] == "fake"
        assert ctx.course["catalog_provider"] == "fake"
        assert ctx.course["catalog_course_id"] == "course-1"
        assert ctx.course["catalog_cached"] is True
        assert ctx.course["lat"] == 36.569
        assert ctx.bbox is not None
    finally:
        PROVIDERS.pop("fake", None)


class EmptyProvider(FakeProvider):
    id = "empty"
    title = "Empty Catalog"

    async def search(self, query: str, *, lat: float | None = None, lon: float | None = None) -> SearchResult:
        return SearchResult(provider=self.id, title=self.title, cached=True, hits=[])


async def _no_osm(lon, lat, name, pad_deg=0.03):
    return []


async def test_discovery_by_name_uses_catalog(work_dir, monkeypatch):
    register_provider("fake", FakeProvider)
    monkeypatch.setattr(settings, "course_catalog_provider", "fake")
    monkeypatch.setattr("app.services.overpass.find_golf_courses", _no_osm)
    FakeProvider.last_search = {}
    try:
        ctx = PipelineContext(
            job_id="name-job",
            work_dir=work_dir,
            request={"name": "Fake Club"},
        )
        await discovery.run(ctx)
        assert FakeProvider.last_search["query"] == "Fake Club"
        assert ctx.course["source"] == "fake"
        assert ctx.course["catalog_course_id"] == "course-1"
    finally:
        PROVIDERS.pop("fake", None)


async def test_discovery_by_coordinates_uses_catalog(work_dir, monkeypatch):
    register_provider("fake", FakeProvider)
    monkeypatch.setattr(settings, "course_catalog_provider", "fake")
    monkeypatch.setattr("app.services.overpass.find_golf_courses", _no_osm)
    FakeProvider.last_search = {}
    try:
        ctx = PipelineContext(
            job_id="coord-job",
            work_dir=work_dir,
            request={"lat": 36.57, "lon": -121.95},
        )
        await discovery.run(ctx)
        assert FakeProvider.last_search == {"query": "", "lat": 36.57, "lon": -121.95}
        assert ctx.course["catalog_course_id"] == "course-1"
        assert ctx.course["source"] == "fake"
    finally:
        PROVIDERS.pop("fake", None)


async def test_discovery_raises_when_catalog_has_no_hits(work_dir, monkeypatch):
    register_provider("empty", EmptyProvider)
    monkeypatch.setattr(settings, "course_catalog_provider", "empty")
    try:
        ctx = PipelineContext(
            job_id="empty-job",
            work_dir=work_dir,
            request={"name": "Unknown Links"},
        )
        with pytest.raises(RuntimeError, match="returned no clubs"):
            await discovery.run(ctx)
    finally:
        PROVIDERS.pop("empty", None)


async def test_search_uses_catalog_only(monkeypatch):
    register_provider("fake", FakeProvider)
    monkeypatch.setattr(settings, "course_catalog_provider", "fake")
    try:
        out = await search_courses(q="pebble")
        assert out.source == "fake"
        assert out.catalog_configured is True
        assert [course.course_id for course in out.courses] == ["course-1"]
        assert all(course.source == "fake" for course in out.courses)
    finally:
        PROVIDERS.pop("fake", None)


def test_chips_from_catalog_cache_attach_vectorized_jobs():
    hits = [
        CourseHit(
            provider="golfapi",
            club_id="club-1",
            club_name="Estoril Golf Club",
            course_id="c-1",
            course_name="Blue Course",
            has_gps=True,
        ),
        CourseHit(
            provider="golfapi",
            club_id="club-1",
            club_name="Estoril Golf Club",
            course_id="c-2",
            course_name="Yellow Course",
        ),
    ]
    done = Job(
        id="job-estoril",
        status="completed",
        request_json='{"catalog_course_id":"c-1"}',
    )
    done.course = Course(
        name="Estoril Golf Club",
        display_name="Estoril Golf Club — Blue Course",
        metadata_json='{"catalog_course_id":"c-1"}',
    )
    done.layers = [LayerArtifact(job_id=done.id, layer_id="green", feature_count=9)]
    items = _chips_from_catalog(hits, [done])
    assert [item.course_id for item in items] == ["c-1", "c-2"]
    assert items[0].job_id == "job-estoril"
    assert items[1].job_id is None
    assert items[1].display_name == "Estoril Golf Club — Yellow Course"


def test_cached_chips_use_completed_jobs_not_search_hits():
    done = Job(
        id="job-estoril",
        status="completed",
        request_json='{"catalog_course_id":"c-1","catalog_club_id":"club-1","name":"Estoril"}',
        result_json='{"course":{"display_name":"Estoril Palácio Golf Course","catalog_provider":"golfapi","catalog_num_holes":18}}',
    )
    done.course = Course(
        name="Estoril Palácio Golf Course",
        display_name="Estoril Palácio Golf Course",
        country="Portugal",
        lat=38.71,
        lon=-9.39,
        metadata_json='{"catalog_course_id":"c-1","catalog_club_id":"club-1","catalog_club_name":"Estoril Palácio","catalog_course_name":"Championship","catalog_has_gps":true}',
    )
    done.layers = [LayerArtifact(job_id=done.id, layer_id="green", feature_count=18)]

    search_only = Job(
        id="job-search-cache",
        status="queued",
        request_json='{"catalog_course_id":"c-2","name":"Pebble Beach"}',
    )
    search_only.layers = []

    unfinished = Job(id="job-empty", status="completed", request_json="{}", result_json="{}")
    unfinished.layers = []

    older = Job(
        id="job-estoril-old",
        status="completed",
        request_json='{"catalog_course_id":"c-1","name":"Estoril old"}',
    )
    older.course = Course(name="Estoril old", display_name="Estoril old")
    older.layers = [LayerArtifact(job_id=older.id, layer_id="boundary", feature_count=1)]

    jamor_a = Job(id="job-jamor-a", status="completed", request_json="{}")
    jamor_a.course = Course(name="Jamor A", display_name="Jamor A", osm_id="123")
    jamor_a.layers = [LayerArtifact(job_id=jamor_a.id, layer_id="green", feature_count=1)]
    jamor_b = Job(id="job-jamor-b", status="completed", request_json="{}")
    jamor_b.course = Course(name="Jamor B", display_name="Jamor B", osm_id="123")
    jamor_b.layers = [LayerArtifact(job_id=jamor_b.id, layer_id="green", feature_count=1)]

    items = _items_from_completed_jobs([done, search_only, unfinished, older, jamor_a, jamor_b])
    assert [item.job_id for item in items] == ["job-estoril", "job-jamor-a"]
    assert items[0].display_name == "Estoril Palácio Golf Course"
    assert items[0].has_gps is True


def test_layer_origin_buckets_api_osm_generated():
    from app.services.layer_origin import layer_origin

    api = feature_collection([to_feature(Point(-9.25, 38.71), {"source": "golfapi"})])
    osm = feature_collection([to_feature(box(-9.26, 38.70, -9.25, 38.71), {"source": "openstreetmap"})])
    generated = feature_collection([to_feature(box(-9.26, 38.70, -9.25, 38.71), {"source": "imagery"})])
    mixed = feature_collection(
        [
            to_feature(Point(-9.25, 38.71), {"source": "golfapi"}),
            to_feature(Point(-9.251, 38.711), {"source": "openstreetmap"}),
        ]
    )
    ai = feature_collection([to_feature(box(-9.26, 38.70, -9.25, 38.71), {"source": "ai"})])
    assert layer_origin(api) == "api"
    assert layer_origin(osm) == "osm"
    assert layer_origin(generated) == "generated"
    assert layer_origin(ai) == "ai"
    assert layer_origin(mixed) == "mixed"
    assert layer_origin(feature_collection([])) == "hybrid"
    assert layer_origin(feature_collection([to_feature(Point(-9.25, 38.71), {})])) == "generated"


async def test_search_requires_configured_catalog(monkeypatch):
    monkeypatch.setattr(settings, "course_catalog_provider", "none")
    with pytest.raises(HTTPException) as err:
        await search_courses(q="pebble")
    assert err.value.status_code == 503
