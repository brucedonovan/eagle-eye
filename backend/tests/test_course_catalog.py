from typing import ClassVar

import pytest
from fastapi import HTTPException
from shapely.geometry import Point, box

from app.api import search_courses
from app.config import settings
from app.pipeline.context import PipelineContext
from app.pipeline.stages import discovery
from app.schemas import CourseCreate
from app.services.course_catalog import PROVIDERS, get_provider, register_provider
from app.services.course_catalog.layers import fuse_layers
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
from app.services.geometry import feature_collection, to_feature

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

    async def newer(path, params):
        calls["n"] += 1
        return {"courseID": "abc", "timestampUpdated": "20", "hasGPS": "0", "courseName": "X"}

    client._http_get = newer  # type: ignore[method-assign]
    stale = await client.get_course("abc", timestamp_updated=20)
    assert stale.cached is False
    assert calls["n"] == 2
    cached_again = await client.get_course("abc", timestamp_updated=20)
    assert cached_again.cached is True
    assert calls["n"] == 2


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
    assert hits2[0].club_id == hits[0].club_id


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
    assert sources.count("fake") == 1
    assert "openstreetmap" in sources
    assert stats["pins_catalog"] == 1
    assert layers["green"]["features"][0]["properties"]["hole"] == 1
    assert layers["hole_centerline"]["features"][0]["properties"]["par"] == 4


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


async def test_search_requires_configured_catalog(monkeypatch):
    monkeypatch.setattr(settings, "course_catalog_provider", "none")
    with pytest.raises(HTTPException) as err:
        await search_courses(q="pebble")
    assert err.value.status_code == 503
