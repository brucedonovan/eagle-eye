from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class CourseCreate(BaseModel):
    name: str | None = Field(default=None, description="Golf course name")
    lat: float | None = None
    lon: float | None = None
    aoi: dict[str, Any] | None = Field(default=None, description="GeoJSON Polygon / MultiPolygon")
    golfapi_course_id: str | None = Field(default=None, description="Deprecated alias for catalog_course_id")
    golfapi_club_id: str | None = Field(default=None, description="Deprecated alias for catalog_club_id")
    golfapi_timestamp_updated: int | None = Field(default=None, description="Deprecated alias")
    catalog_provider: str | None = None
    catalog_course_id: str | None = None
    catalog_club_id: str | None = None
    catalog_timestamp_updated: int | None = None
    include_navigation: bool = True
    include_imagery: bool = True

    @model_validator(mode="after")
    def require_input(self) -> CourseCreate:
        if self.golfapi_course_id and not self.catalog_course_id:
            self.catalog_course_id = self.golfapi_course_id
            self.catalog_provider = self.catalog_provider or "golfapi"
        if self.golfapi_club_id and not self.catalog_club_id:
            self.catalog_club_id = self.golfapi_club_id
        if self.golfapi_timestamp_updated is not None and self.catalog_timestamp_updated is None:
            self.catalog_timestamp_updated = self.golfapi_timestamp_updated
        has_name = bool(self.name and self.name.strip())
        has_point = self.lat is not None and self.lon is not None
        has_aoi = self.aoi is not None
        has_catalog = bool(self.catalog_course_id and self.catalog_course_id.strip())
        if not (has_name or has_point or has_aoi or has_catalog):
            raise ValueError("Provide a course name, catalog course id, lat/lon, or an AOI polygon")
        return self


class CourseSearchItem(BaseModel):
    source: str
    club_id: str | None = None
    club_name: str
    course_id: str | None = None
    course_name: str
    display_name: str
    city: str | None = None
    state: str | None = None
    country: str | None = None
    address: str | None = None
    lat: float | None = None
    lon: float | None = None
    num_holes: int | None = None
    has_gps: bool = False
    distance_km: float | None = None
    timestamp_updated: int | None = None
    job_id: str | None = None


class CourseSearchOut(BaseModel):
    query: str
    source: str
    cached: bool = False
    catalog_configured: bool = False
    catalog_provider: str | None = None
    provider_title: str | None = None
    api_requests_left: str | None = None
    courses: list[CourseSearchItem] = Field(default_factory=list)


class CourseOut(BaseModel):
    id: str
    name: str
    display_name: str | None = None
    country: str | None = None
    address: str | None = None
    lat: float | None = None
    lon: float | None = None
    bbox: list[float] | None = None
    osm_id: str | None = None


class LayerOut(BaseModel):
    layer_id: str
    title: str
    geometry: str
    group: str
    color: str
    feature_count: int
    source: str


class JobOut(BaseModel):
    job_id: str
    status: str
    stage: str
    progress: float
    message: str | None = None
    course: CourseOut | None = None


class StatusOut(JobOut):
    error: str | None = None
    layers: list[LayerOut] = Field(default_factory=list)
    quality: dict[str, Any] = Field(default_factory=dict)
    exports: list[str] = Field(default_factory=list)
