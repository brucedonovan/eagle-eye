from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: Any
    model_config = {"extra": "allow"}


class CourseCreate(BaseModel):
    name: str | None = Field(default=None, description="Golf course name")
    lat: float | None = None
    lon: float | None = None
    aoi: dict[str, Any] | None = Field(default=None, description="GeoJSON Polygon / MultiPolygon")
    include_navigation: bool = True
    include_imagery: bool = True

    @model_validator(mode="after")
    def require_input(self) -> CourseCreate:
        has_name = bool(self.name and self.name.strip())
        has_point = self.lat is not None and self.lon is not None
        has_aoi = self.aoi is not None
        if not (has_name or has_point or has_aoi):
            raise ValueError("Provide a course name, lat/lon, or an AOI polygon")
        return self


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


class DownloadQuery(BaseModel):
    format: Literal[
        "geojson",
        "gpkg",
        "shp",
        "kml",
        "gpx",
        "wkt",
        "dxf",
        "ros_grid",
    ] = "geojson"
    layers: list[str] | None = None
