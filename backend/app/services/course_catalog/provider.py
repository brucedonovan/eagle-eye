"""Course catalog contract: search + cache-first course detail.

Pipeline and search talk only to this interface. A vendor adapter maps its JSON
onto CourseHit / CourseRecord (points are {hole, kind, lat, lon}).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class CatalogError(RuntimeError):
    pass


def club_course_name(club: str, course: str) -> str:
    club = (club or "").strip()
    course = (course or "").strip()
    if not course or course.lower() == club.lower() or course.lower() in club.lower():
        return club or course
    return f"{club} — {course}"


@dataclass
class CourseHit:
    """One selectable course from a catalog search (a club may own several)."""

    provider: str
    club_id: str
    club_name: str
    course_id: str
    course_name: str
    city: str | None = None
    state: str | None = None
    country: str | None = None
    address: str | None = None
    num_holes: int | None = None
    has_gps: bool = False
    distance: float | None = None
    measure_unit: str | None = None
    timestamp_updated: int | None = None
    lat: float | None = None
    lon: float | None = None

    @property
    def display_name(self) -> str:
        return club_course_name(self.club_name, self.course_name)


@dataclass
class CourseRecord:
    """Full course payload after a cache-first fetch."""

    provider: str
    course_id: str
    club_id: str
    club_name: str
    course_name: str
    lat: float | None = None
    lon: float | None = None
    country: str | None = None
    city: str | None = None
    state: str | None = None
    address: str | None = None
    postal_code: str | None = None
    website: str | None = None
    telephone: str | None = None
    num_holes: int | None = None
    has_gps: bool = False
    scorecard: dict[str, Any] = field(default_factory=dict)
    points: list[dict[str, Any]] = field(default_factory=list)
    cached: bool = False
    api_requests_left: str | None = None

    @property
    def display_name(self) -> str:
        return club_course_name(self.club_name, self.course_name)


@dataclass
class SearchResult:
    provider: str
    title: str
    cached: bool
    hits: list[CourseHit]
    api_requests_left: str | None = None


class CourseCatalogProvider(ABC):
    id: str
    title: str

    @property
    @abstractmethod
    def configured(self) -> bool: ...

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        lat: float | None = None,
        lon: float | None = None,
    ) -> SearchResult: ...

    @abstractmethod
    async def load_course(
        self,
        course_id: str,
        *,
        club_id: str | None = None,
        timestamp_updated: int | None = None,
    ) -> CourseRecord: ...
