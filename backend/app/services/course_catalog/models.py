"""Normalized club/course records shared by every catalog provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
        course = (self.course_name or "").strip()
        club = (self.club_name or "").strip()
        if not course or course.lower() == club.lower() or course.lower() in club.lower():
            return club or course
        return f"{club} — {course}"


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
        course = (self.course_name or "").strip()
        club = (self.club_name or "").strip()
        if not course or course.lower() == club.lower() or course.lower() in club.lower():
            return club or course
        return f"{club} — {course}"


@dataclass
class SearchResult:
    provider: str
    title: str
    cached: bool
    hits: list[CourseHit]
    api_requests_left: str | None = None
