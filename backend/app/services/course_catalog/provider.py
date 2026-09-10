"""Course catalog provider contract.

Search, discovery, and OSM fusion talk only to this interface. A vendor API
(golfapi.io or anything with clubs, courses, scorecards, and GPS) is an adapter.

To drop in another API:

1. Subclass ``CourseCatalogProvider`` in ``providers/<name>.py``.
2. Map that vendor's JSON onto ``CourseHit`` / ``CourseRecord``
   (points are ``{hole, kind, lat, lon}`` with kind in green/pin/tee/…).
3. Register it in ``_ensure_builtins()`` (or call
   ``register_provider("name", YourProvider)`` at process start).
4. Set ``COURSE_CATALOG_PROVIDER=name`` and that vendor's credentials.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.services.course_catalog.models import CourseRecord, SearchResult


class CatalogError(RuntimeError):
    pass


class CourseCatalogProvider(ABC):
    """Club/course search + cache-first course detail."""

    id: str
    title: str

    @property
    @abstractmethod
    def configured(self) -> bool:
        """True when credentials (or local data) are present."""

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        lat: float | None = None,
        lng: float | None = None,
    ) -> SearchResult:
        """Return courses grouped under clubs. Must use disk cache when possible."""

    @abstractmethod
    async def load_course(
        self,
        course_id: str,
        *,
        club_id: str | None = None,
        timestamp_updated: int | None = None,
    ) -> CourseRecord:
        """Scorecard + GPS. Must not hit the network when a fresh cache exists."""
