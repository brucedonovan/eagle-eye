"""Course catalog registry. Search and discovery call get_provider()."""

from __future__ import annotations

from collections.abc import Callable

from app.config import settings
from app.services.course_catalog.provider import (
    CatalogError,
    CourseCatalogProvider,
    CourseHit,
    CourseRecord,
    SearchResult,
)

ProviderFactory = Callable[[], CourseCatalogProvider]

PROVIDERS: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    PROVIDERS[name.strip().lower()] = factory


def _ensure_builtins() -> None:
    if "golfapi" not in PROVIDERS:
        from app.services.course_catalog.providers.golfapi import GolfApiProvider

        register_provider("golfapi", GolfApiProvider)


def get_provider(name: str | None = None) -> CourseCatalogProvider | None:
    """Return the active catalog, or None for OpenStreetMap-only search."""
    _ensure_builtins()
    key = (name or settings.course_catalog_provider or "golfapi").strip().lower()
    if key in {"", "none", "off", "osm"}:
        return None
    factory = PROVIDERS.get(key)
    if factory is None:
        known = ", ".join(sorted(PROVIDERS) or ["(none)"])
        raise CatalogError(f"Unknown course catalog provider {key!r}. Registered: {known}")
    provider = factory()
    return provider if provider.configured else None


def status() -> dict:
    _ensure_builtins()
    try:
        provider = get_provider()
    except CatalogError as exc:
        return {
            "provider": settings.course_catalog_provider,
            "configured": False,
            "registered": sorted(PROVIDERS),
            "error": str(exc),
        }
    if provider is None:
        return {
            "provider": settings.course_catalog_provider,
            "configured": False,
            "registered": sorted(PROVIDERS),
            "search": "/search",
            "cache": "disk",
        }
    return {
        "provider": provider.id,
        "title": provider.title,
        "configured": True,
        "registered": sorted(PROVIDERS),
        "search": "/search",
        "cache": "disk",
    }


__all__ = [
    "PROVIDERS",
    "CatalogError",
    "CourseCatalogProvider",
    "CourseHit",
    "CourseRecord",
    "SearchResult",
    "get_provider",
    "register_provider",
    "status",
]
