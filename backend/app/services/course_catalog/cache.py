"""Disk cache for catalog HTTP payloads. Shared by every provider."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


@dataclass
class CacheResult:
    payload: Any
    cached: bool
    api_requests_left: str | None = None


class DiskJsonCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def last_requests_left(self) -> str | None:
        meta = read_json(self.root / "meta.json")
        if not meta:
            return None
        return meta.get("api_requests_left")

    def get(
        self,
        bucket: str,
        key: str,
        *,
        ttl_days: int | None = None,
        newer_than: int | None = None,
    ) -> CacheResult | None:
        path_file = self.root / bucket / f"{safe_id(key)}.json"
        entry = read_json(path_file)
        if not entry or not cache_usable(entry, ttl_days=ttl_days, newer_than=newer_than):
            return None
        return CacheResult(
            payload=entry.get("payload"),
            cached=True,
            api_requests_left=entry.get("api_requests_left") or self.last_requests_left(),
        )

    def put(
        self,
        bucket: str,
        key: str,
        payload: Any,
        *,
        path: str = "",
        params: dict[str, Any] | None = None,
        api_requests_left: Any = None,
    ) -> CacheResult:
        if api_requests_left is not None:
            write_json(self.root / "meta.json", {"api_requests_left": str(api_requests_left)})
        write_json(
            self.root / bucket / f"{safe_id(key)}.json",
            {
                "cached_at": now_iso(),
                "path": path,
                "params": params or {},
                "api_requests_left": api_requests_left,
                "payload": payload,
            },
        )
        return CacheResult(
            payload=payload,
            cached=False,
            api_requests_left=str(api_requests_left) if api_requests_left is not None else None,
        )

    def iter_entries(self, bucket: str, *, ttl_days: int | None = None) -> list[dict[str, Any]]:
        folder = self.root / bucket
        if not folder.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in folder.glob("*.json"):
            if path.name.endswith(".tmp") or path.suffix != ".json":
                continue
            entry = read_json(path)
            if not entry or not cache_usable(entry, ttl_days=ttl_days, newer_than=None):
                continue
            rows.append(entry)
        rows.sort(key=lambda row: str(row.get("cached_at") or ""), reverse=True)
        return rows


def params_key(path: str, params: dict[str, Any]) -> str:
    blob = path + "?" + urlencode(sorted((str(k), str(v)) for k, v in params.items()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def cache_usable(entry: dict[str, Any], *, ttl_days: int | None, newer_than: int | None) -> bool:
    payload = entry.get("payload")
    if newer_than:
        cached_ts = 0
        if isinstance(payload, dict):
            cached_ts = as_int(payload.get("timestampUpdated") or payload.get("timestamp_updated"))
        if cached_ts and cached_ts < int(newer_than):
            return False
    if ttl_days is None:
        return True
    cached_at = entry.get("cached_at")
    if not cached_at:
        return False
    try:
        when = datetime.fromisoformat(str(cached_at).replace("Z", "+00:00"))
    except ValueError:
        return False
    age = datetime.now(UTC) - when.astimezone(UTC)
    return age.total_seconds() <= ttl_days * 86400


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned[:80] or "unknown"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
    tmp.replace(path)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def as_int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        n = as_int(item)
        if n:
            out.append(n)
        elif item in {0, "0"}:
            out.append(0)
    return out


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
