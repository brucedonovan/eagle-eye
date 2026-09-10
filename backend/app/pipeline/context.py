from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineContext:
    job_id: str
    work_dir: Path
    request: dict[str, Any]
    course: dict[str, Any] = field(default_factory=dict)
    bbox: tuple[float, float, float, float] | None = None  # west, south, east, north
    layers: dict[str, dict[str, Any]] = field(default_factory=dict)
    imagery: dict[str, Any] = field(default_factory=dict)
    masks: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    exports: dict[str, str] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.logs.append(message)

    def layer_dir(self) -> Path:
        path = self.work_dir / "layers"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def export_dir(self) -> Path:
        path = self.work_dir / "exports"
        path.mkdir(parents=True, exist_ok=True)
        return path
