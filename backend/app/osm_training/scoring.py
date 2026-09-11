"""OSM golf-course mapping completeness scores.

A well-mapped championship course typically has greens, fairways, tees, hole
ways, bunkers, and cart paths. Optional water / pins / rough add a small bonus.
The score is designed so OSM improvements automatically raise a course's rank
when the census is rerun.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

CORE_WEIGHTS = {
    "green": 24.0,
    "fairway": 20.0,
    "tee": 14.0,
    "hole": 16.0,
    "bunker": 10.0,
    "cart_path": 8.0,
}

BONUS_WEIGHTS = {
    "water": 3.0,
    "pin": 2.0,
    "rough": 2.0,
    "named": 3.0,
}

MINIATURE_TAGS = {"miniature", "pitch_and_putt", "par_3", "par3", "disc_golf"}


@dataclass
class FeatureCounts:
    green: int = 0
    fairway: int = 0
    tee: int = 0
    bunker: int = 0
    hole: int = 0
    cart_path: int = 0
    water: int = 0
    pin: int = 0
    rough: int = 0
    fringe: int = 0
    driving_range: int = 0
    putting_green: int = 0
    green_ways: int = 0
    fairway_ways: int = 0
    tee_ways: int = 0
    hole_ways: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class CompletenessResult:
    score: float
    expected_holes: int
    components: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "expected_holes": self.expected_holes,
            "components": self.components,
            "flags": self.flags,
        }


def infer_expected_holes(counts: FeatureCounts, tags: dict[str, str] | None = None) -> int:
    tags = tags or {}
    tagged = _int_tag(tags, "holes") or _int_tag(tags, "golf:holes")
    if tagged in {9, 18, 27, 36}:
        return tagged
    hint = max(counts.green, counts.hole, counts.fairway, counts.green_ways, counts.hole_ways)
    if hint >= 33:
        return 36
    if hint >= 24:
        return 27
    if hint >= 14:
        return 18
    if hint >= 7:
        return 9
    return 18


def completeness_score(
    counts: FeatureCounts,
    tags: dict[str, str] | None = None,
    *,
    expected_holes: int | None = None,
) -> CompletenessResult:
    tags = {str(k): str(v) for k, v in (tags or {}).items()}
    expected = expected_holes or infer_expected_holes(counts, tags)
    flags: list[str] = []
    components: dict[str, float] = {}

    expected_bunkers = max(6.0, expected * 0.75)
    targets = {
        "green": float(expected),
        "fairway": float(expected),
        "tee": float(expected),
        "hole": float(expected),
        "bunker": expected_bunkers,
        "cart_path": 1.0,
    }
    observed = {
        "green": counts.green,
        "fairway": counts.fairway,
        "tee": counts.tee,
        "hole": counts.hole,
        "bunker": counts.bunker,
        "cart_path": counts.cart_path,
    }

    score = 0.0
    for key, weight in CORE_WEIGHTS.items():
        ratio = _ratio(observed[key], targets[key])
        points = weight * ratio
        components[key] = round(points, 3)
        score += points

    bonus_obs = {
        "water": counts.water,
        "pin": counts.pin,
        "rough": counts.rough,
        "named": 1.0 if (tags.get("name") or "").strip() else 0.0,
    }
    bonus_targets = {"water": 1.0, "pin": float(expected), "rough": 1.0, "named": 1.0}
    for key, weight in BONUS_WEIGHTS.items():
        ratio = _ratio(bonus_obs[key], bonus_targets[key])
        points = weight * ratio
        components[key] = round(points, 3)
        score += points

    if counts.green_ways == 0 and counts.green > 0:
        flags.append("green_nodes_only")
        score *= 0.85
    if counts.green < expected * 0.5:
        flags.append("incomplete_core")
    if not (tags.get("name") or "").strip():
        flags.append("unnamed")
    if _is_miniature(tags):
        flags.append("miniature")
        score *= 0.35
    if counts.green == 0 and counts.driving_range > 0:
        flags.append("driving_range_only")
        score *= 0.3
    if counts.green == 0 and counts.fairway == 0 and counts.hole == 0:
        flags.append("boundary_only")

    score = round(min(max(score, 0.0), 100.0), 3)
    return CompletenessResult(
        score=score,
        expected_holes=expected,
        components=components,
        flags=flags,
    )


def counts_from_layers(layers: dict[str, Any] | None) -> FeatureCounts:
    layers = layers or {}
    counts = FeatureCounts(
        green=_n(layers, "green"),
        fairway=_n(layers, "fairway"),
        tee=_n(layers, "tee"),
        bunker=_n(layers, "bunker"),
        hole=_n(layers, "hole_centerline"),
        cart_path=_n(layers, "cart_path"),
        water=_n(layers, "water") + _n(layers, "lake"),
        pin=_n(layers, "pin"),
        rough=_n(layers, "managed_rough") + _n(layers, "natural_rough"),
        fringe=_n(layers, "green_fringe"),
        driving_range=_n(layers, "driving_range"),
        putting_green=_n(layers, "putting_green"),
    )
    counts.green_ways = _n_geom(layers, "green", {"Polygon", "MultiPolygon"})
    counts.fairway_ways = _n_geom(layers, "fairway", {"Polygon", "MultiPolygon"})
    counts.tee_ways = _n_geom(layers, "tee", {"Polygon", "MultiPolygon"})
    counts.hole_ways = _n_geom(layers, "hole_centerline", {"LineString", "MultiLineString"})
    return counts


def _n(layers: dict[str, Any], layer_id: str) -> int:
    fc = layers.get(layer_id) or {}
    return len(fc.get("features") or [])


def _n_geom(layers: dict[str, Any], layer_id: str, types: set[str]) -> int:
    fc = layers.get(layer_id) or {}
    n = 0
    for feat in fc.get("features") or []:
        geom = (feat.get("geometry") or {}).get("type")
        if geom in types:
            n += 1
    return n


def _ratio(count: float, expected: float) -> float:
    if expected <= 0:
        return 0.0
    return min(float(count) / expected, 1.0)


def _int_tag(tags: dict[str, str], key: str) -> int:
    raw = (tags.get(key) or "").strip()
    if not raw:
        return 0
    try:
        return int(float(raw))
    except ValueError:
        return 0


def _is_miniature(tags: dict[str, str]) -> bool:
    golf = (tags.get("golf") or "").lower()
    if any(part.strip() in MINIATURE_TAGS for part in golf.split(";")):
        return True
    name = (tags.get("name") or "").lower()
    return "mini golf" in name or "miniature" in name or "disc golf" in name
