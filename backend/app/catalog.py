"""Canonical GIS layer catalog for golf-course digital twins.

Every pipeline stage reads/writes layers by these identifiers so OSM priors,
imagery segmentation, topology repair, navigation, and exporters stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerSpec:
    id: str
    title: str
    geometry: str  # polygon | linestring | point | raster
    group: str
    color: str
    osm_filters: tuple[str, ...] = ()
    robot_class: str | None = None  # free | costly | no_go | obstacle
    mvp: bool = False


LAYERS: tuple[LayerSpec, ...] = (
    LayerSpec("boundary", "Golf course boundary", "polygon", "site", "#1f4e3d", ("leisure=golf_course",), "free", True),
    LayerSpec("fairway", "Fairways", "polygon", "play", "#3d8b40", ("golf=fairway",), "free", True),
    LayerSpec("green", "Greens", "polygon", "play", "#7cfc00", ("golf=green",), "costly", True),
    LayerSpec("green_fringe", "Green fringes", "polygon", "play", "#b4e645", ("golf=fringe", "golf=collar"), "costly", True),
    LayerSpec("pin", "Pins", "point", "play", "#ffffff", ("golf=pin",), None),
    LayerSpec("tee", "Tees", "polygon", "play", "#98fb98", ("golf=tee",), "costly", True),
    LayerSpec("bunker", "Bunkers", "polygon", "play", "#edc9af", ("golf=bunker", "golf=sand_bunker"), "no_go", True),
    LayerSpec("water", "Water hazards", "polygon", "hazard", "#1e90ff", ("golf=water_hazard", "natural=water", "waterway=riverbank"), "no_go", True),
    LayerSpec("lake", "Lakes", "polygon", "hazard", "#1874cd", ("natural=water", "water=lake"), "no_go"),
    LayerSpec("stream", "Streams", "linestring", "hazard", "#00bfff", ("waterway=stream", "waterway=river"), "no_go"),
    LayerSpec("cart_path", "Cart paths", "linestring", "access", "#c0c0c0", ("golf=cartpath", "golf=path", "highway=service"), "free", True),
    LayerSpec("walking_path", "Walking paths", "linestring", "access", "#d2b48c", ("highway=path", "highway=footway"), "free"),
    LayerSpec("road", "Roads", "linestring", "access", "#696969", ("highway=residential", "highway=unclassified", "highway=tertiary"), "obstacle"),
    LayerSpec("driving_range", "Driving range", "polygon", "practice", "#6b8e23", ("golf=driving_range",), "free"),
    LayerSpec("putting_green", "Putting green", "polygon", "practice", "#adff2f", ("golf=practice", "golf=putting_green"), "costly"),
    LayerSpec("practice_bunker", "Practice bunker", "polygon", "practice", "#deb887", ("golf=practice_bunker",), "no_go"),
    LayerSpec("clubhouse", "Clubhouse", "polygon", "built", "#8b4513", ("building=yes", "amenity=restaurant"), "obstacle"),
    LayerSpec("building", "Buildings", "polygon", "built", "#a0522d", ("building",), "obstacle"),
    LayerSpec("maintenance", "Maintenance buildings", "polygon", "built", "#5c4033", ("building=industrial",), "obstacle"),
    LayerSpec("parking", "Parking", "polygon", "built", "#778899", ("amenity=parking",), "obstacle"),
    LayerSpec("bridge", "Bridges", "polygon", "built", "#708090", ("man_made=bridge", "bridge=yes"), "free"),
    LayerSpec("fence", "Fences", "linestring", "site", "#2f4f4f", ("barrier=fence",), "obstacle"),
    LayerSpec("tree", "Trees", "point", "veg", "#228b22", ("natural=tree",), "obstacle"),
    LayerSpec("woodland", "Woodland", "polygon", "veg", "#006400", ("landuse=forest", "natural=wood"), "no_go"),
    LayerSpec("oob", "Out of bounds", "polygon", "site", "#ff4500", ("golf=out_of_bounds",), "no_go"),
    LayerSpec("natural_rough", "Natural rough", "polygon", "play", "#556b2f", ("golf=rough", "natural=grassland"), "costly"),
    LayerSpec("managed_rough", "Managed rough", "polygon", "play", "#6b8e23", ("golf=rough", "landuse=grass"), "costly"),
    LayerSpec("hole_centerline", "Hole centerlines", "linestring", "play", "#ffffff", ("golf=hole",), None, True),
    LayerSpec("no_go", "No-go zones", "polygon", "nav", "#ff0000", (), "no_go"),
    LayerSpec("obstacle", "Obstacle polygons", "polygon", "nav", "#8b0000", (), "obstacle"),
    LayerSpec("mowing_sector", "Mowing sectors", "polygon", "nav", "#32cd32", (), "free"),
    LayerSpec("coverage", "Coverage polygons", "polygon", "nav", "#00fa9a", (), "free"),
    LayerSpec("waypoint_graph", "Waypoint graph", "linestring", "nav", "#ffd700", (), None),
    LayerSpec("nav_mesh", "Robot navigation mesh", "polygon", "nav", "#40e0d0", (), "free"),
)

LAYER_BY_ID = {layer.id: layer for layer in LAYERS}
MVP_LAYERS = tuple(layer.id for layer in LAYERS if layer.mvp)

SEMANTIC_CLASSES = (
    "background",
    "fairway",
    "green",
    "tee",
    "bunker",
    "water",
    "cart_path",
    "walking_path",
    "tree",
    "woodland",
    "building",
    "parking",
    "road",
    "bridge",
    "sand",
    "practice",
    "driving_range",
    "putting_green",
    "natural_rough",
    "managed_rough",
)

# OSM golf=* values mapped to catalog ids
OSM_GOLF_TO_LAYER = {
    "fairway": "fairway",
    "green": "green",
    "fringe": "green_fringe",
    "collar": "green_fringe",
    "pin": "pin",
    "tee": "tee",
    "bunker": "bunker",
    "sand_bunker": "bunker",
    "water_hazard": "water",
    "lateral_water_hazard": "water",
    "rough": "managed_rough",
    "driving_range": "driving_range",
    "hole": "hole_centerline",
    "cartpath": "cart_path",
    "path": "cart_path",
    "practice": "putting_green",
    "putting_green": "putting_green",
}
