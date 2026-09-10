from shapely.geometry import box

from app.services.overpass import (
    classify_osm,
    clip_layers_to_boundary,
    elements_to_layers,
    name_score,
)


def test_classify_osm_golf_tags():
    assert classify_osm({"golf": "green"}, "way") == "green"
    assert classify_osm({"golf": "fringe"}, "way") == "green_fringe"
    assert classify_osm({"golf": "bunker"}, "way") == "bunker"
    assert classify_osm({"natural": "tree"}, "node") == "tree"
    assert classify_osm({"golf": "green;fairway"}, "way") == "green"
    assert classify_osm({"golf": "pin"}, "node") == "pin"
    assert classify_osm({"leisure": "golf_course"}, "relation") == "boundary"
    assert classify_osm({"natural": "water"}, "way") == "water"
    assert classify_osm({"natural": "sand"}, "way") == "bunker"
    assert classify_osm({"highway": "path"}, "way") == "walking_path"
    assert classify_osm({"building": "yes", "name": "Club House"}, "way") == "clubhouse"


def test_elements_to_layers_pin_node():
    payload = {
        "elements": [
            {
                "type": "node",
                "id": 42,
                "lon": -9.2515,
                "lat": 38.7141,
                "tags": {"golf": "pin"},
            }
        ]
    }
    layers = elements_to_layers(payload)
    assert layers["pin"]["features"][0]["geometry"]["type"] == "Point"


def test_elements_to_layers_closed_way():
    payload = {
        "elements": [
            {"type": "node", "id": 1, "lon": 0.0, "lat": 0.0},
            {"type": "node", "id": 2, "lon": 0.001, "lat": 0.0},
            {"type": "node", "id": 3, "lon": 0.001, "lat": 0.001},
            {"type": "node", "id": 4, "lon": 0.0, "lat": 0.001},
            {
                "type": "way",
                "id": 10,
                "nodes": [1, 2, 3, 4, 1],
                "tags": {"golf": "green", "ref": "1"},
            },
        ]
    }
    layers = elements_to_layers(payload)
    assert "green" in layers
    assert layers["green"]["features"][0]["geometry"]["type"] == "Polygon"
    assert layers["green"]["features"][0]["properties"]["ref"] == "1"


def test_elements_to_layers_out_geom():
    payload = {
        "elements": [
            {
                "type": "way",
                "id": 99,
                "tags": {"golf": "bunker"},
                "geometry": [
                    {"lat": 36.56, "lon": -121.95},
                    {"lat": 36.56, "lon": -121.949},
                    {"lat": 36.561, "lon": -121.949},
                    {"lat": 36.561, "lon": -121.95},
                    {"lat": 36.56, "lon": -121.95},
                ],
            }
        ]
    }
    layers = elements_to_layers(payload)
    assert layers["bunker"]["features"][0]["geometry"]["type"] == "Polygon"


def test_relation_joins_split_outer_ways():
    payload = {
        "elements": [
            {
                "type": "relation",
                "id": 1,
                "tags": {"leisure": "golf_course", "name": "Split Course"},
                "bounds": {"minlat": 0.0, "minlon": 0.0, "maxlat": 0.002, "maxlon": 0.002},
                "members": [
                    {
                        "type": "way",
                        "role": "outer",
                        "geometry": [
                            {"lat": 0.0, "lon": 0.0},
                            {"lat": 0.0, "lon": 0.002},
                        ],
                    },
                    {
                        "type": "way",
                        "role": "outer",
                        "geometry": [
                            {"lat": 0.0, "lon": 0.002},
                            {"lat": 0.002, "lon": 0.002},
                            {"lat": 0.002, "lon": 0.0},
                            {"lat": 0.0, "lon": 0.0},
                        ],
                    },
                ],
            }
        ]
    }
    layers = elements_to_layers(payload)
    geom = layers["boundary"]["features"][0]["geometry"]
    assert geom["type"] == "Polygon"
    # Full 0.002 x 0.002 square, not a collapsed sliver.
    from shapely.geometry import shape

    assert shape(geom).area > 3e-6


def test_name_score_prefers_pebble_beach():
    assert name_score("Pebble Beach Golf Links", "Pebble Beach Golf Course") > 0.9
    assert name_score("Pebble Beach Golf Links", "The Hay") == 0
    assert name_score("Pebble Beach Golf Links", "Cypress Point Golf Course") < 0.4


def test_name_score_matches_jamor_official_vs_osm():
    assert (
        name_score(
            "Centro Nacional de Formação de Golfe do Jamor",
            "Centro Nacional de Formação do Jamor",
        )
        >= 0.6
    )


def test_clip_drops_features_outside_course():
    layers = {
        "boundary": {"type": "FeatureCollection", "features": []},
        "green": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": box(-121.94, 36.56, -121.939, 36.561).__geo_interface__,
                    "properties": {"id": "in"},
                },
                {
                    "type": "Feature",
                    "geometry": box(-122.2, 36.8, -122.19, 36.81).__geo_interface__,
                    "properties": {"id": "out"},
                },
            ],
        },
    }
    clipped = clip_layers_to_boundary(layers, box(-121.95, 36.55, -121.93, 36.57))
    assert len(clipped["green"]["features"]) == 1
    assert clipped["green"]["features"][0]["properties"]["id"] == "in"
