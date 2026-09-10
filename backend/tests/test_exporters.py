from pathlib import Path

from app.services import exporters


def test_geojson_kml_gpx_roundtrip(tmp_path: Path, fake_course):
    geo = exporters.write_geojson(fake_course.layers, tmp_path / "course.geojson")
    kml = exporters.write_kml(fake_course.layers, tmp_path / "course.kml")
    gpx = exporters.write_gpx(fake_course.layers, tmp_path / "course.gpx")
    wkt = exporters.write_wkt(fake_course.layers, tmp_path / "course.wkt")
    dxf = exporters.write_dxf(fake_course.layers, tmp_path / "course.dxf")
    assert geo.stat().st_size > 20
    assert "<kml" in kml.read_text()
    assert "<gpx" in gpx.read_text()
    assert "POLYGON" in wkt.read_text() or "LINESTRING" in wkt.read_text()
    assert "LWPOLYLINE" in dxf.read_text()


def test_flatten_skips_none_layers(tmp_path: Path, fake_course):
    fake_course.layers["green"] = None  # type: ignore[assignment]
    path = exporters.write_geojson(fake_course.layers, tmp_path / "course.geojson")
    assert path.exists()


def test_schema_requires_input():
    from pydantic import ValidationError

    from app.schemas import CourseCreate

    try:
        CourseCreate()
        raise AssertionError("expected validation error")
    except ValidationError:
        pass
    CourseCreate(name="Pebble Beach Golf Links")
    CourseCreate(lat=36.56, lon=-121.94)
