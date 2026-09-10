# Eagle Eye

Autonomous golf-course vectorization. Give the platform a **course name**, a **lat/lon**, or a **polygon AOI**. It discovers the site, pulls the best available public (and optionally licensed) evidence, and writes a GIS / robotics dataset.

This is a **hybrid geospatial AI system**, not a pure computer-vision demo. Existing vectors (OpenStreetMap golf tagging, official GIS when present) are the prior. **Image analysis is required** to confirm those priors and fill missing greens, fairways, bunkers, tees, and water. Optional SAM/ONNX weights can refine later; the default path is CPU color+texture fusion (`backend=imagery_fusion`). Topology rules and hole-graph heuristics keep the output valid.

```
User → Course Discovery → AOI → Imagery → Preprocess → Segmentation
     → Instances → Hole ID → Vectorize → Topology → Navigation → Export
```

## What the MVP produces

From a single name such as `Pebble Beach Golf Links` or `Centro Nacional de Formação de Golfe do Jamor`:

- Course boundary, fairways, greens, tees, bunkers, water, cart paths
- Hole centerlines (OSM `golf=hole` when present, otherwise tee–green pairing)
- No-go zones, coverage / mowing sectors, waypoint graph, Nav2 occupancy grid
- Downloads: GeoJSON, KML, GPX, WKT, DXF, and GeoPackage / Shapefile when GDAL extras are installed

Later stages (trained SegFormer / SAM2, digital-twin meshes, OpenDRIVE, continuous global refresh) plug into the same catalog and job API.

## Quick start

```bash
# API
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
mkdir -p ../data
DATA_DIR=../data DATABASE_URL=sqlite+aiosqlite:///../data/eagle_eye.db \
  uvicorn app.main:app --reload --port 8000

# UI (second terminal)
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The Vite dev server proxies `/api` to the FastAPI process.

Or: `cp .env.example .env && docker compose up --build` → API on `:8000`, UI on `:8080`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/search?q=` | Search the configured course catalog (cached) and list courses |
| `POST` | `/course` | Discover + run the full pipeline |
| `POST` | `/segment` `/vectorize` `/navigation` | Same job entry (stage aliases) |
| `GET` | `/status/{job_id}` | Stage, progress, quality, layer counts |
| `GET` | `/layers/{job_id}` | All GeoJSON layers |
| `GET` | `/download/{job_id}?format=geojson` | `geojson` `kml` `gpx` `wkt` `dxf` `gpkg` `shp` `ros_grid` |
| `GET` | `/capabilities` | Layer catalog + imagery sources |

```bash
curl -s -X POST http://localhost:8000/course \
  -H 'Content-Type: application/json' \
  -d '{"name":"Pebble Beach Golf Links"}'
```

## Architecture notes

**Discovery** talks to a **course catalog** interface (`CourseCatalogProvider`), not a vendor SDK. Search and course identity use [golfapi.io](https://golfapi.io/) only (`COURSE_CATALOG_PROVIDER=golfapi`): search is clubs (`GET /clubs`); a club owns nested courses; selecting one fetches scorecard + GPS (`GET /courses/{id}`, `GET /coordinates/{id}`). Course/GPS payloads are **cached on disk forever**; club search is cached for 30 days. Another vendor with clubs, courses, scorecards, and GPS is a drop-in: implement `providers/<name>.py`, `register_provider(...)`, set `COURSE_CATALOG_PROVIDER`. Overpass still supplies `leisure=golf_course` and `golf=*` polygons after a catalog hit. The catalog wins for identity (name, hole numbers, pins, scorecard) and for the AOI when its GPS hull is tighter than a multi-course OSM boundary.

**Imagery** is a ranked registry (Nearmap / Maxar / Planet / Google / Mapbox when keys exist, otherwise ESRI World Imagery, then Sentinel-2 / USGS). The course mosaic stays on public ESRI. Pins that lack an OSM green get a z19 crop from Google Static or Mapbox Satellite when a key is set, otherwise ESRI at the same zoom.

**Segmentation** (`backend=imagery_fusion`) reads the enhanced mosaic: HSV/Lab, a simple vegetation index `(G−R)/(G+R)`, morphology, and contours. OSM polygons are rasterized as a prior — they confirm overlapping features and seed hole greens; imagery fills missing greens, bunkers, water, tees, and grass corridors along hole centerlines. Optional `SEGMENTATION_WEIGHTS` (ONNX/SAM) are a refine hook and are never required. No GPU for the default path. Limits of this pass: RGB-only (no NIR), tree/building shadows, and seasonal/dormant grass that looks like sand or rough.

**Hole identification** is deliberately heuristic. OSM hole ways with `ref` are preferred. Otherwise tees are paired to greens with a fairway-intersection bonus and numbered by a nearest-neighbor tour from the clubhouse. A hole with a pin and no OSM green may receive a compact pin-seeded imagery outline (high-zoom crop: Google Static or Mapbox when keyed, else ESRI z19). Circles are never invented; overlap with an OSM green is rejected. Ambiguous 27-hole / shared-green courses, or holes still missing a green, are flagged `holes_need_review`.

**Topology** enforces: greens ∩ bunkers = ∅, water ∩ fairways = ∅, buildings ∩ greens = ∅, each hole ends on one green.

## Milestone map

1. **MVP (this repo)** — end-to-end hybrid pipeline, review map, GeoJSON-first exports.
2. **Production** — PostGIS + Celery/Redis (`JOB_BACKEND=celery`), GPU tile inference, QA scoring, editor tools.
3. **Enterprise** — global coverage, reprocessing when new imagery arrives, glTF/USD twins, ROS2 / Unreal / fleet APIs.

## Tests

```bash
cd backend && .venv/bin/pytest -q
```

## Attribution

Geocoding and vector priors: © OpenStreetMap contributors. Default satellite tiles: Esri World Imagery.
