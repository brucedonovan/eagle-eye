import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { LayerOut, StatusOut } from "../api";

type Props = {
  status: StatusOut | null;
  layers: Record<string, { type: string; features: unknown[] }>;
  hidden: Set<string>;
};

const LAYER_ORDER = [
  "boundary",
  "woodland",
  "natural_rough",
  "managed_rough",
  "fairway",
  "driving_range",
  "water",
  "lake",
  "tee",
  "green_fringe",
  "green",
  "putting_green",
  "pin",
  "bunker",
  "tree",
  "building",
  "clubhouse",
  "parking",
  "nav_mesh",
  "no_go",
  "cart_path",
  "walking_path",
  "hole_centerline",
  "waypoint_graph",
];

export function MapView({ status, layers, hidden }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const fittedJob = useRef<string | null>(null);
  const pinMarkers = useRef<maplibregl.Marker[]>([]);

  useEffect(() => {
    if (!ref.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: ref.current,
      style: {
        version: 8,
        sources: {
          esri: {
            type: "raster",
            tiles: [
              "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            ],
            tileSize: 256,
            maxzoom: 19,
            attribution: "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
          },
        },
        layers: [{ id: "esri", type: "raster", source: "esri" }],
      },
      center: [-9.26, 38.73],
      zoom: 14,
      maxZoom: 22,
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "bottom-right");
    map.on("load", () => map.resize());
    mapRef.current = map;
    return () => {
      for (const marker of pinMarkers.current) marker.remove();
      pinMarkers.current = [];
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const apply = () => {
      paintLayers(map, status, layers, hidden, fittedJob);
      syncPinMarkers(map, layers, hidden, pinMarkers);
    };
    if (!map.isStyleLoaded()) {
      map.once("load", apply);
      return;
    }
    apply();
  }, [status, layers, hidden]);

  return (
    <div className="map-wrap">
      <div className="map" ref={ref} />
      {status?.course && (
        <div className="hud">
          <strong>{status.course.display_name || status.course.name}</strong>
          {status.course.country && <span>{status.course.country}</span>}
          {status.course.address && <span>{status.course.address}</span>}
        </div>
      )}
    </div>
  );
}

function paintLayers(
  map: maplibregl.Map,
  status: StatusOut | null,
  layers: Record<string, { type: string; features: unknown[] }>,
  hidden: Set<string>,
  fittedJob: { current: string | null },
) {
  const ids = Object.keys(layers);
  for (const id of ids) {
    const sourceId = `lyr-${id}`;
    const fillId = `${sourceId}-fill`;
    const lineId = `${sourceId}-line`;
    const circleId = `${sourceId}-circle`;
    if (map.getLayer(fillId)) map.removeLayer(fillId);
    if (map.getLayer(lineId)) map.removeLayer(lineId);
    if (map.getLayer(circleId)) map.removeLayer(circleId);
    if (map.getSource(sourceId)) map.removeSource(sourceId);
  }

  const meta: Record<string, LayerOut> = {};
  for (const layer of status?.layers ?? []) meta[layer.layer_id] = layer;

  const order = [...LAYER_ORDER, ...ids.filter((id) => !LAYER_ORDER.includes(id))];
  for (const id of order) {
    const fc = layers[id];
    if (!fc || hidden.has(id)) continue;
    const sourceId = `lyr-${id}`;
    map.addSource(sourceId, {
      type: "geojson",
      data: fc as never,
      // GeoJSON is internally tiled; the default maxzoom of 18 drops
      // polygons as soon as you inspect a green or bunker up close.
      maxzoom: 24,
      tolerance: 0,
      buffer: 256,
    });
    const color = meta[id]?.color || "#ffffff";
    const geom = meta[id]?.geometry || "polygon";
    if (geom === "polygon") {
      map.addLayer({
        id: `${sourceId}-fill`,
        type: "fill",
        source: sourceId,
        paint: { "fill-color": color, "fill-opacity": id === "boundary" ? 0.08 : 0.32 },
      });
      map.addLayer({
        id: `${sourceId}-line`,
        type: "line",
        source: sourceId,
        paint: { "line-color": color, "line-width": id === "boundary" ? 1.6 : 0.7 },
      });
    } else if (geom === "linestring") {
      map.addLayer({
        id: `${sourceId}-line`,
        type: "line",
        source: sourceId,
        paint: {
          "line-color": color,
          "line-width": id === "hole_centerline" ? 2.6 : 1.6,
          "line-dasharray": id === "hole_centerline" ? [1, 0] : [1, 1],
        },
      });
    } else {
      map.addLayer({
        id: `${sourceId}-circle`,
        type: "circle",
        source: sourceId,
        paint: {
          "circle-color": color,
          "circle-radius": id === "pin" ? 7 : 4,
          "circle-stroke-width": 1.2,
          "circle-stroke-color": "#111",
        },
      });
    }
  }

  const jobId = status?.job_id;
  if (jobId && fittedJob.current !== jobId && status?.status === "completed") {
    fittedJob.current = jobId;
    const bbox = status.course?.bbox;
    if (bbox && bbox.length === 4) {
      map.fitBounds(
        [
          [bbox[0], bbox[1]],
          [bbox[2], bbox[3]],
        ],
        { padding: 48, duration: 800, maxZoom: 16 },
      );
    } else if (status.course?.lon && status.course.lat) {
      map.flyTo({ center: [status.course.lon, status.course.lat], zoom: 16, duration: 800 });
    }
  }
}

type GeoFeat = {
  geometry?: { type?: string; coordinates?: number[] | number[][] };
  properties?: Record<string, unknown>;
};

function syncPinMarkers(
  map: maplibregl.Map,
  layers: Record<string, { type: string; features: unknown[] }>,
  hidden: Set<string>,
  pinMarkers: { current: maplibregl.Marker[] },
) {
  for (const marker of pinMarkers.current) marker.remove();
  pinMarkers.current = [];
  if (hidden.has("pin")) return;
  for (const pin of numberedPins(layers)) {
    const hole = pin.properties?.hole;
    const coords = pin.geometry?.coordinates;
    if (hole == null || !Array.isArray(coords) || coords.length < 2) continue;
    if (typeof coords[0] !== "number" || typeof coords[1] !== "number") continue;
    const el = document.createElement("div");
    el.className = "pin-num";
    el.textContent = String(hole);
    pinMarkers.current.push(
      new maplibregl.Marker({ element: el, anchor: "center" }).setLngLat([coords[0], coords[1]]).addTo(map),
    );
  }
}

function numberedPins(layers: Record<string, { type: string; features: unknown[] }>): GeoFeat[] {
  const pins = (layers.pin?.features ?? []) as GeoFeat[];
  const holes = (layers.hole_centerline?.features ?? []) as GeoFeat[];
  return pins.map((pin) => {
    if (pin.properties?.hole != null) return pin;
    const coords = pin.geometry?.coordinates;
    if (!coords || pin.geometry?.type !== "Point" || typeof coords[0] !== "number" || typeof coords[1] !== "number") return pin;
    let best: { n: number; d: number } | null = null;
    for (const hole of holes) {
      const line = hole.geometry?.coordinates;
      const n = Number(hole.properties?.hole);
      if (!Array.isArray(line) || !Number.isFinite(n)) continue;
      const start = line[0];
      const end = line[line.length - 1];
      for (const pt of [start, end]) {
        if (!Array.isArray(pt) || typeof pt[0] !== "number") continue;
        const d = Math.hypot(pt[0] - coords[0], pt[1] - coords[1]);
        if (!best || d < best.d) best = { n, d };
      }
    }
    if (best && best.d < 0.00045) {
      return { ...pin, properties: { ...pin.properties, hole: best.n } };
    }
    return pin;
  });
}
