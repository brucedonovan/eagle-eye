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
      syncHoleMarkers(map, layers, hidden, pinMarkers);
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
          <Scorecard layers={layers} />
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
  geometry?: { type?: string; coordinates?: unknown };
  properties?: Record<string, unknown>;
};

type HoleCard = {
  hole: number;
  par?: number;
  si?: number;
  tee?: [number, number];
  green?: [number, number];
  mid?: [number, number];
};

function Scorecard({ layers }: { layers: Record<string, { type: string; features: unknown[] }> }) {
  const holes = holeCards(layers);
  if (!holes.length) return null;
  const totalPar = holes.reduce((sum, hole) => sum + (hole.par ?? 0), 0);
  const withPar = holes.filter((hole) => hole.par != null).length;
  return (
    <>
      <div className="scorecard">
        {holes.map((hole) => (
          <div className="sc-hole" key={hole.hole}>
            <b>{hole.hole}</b>
            <em>{hole.par != null ? `Par ${hole.par}` : "—"}</em>
            {hole.si != null ? <small>SI {hole.si}</small> : null}
          </div>
        ))}
      </div>
      {withPar > 0 && (
        <div className="sc-total">
          {holes.length} holes · par {totalPar}
        </div>
      )}
    </>
  );
}

function syncHoleMarkers(
  map: maplibregl.Map,
  layers: Record<string, { type: string; features: unknown[] }>,
  hidden: Set<string>,
  pinMarkers: { current: maplibregl.Marker[] },
) {
  for (const marker of pinMarkers.current) marker.remove();
  pinMarkers.current = [];
  const showGreen = !hidden.has("pin") || !hidden.has("green") || !hidden.has("hole_centerline");
  const showTee = !hidden.has("tee") || !hidden.has("hole_centerline");
  if (!showGreen && !showTee) return;

  const pinByHole = new Map<number, [number, number]>();
  for (const pin of numberedPins(layers)) {
    const hole = Number(pin.properties?.hole);
    const coords = asLngLat(pin.geometry?.coordinates);
    if (!Number.isFinite(hole) || !coords) continue;
    pinByHole.set(hole, coords);
  }

  for (const hole of holeCards(layers)) {
    const green = pinByHole.get(hole.hole) ?? hole.green;
    if (showGreen && green) {
      pinMarkers.current.push(flagMarker(green, hole));
    }
    if (showTee && hole.tee && (!green || distLngLat(hole.tee, green) > 0.00012)) {
      pinMarkers.current.push(teeMarker(hole.tee, hole));
    }
  }
  for (const marker of pinMarkers.current) marker.addTo(map);
}

function teeMarker(lngLat: [number, number], hole: HoleCard) {
  const el = document.createElement("div");
  el.className = "hole-badge tee";
  const num = document.createElement("b");
  num.textContent = String(hole.hole);
  el.append(num);
  if (hole.par != null) {
    const par = document.createElement("small");
    par.textContent = `Par ${hole.par}`;
    el.append(par);
  }
  if (hole.si != null) {
    const si = document.createElement("small");
    si.textContent = `SI ${hole.si}`;
    el.append(si);
  }
  return new maplibregl.Marker({ element: el, anchor: "center" }).setLngLat(lngLat);
}

function flagMarker(lngLat: [number, number], hole: HoleCard) {
  const el = document.createElement("div");
  el.className = "green-flag";
  el.title = hole.par != null ? `Hole ${hole.hole} · Par ${hole.par}` : `Hole ${hole.hole}`;
  el.innerHTML = `<svg viewBox="0 0 18 28" aria-hidden="true">
    <line x1="3.5" y1="1" x2="3.5" y2="26" stroke="#111" stroke-width="1.6" stroke-linecap="round"/>
    <path d="M4 2.2 L16.5 7.2 L4 12.4 Z" fill="#e23b2e" stroke="#111" stroke-width="1"/>
    <circle cx="3.5" cy="26.4" r="1.7" fill="#111"/>
  </svg>`;
  return new maplibregl.Marker({ element: el, anchor: "bottom" }).setLngLat(lngLat);
}

function holeCards(layers: Record<string, { type: string; features: unknown[] }>): HoleCard[] {
  const byHole = new Map<number, HoleCard>();
  for (const feat of (layers.hole_centerline?.features ?? []) as GeoFeat[]) {
    const n = Number(feat.properties?.hole ?? feat.properties?.ref);
    if (!Number.isFinite(n) || n < 1) continue;
    const line = lineCoords(feat.geometry);
    if (!line || line.length < 2) continue;
    const tee = line[0];
    const green = line[line.length - 1];
    const mid = line[Math.floor(line.length / 2)] ?? midpoint(tee, green);
    const par = numProp(feat.properties, "par");
    const si = numProp(feat.properties, "stroke_index");
    byHole.set(n, { hole: n, par, si, tee, green, mid });
  }
  const teeBoxes = new Map<number, [number, number][]>();
  for (const feat of (layers.tee?.features ?? []) as GeoFeat[]) {
    const n = Number(feat.properties?.hole ?? feat.properties?.catalog_hole ?? feat.properties?.ref);
    const coords = featureCentroid(feat);
    if (!Number.isFinite(n) || n < 1 || !coords) continue;
    const list = teeBoxes.get(n) ?? [];
    list.push(coords);
    teeBoxes.set(n, list);
    const prev = byHole.get(n);
    if (!prev) {
      byHole.set(n, {
        hole: n,
        tee: coords,
        par: numProp(feat.properties, "par"),
        si: numProp(feat.properties, "stroke_index"),
      });
    }
  }
  for (const [n, boxes] of teeBoxes) {
    const prev = byHole.get(n);
    if (!prev) continue;
    const target = prev.tee ?? prev.green;
    const best = target
      ? boxes.reduce((a, b) => (distLngLat(a, target) <= distLngLat(b, target) ? a : b))
      : boxes[0];
    byHole.set(n, { ...prev, tee: best });
  }
  for (const pin of numberedPins(layers)) {
    const n = Number(pin.properties?.hole);
    const coords = asLngLat(pin.geometry?.coordinates);
    if (!Number.isFinite(n) || !coords) continue;
    const prev = byHole.get(n) ?? { hole: n };
    byHole.set(n, { ...prev, green: coords, par: prev.par ?? numProp(pin.properties, "par"), si: prev.si ?? numProp(pin.properties, "stroke_index") });
  }
  return [...byHole.values()].sort((a, b) => a.hole - b.hole);
}

function numberedPins(layers: Record<string, { type: string; features: unknown[] }>): GeoFeat[] {
  const pins = (layers.pin?.features ?? []) as GeoFeat[];
  const holes = (layers.hole_centerline?.features ?? []) as GeoFeat[];
  return pins.map((pin) => {
    if (pin.properties?.hole != null) return pin;
    const coords = asLngLat(pin.geometry?.coordinates);
    if (!coords) return pin;
    let best: { n: number; d: number } | null = null;
    for (const hole of holes) {
      const line = lineCoords(hole.geometry);
      const n = Number(hole.properties?.hole);
      if (!line || !Number.isFinite(n)) continue;
      for (const pt of [line[0], line[line.length - 1]]) {
        const d = distLngLat(pt, coords);
        if (!best || d < best.d) best = { n, d };
      }
    }
    if (best && best.d < 0.00045) {
      return { ...pin, properties: { ...pin.properties, hole: best.n } };
    }
    return pin;
  });
}

function lineCoords(geometry?: GeoFeat["geometry"]): [number, number][] | null {
  if (!geometry || geometry.type !== "LineString" || !Array.isArray(geometry.coordinates)) return null;
  const out: [number, number][] = [];
  for (const pt of geometry.coordinates) {
    const pair = asLngLat(pt);
    if (pair) out.push(pair);
  }
  return out.length >= 2 ? out : null;
}

function featureCentroid(feat: GeoFeat): [number, number] | null {
  const geom = feat.geometry;
  if (!geom) return null;
  if (geom.type === "Point") return asLngLat(geom.coordinates);
  const ring = outerRing(geom.coordinates);
  if (!ring || ring.length < 3) return null;
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (const pt of ring) {
    const pair = asLngLat(pt);
    if (!pair) continue;
    sx += pair[0];
    sy += pair[1];
    n += 1;
  }
  return n ? [sx / n, sy / n] : null;
}

function outerRing(coordinates: unknown): unknown[] | null {
  if (!Array.isArray(coordinates) || coordinates.length === 0) return null;
  const first = coordinates[0];
  if (!Array.isArray(first) || first.length === 0) return null;
  if (typeof first[0] === "number") return coordinates;
  const nested = first[0];
  if (Array.isArray(nested) && typeof nested[0] === "number") return first as unknown[];
  if (Array.isArray(nested) && Array.isArray(nested[0])) return nested as unknown[];
  return null;
}

function asLngLat(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length < 2) return null;
  if (typeof value[0] !== "number" || typeof value[1] !== "number") return null;
  return [value[0], value[1]];
}

function midpoint(a: [number, number], b: [number, number]): [number, number] {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}

function distLngLat(a: [number, number], b: [number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function numProp(props: Record<string, unknown> | undefined, key: string): number | undefined {
  const value = Number(props?.[key]);
  return Number.isFinite(value) ? value : undefined;
}
