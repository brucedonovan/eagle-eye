import { useEffect, useMemo, useState } from "react";
import { MapView } from "./components/MapView";
import { createCourse, downloadUrl, getLayers, getStatus } from "./api";
import type { StatusOut } from "./types";

const EXAMPLES = [
  "Pebble Beach Golf Links",
  "St Andrews Old Course",
  "Centro Nacional de Formação de Golfe do Jamor",
];

const FORMATS = ["geojson", "kml", "gpx", "wkt", "dxf", "gpkg", "shp", "ros_grid"] as const;

export default function App() {
  const [name, setName] = useState("Pebble Beach Golf Links");
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusOut | null>(null);
  const [layers, setLayers] = useState<Record<string, { type: string; features: unknown[] }>>({});
  const [hidden, setHidden] = useState<Set<string>>(new Set(["nav_mesh", "no_go", "waypoint_graph", "coverage"]));

  useEffect(() => {
    if (!status || (status.status !== "queued" && status.status !== "running")) return;
    const t = window.setInterval(async () => {
      const next = await getStatus(status.job_id);
      setStatus(next);
    }, 1500);
    return () => window.clearInterval(t);
  }, [status?.job_id, status?.status]);

  useEffect(() => {
    if (status?.status !== "completed") return;
    getLayers(status.job_id).then((payload) => setLayers(payload.layers)).catch((err: Error) => setError(err.message));
  }, [status?.job_id, status?.status]);

  const groups = useMemo(() => {
    const grouped: Record<string, StatusOut["layers"]> = {};
    for (const layer of status?.layers ?? []) {
      (grouped[layer.group] ??= []).push(layer);
    }
    return grouped;
  }, [status]);

  async function submit(courseName?: string) {
    setError(null);
    setBusy(true);
    setLayers({});
    try {
      const body: { name?: string; lat?: number; lon?: number } = {};
      const chosen = (courseName ?? name).trim();
      if (lat && lon) {
        body.lat = Number(lat);
        body.lon = Number(lon);
        if (chosen) body.name = chosen;
      } else {
        body.name = chosen;
      }
      const created = await createCourse(body);
      const next = await getStatus(created.job_id);
      setStatus(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  function toggle(id: string) {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function hideAll() {
    setHidden(new Set((status?.layers ?? []).map((layer) => layer.layer_id)));
  }

  function showAll() {
    setHidden(new Set());
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <h1>Eagle Eye</h1>
          <p>Name a course. The pipeline discovers it, fuses OSM priors with imagery, and exports a GIS / robotics dataset.</p>
        </div>
        <div className="search">
          <label htmlFor="course">Course name</label>
          <input id="course" value={name} onChange={(e) => setName(e.target.value)} placeholder="Pebble Beach Golf Links" />
          <div className="row coord">
            <input value={lat} onChange={(e) => setLat(e.target.value)} placeholder="Latitude" />
            <input value={lon} onChange={(e) => setLon(e.target.value)} placeholder="Longitude" />
          </div>
          <button className="primary" disabled={busy} onClick={() => void submit()}>
            {busy ? "Queuing…" : "Vectorize course"}
          </button>
        </div>
        <div className="examples">
          {EXAMPLES.map((example) => (
            <button key={example} onClick={() => { setName(example); void submit(example); }}>
              {example}
            </button>
          ))}
        </div>
        <div className="scroll">
          {error && <div className="card error">{error}</div>}
          {status && (
            <div className="card">
              <h2>Pipeline</h2>
              <div className="status-row">
                <span>{status.stage}</span>
                <span className={`pill ${status.status === "completed" ? "ok" : status.status === "failed" ? "bad" : "run"}`}>
                  {status.status}
                </span>
              </div>
              <div className="bar"><span style={{ width: `${Math.round(status.progress * 100)}%` }} /></div>
              {status.message && <div className="meta">{status.message}</div>}
              {status.error && <div className="error">{status.error.split("\n")[0]}</div>}
              {!!status.quality && Object.keys(status.quality).length > 0 && (
                <div className="meta">
                  {status.quality.holes_detected != null && <>Holes {String(status.quality.holes_detected)} · </>}
                  {status.quality.polygon_validity != null && <>Validity {(Number(status.quality.polygon_validity) * 100).toFixed(1)}%</>}
                  {status.quality.holes_need_review ? " · hole numbering needs review" : ""}
                </div>
              )}
            </div>
          )}
          {status?.layers?.length ? (
            <div className="card">
              <div className="card-head">
                <h2>Layers</h2>
                <div className="layer-bulk">
                  <button type="button" onClick={hideAll}>Hide all</button>
                  <button type="button" onClick={showAll}>Show all</button>
                </div>
              </div>
              {Object.entries(groups).map(([group, items]) => (
                <div key={group}>
                  <div className="meta" style={{ margin: "8px 0 4px", textTransform: "uppercase", letterSpacing: "0.06em" }}>{group}</div>
                  {items.map((layer) => (
                    <div className="layer" key={layer.layer_id}>
                      <span className="swatch" style={{ background: layer.color }} />
                      <span>{layer.title}</span>
                      <span className="count">{layer.feature_count}</span>
                      <button onClick={() => toggle(layer.layer_id)}>{hidden.has(layer.layer_id) ? "Show" : "Hide"}</button>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          ) : null}
          {status?.status === "completed" && (
            <div className="card">
              <h2>Export</h2>
              <div className="exports">
                {(status.exports.length ? status.exports : FORMATS).map((fmt) => (
                  <a key={fmt} href={downloadUrl(status.job_id, fmt)}>
                    {fmt}
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      </aside>
      <MapView status={status} layers={layers} hidden={hidden} />
    </div>
  );
}
