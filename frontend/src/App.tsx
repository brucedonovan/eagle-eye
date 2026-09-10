import { useEffect, useMemo, useState } from "react";
import { MapView } from "./components/MapView";
import { createCourse, downloadUrl, getLayers, getStatus, listCachedCourses, searchCourses, type CourseSearchItem, type CourseSearchOut, type CreateCourseBody, type StatusOut } from "./api";

const FORMATS = ["geojson", "kml", "gpx", "wkt", "dxf", "gpkg", "shp", "ros_grid"] as const;

type LayerOrigin = "api" | "osm" | "ai" | "generated" | "mixed";

const ORIGIN_LABEL: Record<LayerOrigin, string> = {
  api: "API",
  osm: "OSM",
  ai: "AI",
  generated: "generated",
  mixed: "mixed",
};

const GROUP_ORDER = ["site", "play", "hazard", "access", "practice", "built", "veg", "ai", "nav", "other"];

const GROUP_LABEL: Record<string, string> = {
  site: "Site",
  play: "Play",
  hazard: "Hazard",
  access: "Access",
  practice: "Practice",
  built: "Built",
  veg: "Vegetation",
  ai: "AI",
  nav: "Navigation",
  other: "Other",
};

function cachedCovers(cached: CourseSearchItem[], example: string): boolean {
  const needle = example.trim().toLowerCase();
  return cached.some((course) => {
    const names = [course.display_name, course.club_name, course.course_name]
      .filter(Boolean)
      .map((value) => value.toLowerCase());
    return names.some((name) => name.includes(needle) || needle.includes(name));
  });
}

function originBucket(raw?: string | null): LayerOrigin {
  const value = (raw || "").trim().toLowerCase();
  if (value === "api" || value === "golfapi" || value === "catalog" || value === "fake") return "api";
  if (value === "osm" || value === "openstreetmap" || value === "osm_refined") return "osm";
  if (value === "ai" || value === "ai_layer" || value === "ai_fusion") return "ai";
  if (value === "mixed" || value === "hybrid") return "mixed";
  return "generated";
}

function layerOrigin(
  layerId: string,
  geojson: Record<string, { features?: unknown[] }>,
  fallback?: string | null,
): LayerOrigin {
  const loaded = Object.prototype.hasOwnProperty.call(geojson, layerId);
  const kinds = new Set<LayerOrigin>();
  for (const feature of geojson[layerId]?.features ?? []) {
    const props = feature && typeof feature === "object" && "properties" in feature
      ? (feature as { properties?: { source?: string } }).properties
      : undefined;
    kinds.add(originBucket(props?.source));
  }
  if (kinds.size === 0) {
    if (!loaded) return originBucket(fallback);
    return "generated";
  }
  if (kinds.size === 1) return [...kinds][0];
  return "mixed";
}

export default function App() {
  const [name, setName] = useState("Pebble Beach Golf Links");
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [busy, setBusy] = useState(false);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<CourseSearchOut | null>(null);
  const [selected, setSelected] = useState<CourseSearchItem | null>(null);
  const [cachedCourses, setCachedCourses] = useState<CourseSearchItem[]>([]);
  const [status, setStatus] = useState<StatusOut | null>(null);
  const [layers, setLayers] = useState<Record<string, { type: string; features: unknown[] }>>({});
  const [hidden, setHidden] = useState<Set<string>>(new Set(["nav_mesh", "no_go", "waypoint_graph", "coverage"]));

  useEffect(() => {
    void listCachedCourses().then(setCachedCourses).catch(() => setCachedCourses([]));
  }, []);

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
    void listCachedCourses().then(setCachedCourses).catch(() => undefined);
  }, [status?.job_id, status?.status]);

  const groups = useMemo(() => {
    const grouped: Record<string, StatusOut["layers"]> = {};
    for (const layer of status?.layers ?? []) {
      (grouped[layer.group] ??= []).push(layer);
    }
    const ordered = GROUP_ORDER
      .filter((group) => grouped[group]?.length)
      .map((group) => [group, grouped[group]] as const);
    const extra = Object.entries(grouped).filter(([group]) => !GROUP_ORDER.includes(group));
    return [...ordered, ...extra];
  }, [status]);

  const clubGroups = useMemo(() => {
    const byClub = new Map<string, CourseSearchItem[]>();
    for (const course of results?.courses ?? []) {
      const key = course.club_id || course.club_name;
      const list = byClub.get(key) ?? [];
      list.push(course);
      byClub.set(key, list);
    }
    return [...byClub.entries()];
  }, [results]);

  async function search(query?: string) {
    setError(null);
    setSearching(true);
    setSelected(null);
    try {
      const chosen = (query ?? name).trim();
      const body: { q: string; lat?: number; lon?: number } = { q: chosen };
      if (lat && lon) {
        body.lat = Number(lat);
        body.lon = Number(lon);
      }
      const payload = await searchCourses(body);
      setResults(payload);
      if (payload.courses.length === 1) {
        setSelected(payload.courses[0]);
        setName(payload.courses[0].display_name);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
      setResults(null);
    } finally {
      setSearching(false);
    }
  }

  async function submit() {
    setError(null);
    setBusy(true);
    setLayers({});
    try {
      const body: CreateCourseBody = {};
      const chosenChip = selected?.course_id
        ? selected
        : cachedCourses.find((course) => cachedCovers([course], name.trim()));
      if (chosenChip?.course_id) {
        body.catalog_provider = chosenChip.source;
        body.catalog_course_id = chosenChip.course_id;
        if (chosenChip.club_id) body.catalog_club_id = chosenChip.club_id;
        body.name = chosenChip.display_name;
      } else {
        const chosen = name.trim();
        if (lat && lon) {
          body.lat = Number(lat);
          body.lon = Number(lon);
          if (chosen) body.name = chosen;
        } else {
          body.name = chosen;
        }
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

  function pick(course: CourseSearchItem) {
    setSelected(course);
    setName(course.display_name);
    if (course.lat != null) setLat(String(course.lat));
    if (course.lon != null) setLon(String(course.lon));
  }

  async function openVectorized(course: CourseSearchItem) {
    pick(course);
    if (!course.job_id) return;
    setError(null);
    try {
      const next = await getStatus(course.job_id);
      setStatus(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load vectorized course");
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
          <p>Chips are fully cached golfapi.io course payloads. Search results stay in the list until a course is fetched.</p>
        </div>
        <div className="search">
          <label htmlFor="course">Course or club name</label>
          <input
            id="course"
            value={name}
            onChange={(e) => { setName(e.target.value); setSelected(null); }}
            onKeyDown={(e) => { if (e.key === "Enter") void search(); }}
            placeholder="Pebble Beach Golf Links"
          />
          <div className="row coord">
            <input value={lat} onChange={(e) => setLat(e.target.value)} placeholder="Latitude" />
            <input value={lon} onChange={(e) => setLon(e.target.value)} placeholder="Longitude" />
          </div>
          <div className="row">
            <button className="ghost" disabled={searching || busy} onClick={() => void search()}>
              {searching ? "Searching…" : "Search"}
            </button>
            <button className="primary" disabled={busy} onClick={() => void submit()}>
              {busy ? "Queuing…" : selected ? "Vectorize selected" : "Vectorize course"}
            </button>
          </div>
        </div>
        {cachedCourses.length > 0 && (
          <div className="examples">
            {cachedCourses.map((course) => (
              <button
                key={course.job_id || course.course_id || course.display_name}
                type="button"
                className={
                  (selected?.job_id && selected.job_id === course.job_id)
                  || (selected?.course_id && selected.course_id === course.course_id)
                    ? "cached selected"
                    : "cached"
                }
                title={[
                  course.job_id ? "Vectorized" : "Cached catalog",
                  course.club_name,
                  course.city,
                  course.state,
                  course.country,
                ].filter(Boolean).join(" · ")}
                onClick={() => void openVectorized(course)}
              >
                {course.display_name}
              </button>
            ))}
          </div>
        )}
        <div className="scroll">
          {error && <div className="card error">{error}</div>}
          {results && (
            <div className="card">
              <h2>Search results</h2>
              <div className="meta">
                {results.cached ? "Cached · " : ""}
                {results.provider_title || results.source}
                {results.api_requests_left ? ` · ${results.api_requests_left} API calls left` : ""}
                {results.courses.length === 0 ? " · no matches" : ` · ${results.courses.length} courses`}
              </div>
              <div className="results">
                {clubGroups.map(([clubKey, courses]) => (
                  <div className="result-club" key={clubKey}>
                    <div className="result-club-name">{courses[0].club_name}</div>
                    <div className="result-club-meta">
                      {[courses[0].city, courses[0].state, courses[0].country].filter(Boolean).join(", ")}
                    </div>
                    {courses.map((course) => {
                      const active = selected?.course_id
                        ? selected.course_id === course.course_id
                        : selected?.display_name === course.display_name;
                      return (
                        <button
                          type="button"
                          key={course.course_id || course.display_name}
                          className={`result-course ${active ? "selected" : ""}`}
                          onClick={() => pick(course)}
                        >
                          <span>{course.course_name}</span>
                          <span className="result-tags">
                            {course.num_holes ? `${course.num_holes} holes` : ""}
                            {course.has_gps ? " GPS" : ""}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
          )}
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
              <div className="layer-origin-legend" title="Polygon and POI source">
                <span className="origin-api">API</span>
                <span className="origin-osm">OSM</span>
                <span className="origin-ai">AI</span>
                <span className="origin-generated">generated</span>
                <span className="origin-mixed">mixed</span>
              </div>
              {groups.map(([group, items]) => (
                <div key={group} className={group === "ai" ? "layer-group ai" : "layer-group"}>
                  <div className={`meta group-label ${group === "ai" ? "origin-ai" : ""}`}>
                    {GROUP_LABEL[group] || group}
                  </div>
                  {items.map((layer) => {
                    const origin = group === "ai" ? "ai" : layerOrigin(layer.layer_id, layers, layer.source);
                    return (
                      <div className="layer" key={layer.layer_id}>
                        <span className="swatch" style={{ background: layer.color }} />
                        <span className={`layer-title origin-${origin}`} title={ORIGIN_LABEL[origin]}>
                          {layer.title}
                        </span>
                        <span className="count">{layer.feature_count}</span>
                        <button onClick={() => toggle(layer.layer_id)}>{hidden.has(layer.layer_id) ? "Show" : "Hide"}</button>
                      </div>
                    );
                  })}
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
