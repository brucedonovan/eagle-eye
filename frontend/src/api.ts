const BASE = "/api";

export type CourseOut = {
  id: string;
  name: string;
  display_name?: string | null;
  country?: string | null;
  address?: string | null;
  lat?: number | null;
  lon?: number | null;
  bbox?: number[] | null;
  osm_id?: string | null;
};

export type LayerOut = {
  layer_id: string;
  title: string;
  geometry: string;
  group: string;
  color: string;
  feature_count: number;
  source: string;
};

export type StatusOut = {
  job_id: string;
  status: string;
  stage: string;
  progress: number;
  message?: string | null;
  error?: string | null;
  course?: CourseOut | null;
  layers: LayerOut[];
  quality: Record<string, unknown>;
  exports: string[];
};

export type CourseSearchItem = {
  source: string;
  club_id?: string | null;
  club_name: string;
  course_id?: string | null;
  course_name: string;
  display_name: string;
  city?: string | null;
  state?: string | null;
  country?: string | null;
  address?: string | null;
  lat?: number | null;
  lon?: number | null;
  num_holes?: number | null;
  has_gps?: boolean;
  distance_km?: number | null;
  timestamp_updated?: number | null;
};

export type CourseSearchOut = {
  query: string;
  source: string;
  cached: boolean;
  catalog_configured: boolean;
  catalog_provider?: string | null;
  provider_title?: string | null;
  api_requests_left?: string | null;
  courses: CourseSearchItem[];
};

export type CreateCourseBody = {
  name?: string;
  lat?: number;
  lon?: number;
  catalog_provider?: string;
  catalog_course_id?: string;
  catalog_club_id?: string;
  catalog_timestamp_updated?: number;
};

export async function searchCourses(body: {
  q: string;
  lat?: number;
  lon?: number;
}): Promise<CourseSearchOut> {
  const params = new URLSearchParams();
  if (body.q.trim()) params.set("q", body.q.trim());
  if (body.lat != null) params.set("lat", String(body.lat));
  if (body.lon != null) params.set("lon", String(body.lon));
  const res = await fetch(`${BASE}/search?${params.toString()}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export async function createCourse(body: CreateCourseBody): Promise<{ job_id: string }> {
  const res = await fetch(`${BASE}/course`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ include_imagery: true, include_navigation: true, ...body }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export async function getStatus(jobId: string): Promise<StatusOut> {
  const res = await fetch(`${BASE}/status/${jobId}`);
  if (!res.ok) throw new Error("Status request failed");
  return res.json();
}

export async function getLayers(jobId: string): Promise<{ layers: Record<string, { type: string; features: unknown[] }> }> {
  const res = await fetch(`${BASE}/layers/${jobId}`);
  if (!res.ok) throw new Error("Layers request failed");
  return res.json();
}

export function downloadUrl(jobId: string, format: string): string {
  return `${BASE}/download/${jobId}?format=${format}`;
}
