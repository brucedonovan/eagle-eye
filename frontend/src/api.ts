import type { StatusOut } from "./types";

const BASE = "/api";

export async function createCourse(body: {
  name?: string;
  lat?: number;
  lon?: number;
}): Promise<{ job_id: string }> {
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
