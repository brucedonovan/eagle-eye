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
  source: "golfapi" | "nominatim" | string;
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
  golfapi_configured: boolean;
  api_requests_left?: string | null;
  warning?: string | null;
  courses: CourseSearchItem[];
};

export type LayerCatalogItem = {
  id: string;
  title: string;
  geometry: string;
  group: string;
  color: string;
  mvp: boolean;
};
