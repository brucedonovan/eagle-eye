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

export type LayerCatalogItem = {
  id: string;
  title: string;
  geometry: string;
  group: string;
  color: string;
  mvp: boolean;
};
