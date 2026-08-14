export interface DemandPoint {
  timestamp: string;
  actual: number | null;
  predicted: number | null;
}

export interface OfflineMetric {
  model: string;
  mae: number;
  rmse: number;
  mape: number;
  smape: number;
  n: number;
}

export interface Forecast {
  target_timestamp: string;
  based_on_timestamp: string;
  predicted_demand: number;
  model_version: string;
  predicted_at: string;
}

export interface CollectorHealthRun {
  polled_at: string;
  files_checked: number;
  rows_inserted: number;
  error: string | null;
}

export interface InferenceHealthRun {
  run_at: string;
  predicted: boolean;
  error: string | null;
}

export interface HealthStatus {
  db_found: boolean;
  model_trained: boolean;
  model_version?: string;
  latest_observation?: string | null;
  latest_prediction?: Forecast | null;
  last_collector_run?: CollectorHealthRun | null;
  last_inference_run?: InferenceHealthRun | null;
}

export interface AccuracyPoint {
  timestamp: string;
  actual: number;
  predicted: number;
}

export interface AccuracySummary {
  n: number;
  mae: number | null;
  rmse: number | null;
  mape: number | null;
  points: AccuracyPoint[];
}

// Every live-data fetch returns a typed failure reason instead of throwing, since
// "backend not running" and "not enough data yet" are expected states during
// development, not exceptional ones - each section of the dashboard renders its own
// explanation rather than the whole page crashing.
export type Fetched<T> = { ok: true; data: T } | { ok: false; reason: string };
