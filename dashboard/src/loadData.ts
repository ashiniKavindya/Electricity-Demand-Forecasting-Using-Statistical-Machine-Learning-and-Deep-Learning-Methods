import type {
  AccuracySummary,
  CollectorHealthRun,
  DemandPoint,
  Fetched,
  Forecast,
  HealthStatus,
  InferenceHealthRun,
  OfflineMetric,
} from "./types";

const BACKEND_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const UNREACHABLE = "Can't reach the local backend. Start it with: uvicorn backend.main:app --reload --port 8000";

async function fetchJSON<T>(path: string): Promise<Fetched<T>> {
  try {
    const res = await fetch(`${BACKEND_URL}${path}`);
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      return { ok: false, reason: body?.detail ?? `Request failed (${res.status})` };
    }
    return { ok: true, data: (await res.json()) as T };
  } catch {
    return { ok: false, reason: UNREACHABLE };
  }
}

export function loadHealth(): Promise<Fetched<HealthStatus>> {
  return fetchJSON<HealthStatus>("/health");
}

export function loadDemandChart(hours = 168): Promise<Fetched<DemandPoint[]>> {
  return fetchJSON<DemandPoint[]>(`/demand/chart?hours=${hours}`);
}

export function loadLatestForecast(): Promise<Fetched<Forecast>> {
  return fetchJSON<Forecast>("/forecast/latest");
}

export function loadAccuracy(limit = 200): Promise<Fetched<AccuracySummary>> {
  return fetchJSON<AccuracySummary>(`/monitoring/accuracy?limit=${limit}`);
}

export function loadCollectorHealth(limit = 5): Promise<Fetched<CollectorHealthRun[]>> {
  return fetchJSON<CollectorHealthRun[]>(`/monitoring/collector?limit=${limit}`);
}

export function loadInferenceHealth(limit = 5): Promise<Fetched<InferenceHealthRun[]>> {
  return fetchJSON<InferenceHealthRun[]>(`/monitoring/inference?limit=${limit}`);
}

export function loadOfflineMetrics(): Promise<Fetched<OfflineMetric[]>> {
  return fetchJSON<OfflineMetric[]>("/metrics/offline");
}
