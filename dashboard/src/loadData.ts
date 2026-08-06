import type {
  BackendModelKey,
  DemandPoint,
  FeatureImportance,
  LiveForecastResult,
  ModelMetrics,
} from "./types";
import { mockDemandSeries, mockFeatureImportance, mockMetrics } from "./mockData";

const BACKEND_URL = "http://localhost:8000";

async function fetchJSON<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(path);
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

// Looks for exported model results in public/data/ (see
// src/evaluation/export_dashboard_data.py). Falls back to mock data
// when a file hasn't been exported yet, so the dashboard always renders.
export async function loadDemandSeries(): Promise<DemandPoint[]> {
  return (await fetchJSON<DemandPoint[]>("/data/demand.json")) ?? mockDemandSeries;
}

export async function loadMetrics(): Promise<ModelMetrics[]> {
  return (await fetchJSON<ModelMetrics[]>("/data/metrics.json")) ?? mockMetrics;
}

export async function loadFeatureImportance(): Promise<FeatureImportance[]> {
  return (await fetchJSON<FeatureImportance[]>("/data/feature_importance.json")) ?? mockFeatureImportance;
}

// Calls the local-only FastAPI backend (backend/main.py) for a live
// prediction. Returns a typed failure reason instead of throwing, since
// "backend not running" and "model not trained yet" are expected states
// during development, not exceptional ones.
export async function fetchLivePrediction(model: BackendModelKey, horizon = 24): Promise<LiveForecastResult> {
  try {
    const res = await fetch(`${BACKEND_URL}/predict?model=${model}&horizon=${horizon}`);
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      return { ok: false, reason: body?.detail ?? `Request failed (${res.status})` };
    }
    const data = await res.json();
    return { ok: true, predictions: data.predictions };
  } catch {
    return {
      ok: false,
      reason: "Can't reach the local backend. Start it with: uvicorn backend.main:app --reload --port 8000",
    };
  }
}
