import type { DemandPoint, FeatureImportance, ModelMetrics } from "./types";

// Placeholder data until the dashboard is wired up to real model output
// (exported predictions from the notebooks / a backend API).
export const mockDemandSeries: DemandPoint[] = Array.from({ length: 48 }, (_, i) => {
  const hour = i % 24;
  const base = 20000 + 6000 * Math.sin((hour / 24) * Math.PI * 2 - Math.PI / 2);
  const actual = Math.round(base + (Math.random() - 0.5) * 800);
  return {
    timestamp: `2018-01-${String(1 + Math.floor(i / 24)).padStart(2, "0")}T${String(hour).padStart(2, "0")}:00`,
    actual,
    predicted: i < 24 ? Math.round(actual + (Math.random() - 0.5) * 1200) : null,
  };
});

export const mockMetrics: ModelMetrics[] = [
  { model: "SeasonalNaive", mae: 1450, rmse: 1890, mape: 6.8, trainingTimeSeconds: 0 },
  { model: "SARIMA", mae: 1120, rmse: 1480, mape: 5.1, trainingTimeSeconds: 240 },
  { model: "XGBoost", mae: 890, rmse: 1210, mape: 4.0, trainingTimeSeconds: 45 },
  { model: "LSTM", mae: 860, rmse: 1150, mape: 3.8, trainingTimeSeconds: 900 },
];

export const mockFeatureImportance: FeatureImportance[] = [
  { feature: "demand_lag_24", importance: 0.34 },
  { feature: "demand_lag_1", importance: 0.22 },
  { feature: "hour_sin", importance: 0.14 },
  { feature: "temperature", importance: 0.11 },
  { feature: "demand_roll_mean_24", importance: 0.09 },
  { feature: "is_weekend", importance: 0.06 },
];
