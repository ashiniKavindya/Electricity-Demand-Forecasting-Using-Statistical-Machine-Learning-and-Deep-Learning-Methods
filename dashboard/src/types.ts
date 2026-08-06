export type ModelName = "SeasonalNaive" | "SARIMA" | "XGBoost" | "LSTM";

export interface DemandPoint {
  timestamp: string;
  actual: number;
  predicted: number | null;
}

export interface ModelMetrics {
  model: ModelName;
  mae: number;
  rmse: number;
  mape: number;
  trainingTimeSeconds: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export type BackendModelKey = "xgboost" | "sarima" | "lstm";

export interface LivePredictionPoint {
  timestamp: string;
  predicted: number;
}

export type LiveForecastResult =
  | { ok: true; predictions: LivePredictionPoint[] }
  | { ok: false; reason: string };
