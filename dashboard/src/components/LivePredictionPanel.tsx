import { useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchLivePrediction } from "../loadData";
import type { BackendModelKey, LivePredictionPoint } from "../types";

const BACKEND_MODELS: { key: BackendModelKey; label: string }[] = [
  { key: "xgboost", label: "XGBoost" },
  { key: "sarima", label: "SARIMA" },
  { key: "lstm", label: "LSTM" },
];

type Status = "idle" | "loading" | "error" | "success";

export function LivePredictionPanel() {
  const [model, setModel] = useState<BackendModelKey>("xgboost");
  const [horizon, setHorizon] = useState(24);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");
  const [predictions, setPredictions] = useState<LivePredictionPoint[]>([]);

  async function handleFetch() {
    setStatus("loading");
    const result = await fetchLivePrediction(model, horizon);
    if (result.ok) {
      setPredictions(result.predictions);
      setStatus("success");
    } else {
      setError(result.reason);
      setStatus("error");
    }
  }

  return (
    <div className="live-panel">
      <div className="live-controls">
        <label>
          Model
          <select value={model} onChange={(e) => setModel(e.target.value as BackendModelKey)}>
            {BACKEND_MODELS.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Horizon (hours)
          <input
            type="number"
            min={1}
            max={168}
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
          />
        </label>
        <button type="button" onClick={handleFetch} disabled={status === "loading"}>
          {status === "loading" ? "Forecasting..." : "Get live forecast"}
        </button>
      </div>

      {status === "error" && (
        <p className="live-error">
          {error} — this only works when the local backend is running (see backend/README.md).
        </p>
      )}

      {status === "success" && (
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={predictions}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" tick={{ fontSize: 11 }} minTickGap={40} />
            <YAxis tick={{ fontSize: 11 }} width={60} />
            <Tooltip />
            <Line type="monotone" dataKey="predicted" stroke="#16a34a" dot={false} name="Live forecast" />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
