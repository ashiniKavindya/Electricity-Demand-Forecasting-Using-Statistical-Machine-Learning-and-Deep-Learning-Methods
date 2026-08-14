import type { Forecast } from "../types";

function formatHour(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function ForecastCard({ forecast }: { forecast: Forecast }) {
  return (
    <div className="forecast-card">
      <div className="forecast-value">{Math.round(forecast.predicted_demand).toLocaleString()} MW</div>
      <div className="forecast-meta">
        forecast for {formatHour(forecast.target_timestamp)} · based on the actual demand at{" "}
        {formatHour(forecast.based_on_timestamp)} · model {forecast.model_version}
      </div>
    </div>
  );
}
