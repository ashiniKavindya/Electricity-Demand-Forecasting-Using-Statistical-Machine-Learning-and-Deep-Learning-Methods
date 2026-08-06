import type { ModelMetrics } from "../types";

export function MetricsTable({ metrics }: { metrics: ModelMetrics[] }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>MAE</th>
          <th>RMSE</th>
          <th>MAPE (%)</th>
          <th>Training time (s)</th>
        </tr>
      </thead>
      <tbody>
        {metrics.map((row) => (
          <tr key={row.model}>
            <td>{row.model}</td>
            <td>{row.mae.toFixed(0)}</td>
            <td>{row.rmse.toFixed(0)}</td>
            <td>{row.mape.toFixed(1)}</td>
            <td>{row.trainingTimeSeconds}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
