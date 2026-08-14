import type { OfflineMetric } from "../types";

export function MetricsTable({ metrics }: { metrics: OfflineMetric[] }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>MAE (MW)</th>
          <th>RMSE</th>
          <th>MAPE (%)</th>
          <th>Test hours</th>
        </tr>
      </thead>
      <tbody>
        {metrics.map((row) => (
          <tr key={row.model}>
            <td>{row.model}</td>
            <td>{row.mae.toFixed(0)}</td>
            <td>{row.rmse.toFixed(0)}</td>
            <td>{row.mape.toFixed(1)}</td>
            <td>{row.n}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
