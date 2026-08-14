import type { AccuracySummary } from "../types";

export function AccuracyPanel({ accuracy }: { accuracy: AccuracySummary }) {
  if (accuracy.n === 0) {
    return (
      <p className="accuracy-empty">
        No predictions have matured into an actual observation yet - the collector needs to catch up to a
        prediction's target hour before it can be scored. Check back after the collector's next poll.
      </p>
    );
  }

  return (
    <div className="accuracy-stats">
      <div>
        <span className="stat-value">{accuracy.mae?.toFixed(0)}</span>
        <span className="stat-label">MAE (MW)</span>
      </div>
      <div>
        <span className="stat-value">{accuracy.rmse?.toFixed(0)}</span>
        <span className="stat-label">RMSE</span>
      </div>
      <div>
        <span className="stat-value">{accuracy.mape?.toFixed(1)}%</span>
        <span className="stat-label">MAPE</span>
      </div>
      <div>
        <span className="stat-value">{accuracy.n}</span>
        <span className="stat-label">predictions scored</span>
      </div>
    </div>
  );
}
