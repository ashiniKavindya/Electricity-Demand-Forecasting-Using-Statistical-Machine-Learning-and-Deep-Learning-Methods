import type { CollectorHealthRun, InferenceHealthRun } from "../types";

function timeAgo(iso: string): string {
  const minutes = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000));
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  return `${Math.round(minutes / 60)}h ago`;
}

export function HealthPanel({
  collectorRuns,
  inferenceRuns,
}: {
  collectorRuns: CollectorHealthRun[];
  inferenceRuns: InferenceHealthRun[];
}) {
  const lastCollector = collectorRuns[0];
  const lastInference = inferenceRuns[0];

  return (
    <div className="health-grid">
      <div className={`health-card ${lastCollector && !lastCollector.error ? "ok" : "error"}`}>
        <h3>Collector</h3>
        {lastCollector ? (
          <>
            <p>
              last poll {timeAgo(lastCollector.polled_at)} - checked {lastCollector.files_checked} file(s), inserted{" "}
              {lastCollector.rows_inserted} row(s)
            </p>
            {lastCollector.error && <p className="health-error">{lastCollector.error}</p>}
          </>
        ) : (
          <p>No collector runs recorded yet.</p>
        )}
      </div>

      <div className={`health-card ${lastInference && !lastInference.error ? "ok" : "error"}`}>
        <h3>Inference</h3>
        {lastInference ? (
          <>
            <p>
              last run {timeAgo(lastInference.run_at)} -{" "}
              {lastInference.predicted ? "produced a forecast" : "did not produce a forecast"}
            </p>
            {lastInference.error && <p className="health-error">{lastInference.error}</p>}
          </>
        ) : (
          <p>No inference runs recorded yet.</p>
        )}
      </div>
    </div>
  );
}
