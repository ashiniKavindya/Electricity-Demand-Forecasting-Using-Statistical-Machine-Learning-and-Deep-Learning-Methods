import type { DemandPoint } from "../types";

export function PeakDemandBanner({
  data,
  percentileThreshold = 0.95,
}: {
  data: DemandPoint[];
  percentileThreshold?: number;
}) {
  const actuals = data.map((d) => d.actual).filter((v): v is number => v !== null);
  if (actuals.length === 0) return null;

  const sorted = [...actuals].sort((a, b) => a - b);
  const threshold = sorted[Math.floor(sorted.length * percentileThreshold)];
  const peakCount = actuals.filter((v) => v >= threshold).length;

  return (
    <div className="peak-banner">
      {peakCount} peak-demand hour(s) above the {Math.round(percentileThreshold * 100)}th percentile threshold (
      {Math.round(threshold).toLocaleString()} MW)
    </div>
  );
}
