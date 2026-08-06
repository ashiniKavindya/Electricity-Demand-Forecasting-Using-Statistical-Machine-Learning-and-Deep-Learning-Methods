import type { DemandPoint } from "../types";

export function PeakDemandBanner({
  data,
  percentileThreshold = 0.95,
}: {
  data: DemandPoint[];
  percentileThreshold?: number;
}) {
  const sorted = [...data].map((d) => d.actual).sort((a, b) => a - b);
  const threshold = sorted[Math.floor(sorted.length * percentileThreshold)];
  const peakPoints = data.filter((d) => d.actual >= threshold);

  if (peakPoints.length === 0) return null;

  return (
    <div className="peak-banner">
      {peakPoints.length} peak-demand hour(s) above the {Math.round(percentileThreshold * 100)}th percentile
      threshold ({threshold.toLocaleString()} kW)
    </div>
  );
}
