import { useEffect, useState } from "react";
import { AccuracyPanel } from "./components/AccuracyPanel";
import { DemandChart } from "./components/DemandChart";
import { ForecastCard } from "./components/ForecastCard";
import { HealthPanel } from "./components/HealthPanel";
import { MetricsTable } from "./components/MetricsTable";
import { PeakDemandBanner } from "./components/PeakDemandBanner";
import "./App.css";
import {
  loadAccuracy,
  loadCollectorHealth,
  loadDemandChart,
  loadInferenceHealth,
  loadLatestForecast,
  loadOfflineMetrics,
} from "./loadData";
import type {
  AccuracySummary,
  CollectorHealthRun,
  DemandPoint,
  Fetched,
  Forecast,
  InferenceHealthRun,
  OfflineMetric,
} from "./types";

function Section<T>({
  title,
  state,
  render,
}: {
  title: string;
  state: Fetched<T> | null;
  render: (data: T) => React.ReactNode;
}) {
  return (
    <section>
      <h2>{title}</h2>
      {state === null && <p className="section-loading">Loading...</p>}
      {state !== null && !state.ok && <p className="section-error">{state.reason}</p>}
      {state !== null && state.ok && render(state.data)}
    </section>
  );
}

function App() {
  const [demandChart, setDemandChart] = useState<Fetched<DemandPoint[]> | null>(null);
  const [forecast, setForecast] = useState<Fetched<Forecast> | null>(null);
  const [accuracy, setAccuracy] = useState<Fetched<AccuracySummary> | null>(null);
  const [collectorHealth, setCollectorHealth] = useState<Fetched<CollectorHealthRun[]> | null>(null);
  const [inferenceHealth, setInferenceHealth] = useState<Fetched<InferenceHealthRun[]> | null>(null);
  const [offlineMetrics, setOfflineMetrics] = useState<Fetched<OfflineMetric[]> | null>(null);

  useEffect(() => {
    loadDemandChart().then(setDemandChart);
    loadLatestForecast().then(setForecast);
    loadAccuracy().then(setAccuracy);
    loadCollectorHealth().then(setCollectorHealth);
    loadInferenceHealth().then(setInferenceHealth);
    loadOfflineMetrics().then(setOfflineMetrics);
  }, []);

  return (
    <div className="dashboard">
      <header>
        <h1>AEMO NEM Live Demand Dashboard</h1>
      </header>

      <Section title="Live forecast" state={forecast} render={(f) => <ForecastCard forecast={f} />} />

      <Section
        title="Actual vs predicted demand"
        state={demandChart}
        render={(data) => (
          <>
            <PeakDemandBanner data={data} />
            <DemandChart data={data} />
          </>
        )}
      />

      <Section
        title="Live forecast accuracy"
        state={accuracy}
        render={(data) => <AccuracyPanel accuracy={data} />}
      />

      <section>
        <h2>Collector &amp; inference health</h2>
        {(collectorHealth === null || inferenceHealth === null) && <p className="section-loading">Loading...</p>}
        {collectorHealth && !collectorHealth.ok && <p className="section-error">{collectorHealth.reason}</p>}
        {inferenceHealth && !inferenceHealth.ok && <p className="section-error">{inferenceHealth.reason}</p>}
        {collectorHealth?.ok && inferenceHealth?.ok && (
          <HealthPanel collectorRuns={collectorHealth.data} inferenceRuns={inferenceHealth.data} />
        )}
      </section>

      <Section
        title="Stage 1 offline evaluation (historical backtest)"
        state={offlineMetrics}
        render={(data) => <MetricsTable metrics={data} />}
      />
    </div>
  );
}

export default App;
