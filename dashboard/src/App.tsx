import { useEffect, useState } from "react";
import { DemandChart } from "./components/DemandChart";
import { FeatureImportanceChart } from "./components/FeatureImportanceChart";
import { LivePredictionPanel } from "./components/LivePredictionPanel";
import { MetricsTable } from "./components/MetricsTable";
import { ModelSelector } from "./components/ModelSelector";
import { PeakDemandBanner } from "./components/PeakDemandBanner";
import "./App.css";
import { loadDemandSeries, loadFeatureImportance, loadMetrics } from "./loadData";
import { mockDemandSeries, mockFeatureImportance, mockMetrics } from "./mockData";
import type { DemandPoint, FeatureImportance, ModelMetrics, ModelName } from "./types";

function App() {
  const [model, setModel] = useState<ModelName>("XGBoost");
  const [demandSeries, setDemandSeries] = useState<DemandPoint[]>(mockDemandSeries);
  const [metrics, setMetrics] = useState<ModelMetrics[]>(mockMetrics);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>(mockFeatureImportance);

  useEffect(() => {
    loadDemandSeries().then(setDemandSeries);
    loadMetrics().then(setMetrics);
    loadFeatureImportance().then(setFeatureImportance);
  }, []);

  return (
    <div className="dashboard">
      <header>
        <h1>Electricity Demand Forecast Dashboard</h1>
        <ModelSelector value={model} onChange={setModel} />
      </header>

      <PeakDemandBanner data={demandSeries} />

      <section>
        <h2>Actual vs Predicted Demand</h2>
        <DemandChart data={demandSeries} />
      </section>

      <section>
        <h2>Model Comparison</h2>
        <MetricsTable metrics={metrics} />
      </section>

      <section>
        <h2>Feature Importance ({model})</h2>
        <FeatureImportanceChart data={featureImportance} />
      </section>

      <section>
        <h2>Live Forecast (local backend)</h2>
        <LivePredictionPanel />
      </section>
    </div>
  );
}

export default App;
