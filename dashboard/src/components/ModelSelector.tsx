import type { ModelName } from "../types";

const MODELS: ModelName[] = ["SeasonalNaive", "SARIMA", "XGBoost", "LSTM"];

export function ModelSelector({
  value,
  onChange,
}: {
  value: ModelName;
  onChange: (model: ModelName) => void;
}) {
  return (
    <label>
      Model
      <select value={value} onChange={(e) => onChange(e.target.value as ModelName)}>
        {MODELS.map((model) => (
          <option key={model} value={model}>
            {model}
          </option>
        ))}
      </select>
    </label>
  );
}
