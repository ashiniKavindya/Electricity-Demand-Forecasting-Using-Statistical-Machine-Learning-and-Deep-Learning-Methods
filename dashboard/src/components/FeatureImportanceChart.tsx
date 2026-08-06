import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { FeatureImportance } from "../types";

export function FeatureImportanceChart({ data }: { data: FeatureImportance[] }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} layout="vertical" margin={{ left: 24 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" tick={{ fontSize: 11 }} />
        <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={140} />
        <Tooltip />
        <Bar dataKey="importance" fill="#2563eb" />
      </BarChart>
    </ResponsiveContainer>
  );
}
