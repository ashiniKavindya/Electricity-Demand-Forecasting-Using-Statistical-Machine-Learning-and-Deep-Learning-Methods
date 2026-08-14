import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DemandPoint } from "../types";

export function DemandChart({ data }: { data: DemandPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="timestamp" tick={{ fontSize: 11 }} minTickGap={40} />
        <YAxis tick={{ fontSize: 11 }} width={60} />
        <Tooltip />
        <Line type="monotone" dataKey="actual" stroke="#2563eb" dot={false} name="Actual" connectNulls />
        <Line
          type="monotone"
          dataKey="predicted"
          stroke="#f97316"
          dot={{ r: 3 }}
          name="Predicted"
          connectNulls={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
