import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import { knnAccuracies } from "@/lib/churnData";

const data = knnAccuracies.map((acc, i) => ({ k: i + 1, accuracy: acc }));

export default function KnnChart() {
  return (
    <div className="glass-card p-6">
      <h3 className="text-lg font-heading font-semibold mb-4">KNN: Accuracy vs K Value</h3>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis dataKey="k" stroke="hsl(var(--muted-foreground))" fontSize={12} />
          <YAxis domain={[71, 79]} stroke="hsl(var(--muted-foreground))" fontSize={12} />
          <Tooltip
            contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8 }}
          />
          <ReferenceLine x={16} stroke="hsl(var(--destructive))" strokeDasharray="5 5" label={{ value: "Best K=16", fill: "hsl(var(--destructive))", fontSize: 12 }} />
          <Line type="monotone" dataKey="accuracy" stroke="hsl(var(--chart-blue))" strokeWidth={2} dot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
