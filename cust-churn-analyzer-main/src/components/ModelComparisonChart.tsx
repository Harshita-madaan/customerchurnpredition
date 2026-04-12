import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { modelMetrics } from "@/lib/churnData";

const metrics = ["accuracy", "precision", "recall", "f1"] as const;
const data = metrics.map((m) => ({
  metric: m === "f1" ? "F1 Score" : m.charAt(0).toUpperCase() + m.slice(1),
  "Logistic Regression": modelMetrics["Logistic Regression"][m],
  "KNN (K=16)": modelMetrics["KNN (K=16)"][m],
  "Decision Tree": modelMetrics["Decision Tree"][m],
}));

export default function ModelComparisonChart() {
  return (
    <div className="glass-card p-6">
      <h3 className="text-lg font-heading font-semibold mb-4">Model Comparison: All Metrics</h3>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={data} barGap={4}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis dataKey="metric" stroke="hsl(var(--muted-foreground))" fontSize={13} />
          <YAxis domain={[70, 85]} stroke="hsl(var(--muted-foreground))" fontSize={12} />
          <Tooltip
            contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8 }}
            labelStyle={{ color: "hsl(var(--foreground))" }}
          />
          <Legend />
          <Bar dataKey="Logistic Regression" fill="hsl(var(--chart-blue))" radius={[4, 4, 0, 0]} />
          <Bar dataKey="KNN (K=16)" fill="hsl(var(--chart-coral))" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Decision Tree" fill="hsl(var(--chart-green))" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
