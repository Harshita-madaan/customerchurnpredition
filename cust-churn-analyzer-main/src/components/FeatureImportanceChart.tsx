import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { featureImportance } from "@/lib/churnData";

const data = [...featureImportance].reverse();

export default function FeatureImportanceChart() {
  return (
    <div className="glass-card p-6">
      <h3 className="text-lg font-heading font-semibold mb-4">Decision Tree: Top 8 Important Features</h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} layout="vertical" margin={{ left: 40 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis type="number" stroke="hsl(var(--muted-foreground))" fontSize={12} unit="%" />
          <YAxis type="category" dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={11} width={160} />
          <Tooltip
            contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8 }}
            formatter={(v: number) => `${v}%`}
          />
          <Bar dataKey="importance" fill="hsl(var(--chart-green))" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
