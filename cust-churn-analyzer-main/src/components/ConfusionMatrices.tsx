import { modelMetrics } from "@/lib/churnData";

function ConfusionMatrix({ name, matrix, color }: { name: string; matrix: number[][]; color: string }) {
  const max = Math.max(...matrix.flat());
  return (
    <div className="flex flex-col items-center gap-2">
      <h4 className="font-heading font-semibold text-sm">{name}</h4>
      <div className="grid grid-cols-2 gap-1">
        {matrix.flat().map((val, i) => {
          const opacity = val / max;
          return (
            <div
              key={i}
              className="w-20 h-16 flex flex-col items-center justify-center rounded-lg text-sm font-semibold transition-all"
              style={{
                backgroundColor: `color-mix(in srgb, ${color} ${Math.round(opacity * 80 + 10)}%, hsl(var(--card)))`,
                color: opacity > 0.5 ? "white" : "hsl(var(--foreground))",
              }}
            >
              <span className="text-lg font-bold">{val}</span>
            </div>
          );
        })}
      </div>
      <div className="flex gap-8 text-xs text-muted-foreground mt-1">
        <span>No Churn</span>
        <span>Churn</span>
      </div>
    </div>
  );
}

export default function ConfusionMatrices() {
  return (
    <div className="glass-card p-6">
      <h3 className="text-lg font-heading font-semibold mb-4">Confusion Matrices</h3>
      <div className="flex flex-wrap justify-around gap-6">
        {Object.entries(modelMetrics).map(([name, m]) => (
          <ConfusionMatrix key={name} name={name} matrix={m.confusionMatrix} color={m.color} />
        ))}
      </div>
      <div className="flex justify-center gap-2 mt-3 text-xs text-muted-foreground">
        <span>Rows: Actual</span>
        <span>•</span>
        <span>Columns: Predicted</span>
      </div>
    </div>
  );
}
