import { modelMetrics } from "@/lib/churnData";
import { motion } from "framer-motion";

const descriptions: Record<string, string> = {
  "Logistic Regression": "Draws a linear decision boundary to separate churners from non-churners. Best overall accuracy.",
  "KNN (K=16)": "Checks the 16 most similar past customers and predicts based on majority vote.",
  "Decision Tree": "Asks a series of Yes/No questions (max depth 5) like a flowchart to predict churn.",
};

const icons: Record<string, string> = {
  "Logistic Regression": "📐",
  "KNN (K=16)": "👥",
  "Decision Tree": "🌳",
};

export default function ModelCards() {
  const best = Object.entries(modelMetrics).reduce((a, b) => (a[1].accuracy > b[1].accuracy ? a : b));

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {Object.entries(modelMetrics).map(([name, m], i) => {
        const isBest = name === best[0];
        return (
          <motion.div
            key={name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className={`glass-card p-5 relative ${isBest ? "glow-primary ring-1 ring-primary/30" : ""}`}
          >
            {isBest && (
              <span className="absolute top-3 right-3 text-xs font-semibold bg-primary/20 text-primary px-2 py-0.5 rounded-full">
                🏆 Best
              </span>
            )}
            <div className="text-3xl mb-2">{icons[name]}</div>
            <h3 className="font-heading font-semibold text-base mb-1">{name}</h3>
            <p className="text-xs text-muted-foreground mb-4 leading-relaxed">{descriptions[name]}</p>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-muted-foreground text-xs">Accuracy</span>
                <div className="font-semibold">{m.accuracy}%</div>
              </div>
              <div>
                <span className="text-muted-foreground text-xs">Precision</span>
                <div className="font-semibold">{m.precision}%</div>
              </div>
              <div>
                <span className="text-muted-foreground text-xs">Recall</span>
                <div className="font-semibold">{m.recall}%</div>
              </div>
              <div>
                <span className="text-muted-foreground text-xs">F1 Score</span>
                <div className="font-semibold">{m.f1}%</div>
              </div>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
