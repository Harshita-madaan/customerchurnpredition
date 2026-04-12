import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CustomerInput, simulatePrediction } from "@/lib/churnData";

const toggleFields: { key: keyof CustomerInput; label: string }[] = [
  { key: "gender", label: "Male" },
  { key: "partner", label: "Partner" },
  { key: "dependents", label: "Dependents" },
  { key: "phoneService", label: "Phone Service" },
  { key: "multipleLines", label: "Multiple Lines" },
  { key: "onlineSecurity", label: "Online Security" },
  { key: "onlineBackup", label: "Online Backup" },
  { key: "deviceProtection", label: "Device Protection" },
  { key: "techSupport", label: "Tech Support" },
  { key: "streamingTV", label: "Streaming TV" },
  { key: "streamingMovies", label: "Streaming Movies" },
  { key: "paperlessBilling", label: "Paperless Billing" },
  { key: "internetFiber", label: "Fiber Optic Internet" },
  { key: "internetNo", label: "No Internet Service" },
  { key: "contractOneYear", label: "1 Year Contract" },
  { key: "contractTwoYear", label: "2 Year Contract" },
  { key: "paymentCredit", label: "Credit Card Payment" },
  { key: "paymentElectronic", label: "Electronic Check" },
  { key: "paymentMailed", label: "Mailed Check" },
];

const defaultInput: CustomerInput = {
  tenure: 12, monthlyCharges: 50, totalCharges: 600,
  gender: 1, partner: 0, dependents: 0, phoneService: 1,
  multipleLines: 0, onlineSecurity: 0, onlineBackup: 0,
  deviceProtection: 0, techSupport: 0, streamingTV: 0,
  streamingMovies: 0, paperlessBilling: 1, internetFiber: 1,
  internetNo: 0, contractOneYear: 0, contractTwoYear: 0,
  paymentCredit: 0, paymentElectronic: 1, paymentMailed: 0,
};

type PredictionResult = ReturnType<typeof simulatePrediction>;

export default function CustomerForm() {
  const [input, setInput] = useState<CustomerInput>(defaultInput);
  const [results, setResults] = useState<PredictionResult | null>(null);

  const handleToggle = (key: keyof CustomerInput) => {
    setInput((prev) => ({ ...prev, [key]: prev[key] ? 0 : 1 }));
  };

  const handlePredict = () => {
    setResults(simulatePrediction(input));
  };

  const colorMap: Record<string, string> = {
    "Logistic Regression": "chart-blue",
    "KNN (K=16)": "chart-coral",
    "Decision Tree": "chart-green",
  };

  return (
    <div className="space-y-6">
      <div className="glass-card p-6">
        <h3 className="text-lg font-heading font-semibold mb-4">🔎 Customer Details</h3>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
          {(["tenure", "monthlyCharges", "totalCharges"] as const).map((key) => (
            <div key={key}>
              <label className="text-sm text-muted-foreground mb-1 block capitalize">
                {key === "monthlyCharges" ? "Monthly Charges ($)" : key === "totalCharges" ? "Total Charges ($)" : "Tenure (months)"}
              </label>
              <input
                type="number"
                value={input[key]}
                onChange={(e) => setInput((p) => ({ ...p, [key]: parseFloat(e.target.value) || 0 }))}
                className="w-full rounded-lg bg-secondary px-3 py-2 text-foreground border border-border focus:outline-none focus:ring-2 focus:ring-primary/50 text-sm"
              />
            </div>
          ))}
        </div>

        <div className="flex flex-wrap gap-2 mb-6">
          {toggleFields.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => handleToggle(key)}
              className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all border ${
                input[key]
                  ? "bg-primary/20 border-primary/50 text-primary"
                  : "bg-secondary border-border text-muted-foreground hover:text-foreground"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        <button
          onClick={handlePredict}
          className="w-full py-3 rounded-xl font-heading font-semibold bg-gradient-to-r from-[hsl(var(--primary))] to-[hsl(var(--accent))] text-primary-foreground hover:opacity-90 transition-opacity"
        >
          Predict Churn
        </button>
      </div>

      <AnimatePresence>
        {results && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-4"
          >
            {Object.entries(results).map(([model, r]) => {
              const isChurn = r.prediction === 1;
              return (
                <div key={model} className={`glass-card p-5 ${isChurn ? "glow-coral" : "glow-accent"}`}>
                  <div className={`text-xs font-semibold uppercase tracking-wider mb-2 text-${colorMap[model]}`}>
                    {model}
                  </div>
                  <div className={`text-2xl font-heading font-bold mb-1 ${isChurn ? "text-coral" : "text-emerald"}`}>
                    {isChurn ? "⚠ Will Churn" : "✅ Will Stay"}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Churn Probability: <span className="text-foreground font-semibold">{(r.probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="mt-3 h-2 rounded-full bg-secondary overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${r.probability * 100}%` }}
                      transition={{ duration: 0.8, ease: "easeOut" }}
                      className={`h-full rounded-full ${isChurn ? "bg-coral" : "bg-emerald"}`}
                    />
                  </div>
                </div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
