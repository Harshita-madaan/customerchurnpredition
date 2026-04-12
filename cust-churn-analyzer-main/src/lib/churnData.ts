// Hardcoded model results from the trained models

export const modelMetrics = {
  "Logistic Regression": {
    accuracy: 80.48,
    precision: 79.73,
    recall: 80.48,
    f1: 79.96,
    confusionMatrix: [[925, 110], [165, 209]],
    color: "hsl(var(--chart-blue))",
  },
  "KNN (K=16)": {
    accuracy: 78.0,
    precision: 76.84,
    recall: 78.0,
    f1: 77.17,
    confusionMatrix: [[917, 118], [192, 182]],
    color: "hsl(var(--chart-coral))",
  },
  "Decision Tree": {
    accuracy: 78.85,
    precision: 79.0,
    recall: 78.85,
    f1: 78.92,
    confusionMatrix: [[882, 153], [145, 229]],
    color: "hsl(var(--chart-green))",
  },
};

export const knnAccuracies = [
  72.5, 76.2, 77.0, 77.1, 77.3, 77.8, 77.5, 77.2, 76.8, 76.9,
  76.5, 76.3, 76.5, 76.3, 77.5, 78.0, 77.2, 77.0, 77.1, 77.8,
];

export const featureImportance = [
  { name: "OnlineSecurity", importance: 38.2 },
  { name: "tenure", importance: 23.44 },
  { name: "InternetService_Fiber optic", importance: 13.23 },
  { name: "TotalCharges", importance: 8.02 },
  { name: "Contract_Two year", importance: 4.69 },
  { name: "MonthlyCharges", importance: 3.84 },
  { name: "Contract_One year", importance: 3.79 },
  { name: "PaymentMethod_Electronic check", importance: 2.96 },
];

export interface CustomerInput {
  tenure: number;
  monthlyCharges: number;
  totalCharges: number;
  gender: number;
  partner: number;
  dependents: number;
  phoneService: number;
  multipleLines: number;
  onlineSecurity: number;
  onlineBackup: number;
  deviceProtection: number;
  techSupport: number;
  streamingTV: number;
  streamingMovies: number;
  paperlessBilling: number;
  internetFiber: number;
  internetNo: number;
  contractOneYear: number;
  contractTwoYear: number;
  paymentCredit: number;
  paymentElectronic: number;
  paymentMailed: number;
}

// Simplified prediction simulation based on feature weights
export function simulatePrediction(input: CustomerInput) {
  const churnScore =
    (1 - input.onlineSecurity) * 0.25 +
    (input.tenure < 12 ? 0.2 : input.tenure < 24 ? 0.1 : 0) +
    (input.internetFiber ? 0.12 : 0) +
    (input.monthlyCharges > 70 ? 0.1 : 0) +
    (!input.contractOneYear && !input.contractTwoYear ? 0.15 : 0) +
    (input.paymentElectronic ? 0.05 : 0) +
    (!input.techSupport ? 0.05 : 0) +
    (input.paperlessBilling ? 0.03 : 0) +
    (!input.partner ? 0.03 : 0) +
    (!input.dependents ? 0.02 : 0);

  const lrProb = Math.min(0.95, Math.max(0.05, churnScore + (Math.random() * 0.06 - 0.03)));
  const knnProb = Math.min(0.95, Math.max(0.05, churnScore + (Math.random() * 0.08 - 0.04)));
  const dtProb = Math.min(0.95, Math.max(0.05, churnScore + (Math.random() * 0.1 - 0.05)));

  return {
    "Logistic Regression": { prediction: lrProb >= 0.5 ? 1 : 0, probability: lrProb },
    "KNN (K=16)": { prediction: knnProb >= 0.5 ? 1 : 0, probability: knnProb },
    "Decision Tree": { prediction: dtProb >= 0.5 ? 1 : 0, probability: dtProb },
  };
}
