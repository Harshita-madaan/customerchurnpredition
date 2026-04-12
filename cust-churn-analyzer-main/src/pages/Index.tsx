import { motion } from "framer-motion";
import ModelCards from "@/components/ModelCards";
import ModelComparisonChart from "@/components/ModelComparisonChart";
import ConfusionMatrices from "@/components/ConfusionMatrices";
import KnnChart from "@/components/KnnChart";
import FeatureImportanceChart from "@/components/FeatureImportanceChart";
import CustomerForm from "@/components/CustomerForm";

export default function Index() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 backdrop-blur-sm sticky top-0 z-50 bg-background/80">
        <div className="container max-w-7xl mx-auto px-4 py-4 flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-[hsl(var(--primary))] to-[hsl(var(--accent))] flex items-center justify-center text-lg">
            📡
          </div>
          <div>
            <h1 className="font-heading font-bold text-lg leading-tight">Telecom Churn Predictor</h1>
            <p className="text-xs text-muted-foreground">ML-Powered Customer Retention Analysis</p>
          </div>
        </div>
      </header>

      <main className="container max-w-7xl mx-auto px-4 py-8 space-y-8">
        {/* Hero */}
        <motion.section initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
          <h2 className="text-3xl md:text-4xl font-heading font-bold mb-2">
            Predicting Customer Churn with <span className="gradient-text">Machine Learning</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl">
            A comparative study of Logistic Regression, K-Nearest Neighbors, and Decision Tree classifiers on the Telco Customer Churn dataset — evaluating accuracy, precision, recall &amp; F1 score to identify the optimal model.
          </p>
        </motion.section>

        {/* Model Cards */}
        <section>
          <h2 className="font-heading font-semibold text-xl mb-4">Trained Models Overview</h2>
          <ModelCards />
        </section>

        {/* Charts Grid */}
        <section>
          <h2 className="font-heading font-semibold text-xl mb-4">Performance Dashboard</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ModelComparisonChart />
            <ConfusionMatrices />
            <KnnChart />
            <FeatureImportanceChart />
          </div>
        </section>

        {/* Prediction */}
        <section>
          <h2 className="font-heading font-semibold text-xl mb-4">Predict Customer Churn</h2>
          <CustomerForm />
        </section>
      </main>

      <footer className="border-t border-border/50 py-6 mt-12 text-center text-xs text-muted-foreground">
        Built with Logistic Regression, KNN & Decision Tree • Telecom Customer Churn Dataset
      </footer>
    </div>
  );
}
