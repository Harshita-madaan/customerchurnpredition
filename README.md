 # 📊 Customer Churn Prediction

## 📖 Overview

Developed an end-to-end machine learning pipeline to predict customer churn for a telecom company using multiple classification algorithms.
The goal is to identify customers at high risk of leaving, enabling data-driven retention strategies.

This project demonstrates a complete ML workflow including data preprocessing, feature engineering, multi-model training, evaluation, and visualization.

---

## 🎯 Problem Statement

Customer churn significantly impacts revenue in telecom businesses.
The objective is to build and compare predictive models that classify whether a customer will churn or stay based on service usage and demographic features.

---

## 🔍 Exploratory Data Analysis (EDA)

* Analyzed churn distribution and class imbalance
* Explored relationships between customer features and churn
* Cleaned dataset and handled missing values
* Converted categorical features into numerical format

---

## ⚙️ Data Preprocessing

* Removed irrelevant features (e.g., `customerID`)
* Applied Label Encoding for binary features
* Applied One-Hot Encoding for multi-category variables
* Converted `TotalCharges` to numeric and handled missing values
* Scaled numerical features using **StandardScaler**
* Performed **80-20 stratified train-test split**

---

## 🤖 Models Implemented

* 📘 Logistic Regression
* 📙 K-Nearest Neighbors (KNN) *(best K ≈ 20)*
* 📗 Decision Tree *(best depth ≈ 5)*
* 🌲 Random Forest

---

## 📊 Model Performance

| Metric    | Logistic Regression | KNN       | Decision Tree | Random Forest |
| --------- | ------------------- | --------- | ------------- | ------------- |
| Accuracy  | 73.81%              | **78.0%** | 75.16%        | 76.93%        |
| Precision | 79.66%              | 77.0%     | 79.93%        | 79.51%        |
| Recall    | 73.81%              | 78.0%     | 75.16%        | 76.93%        |
| F1 Score  | 75.19%              | 77.32%    | 76.38%        | **77.76%**    |
| ROC-AUC   | **84.17%**          | 81.22%    | 83.18%        | 84.12%        |
| CV-AUC    | **84.48%**          | 81.39%    | 82.65%        | 84.17%        |

---

## 🏆 Key Insights

* **KNN achieved highest accuracy (~78%)**
* **Random Forest achieved best F1 Score (~77.7%)**
* **Logistic Regression & Random Forest showed highest ROC-AUC (~84%)**
* Model performance is consistent across cross-validation, indicating good reliability

---

## 📈 Visualizations

* 📊 Model comparison (Accuracy, Precision, Recall, F1 Score)
* 🔲 Confusion matrices for all models
* 📉 KNN hyperparameter tuning (K vs performance)
* 🌟 Feature importance (Random Forest)
* 📈 ROC curve comparison
* 🕸 Radar chart for multi-metric comparison
* 📦 Cross-validation stability analysis

---

## 🛠 Tech Stack

* **Python**
* **Pandas, NumPy**
* **Matplotlib**
* **Scikit-learn**

---

## 📁 Dataset

* IBM Telco Customer Churn Dataset
* ~7,000 customers, 21 features
* Target variable: `Churn` (Yes/No)

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
pip install pandas numpy matplotlib scikit-learn
python churnprediction.py
```

---

## 📈 Key Learnings

* Built a complete end-to-end ML pipeline
* Compared multiple classification models
* Performed feature engineering and preprocessing
* Handled class imbalance using weighted models
* Evaluated models using F1 Score and ROC-AUC instead of accuracy alone
* Applied hyperparameter tuning for KNN and Decision Tree

---

## 💡 Conclusion

The models achieved **~84% ROC-AUC**, indicating strong ability to distinguish churn vs non-churn customers.
This project highlights the importance of using multiple evaluation metrics and model comparison in real-world classification problems.
