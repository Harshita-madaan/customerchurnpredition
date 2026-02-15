# 📌 Telecom Customer Churn Prediction using Machine Learning

## 📖 Overview

Developed a machine learning model to predict customer churn for a telecom company using Logistic Regression. The objective is to identify customers at high risk of leaving the service, enabling data-driven retention strategies.

This project demonstrates end-to-end ML workflow including data preprocessing, feature engineering, exploratory data analysis (EDA), model training, evaluation, and interactive deployment.

---

## 🎯 Problem Statement

Customer churn directly impacts revenue in telecom businesses.  
The goal of this project is to build a predictive model that can classify whether a customer will churn (leave) or stay, based on service usage patterns and demographic information.

---

## 🔍 Exploratory Data Analysis (EDA)

- Analyzed churn distribution across customers
- Visualized class imbalance in the dataset
- Identified relationships between service features and churn
- Cleaned and handled missing values
- Converted categorical features into numerical format

---

## ⚙️ Data Preprocessing

- Removed irrelevant features (e.g., customerID)
- Applied:
  - Label Encoding for binary categorical features
  - One-Hot Encoding for multi-category variables
- Converted `TotalCharges` to numeric and handled missing values
- Scaled numerical features using **StandardScaler**
- Performed 80-20 Train-Test Split

---

## 🤖 Model Development

- Algorithm Used: **Logistic Regression**
- Optimized using increased `max_iter` for convergence
- Trained on processed telecom dataset

---

## 📊 Model Evaluation

Evaluated model performance using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix (visualized)

The confusion matrix visualization helps interpret:
- True Positives
- False Positives
- False Negatives
- True Negatives

---

## 💻 Interactive Prediction System

Built an interactive console-based prediction system that:

- Accepts user inputs for customer features
- Implements strict 0/1 input validation
- Applies scaling and feature alignment
- Outputs:
  - Churn Prediction (YES / NO)
  - Churn Probability (%)

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## 📈 Key Learning Outcomes

- End-to-end machine learning pipeline implementation
- Feature engineering and preprocessing techniques
- Handling categorical data effectively
- Model evaluation using classification metrics
- Building user-level prediction interface
- Understanding business impact of churn prediction

---
## Dataset Link
-https://www.kaggle.com/code/farazrahman/telco-customer-churn-logisticregression](https://www.kaggle.com/code/farazrahman/telco-customer-churn-logisticregression/input

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib scikit-learn

