 📌 Telecom Customer Churn Prediction using Machine Learning

## 📖 Overview

Developed a machine learning pipeline to predict customer churn for a telecom company using **three classification algorithms** — **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Decision Tree**. The objective is to identify customers at high risk of leaving the service, enabling data-driven retention strategies.

This project demonstrates a complete end-to-end ML workflow including data preprocessing, feature engineering, exploratory data analysis (EDA), multi-model training, comparative evaluation, and a web-based interactive prediction interface.

---

## 🎯 Problem Statement

Customer churn directly impacts revenue in telecom businesses. The goal of this project is to build and compare predictive models that classify whether a customer will **churn (leave)** or **stay**, based on service usage patterns and demographic information — then identify the best-performing model.

---

## 🔍 Exploratory Data Analysis (EDA)

- Analyzed churn distribution across customers
- Visualized class imbalance in the dataset
- Identified relationships between service features and churn
- Cleaned and handled missing values
- Converted categorical features into numerical format

---

## ⚙️ Data Preprocessing

- Removed irrelevant features (e.g., `customerID`)
- Applied:
  - **Label Encoding** for binary categorical features
  - **One-Hot Encoding** for multi-category variables (`InternetService`, `Contract`, `PaymentMethod`)
- Converted `TotalCharges` to numeric and handled missing values with median imputation
- Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using **StandardScaler**
- Performed **80-20 Train-Test Split** with stratification

---

## 🤖 Models Trained

### 1. 📘 Logistic Regression
- Draws a linear decision boundary to separate churners from non-churners
- Optimized with `max_iter=2000` for convergence
- **Best overall performer** across all metrics

### 2. 📙 K-Nearest Neighbors (KNN)
- Predicts based on the K most similar past customers (majority vote)
- Automatically finds the **best K value** by testing K = 1 to 20
- Best K found: **K = 16**

### 3. 📗 Decision Tree
- Uses a flowchart-like Yes/No question structure to classify customers
- Controlled with `max_depth=5` to prevent overfitting
- Provides **feature importance scores** to explain predictions

---

## 📊 Model Evaluation & Results

All models evaluated using:

| Metric | Logistic Regression | KNN (K=16) | Decision Tree |
|:-------|:-------------------:|:----------:|:-------------:|
| **Accuracy** | **80.48%** | 78.0% | 78.85% |
| **Precision** | **79.73%** | 76.84% | 79.0% |
| **Recall** | **80.48%** | 78.0% | 78.85% |
| **F1 Score** | **79.96%** | 77.17% | 78.92% |

> 🏆 **Winner: Logistic Regression** — leads on every single metric.

### Confusion Matrix Highlights

| Model | Correct Non-Churners | Correct Churners Caught | Missed Churners |
|:------|:--------------------:|:-----------------------:|:---------------:|
| Logistic Regression | 925 | 209 | 165 |
| KNN (K=16) | 917 | 182 | 192 |
| Decision Tree | 882 | 229 | 145 |

### 🌟 Top Features Driving Churn *(from Decision Tree)*

| Rank | Feature | Importance |
|:----:|:--------|:----------:|
| 1 | OnlineSecurity | 38.2% |
| 2 | Tenure | 23.44% |
| 3 | InternetService — Fiber Optic | 13.23% |
| 4 | TotalCharges | 8.02% |
| 5 | Contract — Two Year | 4.69% |

---

## 📈 Visualizations Generated

| # | Graph | Description |
|:-:|:------|:------------|
| 1 | 📊 Grouped Bar Chart | Accuracy, Precision, Recall & F1 Score for all 3 models side by side |
| 2 | 🔲 Confusion Matrices | All 3 models displayed together for direct comparison |
| 3 | 📉 KNN Accuracy vs K | Line chart showing accuracy across K=1 to 20 with best K highlighted |
| 4 | 🌟 Feature Importance | Top 8 features ranked by Decision Tree importance score |
| 5 | 🕸 Radar Chart | Web-shaped overview of all metrics per model simultaneously |
| 6 | 📦 Churn Distribution | Bar chart of churners vs non-churners in the dataset |

---

## 💻 Interactive Prediction System

Built a **web-based prediction interface** that:

- Takes customer details as input 
- Runs the customer data through all 3 trained models simultaneously
- Displays churn prediction **(YES / NO)** for each model
- Shows **churn probability (%)** per model
- Highlights the most reliable model's result

---

## 🛠 Tech Stack

| Tool | Purpose |
|:-----|:--------|
| `Python 3.8+` | Core language |
| `Pandas` | Data loading & manipulation |
| `NumPy` | Numerical operations |
| `Matplotlib` | All visualizations & graphs |
| `Scikit-learn` | Preprocessing, model training & evaluation |
| `Lovable` | Web-based prediction interface |

---

## 📁 Dataset

- **Source:** IBM Sample Dataset — Telco Customer Churn
- **Records:** 7,043 customers, 21 features
- **Target Variable:** `Churn` (Yes / No)
- 🔗 [View Dataset on Kaggle](https://www.kaggle.com/code/farazrahman/telco-customer-churn-logisticregression/input)

---

## 📂 Project Structure

```
telecom-churn-prediction/
│
├── churn_prediction.py              # Main ML script — all 3 models
├── README.md                        # Project documentation
└── dataset/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## 🚀 How to Run

**1. Clone the repository**

```bash
git clone https://github.com/your-username/telecom-churn-prediction.git
cd telecom-churn-prediction
```

**2. Install dependencies**

```bash
pip install pandas numpy matplotlib scikit-learn
```

**3. Update the dataset path in the script**

```python
df = pd.read_csv("path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

**4. Run the script**

```bash
python churn_prediction.py
```

**5.** All 6 graphs will be displayed automatically, followed by the model comparison summary in the terminal.

---

## 📈 Key Learning Outcomes

- End-to-end machine learning pipeline implementation
- Training and comparing multiple ML algorithms on the same dataset
- Feature engineering and preprocessing techniques
- Handling categorical data and class imbalance effectively
- Model evaluation using Accuracy, Precision, Recall, F1 Score & Confusion Matrix
- KNN hyperparameter tuning — finding the optimal K value
- Decision Tree interpretability via feature importance
- Building a web-based multi-model prediction interface with Lovable
- Understanding the business impact of churn prediction in telecom

---

> 💡 **Recommendation:** Logistic Regression is the best model for this dataset — achieving the highest Accuracy **(80.48%)** and F1 Score **(79.96%)** with low complexity and fast inference.
