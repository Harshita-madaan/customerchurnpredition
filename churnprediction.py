import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ---------------- HELPER FUNCTION ----------------

def get_binary_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value in [0, 1]:
                return value
            else:
                print("❌ Error: Please enter only 1 (Yes) or 0 (No)")
        except ValueError:
            print("❌ Error: Enter numeric value 1 or 0")

# ---------------- LOAD DATA ----------------

df = pd.read_csv("/Users/harshita/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv.xls")
df_copy = df.copy()

df_copy.drop('customerID', axis=1, inplace=True)

binary_cols = [
    'gender','Partner','Dependents','PhoneService','MultipleLines',
    'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
    'StreamingTV','StreamingMovies','PaperlessBilling'
]

df_copy[binary_cols] = df_copy[binary_cols].replace({'No Internet Service':'No'})

le = LabelEncoder()
for col in binary_cols:
    df_copy[col] = le.fit_transform(df_copy[col])

df_copy['Churn'] = df_copy['Churn'].map({'Yes':1, 'No':0})

df_copy = pd.get_dummies(
    df_copy,
    columns=['InternetService', 'Contract', 'PaymentMethod'],
    drop_first=True
)

df_copy['TotalCharges'] = pd.to_numeric(df_copy['TotalCharges'], errors='coerce')
df_copy['TotalCharges'].fillna(df_copy['TotalCharges'].median(), inplace=True)

# ---------------- SCALING ----------------

scaler = StandardScaler()
numeric_cols = ['tenure','MonthlyCharges','TotalCharges']
df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])

# ---------------- TRAIN TEST SPLIT ----------------

X = df_copy.drop('Churn', axis=1)
y = df_copy['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================================
# ---------------- TRAIN ALL 3 MODELS ----------------
# ================================================

# --- Model 1: Logistic Regression ---
print("\n⏳ Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=2000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("✅ Done!")

# --- Model 2: KNN - Find Best K first ---
print("\n⏳ Training KNN — Finding Best K (1 to 20)...")
accuracy_list = []
for k in range(1, 21):
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    accuracy_list.append(knn_temp.score(X_test, y_test))

best_k = accuracy_list.index(max(accuracy_list)) + 1
print(f"✅ Best K = {best_k}")

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

# --- Model 3: Decision Tree ---
print("\n⏳ Training Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("✅ Done!")

# ================================================
# ---------------- CALCULATE ALL METRICS ----------------
# ================================================

# Collect metrics for each model in a neat dictionary
# precision/recall/f1 use average='weighted' to handle class imbalance fairly

metrics = {
    'Logistic Regression': {
        'Accuracy'  : round(accuracy_score(y_test, lr_pred)                        * 100, 2),
        'Precision' : round(precision_score(y_test, lr_pred, average='weighted')   * 100, 2),
        'Recall'    : round(recall_score(y_test, lr_pred, average='weighted')      * 100, 2),
        'F1 Score'  : round(f1_score(y_test, lr_pred, average='weighted')          * 100, 2),
    },
    f'KNN (K={best_k})': {
        'Accuracy'  : round(accuracy_score(y_test, knn_pred)                       * 100, 2),
        'Precision' : round(precision_score(y_test, knn_pred, average='weighted')  * 100, 2),
        'Recall'    : round(recall_score(y_test, knn_pred, average='weighted')     * 100, 2),
        'F1 Score'  : round(f1_score(y_test, knn_pred, average='weighted')         * 100, 2),
    },
    'Decision Tree': {
        'Accuracy'  : round(accuracy_score(y_test, dt_pred)                        * 100, 2),
        'Precision' : round(precision_score(y_test, dt_pred, average='weighted')   * 100, 2),
        'Recall'    : round(recall_score(y_test, dt_pred, average='weighted')      * 100, 2),
        'F1 Score'  : round(f1_score(y_test, dt_pred, average='weighted')          * 100, 2),
    }
}

# Print metrics table in terminal
print("\n" + "="*60)
print("          📊 MODEL PERFORMANCE COMPARISON TABLE")
print("="*60)
print(f"{'Metric':<15} {'Log. Reg':>15} {f'KNN(K={best_k})':>15} {'Dec. Tree':>15}")
print("-"*60)
for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
    vals = [metrics[m][metric] for m in metrics]
    print(f"{metric:<15} {vals[0]:>14}%  {vals[1]:>14}%  {vals[2]:>14}%")
print("="*60)

# ================================================
# -------- GRAPH 1: SIDE-BY-SIDE METRIC COMPARISON --------
# ================================================

model_names  = list(metrics.keys())
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors       = ['steelblue', 'coral', 'mediumseagreen']

x = np.arange(len(metric_names))   # positions on x-axis: 0,1,2,3
width = 0.25                        # width of each bar

fig, ax = plt.subplots(figsize=(12, 6))

for i, (model_name, color) in enumerate(zip(model_names, colors)):
    values = [metrics[model_name][m] for m in metric_names]
    bars = ax.bar(x + i * width, values, width, label=model_name, color=color)

    # Print value on top of each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val}%",
            ha='center', va='bottom', fontsize=8, fontweight='bold'
        )

ax.set_xlabel("Metrics", fontsize=12)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("📊 Model Comparison: Accuracy, Precision, Recall & F1 Score", fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylim(0, 115)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ================================================
# -------- GRAPH 2: CONFUSION MATRICES (3 side by side) --------
# ================================================

conf_matrices = [
    confusion_matrix(y_test, lr_pred),
    confusion_matrix(y_test, knn_pred),
    confusion_matrix(y_test, dt_pred)
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("🔲 Confusion Matrices — All 3 Models", fontsize=14, fontweight='bold')

for ax, cm, name in zip(axes, conf_matrices, model_names):
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['No Churn', 'Churn'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['No Churn', 'Churn'])

    # Write numbers inside each box
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]),
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    color='white' if cm[i][j] > cm.max() / 2 else 'black')

plt.colorbar(im, ax=axes[-1])
plt.tight_layout()
plt.show()

# ================================================
# -------- GRAPH 3: KNN ACCURACY vs K VALUE --------
# ================================================

plt.figure(figsize=(9, 5))
plt.plot(range(1, 21), [a * 100 for a in accuracy_list],
         marker='o', color='steelblue', linewidth=2, markersize=6)
plt.axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best K = {best_k}')
plt.scatter(best_k, max(accuracy_list) * 100, color='red', zorder=5, s=100)
plt.text(best_k + 0.3, max(accuracy_list) * 100,
         f"Best: {round(max(accuracy_list)*100,2)}%", color='red', fontsize=10)
plt.xlabel("K Value", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("🔍 KNN: Accuracy vs K Value", fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ================================================
# -------- GRAPH 4: DECISION TREE FEATURE IMPORTANCE --------
# ================================================

feature_importance = pd.Series(dt_model.feature_importances_, index=X.columns)
top_features = feature_importance.sort_values(ascending=False).head(8)

plt.figure(figsize=(9, 5))
bars = plt.barh(top_features.index[::-1], top_features.values[::-1] * 100, color='mediumseagreen')

for bar, val in zip(bars, top_features.values[::-1] * 100):
    plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
             f"{round(val, 2)}%", va='center', fontsize=9)

plt.xlabel("Importance (%)", fontsize=12)
plt.title("🌟 Decision Tree: Top 8 Important Features", fontsize=13, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ================================================
# -------- GRAPH 5: RADAR CHART - OVERALL BEST MODEL --------
# ================================================

# Radar chart gives a "web" view of all metrics at once per model
from matplotlib.patches import FancyArrowPatch

categories = metric_names
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]   # close the loop

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for model_name, color in zip(model_names, colors):
    values = [metrics[model_name][m] for m in metric_names]
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, label=model_name)
    ax.fill(angles, values, color=color, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 100)
ax.set_title("🕸 Radar Chart: All Models vs All Metrics", fontsize=13,
             fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
plt.tight_layout()
plt.show()

# ================================================
# -------- PRINT BEST MODEL --------
# ================================================

best_model_name = max(metrics, key=lambda m: metrics[m]['F1 Score'])
print(f"\n🏆 BEST MODEL (by F1 Score): {best_model_name}")
print(f"   Accuracy  : {metrics[best_model_name]['Accuracy']}%")
print(f"   Precision : {metrics[best_model_name]['Precision']}%")
print(f"   Recall    : {metrics[best_model_name]['Recall']}%")
print(f"   F1 Score  : {metrics[best_model_name]['F1 Score']}%")

# ================================================
# ---------------- USER INPUT ----------------
# ================================================

print("\n🔎 Enter Customer Details To Predict Churn:\n")

tenure          = float(input("Enter Tenure (in months): "))
monthly_charges = float(input("Enter Monthly Charges: "))
total_charges   = float(input("Enter Total Charges: "))

gender             = get_binary_input("Gender (Male=1, Female=0): ")
partner            = get_binary_input("Partner (Yes=1, No=0): ")
dependents         = get_binary_input("Dependents (Yes=1, No=0): ")
phone_service      = get_binary_input("Phone Service (Yes=1, No=0): ")
multiple_lines     = get_binary_input("Multiple Lines (Yes=1, No=0): ")
online_security    = get_binary_input("Online Security (Yes=1, No=0): ")
online_backup      = get_binary_input("Online Backup (Yes=1, No=0): ")
device_protection  = get_binary_input("Device Protection (Yes=1, No=0): ")
tech_support       = get_binary_input("Tech Support (Yes=1, No=0): ")
streaming_tv       = get_binary_input("Streaming TV (Yes=1, No=0): ")
streaming_movies   = get_binary_input("Streaming Movies (Yes=1, No=0): ")
paperless_billing  = get_binary_input("Paperless Billing (Yes=1, No=0): ")
internet_fiber     = get_binary_input("Internet Service Fiber Optic (Yes=1, No=0): ")
internet_no        = get_binary_input("Internet Service No (Yes=1, No=0): ")
contract_one_year  = get_binary_input("Contract One Year (Yes=1, No=0): ")
contract_two_year  = get_binary_input("Contract Two Year (Yes=1, No=0): ")
payment_credit     = get_binary_input("Payment Method Credit Card (Yes=1, No=0): ")
payment_electronic = get_binary_input("Payment Method Electronic Check (Yes=1, No=0): ")
payment_mailed     = get_binary_input("Payment Method Mailed Check (Yes=1, No=0): ")

# ---------------- CREATE USER DATA ----------------

user_data = pd.DataFrame(0, index=[0], columns=X.columns)

user_data['gender']          = gender
user_data['Partner']         = partner
user_data['Dependents']      = dependents
user_data['tenure']          = tenure
user_data['PhoneService']    = phone_service
user_data['MultipleLines']   = multiple_lines
user_data['OnlineSecurity']  = online_security
user_data['OnlineBackup']    = online_backup
user_data['DeviceProtection']= device_protection
user_data['TechSupport']     = tech_support
user_data['StreamingTV']     = streaming_tv
user_data['StreamingMovies'] = streaming_movies
user_data['PaperlessBilling']= paperless_billing
user_data['MonthlyCharges']  = monthly_charges
user_data['TotalCharges']    = total_charges

for col in user_data.columns:
    if "Fiber optic" in col:  user_data[col] = internet_fiber
    if "InternetService_No"   in col: user_data[col] = internet_no
    if "One year"  in col:    user_data[col] = contract_one_year
    if "Two year"  in col:    user_data[col] = contract_two_year
    if "Credit card"  in col: user_data[col] = payment_credit
    if "Electronic check" in col: user_data[col] = payment_electronic
    if "Mailed check" in col: user_data[col] = payment_mailed

user_data[numeric_cols] = scaler.transform(user_data[numeric_cols])

# ================================================
# ---------------- PREDICTIONS FROM ALL 3 MODELS ----------------
# ================================================

print("\n" + "="*55)
print("       📢 CHURN PREDICTIONS FROM ALL 3 MODELS")
print("="*55)

all_models = {
    "Logistic Regression": lr_model,
    f"KNN (K={best_k})"  : knn_model,
    "Decision Tree"      : dt_model
}

for model_name, trained_model in all_models.items():
    prediction  = trained_model.predict(user_data)
    probability = trained_model.predict_proba(user_data)[0][1]
    verdict     = "⚠  YES — WILL CHURN" if prediction[0] == 1 else "✅  NO  — WILL STAY"
    print(f"\n🔷 {model_name}")
    print(f"   Prediction       : {verdict}")
    print(f"   Churn Probability: {round(probability * 100, 2)}%")

print("\n" + "="*55)
print(f"🏆 Most Reliable Model: {best_model_name}")
print("="*55)
