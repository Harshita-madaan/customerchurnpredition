iimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             roc_curve)
 
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
 
# ---------------- CHECK CLASS BALANCE ----------------
 
print("\n📊 Class Distribution:")
print(df_copy['Churn'].value_counts())
print(df_copy['Churn'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
# If ~80% are "No Churn", class_weight='balanced' below is important!
 
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
# ✅ IMPROVEMENT: Added class_weight='balanced' to LR and DT
#    This fixes the class imbalance problem without any new library
# ================================================
 
# --- Model 1: Logistic Regression ---
print("\n⏳ Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=2000, class_weight='balanced')  # ✅ IMPROVED
lr_model.fit(X_train, y_train)
lr_pred  = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)[:, 1]
print("✅ Done!")
 
# --- Model 2: KNN - Find Best K using F1 Score (not just accuracy) ---
print("\n⏳ Training KNN — Finding Best K (1 to 20) using F1 Score...")
f1_list       = []  # ✅ IMPROVED: track F1 instead of accuracy
accuracy_list = []
for k in range(1, 21):
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    y_temp = knn_temp.predict(X_test)
    f1_list.append(f1_score(y_test, y_temp, average='weighted'))
    accuracy_list.append(knn_temp.score(X_test, y_test))
 
best_k = f1_list.index(max(f1_list)) + 1  # ✅ Best K by F1 now
print(f"✅ Best K = {best_k} (by F1 Score)")
 
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
knn_pred  = knn_model.predict(X_test)
knn_proba = knn_model.predict_proba(X_test)[:, 1]
 
# --- Model 3: Decision Tree ---
# ✅ IMPROVEMENT: Try multiple depths and pick best by F1
print("\n⏳ Training Decision Tree — Finding Best Depth (3 to 10)...")
best_depth   = 5
best_dt_f1   = 0
depth_scores = {}
 
for depth in range(3, 11):
    dt_temp = DecisionTreeClassifier(max_depth=depth,
                                     class_weight='balanced',   # ✅ IMPROVED
                                     random_state=42)
    dt_temp.fit(X_train, y_train)
    score = f1_score(y_test, dt_temp.predict(X_test), average='weighted')
    depth_scores[depth] = round(score * 100, 2)
    if score > best_dt_f1:
        best_dt_f1 = score
        best_depth = depth
 
print(f"✅ Best Depth = {best_depth} (by F1 Score)")
 
dt_model = DecisionTreeClassifier(max_depth=best_depth,
                                  class_weight='balanced',       # ✅ IMPROVED
                                  random_state=42)
dt_model.fit(X_train, y_train)
dt_pred  = dt_model.predict(X_test)
dt_proba = dt_model.predict_proba(X_test)[:, 1]
 
# ================================================
# ---------------- CALCULATE ALL METRICS ----------------
# ✅ IMPROVEMENT: Added ROC-AUC score for each model
# ================================================
 
metrics = {
    'Logistic Regression': {
        'Accuracy'  : round(accuracy_score(y_test, lr_pred)                        * 100, 2),
        'Precision' : round(precision_score(y_test, lr_pred, average='weighted')   * 100, 2),
        'Recall'    : round(recall_score(y_test, lr_pred, average='weighted')      * 100, 2),
        'F1 Score'  : round(f1_score(y_test, lr_pred, average='weighted')          * 100, 2),
        'ROC-AUC'   : round(roc_auc_score(y_test, lr_proba)                        * 100, 2),  # ✅ NEW
    },
    f'KNN (K={best_k})': {
        'Accuracy'  : round(accuracy_score(y_test, knn_pred)                       * 100, 2),
        'Precision' : round(precision_score(y_test, knn_pred, average='weighted')  * 100, 2),
        'Recall'    : round(recall_score(y_test, knn_pred, average='weighted')     * 100, 2),
        'F1 Score'  : round(f1_score(y_test, knn_pred, average='weighted')         * 100, 2),
        'ROC-AUC'   : round(roc_auc_score(y_test, knn_proba)                       * 100, 2),  # ✅ NEW
    },
    f'Decision Tree (Depth={best_depth})': {
        'Accuracy'  : round(accuracy_score(y_test, dt_pred)                        * 100, 2),
        'Precision' : round(precision_score(y_test, dt_pred, average='weighted')   * 100, 2),
        'Recall'    : round(recall_score(y_test, dt_pred, average='weighted')      * 100, 2),
        'F1 Score'  : round(f1_score(y_test, dt_pred, average='weighted')          * 100, 2),
        'ROC-AUC'   : round(roc_auc_score(y_test, dt_proba)                        * 100, 2),  # ✅ NEW
    }
}
 
model_names = list(metrics.keys())
 
# Print metrics table in terminal
print("\n" + "="*70)
print("               📊 MODEL PERFORMANCE COMPARISON TABLE")
print("="*70)
print(f"{'Metric':<15} {'Log. Reg':>15} {f'KNN(K={best_k})':>15} {f'DT(D={best_depth})':>15}")
print("-"*70)
for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']:
    vals = [metrics[m][metric] for m in metrics]
    print(f"{metric:<15} {vals[0]:>14}%  {vals[1]:>14}%  {vals[2]:>14}%")
print("="*70)
 
# ================================================
# -------- GRAPH 1: SIDE-BY-SIDE METRIC COMPARISON --------
# ✅ IMPROVEMENT: Now includes ROC-AUC in the bar chart
# ================================================
 
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']  # ✅ Added ROC-AUC
colors       = ['steelblue', 'coral', 'mediumseagreen']
 
x     = np.arange(len(metric_names))
width = 0.25
 
fig, ax = plt.subplots(figsize=(14, 6))
 
for i, (model_name, color) in enumerate(zip(model_names, colors)):
    values = [metrics[model_name][m] for m in metric_names]
    bars   = ax.bar(x + i * width, values, width, label=model_name, color=color)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val}%",
            ha='center', va='bottom', fontsize=7, fontweight='bold'
        )
 
ax.set_xlabel("Metrics", fontsize=12)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("📊 Model Comparison: Accuracy, Precision, Recall, F1 & ROC-AUC", fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metric_names, fontsize=10)
ax.set_ylim(0, 115)
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
 
# ================================================
# -------- GRAPH 2: CONFUSION MATRICES --------
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
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['No Churn', 'Churn'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['No Churn', 'Churn'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]),
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    color='white' if cm[i][j] > cm.max() / 2 else 'black')
 
plt.colorbar(im, ax=axes[-1])
plt.tight_layout()
plt.show()
 
# ================================================
# -------- GRAPH 3: KNN — F1 Score vs K Value --------
# ✅ IMPROVEMENT: Shows F1 instead of just accuracy for K selection
# ================================================
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# Left: F1 Score vs K
axes[0].plot(range(1, 21), [f * 100 for f in f1_list],
             marker='o', color='steelblue', linewidth=2, markersize=6)
axes[0].axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best K = {best_k}')
axes[0].scatter(best_k, max(f1_list) * 100, color='red', zorder=5, s=100)
axes[0].set_xlabel("K Value", fontsize=12)
axes[0].set_ylabel("F1 Score (%)", fontsize=12)
axes[0].set_title("🔍 KNN: F1 Score vs K Value", fontsize=13, fontweight='bold')
axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.6)
 
# Right: Accuracy vs K (kept for reference)
axes[1].plot(range(1, 21), [a * 100 for a in accuracy_list],
             marker='s', color='coral', linewidth=2, markersize=6)
axes[1].axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best K = {best_k}')
axes[1].set_xlabel("K Value", fontsize=12)
axes[1].set_ylabel("Accuracy (%)", fontsize=12)
axes[1].set_title("🔍 KNN: Accuracy vs K Value", fontsize=13, fontweight='bold')
axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.6)
 
plt.suptitle("KNN Hyperparameter Tuning", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
 
# ================================================
# -------- GRAPH 4: DECISION TREE DEPTH TUNING --------
# ✅ NEW GRAPH: Shows how F1 changes with tree depth
# ================================================
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# Left: Depth vs F1
depths = list(depth_scores.keys())
scores = list(depth_scores.values())
axes[0].plot(depths, scores, marker='o', color='mediumseagreen', linewidth=2, markersize=8)
axes[0].axvline(x=best_depth, color='red', linestyle='--', linewidth=2, label=f'Best Depth = {best_depth}')
axes[0].scatter(best_depth, depth_scores[best_depth], color='red', zorder=5, s=120)
axes[0].set_xlabel("Max Depth", fontsize=12)
axes[0].set_ylabel("F1 Score (%)", fontsize=12)
axes[0].set_title("🌳 Decision Tree: Depth Tuning", fontsize=13, fontweight='bold')
axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.6)
 
# Right: Feature Importance
feature_importance = pd.Series(dt_model.feature_importances_, index=X.columns)
top_features = feature_importance.sort_values(ascending=False).head(8)
 
bars = axes[1].barh(top_features.index[::-1], top_features.values[::-1] * 100, color='mediumseagreen')
for bar, val in zip(bars, top_features.values[::-1] * 100):
    axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{round(val, 2)}%", va='center', fontsize=9)
axes[1].set_xlabel("Importance (%)", fontsize=12)
axes[1].set_title("🌟 Top 8 Important Features", fontsize=13, fontweight='bold')
axes[1].grid(axis='x', linestyle='--', alpha=0.5)
 
plt.tight_layout()
plt.show()
 
# ================================================
# -------- GRAPH 5: ROC CURVE --------
# ✅ NEW GRAPH: Professional metric — great for CV/presentation
# ================================================
 
plt.figure(figsize=(8, 6))
 
for (model_name, proba, color) in [
    ('Logistic Regression', lr_proba,  'steelblue'),
    (f'KNN (K={best_k})',   knn_proba, 'coral'),
    (f'DT (Depth={best_depth})', dt_proba, 'mediumseagreen')
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_score   = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, color=color, linewidth=2,
             label=f"{model_name} (AUC = {round(auc_score, 2)})")
 
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess')
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("📈 ROC Curve — All 3 Models", fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
 
# ================================================
# -------- GRAPH 6: RADAR CHART --------
# ================================================
 
categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']  # ✅ Added ROC-AUC
N      = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
 
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
 
for model_name, color in zip(model_names, colors):
    values = [metrics[model_name][m] for m in categories]
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, label=model_name)
    ax.fill(angles, values, color=color, alpha=0.15)
 
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 100)
ax.set_title("🕸 Radar Chart: All Models vs All Metrics", fontsize=13,
             fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout()
plt.show()
 
# ================================================
# -------- PRINT BEST MODEL --------
# ================================================
 
best_model_name = max(metrics, key=lambda m: metrics[m]['F1 Score'])
print(f"\n🏆 BEST MODEL (by F1 Score): {best_model_name}")
for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']:
    print(f"   {metric:<12}: {metrics[best_model_name][metric]}%")
 
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
 
user_data['gender']           = gender
user_data['Partner']          = partner
user_data['Dependents']       = dependents
user_data['tenure']           = tenure
user_data['PhoneService']     = phone_service
user_data['MultipleLines']    = multiple_lines
user_data['OnlineSecurity']   = online_security
user_data['OnlineBackup']     = online_backup
user_data['DeviceProtection'] = device_protection
user_data['TechSupport']      = tech_support
user_data['StreamingTV']      = streaming_tv
user_data['StreamingMovies']  = streaming_movies
user_data['PaperlessBilling'] = paperless_billing
user_data['MonthlyCharges']   = monthly_charges
user_data['TotalCharges']     = total_charges
 
for col in user_data.columns:
    if "Fiber optic"       in col: user_data[col] = internet_fiber
    if "InternetService_No" in col: user_data[col] = internet_no
    if "One year"          in col: user_data[col] = contract_one_year
    if "Two year"          in col: user_data[col] = contract_two_year
    if "Credit card"       in col: user_data[col] = payment_credit
    if "Electronic check"  in col: user_data[col] = payment_electronic
    if "Mailed check"      in col: user_data[col] = payment_mailed
 
user_data[numeric_cols] = scaler.transform(user_data[numeric_cols])
 
# ================================================
# ---------------- PREDICTIONS FROM ALL 3 MODELS ----------------
# ================================================
 
print("\n" + "="*55)
print("       📢 CHURN PREDICTIONS FROM ALL 3 MODELS")
print("="*55)
 
all_models = {
    "Logistic Regression"         : lr_model,
    f"KNN (K={best_k})"           : knn_model,
    f"Decision Tree (D={best_depth})": dt_model
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
