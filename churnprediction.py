import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve)

# ================================================================
# ---------------- LOAD & PREPROCESS DATA ----------------
# ================================================================

df      = pd.read_csv("/Users/harshita/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv.xls")
df_copy = df.copy()

df_copy.drop('customerID', axis=1, inplace=True)

binary_cols = [
    'gender','Partner','Dependents','PhoneService','MultipleLines',
    'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
    'StreamingTV','StreamingMovies','PaperlessBilling'
]
df_copy[binary_cols] = df_copy[binary_cols].replace({'No Internet Service': 'No'})

le = LabelEncoder()
for col in binary_cols:
    df_copy[col] = le.fit_transform(df_copy[col])

df_copy['Churn'] = df_copy['Churn'].map({'Yes': 1, 'No': 0})

df_copy = pd.get_dummies(df_copy,
                         columns=['InternetService', 'Contract', 'PaymentMethod'],
                         drop_first=True)

df_copy['TotalCharges'] = pd.to_numeric(df_copy['TotalCharges'], errors='coerce')
df_copy['TotalCharges'].fillna(df_copy['TotalCharges'].median(), inplace=True)

# Print class balance
print("\nClass Distribution:")
print(df_copy['Churn'].value_counts())
print(df_copy['Churn'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

# ----------------------------------------------------------------
# Scaling
# ----------------------------------------------------------------
scaler       = StandardScaler()
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])

X = df_copy.drop('Churn', axis=1)
y = df_copy['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================================================
# ---------------- TRAIN ALL 4 MODELS ----------------
# ================================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Model 1: Logistic Regression ---
print("\nTraining Logistic Regression...")
lr_model  = LogisticRegression(max_iter=2000, class_weight='balanced')
lr_model.fit(X_train, y_train)
lr_pred   = lr_model.predict(X_test)
lr_proba  = lr_model.predict_proba(X_test)[:, 1]
lr_cv     = cross_val_score(lr_model, X, y, cv=cv, scoring='roc_auc').mean()
print("Done!")

# --- Model 2: KNN (best K by F1) ---
print("\nTraining KNN — Finding Best K (1 to 20)...")
f1_list = []
acc_list = []
for k in range(1, 21):
    knn_tmp = KNeighborsClassifier(n_neighbors=k)
    knn_tmp.fit(X_train, y_train)
    y_tmp = knn_tmp.predict(X_test)
    f1_list.append(f1_score(y_test, y_tmp, average='weighted'))
    acc_list.append(accuracy_score(y_test, y_tmp))

best_k   = f1_list.index(max(f1_list)) + 1
print(f"Best K = {best_k}")
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
knn_pred  = knn_model.predict(X_test)
knn_proba = knn_model.predict_proba(X_test)[:, 1]
knn_cv    = cross_val_score(knn_model, X, y, cv=cv, scoring='roc_auc').mean()

# --- Model 3: Decision Tree (best depth by F1) ---
print("\nTraining Decision Tree — Finding Best Depth (3 to 10)...")
best_depth   = 5
best_dt_f1   = 0
depth_scores = {}
for depth in range(3, 11):
    dt_tmp = DecisionTreeClassifier(max_depth=depth, class_weight='balanced', random_state=42)
    dt_tmp.fit(X_train, y_train)
    score = f1_score(y_test, dt_tmp.predict(X_test), average='weighted')
    depth_scores[depth] = round(score * 100, 2)
    if score > best_dt_f1:
        best_dt_f1 = score
        best_depth = depth

print(f"Best Depth = {best_depth}")
dt_model = DecisionTreeClassifier(max_depth=best_depth, class_weight='balanced', random_state=42)
dt_model.fit(X_train, y_train)
dt_pred  = dt_model.predict(X_test)
dt_proba = dt_model.predict_proba(X_test)[:, 1]
dt_cv    = cross_val_score(dt_model, X, y, cv=cv, scoring='roc_auc').mean()

# --- Model 4: Random Forest ---
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                  max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred  = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_cv    = cross_val_score(rf_model, X, y, cv=cv, scoring='roc_auc').mean()
print("Done!")

# ================================================================
# ---------------- METRICS TABLE ----------------
# ================================================================

def get_metrics(y_true, y_pred, y_proba, cv_auc):
    return {
        'Accuracy' : round(accuracy_score(y_true, y_pred)                      * 100, 2),
        'Precision': round(precision_score(y_true, y_pred, average='weighted') * 100, 2),
        'Recall'   : round(recall_score(y_true, y_pred, average='weighted')    * 100, 2),
        'F1 Score' : round(f1_score(y_true, y_pred, average='weighted')        * 100, 2),
        'ROC-AUC'  : round(roc_auc_score(y_true, y_proba)                      * 100, 2),
        'CV-AUC'   : round(cv_auc * 100, 2),
    }

metrics = {
    'Logistic Reg'                   : get_metrics(y_test, lr_pred,  lr_proba,  lr_cv),
    f'KNN (K={best_k})'              : get_metrics(y_test, knn_pred, knn_proba, knn_cv),
    f'Decision Tree (D={best_depth})': get_metrics(y_test, dt_pred,  dt_proba,  dt_cv),
    'Random Forest'                  : get_metrics(y_test, rf_pred,  rf_proba,  rf_cv),
}

model_names = list(metrics.keys())

# Print table
col_w = 18
print("\n" + "="*80)
print("                   MODEL PERFORMANCE COMPARISON TABLE")
print("="*80)
header = f"{'Metric':<13}" + "".join(f"{m:>{col_w}}" for m in model_names)
print(header)
print("-"*80)
for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'CV-AUC']:
    row = f"{metric:<13}" + "".join(f"{metrics[m][metric]:>{col_w-1}}%" for m in model_names)
    print(row)
print("="*80)

best_model_name = max(metrics, key=lambda m: metrics[m]['F1 Score'])
print(f"\nBEST MODEL (by F1): {best_model_name}")

# ================================================================
# -------- GRAPH 1: BAR CHART — ALL METRICS ALL MODELS ----------
# ================================================================

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'CV-AUC']
colors       = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
x     = np.arange(len(metric_names))
width = 0.2

fig, ax = plt.subplots(figsize=(15, 6))
for i, (mname, color) in enumerate(zip(model_names, colors)):
    vals = [metrics[mname][m] for m in metric_names]
    bars = ax.bar(x + i * width, vals, width, label=mname, color=color)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val}%", ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xlabel("Metrics", fontsize=12)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Model Comparison: All Metrics (4 Models)", fontsize=13, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metric_names, fontsize=10)
ax.set_ylim(0, 115)
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ================================================================
# -------- GRAPH 2: CONFUSION MATRICES (2x2 layout) -------------
# ================================================================

conf_matrices = [confusion_matrix(y_test, p) for p in [lr_pred, knn_pred, dt_pred, rf_pred]]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Confusion Matrices — All 4 Models", fontsize=14, fontweight='bold')

for ax, cm, name in zip(axes.flat, conf_matrices, model_names):
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_xticklabels(['No Churn', 'Churn'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['No Churn', 'Churn'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                    fontsize=18, fontweight='bold',
                    color='white' if cm[i][j] > cm.max() / 2 else 'black')

plt.tight_layout()
plt.show()

# ================================================================
# -------- GRAPH 3: KNN Tuning (F1 + Accuracy side by side) -----
# ================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, 21), [f * 100 for f in f1_list], marker='o', color='steelblue', linewidth=2)
axes[0].axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best K = {best_k}')
axes[0].scatter(best_k, max(f1_list) * 100, color='red', zorder=5, s=100)
axes[0].set_xlabel("K Value"); axes[0].set_ylabel("F1 Score (%)")
axes[0].set_title("KNN: F1 Score vs K Value", fontweight='bold')
axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.6)

axes[1].plot(range(1, 21), [a * 100 for a in acc_list], marker='s', color='coral', linewidth=2)
axes[1].axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best K = {best_k}')
axes[1].set_xlabel("K Value"); axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("KNN: Accuracy vs K Value", fontweight='bold')
axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.6)

plt.suptitle("KNN Hyperparameter Tuning", fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

# ================================================================
# -------- GRAPH 4: Decision Tree Depth + Feature Importance ----
# ================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

depths = list(depth_scores.keys())
scores = list(depth_scores.values())
axes[0].plot(depths, scores, marker='o', color='mediumseagreen', linewidth=2, markersize=8)
axes[0].axvline(x=best_depth, color='red', linestyle='--', linewidth=2, label=f'Best Depth = {best_depth}')
axes[0].scatter(best_depth, depth_scores[best_depth], color='red', zorder=5, s=120)
axes[0].set_xlabel("Max Depth"); axes[0].set_ylabel("F1 Score (%)")
axes[0].set_title("Decision Tree: Depth Tuning", fontweight='bold')
axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.6)

fi  = pd.Series(rf_model.feature_importances_, index=X.columns)
top = fi.sort_values(ascending=False).head(10)
bars = axes[1].barh(top.index[::-1], top.values[::-1] * 100, color='mediumpurple')
for bar, val in zip(bars, top.values[::-1] * 100):
    axes[1].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                 f"{round(val, 2)}%", va='center', fontsize=9)
axes[1].set_xlabel("Importance (%)")
axes[1].set_title("Random Forest: Top 10 Important Features", fontweight='bold')
axes[1].grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout(); plt.show()

# ================================================================
# -------- GRAPH 5: ROC CURVE — All 4 Models --------------------
# ================================================================

plt.figure(figsize=(8, 6))
combos = [
    ('Logistic Reg',         lr_proba,  'steelblue'),
    (f'KNN (K={best_k})',    knn_proba, 'coral'),
    (f'DT (D={best_depth})', dt_proba,  'mediumseagreen'),
    ('Random Forest',        rf_proba,  'mediumpurple'),
]
for name, proba, color in combos:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={round(auc,2)})")

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess')
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve — All 4 Models", fontsize=13, fontweight='bold')
plt.legend(fontsize=10); plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(); plt.show()

# ================================================================
# -------- GRAPH 6: RADAR CHART — All 4 Models ------------------
# ================================================================

categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
N      = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for mname, color in zip(model_names, colors):
    vals = [metrics[mname][m] for m in categories] + [metrics[mname][categories[0]]]
    ax.plot(angles, vals, color=color, linewidth=2, label=mname)
    ax.fill(angles, vals, color=color, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 100)
ax.set_title("Radar Chart: All Models vs All Metrics", fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=9)
plt.tight_layout(); plt.show()

# ================================================================
# -------- GRAPH 7: CV-AUC Bar — Shows model stability ----------
# ================================================================

cv_scores = {m: metrics[m]['CV-AUC'] for m in model_names}
plt.figure(figsize=(8, 5))
bars = plt.bar(cv_scores.keys(), cv_scores.values(), color=colors, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, cv_scores.values()):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{val}%", ha='center', fontsize=11, fontweight='bold')
plt.ylabel("5-Fold Cross-Validation AUC (%)", fontsize=12)
plt.title("Model Stability: 5-Fold Cross-Validation AUC", fontsize=13, fontweight='bold')
plt.ylim(0, 105)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout(); plt.show()


# ================================================================
# ----------------  USER INPUT HELPER FUNCTIONS  -----------------
# ================================================================

def get_binary_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value in [0, 1]:
                return value
            print("  Please enter only 1 (Yes) or 0 (No)")
        except ValueError:
            print("  Enter numeric value 1 or 0")

def get_float_input(prompt, min_val=0):
    while True:
        try:
            value = float(input(prompt))
            if value >= min_val:
                return value
            print(f"  Value must be >= {min_val}")
        except ValueError:
            print("  Enter a valid number")

def get_choice_input(prompt, choices):
    """Numbered menu for mutually exclusive options (contract, internet, payment)."""
    print(prompt)
    for i, (label, _) in enumerate(choices, 1):
        print(f"  {i}. {label}")
    while True:
        try:
            val = int(input("  Enter choice number: "))
            if 1 <= val <= len(choices):
                return choices[val - 1][1]   # return the dict of one-hot flags
            print(f"  Please enter a number between 1 and {len(choices)}")
        except ValueError:
            print("  Enter a valid number")


# ================================================================
# ----------------  PREDICTION LOOP  ----------------------------
# ================================================================

def run_prediction_loop(all_models_map, X_columns, scaler, numeric_cols,
                        best_k, best_depth, best_model_name):
    while True:
        print("\n" + "="*62)
        print("         CUSTOMER CHURN PREDICTION TOOL")
        print("="*62)
        print("  Enter customer details below.")
        print("  Press Ctrl+C at any time to quit.\n")

        # ── Numeric inputs ──────────────────────────────────────────
        tenure          = get_float_input("  Tenure          (months, e.g. 24)   : ", min_val=0)
        monthly_charges = get_float_input("  Monthly Charges (e.g. 65.50)        : ", min_val=0)
        total_charges   = get_float_input("  Total Charges   (e.g. 1500)         : ", min_val=0)

        # ── Personal details ────────────────────────────────────────
        print("\n  -- Personal Details (1 = Yes / 0 = No) --")
        gender            = get_binary_input("  Gender            (Male=1, Female=0) : ")
        partner           = get_binary_input("  Partner                              : ")
        dependents        = get_binary_input("  Dependents                           : ")
        paperless_billing = get_binary_input("  Paperless Billing                    : ")

        # ── Phone & internet ────────────────────────────────────────
        print("\n  -- Phone & Internet --")
        phone_service  = get_binary_input("  Phone Service                        : ")
        multiple_lines = get_binary_input("  Multiple Lines                       : ")

        internet_flags = get_choice_input(
            "\n  Internet Service type:",
            [
                ("DSL",         {"InternetService_Fiber optic": 0, "InternetService_No": 0}),
                ("Fiber Optic", {"InternetService_Fiber optic": 1, "InternetService_No": 0}),
                ("No Internet", {"InternetService_Fiber optic": 0, "InternetService_No": 1}),
            ]
        )

        # ── Add-on services ─────────────────────────────────────────
        print("\n  -- Add-on Services (1 = Yes / 0 = No) --")
        online_security   = get_binary_input("  Online Security                      : ")
        online_backup     = get_binary_input("  Online Backup                        : ")
        device_protection = get_binary_input("  Device Protection                    : ")
        tech_support      = get_binary_input("  Tech Support                         : ")
        streaming_tv      = get_binary_input("  Streaming TV                         : ")
        streaming_movies  = get_binary_input("  Streaming Movies                     : ")

        # ── Contract & payment ──────────────────────────────────────
        contract_flags = get_choice_input(
            "\n  Contract Type:",
            [
                ("Month-to-Month", {"Contract_One year": 0, "Contract_Two year": 0}),
                ("One Year",       {"Contract_One year": 1, "Contract_Two year": 0}),
                ("Two Year",       {"Contract_One year": 0, "Contract_Two year": 1}),
            ]
        )

        payment_flags = get_choice_input(
            "\n  Payment Method:",
            [
                ("Bank Transfer (auto)", {"PaymentMethod_Credit card (automatic)": 0,
                                          "PaymentMethod_Electronic check": 0,
                                          "PaymentMethod_Mailed check": 0}),
                ("Credit Card (auto)",   {"PaymentMethod_Credit card (automatic)": 1,
                                          "PaymentMethod_Electronic check": 0,
                                          "PaymentMethod_Mailed check": 0}),
                ("Electronic Check",     {"PaymentMethod_Credit card (automatic)": 0,
                                          "PaymentMethod_Electronic check": 1,
                                          "PaymentMethod_Mailed check": 0}),
                ("Mailed Check",         {"PaymentMethod_Credit card (automatic)": 0,
                                          "PaymentMethod_Electronic check": 0,
                                          "PaymentMethod_Mailed check": 1}),
            ]
        )

        # ── Build user dataframe ────────────────────────────────────
        user_data = pd.DataFrame(0, index=[0], columns=X_columns)

        user_data['gender']           = gender
        user_data['Partner']          = partner
        user_data['Dependents']       = dependents
        user_data['PhoneService']     = phone_service
        user_data['MultipleLines']    = multiple_lines
        user_data['OnlineSecurity']   = online_security
        user_data['OnlineBackup']     = online_backup
        user_data['DeviceProtection'] = device_protection
        user_data['TechSupport']      = tech_support
        user_data['StreamingTV']      = streaming_tv
        user_data['StreamingMovies']  = streaming_movies
        user_data['PaperlessBilling'] = paperless_billing
        user_data['tenure']           = tenure
        user_data['MonthlyCharges']   = monthly_charges
        user_data['TotalCharges']     = total_charges

        # Apply one-hot flag dicts
        for col, val in {**internet_flags, **contract_flags, **payment_flags}.items():
            if col in user_data.columns:
                user_data[col] = val

        # Scale numeric columns
        user_data[numeric_cols] = scaler.transform(user_data[numeric_cols])

        # ── Predictions from all 4 models ───────────────────────────
        print("\n" + "="*62)
        print("        CHURN PREDICTIONS — ALL 4 MODELS")
        print("="*62)

        votes      = []
        proba_list = []

        for model_label, model in all_models_map.items():
            pred  = model.predict(user_data)[0]
            proba = model.predict_proba(user_data)[0][1]
            votes.append(pred)
            proba_list.append(proba)
            verdict = "YES — WILL CHURN ⚠️" if pred == 1 else "NO  — WILL STAY  ✅"
            print(f"\n  {model_label}")
            print(f"    Prediction       : {verdict}")
            print(f"    Churn Probability: {round(proba * 100, 2)}%")

        # ── Majority vote & summary ─────────────────────────────────
        churn_votes   = sum(votes)
        avg_proba     = round(np.mean(proba_list) * 100, 2)
        final_verdict = "⚠️  WILL CHURN" if churn_votes >= 3 else "✅  WILL STAY"
        risk_level    = "🔴 HIGH RISK"   if avg_proba > 60 else \
                        "🟡 MEDIUM RISK" if avg_proba > 35 else \
                        "🟢 LOW RISK"

        print("\n" + "="*62)
        print("            FINAL COMBINED PREDICTION")
        print("="*62)
        print(f"  Models predicting churn  : {churn_votes} / 4")
        print(f"  Average churn probability: {avg_proba}%")
        print(f"  Risk Level               : {risk_level}")
        print(f"  Final Verdict            : {final_verdict}")
        print(f"  Best Individual Model    : {best_model_name}")
        print("="*62)

        # ── Graph 8: Per-model churn probability bars ───────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        mnames     = list(all_models_map.keys())
        probas     = [round(p * 100, 2) for p in proba_list]
        bar_colors = ['tomato' if p > 50 else 'steelblue' for p in probas]

        bars = ax.barh(mnames, probas, color=bar_colors, edgecolor='black', linewidth=0.7)
        ax.axvline(x=50, color='red', linestyle='--', linewidth=1.5, label='50% Threshold')

        for bar, val in zip(bars, probas):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val}%", va='center', fontsize=11, fontweight='bold')

        ax.set_xlabel("Churn Probability (%)", fontsize=12)
        ax.set_title(
            f"Customer Churn Probability by Model\n"
            f"Final: {final_verdict} ({avg_proba}% avg) — {risk_level}",
            fontsize=12, fontweight='bold'
        )
        ax.set_xlim(0, 115)
        ax.legend(fontsize=10)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # ── Ask to predict another customer ────────────────────────
        print("\n  Predict another customer?")
        again = input("  Enter 'y' to continue or any other key to exit: ").strip().lower()
        if again != 'y':
            print("\n  Goodbye!\n")
            break


# ================================================================
# -------- RUN PREDICTION LOOP ----------------------------------
# ================================================================

all_models_map = {
    "Logistic Regression"              : lr_model,
    f"KNN (K={best_k})"               : knn_model,
    f"Decision Tree (D={best_depth})" : dt_model,
    "Random Forest"                   : rf_model,
}

try:
    run_prediction_loop(
        all_models_map  = all_models_map,
        X_columns       = X.columns,
        scaler          = scaler,
        numeric_cols    = numeric_cols,
        best_k          = best_k,
        best_depth      = best_depth,
        best_model_name = best_model_name,
    )
except KeyboardInterrupt:
    print("\n\n  Interrupted. Goodbye!\n")
