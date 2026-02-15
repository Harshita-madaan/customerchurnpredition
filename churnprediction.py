import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

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

df_copy[binary_cols] = df_copy[binary_cols].replace({
    'No Internet Service':'No'
})

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

# ---------------- TRAIN MODEL ----------------

X = df_copy.drop('Churn', axis=1)
y = df_copy['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n📊 Classification Report")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# ---------------- CONFUSION MATRIX GRAPH ----------------

plt.figure()
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0,1], ['No Churn','Churn'])
plt.yticks([0,1], ['No Churn','Churn'])
plt.show()

# ---------------- CHURN DISTRIBUTION GRAPH ----------------

churn_counts = df_copy['Churn'].value_counts()

plt.figure()
plt.bar(['No Churn','Churn'], churn_counts.values)
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.title("Churn Distribution in Telecom Dataset")
plt.show()

# ---------------- USER INPUT ----------------

print("\n🔎 Enter Customer Details To Predict Churn:\n")

tenure = float(input("Enter Tenure (in months): "))
monthly_charges = float(input("Enter Monthly Charges: "))
total_charges = float(input("Enter Total Charges: "))

gender = get_binary_input("Gender (Male=1, Female=0): ")
partner = get_binary_input("Partner (Yes=1, No=0): ")
dependents = get_binary_input("Dependents (Yes=1, No=0): ")
phone_service = get_binary_input("Phone Service (Yes=1, No=0): ")
multiple_lines = get_binary_input("Multiple Lines (Yes=1, No=0): ")
online_security = get_binary_input("Online Security (Yes=1, No=0): ")
online_backup = get_binary_input("Online Backup (Yes=1, No=0): ")
device_protection = get_binary_input("Device Protection (Yes=1, No=0): ")
tech_support = get_binary_input("Tech Support (Yes=1, No=0): ")
streaming_tv = get_binary_input("Streaming TV (Yes=1, No=0): ")
streaming_movies = get_binary_input("Streaming Movies (Yes=1, No=0): ")
paperless_billing = get_binary_input("Paperless Billing (Yes=1, No=0): ")

internet_fiber = get_binary_input("Internet Service Fiber Optic (Yes=1, No=0): ")
internet_no = get_binary_input("Internet Service No (Yes=1, No=0): ")

contract_one_year = get_binary_input("Contract One Year (Yes=1, No=0): ")
contract_two_year = get_binary_input("Contract Two Year (Yes=1, No=0): ")

payment_credit = get_binary_input("Payment Method Credit Card (Yes=1, No=0): ")
payment_electronic = get_binary_input("Payment Method Electronic Check (Yes=1, No=0): ")
payment_mailed = get_binary_input("Payment Method Mailed Check (Yes=1, No=0): ")

# ---------------- CREATE USER DATA ----------------

user_data = pd.DataFrame(0, index=[0], columns=X.columns)

user_data['gender'] = gender
user_data['Partner'] = partner
user_data['Dependents'] = dependents
user_data['tenure'] = tenure
user_data['PhoneService'] = phone_service
user_data['MultipleLines'] = multiple_lines
user_data['OnlineSecurity'] = online_security
user_data['OnlineBackup'] = online_backup
user_data['DeviceProtection'] = device_protection
user_data['TechSupport'] = tech_support
user_data['StreamingTV'] = streaming_tv
user_data['StreamingMovies'] = streaming_movies
user_data['PaperlessBilling'] = paperless_billing
user_data['MonthlyCharges'] = monthly_charges
user_data['TotalCharges'] = total_charges

for col in user_data.columns:
    if "Fiber optic" in col:
        user_data[col] = internet_fiber
    if "InternetService_No" in col:
        user_data[col] = internet_no
    if "One year" in col:
        user_data[col] = contract_one_year
    if "Two year" in col:
        user_data[col] = contract_two_year
    if "Credit card" in col:
        user_data[col] = payment_credit
    if "Electronic check" in col:
        user_data[col] = payment_electronic
    if "Mailed check" in col:
        user_data[col] = payment_mailed

user_data[numeric_cols] = scaler.transform(user_data[numeric_cols])

# ---------------- PREDICTION ----------------

prediction = model.predict(user_data)
probability = model.predict_proba(user_data)[0][1]

print("\n📢 Final Prediction:")

if prediction[0] == 1:
    print("⚠ YES — Customer WILL CHURN")
else:
    print("✅ NO — Customer WILL STAY")

print(f"Churn Probability: {round(probability*100,2)}%")
