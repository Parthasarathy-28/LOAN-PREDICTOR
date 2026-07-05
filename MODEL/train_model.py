import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ---------------- LOAD DATASET ---------------- #

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "Loan_Default.csv")

print("Trying to read from:", file_path)

df = pd.read_csv(file_path)

# ---------------- BASIC DATASET INFO ---------------- #

print("\nFirst 5 rows:\n")
print(df.head())

print("\nColumn names:\n")
print(df.columns)

print("\nMissing values before cleaning:\n")
print(df.isnull().sum())

print("\nShape of dataset before cleaning:\n")
print(df.shape)

# ---------------- DATA CLEANING ---------------- #

df = df.drop_duplicates()

# Fill numeric columns with median
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after cleaning:\n")
print(df.isnull().sum())

print("\nShape of dataset after cleaning:\n")
print(df.shape)

# ---------------- SAVE CLEAN COPY FOR TABLEAU ---------------- #

df_tableau = df.copy()

# Analytics columns for Tableau
df_tableau["Income_Group"] = pd.cut(
    df_tableau["income"],
    bins=[0, 30000, 70000, 1000000],
    labels=["Low Income", "Medium Income", "High Income"]
)

df_tableau["CreditScore_Group"] = pd.cut(
    df_tableau["Credit_Score"],
    bins=[0, 600, 700, 800, 900],
    labels=["Poor", "Fair", "Good", "Excellent"]
)

df_tableau["LoanAmount_Group"] = pd.cut(
    df_tableau["loan_amount"],
    bins=[0, 100000, 300000, 10000000],
    labels=["Small Loan", "Medium Loan", "Large Loan"]
)

df_tableau["LTV_Group"] = pd.cut(
    df_tableau["LTV"],
    bins=[0, 60, 80, 100],
    labels=["Low LTV", "Moderate LTV", "High LTV"]
)

df_tableau["DTI_Group"] = pd.cut(
    df_tableau["dtir1"],
    bins=[0, 30, 40, 100],
    labels=["Safe DTI", "Moderate DTI", "Risky DTI"]
)

tableau_path = os.path.join(current_dir, "..", "data", "loan_tableau.csv")
df_tableau.to_csv(tableau_path, index=False)
print("\nTableau dataset saved successfully!")

# ---------------- TARGET COLUMN ---------------- #

target_column = "Status"

print("\nTarget column:", target_column)
print("\nTarget value counts:\n")
print(df[target_column].value_counts())

# ---------------- FEATURE ENCODING ---------------- #

df_encoded = pd.get_dummies(df, drop_first=True)

# ---------------- SPLIT INTO X AND y ---------------- #

X = df_encoded.drop(target_column, axis=1)
y = df_encoded[target_column]

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)

# ---------------- TRAIN TEST SPLIT ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# ---------------- TRAIN MODELS ---------------- #

lr = LogisticRegression(max_iter=2000, class_weight="balanced")
dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2
)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ---------------- EVALUATE MODELS ---------------- #

models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf
}

best_model = None
best_recall = -1

for name, model in models.items():
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{name}")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print(classification_report(y_test, y_pred, zero_division=0))

    # choose best model based on recall first, then F1 if needed
    if rec > best_recall:
        best_recall = rec
        best_model = model

# ---------------- SAVE BEST MODEL ---------------- #

model_path = os.path.join(current_dir, "..", "model", "loan_model.pkl")
feature_path = os.path.join(current_dir, "..", "model", "feature_columns.pkl")

with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

with open(feature_path, "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("\nModel saved successfully!")
print("Feature columns saved successfully!")

# ---------------- DEBUG TEST PREDICTIONS ---------------- #

print("\nTesting model predictions on first 10 test rows:")
test_preds = best_model.predict(X_test[:10])
print(test_preds)

if hasattr(best_model, "predict_proba"):
    print("\nPrediction probabilities on first 10 test rows:")
    print(best_model.predict_proba(X_test[:10])[:, 1])