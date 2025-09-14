import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

df = pd.read_csv("loan_data.csv")

print(df.head())
print(df.info())

X = df.drop(columns=["customer_id", "default"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
log_reg_probs = log_reg.predict_proba(X_test_scaled)[:, 1]
print("Logistic Regression AUC:", roc_auc_score(y_test, log_reg_probs))

rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
print("Random Forest AUC:", roc_auc_score(y_test, rf_probs))

print("\nRandom Forest Report:")
print(classification_report(y_test, rf.predict(X_test_scaled)))

def expected_loss(model, scaler, borrower_info, recovery_rate=0.1):
    """
    borrower_info: dict with borrower features
    model: trained ML model
    scaler: trained scaler
    """
    borrower_df = pd.DataFrame([borrower_info])
    borrower_scaled = scaler.transform(borrower_df)
    pd_default = model.predict_proba(borrower_scaled)[:, 1][0]
    
    EAD = borrower_info["loan_amt_outstanding"]
    LGD = 1 - recovery_rate
    EL = pd_default * LGD * EAD
    
    return {
        "PD": pd_default,
        "Expected_Loss": EL
    }


borrower = {
    "credit_lines_outstanding": 4,
    "loan_amt_outstanding": 15000,
    "total_debt_outstanding": 30000,
    "income": 55000,
    "years_employed": 3,
    "fico_score": 680
}

print("\nExample Expected Loss:")
print(expected_loss(rf, scaler, borrower))
