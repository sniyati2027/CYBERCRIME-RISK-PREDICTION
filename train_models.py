import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

print("Loading Data...")
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train_loc = pd.read_csv('y_train.csv').values.ravel()
y_test_loc = pd.read_csv('y_test.csv').values.ravel()

# --- Task 1: Predict High Risk (Binary Classification) ---
print("\n=== Task 1: Predicting 'is_high_risk' ===")

# Create target variable for risk (assuming is_high_risk is in X)
# We need to extract it and remove it from features to prevent leakage if we were doing a real "unknown" prediction, 
# but here we want to see if the model learns the rule.
if 'is_high_risk' in X_train.columns:
    y_train_risk = X_train['is_high_risk']
    y_test_risk = X_test['is_high_risk']
    # Drop from features for this specific task? 
    # The prompt asks to "Identify top contributing features". 
    # If we keep 'fraud_amount' and 'time_to_withdrawal' (which define risk), they should be top.
    # We remove 'is_high_risk' from the input features itself (obviously).
    X_train_risk = X_train.drop(columns=['is_high_risk'])
    X_test_risk = X_test.drop(columns=['is_high_risk'])
else:
    print("Error: 'is_high_risk' column not found in features.")
    exit()

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_risk, y_train_risk)

# Evaluate
y_pred_risk = lr_model.predict(X_test_risk)
acc_risk = accuracy_score(y_test_risk, y_pred_risk)
print(f"Logistic Regression Accuracy: {acc_risk:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test_risk, y_pred_risk))
print("\nTop Contributing Features (Coefficients):")
coeffs = pd.DataFrame({'Feature': X_train_risk.columns, 'Coefficient': lr_model.coef_[0]})
print(coeffs.sort_values(by='Coefficient', key=abs, ascending=False).head(5))


# --- Task 2: Predict Withdrawal City (Multiclass Classification) ---
print("\n=== Task 2: Predicting 'withdrawal_city' ===")

# Features: Use all available features (including is_high_risk if helpful)
# We use X_train directly (which includes is_high_risk)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_loc)

# Evaluate
y_pred_loc = rf_model.predict(X_test)
acc_loc = accuracy_score(y_test_loc, y_pred_loc)
print(f"Random Forest Accuracy: {acc_loc:.4f}")

print("\nTop Important Features:")
importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf_model.feature_importances_})
print(importances.sort_values(by='Importance', ascending=False).head(5))

print("\nDetailed Classification Report (Top Cities):")
# Filter report for readability if many classes
print(classification_report(y_test_loc, y_pred_loc))
