"""
================================================================================
LOGISTIC REGRESSION - Real-World Example: Loan Default Prediction
================================================================================
REAL-TIME USE CASE:
    A bank wants to predict whether a loan applicant will DEFAULT (fail to repay)
    based on age, income, credit score, loan amount, and employment years.

ALGORITHM:
    Logistic Regression applies a sigmoid function to a linear combination of features
    to output a PROBABILITY between 0 and 1. If probability > 0.5 -> class 1 (default).

MODEL TYPE AFTER TRAINING:
    -> A set of COEFFICIENTS (weights) + INTERCEPT, same as linear regression,
       but passed through a SIGMOID function to get probabilities.
    -> Formula: P(default) = sigmoid(w1*Age + w2*Income + ... + b)
    -> Saved as .pkl, contains weights that map features to default probability.
    -> PARAMETRIC model, suitable for binary and multi-class classification.
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (loan default data)
    # Each row is a loan applicant with financial features and default status
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nTarget distribution:\n{df['Defaulted'].value_counts().to_string()}")
    print(f"  0 = No Default (good), 1 = Defaulted (bad)")

    # -------------------------------------------------------------------------
    # STEP 2: Separate features and target
    # -------------------------------------------------------------------------
    feature_cols = ["Age", "AnnualIncome_USD", "CreditScore", "LoanAmount_USD", "EmploymentYears"]
    X = df[feature_cols].values
    y = df["Defaulted"].values

    print(f"\n=== STEP 2: Feature/Target Split ===")
    print(f"Features: {feature_cols}")
    print(f"Target  : Defaulted (0 or 1)")

    # -------------------------------------------------------------------------
    # STEP 3: Train/Test split - 80/20
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n=== STEP 3: Train/Test Split ===")
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # -------------------------------------------------------------------------
    # STEP 4: Feature scaling
    # Logistic Regression converges faster with scaled features
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"\n=== STEP 4: Feature Scaling Done ===")

    # -------------------------------------------------------------------------
    # STEP 5: Train Logistic Regression
    # The model learns: log(P/(1-P)) = w1*x1 + w2*x2 + ... + b
    # Then converts to probability: P = 1 / (1 + e^(-z))
    # -------------------------------------------------------------------------
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    print(f"\n=== STEP 5: Model Training Complete ===")
    print(f"Model type: {type(model).__name__}")
    print(f"\nLearned Coefficients (how each feature affects default probability):")
    for feat, coef in zip(feature_cols, model.coef_[0]):
        effect = "INCREASES default risk" if coef > 0 else "DECREASES default risk"
        print(f"  {feat:25s} -> Weight = {coef:>8.4f} ({effect})")
    print(f"  {'Intercept':25s} -> Value  = {model.intercept_[0]:>8.4f}")

    # -------------------------------------------------------------------------
    # STEP 6: Predict on test set (both class and probability)
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    print(f"\n=== STEP 6: Predictions ===")
    print(f"{'Actual':>8s} | {'Predicted':>10s} | {'P(No Default)':>14s} | {'P(Default)':>12s}")
    print("-" * 52)
    for actual, pred, prob in zip(y_test, y_pred, y_proba):
        label = "Default" if pred == 1 else "No Def."
        print(f"{'Default' if actual==1 else 'No Def.':>8s} | {label:>10s} | {prob[0]:>13.2%} | {prob[1]:>11.2%}")

    # -------------------------------------------------------------------------
    # STEP 7: Evaluate the model
    # Confusion Matrix: shows true/false positives and negatives
    # -------------------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== STEP 7: Evaluation ===")
    print(f"Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted No Default  Predicted Default")
    print(f"  Actual No Def.      {cm[0][0]:>5d}              {cm[0][1]:>5d}")
    print(f"  Actual Default      {cm[1][0]:>5d}              {cm[1][1]:>5d}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Default','Default'], zero_division=0)}")

    # -------------------------------------------------------------------------
    # STEP 8: Save the trained model
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "logistic_regression_model.pkl")
    joblib.dump({"model": model, "scaler": scaler, "features": feature_cols}, model_path)

    print(f"=== STEP 8: Model Saved ===")
    print(f"Saved to: {model_path}")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type       : Logistic Regression (Parametric Binary Classifier)")
    print(f"Contents   : {len(model.coef_[0])} coefficients + 1 intercept + sigmoid activation")
    print(f"Output     : Probability of default (0.0 to 1.0)")
    print(f"Decision   : If P(default) > 0.5 -> predict 'Default', else 'No Default'")
    print(f"Parameters : {len(model.coef_[0])+1} total (tiny model, fast predictions)")

if __name__ == "__main__":
    main()
