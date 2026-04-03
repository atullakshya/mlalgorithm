"""
================================================================================
SUPPORT VECTOR MACHINE (SVM) - Real-World Example: Cancer Diagnosis
================================================================================
REAL-TIME USE CASE:
    A hospital wants to classify tumors as Malignant or Benign based on cell
    measurements from biopsy images (radius, texture, perimeter, area, smoothness).

ALGORITHM:
    SVM finds the OPTIMAL HYPERPLANE that separates two classes with the maximum
    margin. Uses kernel trick (RBF) to handle non-linearly separable data.

MODEL TYPE AFTER TRAINING:
    -> A set of SUPPORT VECTORS (key data points near the decision boundary),
       plus kernel parameters and bias.
    -> The model stores actual training samples that define the boundary.
    -> Saved as .pkl, contains support vectors + learned alpha weights.
    -> NON-PARAMETRIC in the sense that #support vectors depends on data.
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (tumor measurements)
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nDiagnosis distribution:\n{df['Diagnosis'].value_counts().to_string()}")

    # -------------------------------------------------------------------------
    # STEP 2: Encode target and split features
    # -------------------------------------------------------------------------
    le = LabelEncoder()
    df["Target"] = le.fit_transform(df["Diagnosis"])

    feature_cols = ["MeanRadius", "MeanTexture", "MeanPerimeter", "MeanArea", "MeanSmoothness"]
    X = df[feature_cols].values
    y = df["Target"].values

    print(f"\n=== STEP 2: Encoding ===")
    print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # -------------------------------------------------------------------------
    # STEP 3: Train/Test split
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n=== STEP 3: Train: {len(X_train)}, Test: {len(X_test)} ===")

    # -------------------------------------------------------------------------
    # STEP 4: Feature scaling (CRITICAL for SVM - it uses distance calculations)
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"\n=== STEP 4: Features Scaled (StandardScaler) ===")

    # -------------------------------------------------------------------------
    # STEP 5: Train SVM with RBF kernel
    # C=10: controls trade-off between smooth boundary and classifying correctly
    # kernel='rbf': maps data to higher dimensions for non-linear separation
    # probability=True: enables probability output (slower but useful)
    # -------------------------------------------------------------------------
    model = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    print(f"\n=== STEP 5: Model Training Complete ===")
    print(f"Model type         : {type(model).__name__}")
    print(f"Kernel             : {model.kernel}")
    print(f"C (regularization) : {model.C}")
    print(f"Support vectors    : {model.n_support_} (per class)")
    print(f"Total support vec. : {len(model.support_vectors_)} out of {len(X_train)} training samples")
    print(f"  -> These {len(model.support_vectors_)} points DEFINE the decision boundary")

    # -------------------------------------------------------------------------
    # STEP 6: Predict and show probabilities
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    print(f"\n=== STEP 6: Predictions ===")
    print(f"{'Actual':>12s} | {'Predicted':>12s} | {'P(Benign)':>10s} | {'P(Malignant)':>12s}")
    print("-" * 55)
    for actual, pred, prob in zip(y_test, y_pred, y_proba):
        a_name = le.inverse_transform([actual])[0]
        p_name = le.inverse_transform([pred])[0]
        print(f"{a_name:>12s} | {p_name:>12s} | {prob[0]:>9.2%} | {prob[1]:>11.2%}")

    # -------------------------------------------------------------------------
    # STEP 7: Evaluate
    # -------------------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== STEP 7: Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

    # -------------------------------------------------------------------------
    # STEP 8: Save model
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "svm_model.pkl")
    joblib.dump({"model": model, "scaler": scaler, "label_encoder": le}, model_path)

    print(f"=== STEP 8: Model Saved ===")
    print(f"Saved to: {model_path}")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type     : SVM with RBF Kernel (Support Vector Classifier)")
    print(f"Contents : {len(model.support_vectors_)} support vectors + kernel params + bias")
    print(f"How      : Stores KEY training points that define the decision boundary")
    print(f"Predict  : Maps new point to high-dim space, checks which side of boundary")
    print(f"Size     : Medium (~few KB) - stores subset of training data")

if __name__ == "__main__":
    main()
