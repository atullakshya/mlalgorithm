"""
================================================================================
K-NEAREST NEIGHBORS (KNN) - Real-World Example: Diabetes Prediction
================================================================================
REAL-TIME USE CASE:
    A healthcare system predicts whether a patient has diabetes based on medical
    measurements: glucose, blood pressure, BMI, age, insulin levels.

ALGORITHM:
    KNN makes predictions by finding the K closest training samples to a new point
    (using Euclidean distance) and voting on the class. No learning phase - it
    memorizes all training data (lazy learner).

MODEL TYPE AFTER TRAINING:
    -> THE ENTIRE TRAINING DATASET is the model.
    -> No weights, no rules, no tree - just stored data points.
    -> At prediction time, it computes distances to ALL training points.
    -> Saved as .pkl, contains all training data + K value.
    -> NON-PARAMETRIC, INSTANCE-BASED (lazy) model.
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (diabetes patient data)
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nDiabetes distribution:\n{df['DiabetesOutcome'].value_counts().to_string()}")
    print(f"  0 = No Diabetes, 1 = Diabetes")

    # -------------------------------------------------------------------------
    # STEP 2: Feature/Target split
    # -------------------------------------------------------------------------
    feature_cols = ["Glucose", "BloodPressure", "BMI", "Age", "Insulin"]
    X = df[feature_cols].values
    y = df["DiabetesOutcome"].values
    print(f"\n=== STEP 2: Features: {feature_cols} ===")

    # -------------------------------------------------------------------------
    # STEP 3: Train/Test split
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n=== STEP 3: Train: {len(X_train)}, Test: {len(X_test)} ===")

    # -------------------------------------------------------------------------
    # STEP 4: Feature scaling (CRITICAL for KNN - uses distance calculations)
    # Without scaling, features with large ranges dominate distance computation
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"\n=== STEP 4: Features Scaled (essential for KNN) ===")

    # -------------------------------------------------------------------------
    # STEP 5: Find the best K by trying different values
    # K = number of neighbors to consult for each prediction
    # Small K = noisy, Large K = too smooth
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 5: Finding Best K ===")
    results = []
    for k in range(1, 16):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test_scaled))
        results.append((k, acc))
        bar = "#" * int(acc * 40)
        print(f"  K={k:2d}: Accuracy={acc:.4f} {bar}")

    best_k = max(results, key=lambda x: x[1])[0]
    print(f"\n  Best K = {best_k}")

    # -------------------------------------------------------------------------
    # STEP 6: Train final model with best K
    # "Training" for KNN = simply storing the data (lazy learner)
    # -------------------------------------------------------------------------
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_scaled, y_train)

    print(f"\n=== STEP 6: Model 'Trained' (data stored) ===")
    print(f"Model type         : {type(model).__name__}")
    print(f"K (neighbors)      : {best_k}")
    print(f"Training samples   : {len(X_train)} (ALL stored as the model)")
    print(f"Distance metric    : Euclidean (default)")
    print(f"  -> KNN does NOT learn parameters. It memorizes training data.")

    # -------------------------------------------------------------------------
    # STEP 7: Predict and show neighbor details
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    distances, indices = model.kneighbors(X_test_scaled)

    print(f"\n=== STEP 7: Predictions (showing nearest neighbors) ===")
    for i in range(min(4, len(X_test))):
        actual = "Diabetes" if y_test[i] == 1 else "No Diab."
        predicted = "Diabetes" if y_pred[i] == 1 else "No Diab."
        neighbor_labels = y_train[indices[i]]
        votes = f"{sum(neighbor_labels)}/{best_k} say Diabetes"
        print(f"  Patient {i+1}: Actual={actual:>10s}, Predicted={predicted:>10s} ({votes})")
        print(f"    Neighbor distances: {distances[i].round(2)}")

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Diabetes','Diabetes'])}")

    # -------------------------------------------------------------------------
    # STEP 8: Save model
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "knn_model.pkl")
    joblib.dump({"model": model, "scaler": scaler, "features": feature_cols, "k": best_k}, model_path)

    print(f"=== STEP 8: Model Saved ===")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type    : KNN (Instance-Based / Lazy Learner)")
    print(f"Contents: ALL {len(X_train)} training samples stored in memory")
    print(f"No      : coefficients, weights, rules, or trees - just raw data")
    print(f"Predict : Find {best_k} closest stored samples, majority vote wins")
    print(f"Size    : Grows with training data (large dataset = large model)")
    print(f"Speed   : Slow prediction (must compare to all stored points)")

if __name__ == "__main__":
    main()
