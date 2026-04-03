"""
================================================================================
DBSCAN CLUSTERING - Real-World Example: Credit Card Fraud Detection (Anomaly)
================================================================================
REAL-TIME USE CASE:
    A bank wants to detect fraudulent credit card transactions by finding
    OUTLIERS (anomalies) that don't belong to any normal spending cluster.
    Fraudulent transactions have unusual amounts, times, and distances.

ALGORITHM:
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
    - Groups points that are closely packed together (core points)
    - Points in sparse regions are labeled as NOISE (outliers/anomalies)
    - Does NOT require specifying K (number of clusters) in advance
    - Finds clusters of ARBITRARY SHAPE (unlike K-Means which finds spherical clusters)

MODEL TYPE AFTER TRAINING:
    -> CLUSTER LABELS + NOISE LABELS (-1 for outliers).
    -> Core samples (dense region points) and their neighborhoods.
    -> No centroids, no tree - just density-based assignments.
    -> Saved as .pkl, contains labels and core sample indices.
    -> UNSUPERVISED, density-based, non-parametric model.
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import os

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (credit card transactions)
    # Normal transactions: small amounts, daytime, close to home, low risk
    # Fraudulent: large amounts, late night, far from home, high risk
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nAmount stats:\n{df['Amount_USD'].describe()}")

    # -------------------------------------------------------------------------
    # STEP 2: Select features for anomaly detection
    # -------------------------------------------------------------------------
    feature_cols = ["Amount_USD", "Hour", "DistanceFromHome_km", "MerchantRiskScore"]
    X = df[feature_cols].values
    print(f"\n=== STEP 2: Features: {feature_cols} ===")

    # -------------------------------------------------------------------------
    # STEP 3: Feature scaling (CRITICAL for DBSCAN - uses distance)
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n=== STEP 3: Features Scaled ===")

    # -------------------------------------------------------------------------
    # STEP 4: Parameter search for DBSCAN
    # eps = maximum distance between two points to be "neighbors"
    # min_samples = minimum points needed to form a dense region (core point)
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 4: Parameter Search ===")
    print(f"{'eps':>5s} | {'min_s':>5s} | {'Clusters':>8s} | {'Noise':>5s} | {'Silhouette':>10s}")
    print("-" * 45)
    best_params = {"eps": 0.5, "min_samples": 3}
    best_sil = -1
    for eps in [0.3, 0.5, 0.8, 1.0]:
        for min_s in [2, 3, 5]:
            db = DBSCAN(eps=eps, min_samples=min_s)
            labels = db.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            if n_clusters >= 2:
                sil = silhouette_score(X_scaled[labels != -1], labels[labels != -1])
                if sil > best_sil:
                    best_sil = sil
                    best_params = {"eps": eps, "min_samples": min_s}
                print(f" {eps:>4.1f} | {min_s:>5d} | {n_clusters:>8d} | {n_noise:>5d} | {sil:>10.4f}")
            else:
                print(f" {eps:>4.1f} | {min_s:>5d} | {n_clusters:>8d} | {n_noise:>5d} | {'N/A':>10s}")

    # -------------------------------------------------------------------------
    # STEP 5: Train final DBSCAN model
    # -------------------------------------------------------------------------
    model = DBSCAN(**best_params)
    labels = model.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    print(f"\n=== STEP 5: Model Training Complete ===")
    print(f"Model type      : {type(model).__name__}")
    print(f"eps             : {best_params['eps']}")
    print(f"min_samples     : {best_params['min_samples']}")
    print(f"Clusters found  : {n_clusters}")
    print(f"Noise (outliers): {n_noise} -> POTENTIAL FRAUD!")
    print(f"Core samples    : {len(model.core_sample_indices_)}")

    # -------------------------------------------------------------------------
    # STEP 6: Analyze clusters and anomalies
    # -------------------------------------------------------------------------
    df["Cluster"] = labels

    print(f"\n=== STEP 6: Cluster & Anomaly Analysis ===")
    for c in sorted(df["Cluster"].unique()):
        mask = df["Cluster"] == c
        subset = df[mask]
        if c == -1:
            label = "NOISE/ANOMALY (Potential Fraud)"
        else:
            label = f"Normal Cluster {c}"
        print(f"\n  {label} ({mask.sum()} transactions):")
        print(f"    Avg Amount     : ${subset['Amount_USD'].mean():>10,.2f}")
        print(f"    Avg Hour       : {subset['Hour'].mean():>10.1f}")
        print(f"    Avg Distance   : {subset['DistanceFromHome_km'].mean():>10.1f} km")
        print(f"    Avg Risk Score : {subset['MerchantRiskScore'].mean():>10.3f}")

    # -------------------------------------------------------------------------
    # STEP 7: Flag suspicious transactions
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 7: Flagged Transactions ===")
    anomalies = df[df["Cluster"] == -1]
    if len(anomalies) > 0:
        print(f"ALERT: {len(anomalies)} suspicious transactions detected!")
        for _, row in anomalies.iterrows():
            print(f"  TX #{int(row['TransactionID'])}: ${row['Amount_USD']:,.2f} at {int(row['Hour'])}:00, "
                  f"{row['DistanceFromHome_km']:.0f}km from home, risk={row['MerchantRiskScore']:.2f}")
    else:
        print("No anomalies detected.")

    # -------------------------------------------------------------------------
    # STEP 8: Save model
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "dbscan_model.pkl")
    joblib.dump({"model": model, "scaler": scaler, "features": feature_cols}, model_path)

    print(f"\n=== STEP 8: Model Saved ===")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type     : DBSCAN (Density-Based Clustering / Anomaly Detection)")
    print(f"Contents : {len(model.core_sample_indices_)} core sample indices + {n_clusters} cluster labels")
    print(f"Key      : Points labeled -1 are NOISE (anomalies/outliers)")
    print(f"Strength : Detects clusters of ANY shape + automatic anomaly detection")
    print(f"           Does NOT need K (number of clusters) specified in advance")
    print(f"Use case : Fraud detection, intrusion detection, spatial analysis")

if __name__ == "__main__":
    main()
