"""
================================================================================
K-MEANS CLUSTERING - Real-World Example: Customer Segmentation
================================================================================
REAL-TIME USE CASE:
    A retail company wants to segment customers into groups based on income,
    spending behavior, age, and visit frequency - for targeted marketing.

ALGORITHM:
    K-Means partitions data into K clusters by iteratively:
    1. Assigning each point to the nearest cluster center (centroid)
    2. Recalculating centroids as the mean of assigned points
    Repeats until centroids stabilize.

MODEL TYPE AFTER TRAINING:
    -> K CLUSTER CENTROIDS (center points in feature space).
    -> Each centroid defines a cluster - new points assigned to nearest centroid.
    -> Saved as .pkl, contains K centroid coordinates.
    -> UNSUPERVISED model - no labels needed, discovers structure automatically.
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import os

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (customer mall data)
    # Note: This is UNSUPERVISED - there are NO labels to predict
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nStats:\n{df.describe()}")

    # -------------------------------------------------------------------------
    # STEP 2: Select features for clustering (no target variable!)
    # -------------------------------------------------------------------------
    feature_cols = ["AnnualIncome_K", "SpendingScore", "Age", "VisitsPerMonth"]
    X = df[feature_cols].values
    print(f"\n=== STEP 2: Features for clustering: {feature_cols} ===")
    print(f"  NO target variable - unsupervised learning!")

    # -------------------------------------------------------------------------
    # STEP 3: Feature scaling (CRITICAL for K-Means - uses distance calculations)
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n=== STEP 3: Features Scaled ===")

    # -------------------------------------------------------------------------
    # STEP 4: Find optimal K using Elbow Method + Silhouette Score
    # Elbow: plot inertia (sum of distances to centroids) vs K
    # Silhouette: measures how well-separated clusters are (-1 to 1, higher=better)
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 4: Finding Optimal K ===")
    print(f"{'K':>3s} | {'Inertia':>10s} | {'Silhouette':>10s} | Visual")
    print("-" * 50)
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        bar = "#" * int(sil * 40)
        print(f"  {k} | {km.inertia_:>10.2f} | {sil:>10.4f} | {bar}")

    # -------------------------------------------------------------------------
    # STEP 5: Train final model with chosen K
    # -------------------------------------------------------------------------
    best_k = 4
    model = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    model.fit(X_scaled)

    print(f"\n=== STEP 5: Model Training Complete (K={best_k}) ===")
    print(f"Model type         : {type(model).__name__}")
    print(f"Clusters           : {best_k}")
    print(f"Iterations to fit  : {model.n_iter_}")
    print(f"Inertia (lower=better): {model.inertia_:.2f}")
    print(f"Silhouette Score   : {silhouette_score(X_scaled, model.labels_):.4f}")

    # -------------------------------------------------------------------------
    # STEP 6: Analyze what each cluster represents
    # The centroids (in original scale) tell us the "profile" of each group
    # -------------------------------------------------------------------------
    centroids_original = scaler.inverse_transform(model.cluster_centers_)
    df["Cluster"] = model.labels_

    print(f"\n=== STEP 6: Cluster Profiles (Centroids) ===")
    for i in range(best_k):
        c = centroids_original[i]
        size = (model.labels_ == i).sum()
        print(f"\n  CLUSTER {i} ({size} customers):")
        for feat, val in zip(feature_cols, c):
            print(f"    {feat:20s}: {val:.1f}")
        if c[0] > 60 and c[1] > 70:
            print(f"    -> Profile: HIGH Income, HIGH Spending (VIP customers)")
        elif c[0] > 60 and c[1] < 30:
            print(f"    -> Profile: HIGH Income, LOW Spending (potential upsell)")
        elif c[0] < 30 and c[1] > 70:
            print(f"    -> Profile: LOW Income, HIGH Spending (impulse buyers)")
        elif c[0] < 30 and c[1] < 30:
            print(f"    -> Profile: LOW Income, LOW Spending (budget-conscious)")
        else:
            print(f"    -> Profile: MODERATE Income & Spending (average customers)")

    # -------------------------------------------------------------------------
    # STEP 7: Show cluster assignments for each customer
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 7: Customer Cluster Assignments ===")
    for _, row in df.head(10).iterrows():
        print(f"  Customer {int(row['CustomerID']):2d}: Income=${row['AnnualIncome_K']:.0f}K, "
              f"Spending={row['SpendingScore']:.0f} -> Cluster {int(row['Cluster'])}")

    # -------------------------------------------------------------------------
    # STEP 8: Save model
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "kmeans_model.pkl")
    joblib.dump({"model": model, "scaler": scaler, "features": feature_cols}, model_path)

    print(f"\n=== STEP 8: Model Saved ===")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type     : K-Means Clustering (Unsupervised, Centroid-Based)")
    print(f"Contents : {best_k} cluster centroids (center coordinates)")
    for i, c in enumerate(model.cluster_centers_):
        print(f"           Centroid {i}: [{', '.join(f'{v:.2f}' for v in c)}]")
    print(f"Predict  : Assign new customer to nearest centroid (Euclidean distance)")
    print(f"Size     : Tiny - only {best_k} points with {X.shape[1]} coordinates each")
    print(f"Note     : No labels needed! The model discovers groups on its own.")

if __name__ == "__main__":
    main()
