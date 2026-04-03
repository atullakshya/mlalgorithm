"""
================================================================================
HIERARCHICAL CLUSTERING - Real-World Example: Patient Health Risk Grouping
================================================================================
REAL-TIME USE CASE:
    A hospital wants to group patients into health risk categories based on
    cholesterol, blood sugar, blood pressure, BMI, and heart rate - WITHOUT
    predefined categories. The hierarchy shows which patients are most similar.

ALGORITHM:
    Agglomerative (bottom-up): Start with each patient as its own cluster,
    then MERGE the two closest clusters iteratively until one cluster remains.
    Creates a DENDROGRAM (tree) showing the merge history.

MODEL TYPE AFTER TRAINING:
    -> A DENDROGRAM (hierarchical tree structure) + cluster labels.
    -> No centroids like K-Means. Instead, a merge history (linkage matrix).
    -> Cut the tree at any level to get desired number of clusters.
    -> Saved as .pkl, contains linkage matrix and cluster assignments.
    -> UNSUPERVISED, hierarchical, connectivity-based model.
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
import joblib
import os

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (patient health metrics)
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")

    # -------------------------------------------------------------------------
    # STEP 2: Prepare features (no target - unsupervised)
    # -------------------------------------------------------------------------
    feature_cols = ["Cholesterol", "BloodSugar", "BloodPressureSystolic", "BMI", "HeartRate"]
    X = df[feature_cols].values
    print(f"\n=== STEP 2: Features: {feature_cols} ===")

    # -------------------------------------------------------------------------
    # STEP 3: Feature scaling
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n=== STEP 3: Features Scaled ===")

    # -------------------------------------------------------------------------
    # STEP 4: Compute linkage matrix (the full merge history)
    # 'ward' minimizes within-cluster variance at each merge step
    # The linkage matrix records: [cluster1, cluster2, distance, size]
    # -------------------------------------------------------------------------
    Z = linkage(X_scaled, method="ward")
    print(f"\n=== STEP 4: Linkage Matrix Computed ===")
    print(f"Shape: {Z.shape} (each row = one merge step)")
    print(f"Total merge steps: {len(Z)} (n_samples - 1)")
    print(f"\nFirst 5 merges (closest patients merged first):")
    for i in range(5):
        print(f"  Merge {i+1}: Cluster {int(Z[i,0])} + Cluster {int(Z[i,1])}, "
              f"distance={Z[i,2]:.3f}, new_size={int(Z[i,3])}")

    # -------------------------------------------------------------------------
    # STEP 5: Try different numbers of clusters and evaluate
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 5: Finding Optimal Clusters ===")
    for n in [2, 3, 4, 5]:
        model = AgglomerativeClustering(n_clusters=n, linkage="ward")
        labels = model.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        print(f"  n_clusters={n}: Silhouette={sil:.4f}")

    # -------------------------------------------------------------------------
    # STEP 6: Train final model with n=3 (Low/Medium/High risk groups)
    # -------------------------------------------------------------------------
    best_n = 3
    model = AgglomerativeClustering(n_clusters=best_n, linkage="ward")
    labels = model.fit_predict(X_scaled)
    df["RiskGroup"] = labels

    print(f"\n=== STEP 6: Final Model (n={best_n}) ===")
    print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")

    print(f"\nCluster Profiles (average health metrics):")
    for c in range(best_n):
        mask = labels == c
        size = mask.sum()
        means = X[mask].mean(axis=0)
        print(f"\n  RISK GROUP {c} ({size} patients):")
        for feat, val in zip(feature_cols, means):
            print(f"    {feat:30s}: {val:.1f}")
        avg_chol = means[0]
        if avg_chol > 300:
            print(f"    -> HIGH RISK (elevated cholesterol, BP, sugar)")
        elif avg_chol > 230:
            print(f"    -> MEDIUM RISK (moderate levels)")
        else:
            print(f"    -> LOW RISK (healthy range)")

    # -------------------------------------------------------------------------
    # STEP 7: Show patient assignments
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 7: Patient Assignments ===")
    for _, row in df.head(10).iterrows():
        print(f"  Patient {int(row['PatientID']):2d}: Chol={row['Cholesterol']:.0f}, "
              f"BP={row['BloodPressureSystolic']:.0f} -> Risk Group {int(row['RiskGroup'])}")

    # -------------------------------------------------------------------------
    # STEP 8: Save model
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "hierarchical_model.pkl")
    joblib.dump({"model": model, "linkage_matrix": Z, "scaler": scaler, "features": feature_cols}, model_path)

    print(f"\n=== STEP 8: Model Saved ===")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type     : Agglomerative Hierarchical Clustering (Unsupervised)")
    print(f"Contents : Linkage matrix ({Z.shape[0]} merge steps) + {best_n} cluster labels")
    print(f"Structure: A DENDROGRAM (tree) showing how patients are grouped")
    print(f"           Can be cut at ANY level to get 2, 3, 4, ... clusters")
    print(f"Predict  : New patient compared to existing cluster profiles")
    print(f"Strength : Shows HIERARCHY of similarity (unlike flat K-Means)")

if __name__ == "__main__":
    main()
