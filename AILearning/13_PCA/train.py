"""
================================================================================
PCA - Real-World Example: Student Performance Dimensionality Reduction
================================================================================
REAL-TIME USE CASE:
    A school has 10 subject scores per student. PCA reduces these 10 dimensions
    to 2-3 principal components that capture the most variance, revealing hidden
    patterns (e.g., "science aptitude" vs "arts aptitude").

ALGORITHM:
    PCA (Principal Component Analysis):
    1. Standardize features
    2. Compute covariance matrix
    3. Find eigenvectors (principal components) and eigenvalues
    4. Project data onto top K eigenvectors (retain most variance)

MODEL TYPE AFTER TRAINING:
    -> A set of PRINCIPAL COMPONENTS (eigenvectors) + explained variance.
    -> Each component is a LINEAR COMBINATION of original features.
    -> E.g., PC1 = 0.4*Math + 0.35*Physics + 0.3*Chemistry - 0.2*Art
    -> Saved as .pkl, contains the transformation matrix.
    -> UNSUPERVISED, linear, parametric dimensionality reduction model.
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (student scores across 10 subjects)
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Shape: {df.shape} (30 students x 10 subjects)")
    print(f"\nFirst 5 rows:\n{df.head()}")

    # -------------------------------------------------------------------------
    # STEP 2: Prepare features
    # -------------------------------------------------------------------------
    feature_cols = ["Math", "Physics", "Chemistry", "Biology", "English",
                    "History", "Geography", "Art", "Music", "PE"]
    X = df[feature_cols].values
    print(f"\n=== STEP 2: {len(feature_cols)} features (subjects) ===")

    # -------------------------------------------------------------------------
    # STEP 3: Standardize features (PCA is sensitive to scale)
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n=== STEP 3: Features Standardized ===")

    # -------------------------------------------------------------------------
    # STEP 4: Apply PCA to find all principal components first
    # See how much variance each component explains
    # -------------------------------------------------------------------------
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    print(f"\n=== STEP 4: All Principal Components ===")
    print(f"{'PC':>5s} | {'Variance %':>10s} | {'Cumulative %':>12s} | Visual")
    print("-" * 55)
    for i, (var, cum) in enumerate(zip(pca_full.explained_variance_ratio_, cumvar)):
        bar = "#" * int(var * 80)
        print(f"  PC{i+1} | {var*100:>9.2f}% | {cum*100:>11.2f}% | {bar}")

    # -------------------------------------------------------------------------
    # STEP 5: Choose number of components (95% variance threshold)
    # -------------------------------------------------------------------------
    n_components = int(np.argmax(cumvar >= 0.95) + 1)
    print(f"\n=== STEP 5: Keeping {n_components} components (95% variance) ===")
    print(f"Reduced from {len(feature_cols)} dimensions -> {n_components} dimensions!")

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # -------------------------------------------------------------------------
    # STEP 6: Interpret what each principal component means
    # Each component is a linear combination of original features
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 6: Component Interpretation ===")
    for i in range(n_components):
        loadings = pca.components_[i]
        print(f"\n  PC{i+1} (explains {pca.explained_variance_ratio_[i]*100:.1f}% variance):")
        sorted_idx = np.argsort(np.abs(loadings))[::-1]
        for j in sorted_idx[:5]:
            direction = "+" if loadings[j] > 0 else "-"
            print(f"    {direction}{abs(loadings[j]):.3f} * {feature_cols[j]}")

        # Interpret the component
        top_positive = [(feature_cols[j], loadings[j]) for j in sorted_idx if loadings[j] > 0.2]
        top_negative = [(feature_cols[j], loadings[j]) for j in sorted_idx if loadings[j] < -0.2]
        if top_positive and top_negative:
            pos_names = [f[0] for f in top_positive[:3]]
            neg_names = [f[0] for f in top_negative[:3]]
            print(f"    -> Contrasts: {', '.join(pos_names)} vs {', '.join(neg_names)}")

    # -------------------------------------------------------------------------
    # STEP 7: Show transformed student data
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 7: Transformed Student Data ===")
    print(f"Original shape: {X.shape} -> PCA shape: {X_pca.shape}")
    print(f"\n{'Student':>8s} | " + " | ".join(f"{'PC'+str(i+1):>8s}" for i in range(min(3, n_components))))
    print("-" * 50)
    for i in range(min(10, len(X_pca))):
        vals = " | ".join(f"{X_pca[i,j]:>8.3f}" for j in range(min(3, n_components)))
        print(f"     {df['StudentID'].iloc[i]:>3d} | {vals}")

    # -------------------------------------------------------------------------
    # STEP 8: Save model
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "pca_model.pkl")
    joblib.dump({"pca": pca, "scaler": scaler, "features": feature_cols}, model_path)

    print(f"\n=== STEP 8: Model Saved ===")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type       : PCA (Principal Component Analysis)")
    print(f"Contents   : {n_components} principal components (eigenvectors)")
    print(f"             Each component = linear combination of {len(feature_cols)} features")
    print(f"Transform  : {len(feature_cols)} features -> {n_components} components")
    print(f"             (components matrix shape: {pca.components_.shape})")
    print(f"Variance   : {cumvar[n_components-1]*100:.1f}% of information retained")
    print(f"Use case   : Reduces complexity, removes noise, enables visualization")
    print(f"Size       : Tiny - just a {pca.components_.shape} matrix + mean vector")

if __name__ == "__main__":
    main()
