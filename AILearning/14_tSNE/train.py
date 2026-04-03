"""
================================================================================
t-SNE - Real-World Example: Product Catalog Visualization
================================================================================
REAL-TIME USE CASE:
    An e-commerce company has products with 8 numerical features. They want to
    VISUALIZE all products on a 2D map to see natural groupings and categories.
    t-SNE projects 8D data into 2D while preserving local neighborhoods.

ALGORITHM:
    t-SNE (t-distributed Stochastic Neighbor Embedding):
    1. Compute pairwise similarities in high-dimensional space (Gaussian kernel)
    2. Initialize random 2D points
    3. Iteratively adjust 2D positions to match high-dim similarities (using t-distribution)
    Preserves LOCAL structure (nearby points stay nearby) better than PCA.

MODEL TYPE AFTER TRAINING:
    -> A 2D EMBEDDING MATRIX (coordinates for each data point).
    -> NOT a reusable transformation - specific to the input data.
    -> Cannot transform new data (must re-run on all data).
    -> Saved as .npz, contains 2D coordinates + labels.
    -> UNSUPERVISED, non-linear visualization tool (NOT a predictive model).
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (product features)
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Shape: {df.shape} (30 products x 8 features + category)")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nCategories: {df['Category'].value_counts().to_dict()}")

    # -------------------------------------------------------------------------
    # STEP 2: Prepare features
    # -------------------------------------------------------------------------
    feature_cols = [f"Feature{i}" for i in range(1, 9)]
    X = df[feature_cols].values
    categories = df["Category"].values
    print(f"\n=== STEP 2: {len(feature_cols)} features selected ===")

    # -------------------------------------------------------------------------
    # STEP 3: Standardize features
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n=== STEP 3: Features Standardized ===")

    # -------------------------------------------------------------------------
    # STEP 4: Try different perplexity values
    # Perplexity ~ number of nearest neighbors considered
    # Low perplexity = focus on very local structure
    # High perplexity = consider broader patterns
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 4: t-SNE with Different Perplexities ===")
    for perp in [5, 10]:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
        X_emb = tsne.fit_transform(X_scaled)
        print(f"\n  Perplexity={perp}, KL Divergence={tsne.kl_divergence_:.4f}")
        for cat in sorted(set(categories)):
            mask = categories == cat
            center = X_emb[mask].mean(axis=0)
            spread = X_emb[mask].std(axis=0).mean()
            print(f"    {cat:>12s}: center=({center[0]:>7.2f}, {center[1]:>7.2f}), spread={spread:.2f}")

    # -------------------------------------------------------------------------
    # STEP 5: Final t-SNE embedding
    # -------------------------------------------------------------------------
    best_perp = 10
    tsne_final = TSNE(n_components=2, perplexity=best_perp, random_state=42, max_iter=1000)
    X_final = tsne_final.fit_transform(X_scaled)

    print(f"\n=== STEP 5: Final Embedding (perplexity={best_perp}) ===")
    print(f"Input shape : {X_scaled.shape} (8 dimensions)")
    print(f"Output shape: {X_final.shape} (2 dimensions)")
    print(f"KL Divergence: {tsne_final.kl_divergence_:.4f}")

    # -------------------------------------------------------------------------
    # STEP 6: Show product positions in 2D space
    # Products in the same category should cluster together
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 6: Product 2D Positions ===")
    print(f"{'Product':>8s} | {'Category':>12s} | {'X':>8s} | {'Y':>8s}")
    print("-" * 45)
    for i in range(len(df)):
        print(f"     {df['ProductID'].iloc[i]:>3d} | {categories[i]:>12s} | "
              f"{X_final[i,0]:>8.2f} | {X_final[i,1]:>8.2f}")

    # -------------------------------------------------------------------------
    # STEP 7: Measure cluster separation in 2D
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 7: Category Separation Analysis ===")
    unique_cats = sorted(set(categories))
    for i, cat1 in enumerate(unique_cats):
        for cat2 in unique_cats[i+1:]:
            c1 = X_final[categories == cat1].mean(axis=0)
            c2 = X_final[categories == cat2].mean(axis=0)
            dist = np.linalg.norm(c1 - c2)
            print(f"  {cat1:>12s} <-> {cat2:>12s}: distance = {dist:.2f}")

    # -------------------------------------------------------------------------
    # STEP 8: Save results
    # -------------------------------------------------------------------------
    result_path = os.path.join(script_dir, "tsne_results.npz")
    np.savez(result_path, embedding=X_final, categories=categories)

    print(f"\n=== STEP 8: Results Saved ===")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type      : t-SNE (Non-linear Dimensionality Reduction / Visualization)")
    print(f"Output    : {X_final.shape} embedding matrix (2D coordinates per product)")
    print(f"Contents  : Optimized 2D positions that preserve neighborhood structure")
    print(f"IMPORTANT : t-SNE is NOT a reusable model!")
    print(f"            - Cannot transform new data points")
    print(f"            - Must re-run on entire dataset including new points")
    print(f"            - Used for VISUALIZATION, not prediction")
    print(f"Vs PCA    : PCA is linear & reusable; t-SNE is non-linear & one-time")
    print(f"Strength  : Best for visualizing clusters in high-dimensional data")

if __name__ == "__main__":
    main()
