"""
================================================================================
RANDOM FOREST - Real-World Example: Employee Attrition Prediction
================================================================================
REAL-TIME USE CASE:
    An HR department wants to predict which employees are likely to LEAVE the
    company based on satisfaction, evaluations, projects, hours, and tenure.

ALGORITHM:
    Random Forest builds MANY decision trees on random subsets of data and features,
    then aggregates their predictions (majority vote). This "bagging" approach
    reduces overfitting compared to a single decision tree.

MODEL TYPE AFTER TRAINING:
    -> An ENSEMBLE of decision trees (e.g., 100 trees).
    -> Each tree learned different patterns from random data subsets.
    -> Final prediction = majority vote of all trees.
    -> Saved as .pkl, contains all 100 tree structures.
    -> NON-PARAMETRIC ensemble model - more robust than single tree.
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (HR employee data)
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nAttrition: Left={df['LeftCompany'].sum()}, Stayed={len(df)-df['LeftCompany'].sum()}")

    # -------------------------------------------------------------------------
    # STEP 2: Feature/Target split
    # -------------------------------------------------------------------------
    feature_cols = [c for c in df.columns if c != "LeftCompany"]
    X = df[feature_cols].values
    y = df["LeftCompany"].values
    print(f"\n=== STEP 2: Features: {feature_cols} ===")

    # -------------------------------------------------------------------------
    # STEP 3: Train/Test split
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n=== STEP 3: Train: {len(X_train)}, Test: {len(X_test)} ===")

    # -------------------------------------------------------------------------
    # STEP 4: Train Random Forest (no scaling needed - tree-based models
    # are not affected by feature scale)
    # n_estimators=100: build 100 decision trees
    # max_depth=5: each tree goes max 5 levels deep
    # max_features='sqrt': each split considers sqrt(n_features) random features
    # -------------------------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, max_features="sqrt", random_state=42
    )
    model.fit(X_train, y_train)

    print(f"\n=== STEP 4: Model Training Complete ===")
    print(f"Model type     : {type(model).__name__}")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Max depth/tree : {model.max_depth}")
    total_nodes = sum(t.tree_.node_count for t in model.estimators_)
    total_leaves = sum(t.tree_.n_leaves for t in model.estimators_)
    print(f"Total nodes    : {total_nodes} across all trees")
    print(f"Total leaves   : {total_leaves} (decision endpoints)")

    # -------------------------------------------------------------------------
    # STEP 5: Feature importance (which features matter most for prediction)
    # Random Forest naturally ranks features by how much they reduce impurity
    # -------------------------------------------------------------------------
    importances = model.feature_importances_
    print(f"\n=== STEP 5: Feature Importances ===")
    for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"  {feat:25s}: {imp:.4f} {bar}")

    # -------------------------------------------------------------------------
    # STEP 6: Predict and show individual tree votes
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test)

    print(f"\n=== STEP 6: Predictions (with tree voting) ===")
    for i in range(min(4, len(X_test))):
        actual = "Left" if y_test[i] == 1 else "Stayed"
        predicted = "Left" if y_pred[i] == 1 else "Stayed"
        tree_preds = np.array([t.predict(X_test[i:i+1])[0] for t in model.estimators_])
        votes_leave = tree_preds.sum()
        votes_stay = len(tree_preds) - votes_leave
        print(f"  Employee {i+1}: Actual={actual:>6s}, Predicted={predicted:>6s}")
        print(f"    Tree votes: {votes_stay} say Stay, {votes_leave} say Leave -> Majority wins")

    # -------------------------------------------------------------------------
    # STEP 7: Evaluate
    # -------------------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== STEP 7: Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Stayed','Left'])}")

    # -------------------------------------------------------------------------
    # STEP 8: Save model
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "random_forest_model.pkl")
    joblib.dump({"model": model, "features": feature_cols}, model_path)

    print(f"=== STEP 8: Model Saved ===")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type     : Random Forest (Ensemble of {model.n_estimators} Decision Trees)")
    print(f"Contents : {model.n_estimators} independent decision trees")
    print(f"           Total {total_nodes} nodes / {total_leaves} leaf nodes")
    print(f"Predict  : Each tree votes independently, majority vote wins")
    print(f"Strength : Handles overfitting better than single tree")
    print(f"Size     : Larger than single tree (~{total_nodes * 8 // 1024} KB approx)")

if __name__ == "__main__":
    main()
