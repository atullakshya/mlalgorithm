"""
================================================================================
DECISION TREE - Real-World Example: Play Tennis Decision
================================================================================
REAL-TIME USE CASE:
    A sports club wants to predict whether members will play tennis based on
    weather conditions: temperature, humidity, wind speed, and outlook.

ALGORITHM:
    Decision Tree recursively splits data on feature thresholds that best separate
    the classes. It uses GINI impurity or Information Gain to pick the best split.

MODEL TYPE AFTER TRAINING:
    -> A TREE STRUCTURE with nodes, branches, and leaves.
    -> Each internal node = a question (e.g., "Is Humidity > 80%?")
    -> Each leaf node = a prediction (e.g., "Yes" or "No")
    -> Saved as .pkl, contains the entire tree of if-else rules.
    -> NON-PARAMETRIC model (tree depth grows with data complexity).
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (weather + tennis decision)
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nTarget distribution:\n{df['PlayTennis'].value_counts().to_string()}")

    # -------------------------------------------------------------------------
    # STEP 2: Encode categorical features
    # 'Outlook' is a text column -> convert to numbers (Sunny=2, Overcast=0, Rainy=1)
    # -------------------------------------------------------------------------
    le_outlook = LabelEncoder()
    df["Outlook_encoded"] = le_outlook.fit_transform(df["Outlook"])

    le_target = LabelEncoder()
    df["Target"] = le_target.fit_transform(df["PlayTennis"])

    print(f"\n=== STEP 2: Encoding ===")
    print(f"Outlook mapping: {dict(zip(le_outlook.classes_, le_outlook.transform(le_outlook.classes_)))}")
    print(f"Target mapping : {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

    # -------------------------------------------------------------------------
    # STEP 3: Feature/Target split
    # -------------------------------------------------------------------------
    feature_cols = ["Temperature_C", "Humidity_pct", "WindSpeed_kmh", "Outlook_encoded"]
    feature_names = ["Temperature_C", "Humidity_pct", "WindSpeed_kmh", "Outlook"]
    X = df[feature_cols].values
    y = df["Target"].values

    print(f"\n=== STEP 3: Features: {feature_names}, Target: PlayTennis ===")

    # -------------------------------------------------------------------------
    # STEP 4: Train/Test split
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n=== STEP 4: Train: {len(X_train)} samples, Test: {len(X_test)} samples ===")

    # -------------------------------------------------------------------------
    # STEP 5: Train the Decision Tree
    # max_depth=4 prevents overfitting (too many if-else rules)
    # criterion='gini' uses Gini Impurity to measure split quality
    # -------------------------------------------------------------------------
    model = DecisionTreeClassifier(max_depth=4, criterion="gini", random_state=42)
    model.fit(X_train, y_train)

    print(f"\n=== STEP 5: Model Training Complete ===")
    print(f"Model type : {type(model).__name__}")
    print(f"Tree depth : {model.get_depth()}")
    print(f"Leaf nodes : {model.get_n_leaves()}")
    print(f"Total nodes: {model.tree_.node_count}")

    # -------------------------------------------------------------------------
    # STEP 6: Visualize the tree rules (human-readable)
    # THIS IS the model - a set of if-else rules learned from data
    # -------------------------------------------------------------------------
    tree_rules = export_text(model, feature_names=feature_names)
    print(f"\n=== STEP 6: Learned Decision Tree (THE MODEL) ===")
    print(tree_rules)

    # -------------------------------------------------------------------------
    # STEP 7: Predict and evaluate
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"=== STEP 7: Predictions ===")
    for i, (actual, predicted) in enumerate(zip(y_test, y_pred)):
        a_label = le_target.inverse_transform([actual])[0]
        p_label = le_target.inverse_transform([predicted])[0]
        match = "OK" if actual == predicted else "WRONG"
        print(f"  Sample {i+1}: Actual={a_label:>3s}, Predicted={p_label:>3s} [{match}]")

    print(f"\nAccuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"\nFeature Importances (which feature matters most):")
    for feat, imp in sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1]):
        bar = "#" * int(imp * 40)
        print(f"  {feat:20s}: {imp:.4f} {bar}")

    # -------------------------------------------------------------------------
    # STEP 8: Save the model
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "decision_tree_model.pkl")
    joblib.dump({"model": model, "label_encoders": {"outlook": le_outlook, "target": le_target}}, model_path)

    print(f"\n=== STEP 8: Model Saved ===")
    print(f"Saved to: {model_path}")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type     : Decision Tree (Non-Parametric, Rule-Based Classifier)")
    print(f"Structure: A tree with {model.get_depth()} levels and {model.get_n_leaves()} leaf nodes")
    print(f"Contents : A set of IF-ELSE rules learned from training data")
    print(f"Example  : IF Humidity <= 80 AND WindSpeed <= 20 THEN PlayTennis = Yes")
    print(f"Strength : Easy to interpret and explain to non-technical users")

if __name__ == "__main__":
    main()
