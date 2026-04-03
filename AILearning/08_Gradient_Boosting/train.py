"""
================================================================================
GRADIENT BOOSTING - Real-World Example: Used Car Price Prediction
================================================================================
REAL-TIME USE CASE:
    A used car marketplace wants to predict fair prices for cars based on
    engine size, horsepower, weight, fuel type, cylinders, mileage, and age.

ALGORITHM:
    Gradient Boosting builds trees SEQUENTIALLY - each new tree corrects the
    ERRORS of the previous ensemble. Unlike Random Forest (parallel trees),
    boosting focuses on hard-to-predict samples.

MODEL TYPE AFTER TRAINING:
    -> A SEQUENCE of small decision trees (weak learners).
    -> Each tree is shallow (depth 3-5) and predicts RESIDUAL ERRORS.
    -> Final prediction = sum of all tree predictions * learning_rate.
    -> Saved as .pkl, contains ordered sequence of trees + learning rate.
    -> NON-PARAMETRIC, sequential ensemble model.
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (used car data)
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nPrice stats:\n{df['Price_USD'].describe()}")

    # -------------------------------------------------------------------------
    # STEP 2: Feature/Target split
    # -------------------------------------------------------------------------
    feature_cols = ["EngineSize_L", "Horsepower", "Weight_kg", "FuelType", "Cylinders", "Mileage_km", "CarAge_years"]
    X = df[feature_cols].values
    y = df["Price_USD"].values
    print(f"\n=== STEP 2: Features: {feature_cols} ===")

    # -------------------------------------------------------------------------
    # STEP 3: Train/Test split
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n=== STEP 3: Train: {len(X_train)}, Test: {len(X_test)} ===")

    # -------------------------------------------------------------------------
    # STEP 4: Train Gradient Boosting Regressor
    # n_estimators=200: build 200 trees sequentially
    # learning_rate=0.1: each tree contributes only 10% (prevents overfitting)
    # max_depth=3: each tree is small ("weak learner")
    # -------------------------------------------------------------------------
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
    )
    model.fit(X_train, y_train)

    print(f"\n=== STEP 4: Model Training Complete ===")
    print(f"Model type    : {type(model).__name__}")
    print(f"Trees built   : {model.n_estimators}")
    print(f"Learning rate : {model.learning_rate}")
    print(f"Max depth     : {model.max_depth}")

    # -------------------------------------------------------------------------
    # STEP 5: Show how boosting learns progressively
    # Each stage corrects previous errors, loss decreases over iterations
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 5: Training Progress (loss reduction) ===")
    train_loss = model.train_score_
    milestones = [0, 9, 49, 99, 149, 199]
    for m in milestones:
        if m < len(train_loss):
            print(f"  After tree {m+1:3d}: Training Loss = {train_loss[m]:.2f}")

    # -------------------------------------------------------------------------
    # STEP 6: Feature importance
    # -------------------------------------------------------------------------
    importances = model.feature_importances_
    print(f"\n=== STEP 6: Feature Importances ===")
    for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"  {feat:20s}: {imp:.4f} {bar}")

    # -------------------------------------------------------------------------
    # STEP 7: Predict and evaluate
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== STEP 7: Predictions ===")
    print(f"{'Actual':>12s} | {'Predicted':>12s} | {'Error':>10s}")
    print("-" * 40)
    for actual, predicted in zip(y_test, y_pred):
        print(f"${actual:>10,.0f} | ${predicted:>10,.0f} | ${actual-predicted:>8,.0f}")

    print(f"\nMSE  : {mse:>12,.0f}")
    print(f"RMSE : {np.sqrt(mse):>12,.0f}")
    print(f"MAE  : {mae:>12,.0f}")
    print(f"R2   : {r2:>12.4f}")

    # -------------------------------------------------------------------------
    # STEP 8: Save model
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "gradient_boosting_model.pkl")
    joblib.dump({"model": model, "features": feature_cols}, model_path)

    print(f"\n=== STEP 8: Model Saved ===")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type     : Gradient Boosting Regressor (Sequential Ensemble)")
    print(f"Contents : {model.n_estimators} small decision trees built in ORDER")
    print(f"           Each tree fixes errors of all previous trees")
    print(f"Predict  : Sum of all tree predictions * learning_rate({model.learning_rate})")
    print(f"Formula  : price = tree1(x)*0.1 + tree2(x)*0.1 + ... + tree200(x)*0.1")
    print(f"Strength : Often the BEST performing model on structured/tabular data")

if __name__ == "__main__":
    main()
