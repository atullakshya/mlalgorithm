"""
================================================================================
LINEAR REGRESSION - Real-World Example: House Price Prediction
================================================================================
REAL-TIME USE CASE:
    A real estate company wants to predict house prices based on features like
    square footage, number of bedrooms, age of house, and distance to city center.

ALGORITHM:
    Ordinary Least Squares (OLS) Linear Regression finds the best-fit line/hyperplane
    by minimizing the sum of squared differences between predicted and actual values.

MODEL TYPE AFTER TRAINING:
    -> A set of COEFFICIENTS (weights) and an INTERCEPT (bias).
    -> Formula: Price = w1*SquareFeet + w2*Bedrooms + w3*Age + w4*Distance + b
    -> Saved as a .pkl file containing the learned weights.
    -> This is a PARAMETRIC model (fixed number of parameters regardless of data size).
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (house prices)
    # We use a CSV file with realistic house data: size, bedrooms, age, distance
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Shape: {df.shape} (rows, columns)")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nBasic Stats:\n{df.describe()}")

    # -------------------------------------------------------------------------
    # STEP 2: Separate features (X) and target (y)
    # Features = what we use to predict, Target = what we want to predict
    # -------------------------------------------------------------------------
    feature_columns = ["SquareFeet", "Bedrooms", "Age", "DistanceToCity_km"]
    target_column = "Price_USD"

    X = df[feature_columns].values
    y = df[target_column].values

    print(f"\n=== STEP 2: Feature/Target Split ===")
    print(f"Features (X) shape: {X.shape} -> {feature_columns}")
    print(f"Target   (y) shape: {y.shape} -> {target_column}")

    # -------------------------------------------------------------------------
    # STEP 3: Split into training and testing sets
    # 80% for training the model, 20% for evaluating it on unseen data
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n=== STEP 3: Train/Test Split ===")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")

    # -------------------------------------------------------------------------
    # STEP 4: Feature scaling (optional for Linear Regression but good practice)
    # Standardize features to have mean=0, std=1 for better coefficient interpretation
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"\n=== STEP 4: Feature Scaling (StandardScaler) ===")
    print(f"Train mean after scaling: {X_train_scaled.mean(axis=0).round(4)}")
    print(f"Train std  after scaling: {X_train_scaled.std(axis=0).round(4)}")

    # -------------------------------------------------------------------------
    # STEP 5: Train the Linear Regression model
    # The model learns weights (coefficients) for each feature and a bias (intercept)
    # It minimizes: Sum of (actual_price - predicted_price)^2
    # -------------------------------------------------------------------------
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    print(f"\n=== STEP 5: Model Training Complete ===")
    print(f"Model type: {type(model).__name__}")
    print(f"\nLearned Parameters (what the model contains after training):")
    for feat, coef in zip(feature_columns, model.coef_):
        direction = "increases" if coef > 0 else "decreases"
        print(f"  {feat:25s} -> Coefficient = {coef:>12.2f} (price {direction} with this feature)")
    print(f"  {'Intercept (bias)':25s} -> Value      = {model.intercept_:>12.2f}")

    # -------------------------------------------------------------------------
    # STEP 6: Make predictions on the test set
    # Use the learned formula: y_pred = X @ coefficients + intercept
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)

    print(f"\n=== STEP 6: Predictions on Test Set ===")
    print(f"{'Actual Price':>15s} | {'Predicted Price':>15s} | {'Difference':>12s}")
    print("-" * 48)
    for actual, predicted in zip(y_test, y_pred):
        diff = actual - predicted
        print(f"${actual:>13,.0f} | ${predicted:>13,.0f} | ${diff:>10,.0f}")

    # -------------------------------------------------------------------------
    # STEP 7: Evaluate the model
    # MSE = average of squared errors, MAE = average of absolute errors
    # R2 = how much variance is explained (1.0 = perfect, 0.0 = useless)
    # -------------------------------------------------------------------------
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== STEP 7: Model Evaluation ===")
    print(f"Mean Squared Error (MSE) : {mse:>15,.2f}")
    print(f"Root MSE (RMSE)          : {np.sqrt(mse):>15,.2f}")
    print(f"Mean Absolute Error (MAE): {mae:>15,.2f}")
    print(f"R-Squared (R2)           : {r2:>15.4f}")
    if r2 > 0.9:
        print("  -> Excellent fit! The model explains >90% of price variance.")
    elif r2 > 0.7:
        print("  -> Good fit. The model explains >70% of price variance.")
    else:
        print("  -> Moderate fit. Consider adding more features.")

    # -------------------------------------------------------------------------
    # STEP 8: Save the trained model
    # The saved .pkl file contains: coefficients, intercept, and model metadata.
    # This is the "trained model" - a simple formula with learned weights.
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "linear_regression_model.pkl")
    joblib.dump({"model": model, "scaler": scaler, "features": feature_columns}, model_path)

    print(f"\n=== STEP 8: Model Saved ===")
    print(f"Saved to: {model_path}")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type      : Linear Regression (Parametric Model)")
    print(f"Contents  : {len(model.coef_)} coefficients + 1 intercept = {len(model.coef_)+1} parameters")
    print(f"Formula   : Price = {model.coef_[0]:.1f}*SquareFeet_scaled + {model.coef_[1]:.1f}*Bedrooms_scaled")
    print(f"                   + {model.coef_[2]:.1f}*Age_scaled + {model.coef_[3]:.1f}*Distance_scaled + {model.intercept_:.1f}")
    print(f"File size : Tiny (~1 KB) because it only stores {len(model.coef_)+1} numbers")
    print(f"Prediction: Instant (just multiply & add)")

if __name__ == "__main__":
    main()
