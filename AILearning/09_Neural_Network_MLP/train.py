"""
================================================================================
NEURAL NETWORK (MLP) - Real-World Example: Banknote Forgery Detection
================================================================================
REAL-TIME USE CASE:
    A bank wants to detect forged banknotes using image features extracted from
    photos of banknotes (variance, skewness, kurtosis, entropy of wavelet transform).

ALGORITHM:
    Multi-Layer Perceptron (MLP) - a neural network with input, hidden, and output
    layers. Each neuron applies: output = activation(weights * input + bias).
    Trained via backpropagation (gradient descent on error).

MODEL TYPE AFTER TRAINING:
    -> A set of WEIGHT MATRICES and BIAS VECTORS for each layer.
    -> Layer structure: Input(4) -> Hidden(128) -> Hidden(64) -> Output(2)
    -> Each connection has a learned weight value.
    -> Saved as .pkl, contains all weight matrices and biases.
    -> PARAMETRIC model with potentially many parameters.
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (banknote authentication features)
    # Features are extracted from wavelet transform of banknote images
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nTarget: 0=Genuine, 1=Forgery")
    print(f"Distribution:\n{df['IsForgery'].value_counts().to_string()}")

    # -------------------------------------------------------------------------
    # STEP 2: Feature/Target split
    # -------------------------------------------------------------------------
    feature_cols = ["Variance", "Skewness", "Kurtosis", "Entropy"]
    X = df[feature_cols].values
    y = df["IsForgery"].values
    print(f"\n=== STEP 2: Features: {feature_cols} ===")

    # -------------------------------------------------------------------------
    # STEP 3: Train/Test split
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n=== STEP 3: Train: {len(X_train)}, Test: {len(X_test)} ===")

    # -------------------------------------------------------------------------
    # STEP 4: Feature scaling (CRITICAL for neural networks)
    # Neural networks use gradient descent - features must be on same scale
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"\n=== STEP 4: Features Scaled ===")

    # -------------------------------------------------------------------------
    # STEP 5: Build and train the MLP Neural Network
    # Architecture: Input(4) -> Hidden1(128 neurons) -> Hidden2(64 neurons) -> Output(2)
    # activation='relu': max(0, x) - helps with non-linear patterns
    # solver='adam': adaptive learning rate optimizer
    # -------------------------------------------------------------------------
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    n_input = X_train.shape[1]
    layers = [n_input] + list(model.hidden_layer_sizes) + [len(np.unique(y))]
    total_params = 0
    for i in range(len(layers) - 1):
        weights = layers[i] * layers[i + 1]
        biases = layers[i + 1]
        total_params += weights + biases

    print(f"\n=== STEP 5: Neural Network Training Complete ===")
    print(f"Model type  : {type(model).__name__}")
    print(f"Architecture: {' -> '.join(str(l) for l in layers)}")
    print(f"  Layer breakdown:")
    for i in range(len(layers) - 1):
        w = layers[i] * layers[i + 1]
        b = layers[i + 1]
        print(f"    Layer {i}: {layers[i]} -> {layers[i+1]} neurons = {w} weights + {b} biases = {w+b} params")
    print(f"  Total parameters: {total_params}")
    print(f"  Training iterations: {model.n_iter_}")
    print(f"  Final loss: {model.loss_:.6f}")

    # -------------------------------------------------------------------------
    # STEP 6: Predict and show probabilities
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    print(f"\n=== STEP 6: Predictions ===")
    for i in range(min(6, len(X_test))):
        actual = "Forgery" if y_test[i] == 1 else "Genuine"
        predicted = "Forgery" if y_pred[i] == 1 else "Genuine"
        conf = max(y_proba[i]) * 100
        print(f"  Note {i+1}: Actual={actual:>8s}, Predicted={predicted:>8s} (Confidence: {conf:.1f}%)")

    # -------------------------------------------------------------------------
    # STEP 7: Evaluate
    # -------------------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== STEP 7: Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Genuine','Forgery'], zero_division=0)}")

    # -------------------------------------------------------------------------
    # STEP 8: Save model
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "mlp_model.pkl")
    joblib.dump({"model": model, "scaler": scaler, "features": feature_cols}, model_path)

    print(f"=== STEP 8: Model Saved ===")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type       : MLP Neural Network (Multi-Layer Perceptron)")
    print(f"Architecture: {' -> '.join(str(l) for l in layers)}")
    print(f"Contents   : {total_params} learned parameters (weights + biases)")
    print(f"             {len(model.coefs_)} weight matrices + {len(model.intercepts_)} bias vectors")
    for i, (w, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        print(f"             Layer {i}: weights {w.shape}, biases {b.shape}")
    print(f"Predict    : Forward pass through all layers with ReLU activation")
    print(f"Strength   : Can learn COMPLEX non-linear patterns")

if __name__ == "__main__":
    main()
