"""
================================================================================
AUTOENCODER - Real-World Example: Industrial Equipment Anomaly Detection
================================================================================
REAL-TIME USE CASE:
    A factory monitors equipment sensors (temperature, pressure, vibration, RPM,
    current, voltage, humidity, flow rate). An autoencoder learns what NORMAL
    readings look like. When readings deviate, reconstruction error spikes,
    indicating a POTENTIAL EQUIPMENT FAILURE.

ALGORITHM:
    Autoencoder is a neural network that learns to COMPRESS and RECONSTRUCT data:
    1. ENCODER: Input(8 features) -> Bottleneck(3 features) - compress
    2. DECODER: Bottleneck(3 features) -> Output(8 features) - reconstruct
    The network is trained to minimize reconstruction error (input ≈ output).
    Anomalies have HIGH reconstruction error because they differ from training data.

MODEL TYPE AFTER TRAINING:
    -> WEIGHT MATRICES for encoder and decoder layers.
    -> Encoder compresses 8 features -> 3 latent features
    -> Decoder reconstructs 3 latent -> 8 features
    -> High reconstruction error = anomaly detected
    -> Saved as .npz, contains encoder/decoder weights.
    -> UNSUPERVISED, neural network-based model.
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class Autoencoder:
    """Simple autoencoder with one hidden layer (bottleneck)."""
    def __init__(self, input_dim, bottleneck_dim, learning_rate=0.01):
        self.lr = learning_rate
        # Encoder: input_dim -> bottleneck_dim
        scale_enc = np.sqrt(2.0 / input_dim)
        self.W_enc = np.random.randn(input_dim, bottleneck_dim) * scale_enc
        self.b_enc = np.zeros(bottleneck_dim)
        # Decoder: bottleneck_dim -> input_dim
        scale_dec = np.sqrt(2.0 / bottleneck_dim)
        self.W_dec = np.random.randn(bottleneck_dim, input_dim) * scale_dec
        self.b_dec = np.zeros(input_dim)

    def encode(self, X):
        """Compress input to bottleneck representation."""
        self.z_enc = X @ self.W_enc + self.b_enc
        self.encoded = relu(self.z_enc)
        return self.encoded

    def decode(self, H):
        """Reconstruct from bottleneck representation."""
        self.z_dec = H @ self.W_dec + self.b_dec
        self.decoded = sigmoid(self.z_dec)
        return self.decoded

    def forward(self, X):
        encoded = self.encode(X)
        decoded = self.decode(encoded)
        return decoded

    def train_step(self, X):
        output = self.forward(X)
        loss = np.mean((X - output) ** 2)

        # Backpropagation
        d_output = -(X - output) * sigmoid_derivative(self.z_dec)
        d_W_dec = self.encoded.T @ d_output / len(X)
        d_b_dec = np.mean(d_output, axis=0)

        d_hidden = d_output @ self.W_dec.T * relu_derivative(self.z_enc)
        d_W_enc = X.T @ d_hidden / len(X)
        d_b_enc = np.mean(d_hidden, axis=0)

        self.W_dec -= self.lr * d_W_dec
        self.b_dec -= self.lr * d_b_dec
        self.W_enc -= self.lr * d_W_enc
        self.b_enc -= self.lr * d_b_enc

        return loss

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (sensor readings)
    # Rows 16-19, 25, 30 are ANOMALIES (equipment issues)
    # -------------------------------------------------------------------------
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Shape: {df.shape} (30 sensor readings x 8 features)")
    print(f"\nFirst 5 rows (NORMAL readings):\n{df.head()}")
    print(f"\nAnomaly rows (equipment issues):\n{df.iloc[[15,16,17,18,24,29]]}")

    # -------------------------------------------------------------------------
    # STEP 2: Prepare features
    # -------------------------------------------------------------------------
    feature_cols = ["Temp_C", "Pressure_kPa", "Vibration_mm", "RPM",
                    "Current_A", "Voltage_V", "Humidity_pct", "FlowRate_Lpm"]
    X = df[feature_cols].values
    print(f"\n=== STEP 2: {len(feature_cols)} sensor features selected ===")

    # -------------------------------------------------------------------------
    # STEP 3: Scale features to [0, 1] for the sigmoid output layer
    # -------------------------------------------------------------------------
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n=== STEP 3: Features Scaled to [0, 1] ===")

    # -------------------------------------------------------------------------
    # STEP 4: Train ONLY on normal data (first 15 rows)
    # The autoencoder learns what "normal" looks like
    # -------------------------------------------------------------------------
    X_train = X_scaled[:15]  # Normal readings only
    X_all = X_scaled          # All readings (for testing anomaly detection)

    input_dim = X_train.shape[1]   # 8 features
    bottleneck_dim = 3              # Compress to 3 dimensions
    epochs = 300
    batch_size = 8

    ae = Autoencoder(input_dim, bottleneck_dim, learning_rate=0.1)

    print(f"\n=== STEP 4: Autoencoder Architecture ===")
    print(f"  Input layer     : {input_dim} neurons (8 sensor features)")
    print(f"  Bottleneck layer: {bottleneck_dim} neurons (compressed representation)")
    print(f"  Output layer    : {input_dim} neurons (reconstructed features)")
    print(f"  Total params    : {input_dim*bottleneck_dim + bottleneck_dim + bottleneck_dim*input_dim + input_dim}")
    print(f"\nTraining on {len(X_train)} NORMAL samples only...")

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train))
        epoch_loss = 0
        n_batches = 0
        for i in range(0, len(X_train), batch_size):
            batch = X_train[idx[i:i+batch_size]]
            loss = ae.train_step(batch)
            epoch_loss += loss
            n_batches += 1
        if (epoch + 1) % 100 == 0:
            avg_loss = epoch_loss / n_batches
            print(f"  Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.6f}")

    # -------------------------------------------------------------------------
    # STEP 5: Compute reconstruction error for ALL readings
    # Normal readings: low error. Anomalous readings: HIGH error.
    # -------------------------------------------------------------------------
    reconstructed = ae.forward(X_all)
    reconstruction_errors = np.mean((X_all - reconstructed) ** 2, axis=1)

    # Set threshold as mean + 2*std of normal errors
    normal_errors = reconstruction_errors[:15]
    threshold = normal_errors.mean() + 2 * normal_errors.std()

    print(f"\n=== STEP 5: Reconstruction Error Analysis ===")
    print(f"Normal readings avg error  : {normal_errors.mean():.6f}")
    print(f"Normal readings std error  : {normal_errors.std():.6f}")
    print(f"Anomaly detection threshold: {threshold:.6f}")

    print(f"\n{'Reading':>8s} | {'Recon Error':>12s} | {'Threshold':>10s} | {'Status':>10s}")
    print("-" * 52)
    for i in range(len(X_all)):
        err = reconstruction_errors[i]
        status = "ANOMALY!" if err > threshold else "Normal"
        marker = " ***" if err > threshold else ""
        print(f"     {df['SensorID'].iloc[i]:>3d} | {err:>12.6f} | {threshold:>10.6f} | {status:>10s}{marker}")

    # -------------------------------------------------------------------------
    # STEP 6: Show compressed representation (bottleneck values)
    # -------------------------------------------------------------------------
    encoded = ae.encode(X_all)
    print(f"\n=== STEP 6: Compressed Representation (8D -> 3D) ===")
    print(f"{'Reading':>8s} | {'Dim1':>8s} | {'Dim2':>8s} | {'Dim3':>8s} | Status")
    print("-" * 55)
    for i in [0, 5, 10, 15, 16, 24]:
        status = "ANOMALY" if reconstruction_errors[i] > threshold else "Normal"
        print(f"     {df['SensorID'].iloc[i]:>3d} | {encoded[i,0]:>8.4f} | "
              f"{encoded[i,1]:>8.4f} | {encoded[i,2]:>8.4f} | {status}")

    # -------------------------------------------------------------------------
    # STEP 7: Summary of detected anomalies
    # -------------------------------------------------------------------------
    anomaly_mask = reconstruction_errors > threshold
    n_anomalies = anomaly_mask.sum()
    print(f"\n=== STEP 7: Anomaly Detection Summary ===")
    print(f"Total readings  : {len(X_all)}")
    print(f"Normal detected : {len(X_all) - n_anomalies}")
    print(f"Anomalies found : {n_anomalies}")
    if n_anomalies > 0:
        print(f"\nAnomaly details:")
        for i in np.where(anomaly_mask)[0]:
            print(f"  Sensor {df['SensorID'].iloc[i]}: Temp={df['Temp_C'].iloc[i]}, "
                  f"Vibration={df['Vibration_mm'].iloc[i]}, RPM={df['RPM'].iloc[i]} "
                  f"(error={reconstruction_errors[i]:.4f})")

    # -------------------------------------------------------------------------
    # STEP 8: Save model
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "autoencoder_weights.npz")
    np.savez(model_path,
             W_enc=ae.W_enc, b_enc=ae.b_enc,
             W_dec=ae.W_dec, b_dec=ae.b_dec,
             threshold=threshold)

    print(f"\n=== STEP 8: Model Saved ===")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type       : Autoencoder (Unsupervised Neural Network)")
    print(f"Architecture: {input_dim} -> {bottleneck_dim} -> {input_dim}")
    print(f"Contents   : Encoder weights ({ae.W_enc.shape}) + Decoder weights ({ae.W_dec.shape})")
    print(f"             + biases + anomaly threshold ({threshold:.6f})")
    n_params = ae.W_enc.size + ae.b_enc.size + ae.W_dec.size + ae.b_dec.size
    print(f"Parameters : {n_params} total")
    print(f"How it works:")
    print(f"  1. Compress new reading: 8 features -> 3 values (encoder)")
    print(f"  2. Reconstruct: 3 values -> 8 features (decoder)")
    print(f"  3. If reconstruction error > threshold -> ANOMALY ALERT")
    print(f"Key benefit: Learns 'normal' without any labels (unsupervised)")

if __name__ == "__main__":
    main()
