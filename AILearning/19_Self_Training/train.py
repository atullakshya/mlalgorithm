"""
================================================================================
SELF-TRAINING - Real-World Example: Product Review Sentiment with Few Labels
================================================================================
REAL-TIME USE CASE:
    An e-commerce platform has thousands of product reviews but only a FEW are
    labeled as Positive/Negative. Labeling is expensive (requires human reviewers).
    Self-Training uses the few labeled reviews to label the rest automatically.

ALGORITHM:
    Self-Training (Semi-Supervised Learning):
    1. Train a classifier on the SMALL set of labeled data
    2. Use it to predict labels for UNLABELED data
    3. Add the most CONFIDENT predictions as new labeled data
    4. Retrain and repeat until no more confident predictions
    Gradually expands the labeled dataset using its own predictions.

MODEL TYPE AFTER TRAINING:
    -> The BASE CLASSIFIER (e.g., SVM) trained on expanded labeled data.
    -> Same model type as the base classifier, but trained on MORE data.
    -> The self-training wrapper is just the training PROCESS, not the model.
    -> Saved as .pkl, contains the base classifier after iterative training.
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (product reviews)
    # Some reviews have labels (Positive/Negative), most are UNLABELED (empty)
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))

    labeled_mask = df["Sentiment"].notna()
    n_labeled = labeled_mask.sum()
    n_unlabeled = (~labeled_mask).sum()

    print("=== STEP 1: Dataset Loaded ===")
    print(f"Total reviews   : {len(df)}")
    print(f"Labeled         : {n_labeled} ({n_labeled/len(df)*100:.0f}%)")
    print(f"Unlabeled       : {n_unlabeled} ({n_unlabeled/len(df)*100:.0f}%)")
    print(f"\nLabeled examples:")
    for _, row in df[labeled_mask].head(4).iterrows():
        print(f"  [{row['Sentiment']:>8s}] {row['ReviewText'][:55]}...")
    print(f"\nUnlabeled examples:")
    for _, row in df[~labeled_mask].head(3).iterrows():
        print(f"  [       ?] {row['ReviewText'][:55]}...")

    # -------------------------------------------------------------------------
    # STEP 2: Encode labels (Positive=1, Negative=0, Unlabeled=-1)
    # sklearn SelfTraining requires -1 for unlabeled samples
    # -------------------------------------------------------------------------
    label_map = {"Positive": 1, "Negative": 0}
    y_full = np.array([label_map.get(s, -1) for s in df["Sentiment"]])
    X_text = df["ReviewText"].values

    print(f"\n=== STEP 2: Label Encoding ===")
    print(f"Positive (1): {(y_full==1).sum()}")
    print(f"Negative (0): {(y_full==0).sum()}")
    print(f"Unlabeled(-1): {(y_full==-1).sum()}")

    # -------------------------------------------------------------------------
    # STEP 3: Vectorize text using TF-IDF
    # -------------------------------------------------------------------------
    tfidf = TfidfVectorizer(max_features=200, stop_words="english")
    X_vectors = tfidf.fit_transform(X_text)
    print(f"\n=== STEP 3: TF-IDF Vectorization ===")
    print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"Feature matrix : {X_vectors.shape}")

    # -------------------------------------------------------------------------
    # STEP 4: Split data (keep some labeled data for testing)
    # We use labeled data for test evaluation, train on mix of labeled+unlabeled
    # -------------------------------------------------------------------------
    labeled_idx = np.where(y_full != -1)[0]
    test_idx = labeled_idx[::3]  # Every 3rd labeled sample for testing
    train_idx = np.array([i for i in range(len(y_full)) if i not in test_idx])

    X_train = X_vectors[train_idx]
    y_train = y_full[train_idx]
    X_test = X_vectors[test_idx]
    y_test = y_full[test_idx]

    n_train_labeled = (y_train != -1).sum()
    n_train_unlabeled = (y_train == -1).sum()

    print(f"\n=== STEP 4: Data Split ===")
    print(f"Train: {len(y_train)} total ({n_train_labeled} labeled + {n_train_unlabeled} unlabeled)")
    print(f"Test : {len(y_test)} labeled samples (for evaluation)")

    # -------------------------------------------------------------------------
    # STEP 5A: Baseline - train ONLY on labeled data
    # -------------------------------------------------------------------------
    baseline_svm = SVC(kernel="rbf", probability=True, random_state=42)
    labeled_train_mask = y_train != -1
    baseline_svm.fit(X_train[labeled_train_mask], y_train[labeled_train_mask])
    baseline_pred = baseline_svm.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    print(f"\n=== STEP 5A: Baseline (Supervised Only) ===")
    print(f"Trained on {n_train_labeled} labeled samples only")
    print(f"Accuracy: {baseline_acc:.4f}")

    # -------------------------------------------------------------------------
    # STEP 5B: Self-Training - uses BOTH labeled and unlabeled data
    # threshold=0.8: only add predictions with >80% confidence
    # -------------------------------------------------------------------------
    base_svm = SVC(kernel="rbf", probability=True, random_state=42)
    self_trainer = SelfTrainingClassifier(base_svm, threshold=0.8, max_iter=20)
    self_trainer.fit(X_train, y_train)
    st_pred = self_trainer.predict(X_test)
    st_acc = accuracy_score(y_test, st_pred)

    print(f"\n=== STEP 5B: Self-Training (Semi-Supervised) ===")
    print(f"Trained on {n_train_labeled} labeled + {n_train_unlabeled} unlabeled samples")
    print(f"Self-training iterations: {self_trainer.n_iter_}")
    print(f"Accuracy: {st_acc:.4f}")
    print(f"\nImprovement over baseline: {(st_acc-baseline_acc)*100:+.2f}%")

    # -------------------------------------------------------------------------
    # STEP 6: Show predictions on test data
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 6: Test Predictions ===")
    label_names = {0: "Negative", 1: "Positive"}
    for i in range(len(y_test)):
        actual = label_names[y_test[i]]
        predicted = label_names[st_pred[i]]
        match = "OK" if y_test[i] == st_pred[i] else "WRONG"
        text = X_text[test_idx[i]][:50]
        print(f"  [{match:>5s}] Actual={actual:>8s}, Pred={predicted:>8s} | \"{text}...\"")

    # -------------------------------------------------------------------------
    # STEP 7: Show what Self-Training labeled for unlabeled data
    # -------------------------------------------------------------------------
    final_labels = self_trainer.transduction_
    newly_labeled = y_train == -1
    print(f"\n=== STEP 7: Auto-Labeled Reviews ===")
    for idx in np.where(newly_labeled)[0][:5]:
        orig_idx = train_idx[idx]
        label = label_names.get(final_labels[idx], "Unknown")
        text = X_text[orig_idx][:55]
        print(f"  [{label:>8s}] \"{text}...\"")

    # -------------------------------------------------------------------------
    # STEP 8: Save model
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "self_training_model.pkl")
    joblib.dump({"model": self_trainer, "tfidf": tfidf}, model_path)

    print(f"\n=== STEP 8: Model Saved ===")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type       : Self-Training (Semi-Supervised Wrapper)")
    print(f"Base model : SVC (Support Vector Classifier)")
    print(f"Process    : Iteratively expanded {n_train_labeled} labels to {(final_labels!=-1).sum()} labels")
    print(f"Iterations : {self_trainer.n_iter_}")
    print(f"Threshold  : Only added predictions with >80% confidence")
    print(f"Key benefit: Uses UNLABELED data to improve accuracy")
    print(f"             Reduces need for expensive human labeling")

if __name__ == "__main__":
    main()
