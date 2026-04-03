"""
================================================================================
NAIVE BAYES - Real-World Example: Email Spam Detection
================================================================================
REAL-TIME USE CASE:
    An email provider wants to automatically classify incoming emails as SPAM or
    NOT SPAM based on the text content of the email.

ALGORITHM:
    Naive Bayes uses Bayes' Theorem: P(Spam|words) = P(words|Spam) * P(Spam) / P(words)
    "Naive" because it assumes all words are INDEPENDENT of each other.
    It calculates the probability of each class given the words in the email.

MODEL TYPE AFTER TRAINING:
    -> A PROBABILITY TABLE: P(each word | Spam) and P(each word | Not Spam)
    -> Plus class priors: P(Spam) and P(Not Spam)
    -> Very compact: just stores word frequencies per class.
    -> Saved as .pkl, contains word probabilities and priors.
    -> PARAMETRIC, PROBABILISTIC model - extremely fast to train and predict.
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (emails with spam labels)
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Shape: {df.shape}")
    print(f"\nSample emails:")
    for i, row in df.head(4).iterrows():
        label = "SPAM" if row["IsSpam"] == 1 else "HAM "
        print(f"  [{label}] {row['EmailText'][:60]}...")
    print(f"\nDistribution: Spam={df['IsSpam'].sum()}, Not Spam={len(df)-df['IsSpam'].sum()}")

    # -------------------------------------------------------------------------
    # STEP 2: Feature/Target split
    # For text data, features = raw text strings (will be vectorized later)
    # -------------------------------------------------------------------------
    X = df["EmailText"].values
    y = df["IsSpam"].values
    print(f"\n=== STEP 2: X = email text, y = spam/not spam ===")

    # -------------------------------------------------------------------------
    # STEP 3: Train/Test split
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n=== STEP 3: Train: {len(X_train)}, Test: {len(X_test)} ===")

    # -------------------------------------------------------------------------
    # STEP 4: Build a Pipeline (TF-IDF Vectorizer + Naive Bayes)
    # TF-IDF converts text to numbers: each word gets a score based on frequency
    # TF = how often word appears in document
    # IDF = how rare the word is across all documents
    # -------------------------------------------------------------------------
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500, stop_words="english")),
        ("nb", MultinomialNB(alpha=1.0)),  # alpha=1.0 = Laplace smoothing
    ])
    print(f"\n=== STEP 4: Pipeline Created ===")
    print(f"  Step 1: TfidfVectorizer (converts text -> numbers)")
    print(f"  Step 2: MultinomialNB (learns word probabilities per class)")

    # -------------------------------------------------------------------------
    # STEP 5: Train the model
    # Naive Bayes learns: P(word|spam) and P(word|not_spam) for each word
    # -------------------------------------------------------------------------
    pipeline.fit(X_train, y_train)

    tfidf = pipeline.named_steps["tfidf"]
    nb = pipeline.named_steps["nb"]
    vocab_size = len(tfidf.vocabulary_)

    print(f"\n=== STEP 5: Model Training Complete ===")
    print(f"Vocabulary size : {vocab_size} words")
    print(f"Class priors    : P(NotSpam)={np.exp(nb.class_log_prior_[0]):.3f}, P(Spam)={np.exp(nb.class_log_prior_[1]):.3f}")

    # Show top spam indicator words
    feature_names = np.array(tfidf.get_feature_names_out())
    spam_word_probs = nb.feature_log_prob_[1]
    ham_word_probs = nb.feature_log_prob_[0]
    spam_indicators = np.argsort(spam_word_probs - ham_word_probs)[::-1][:10]
    print(f"\nTop 10 SPAM indicator words (highest P(word|spam)/P(word|ham)):")
    for idx in spam_indicators:
        print(f"  '{feature_names[idx]}' -> spam_score={spam_word_probs[idx]:.3f}, ham_score={ham_word_probs[idx]:.3f}")

    # -------------------------------------------------------------------------
    # STEP 6: Predict and show probabilities
    # -------------------------------------------------------------------------
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    print(f"\n=== STEP 6: Predictions ===")
    for i, (text, actual, pred, prob) in enumerate(zip(X_test, y_test, y_pred, y_proba)):
        a_label = "SPAM" if actual == 1 else "HAM"
        p_label = "SPAM" if pred == 1 else "HAM"
        print(f"  [{a_label}->{p_label}] P(Ham)={prob[0]:.2%}, P(Spam)={prob[1]:.2%}")
        print(f"    \"{text[:55]}...\"")

    # -------------------------------------------------------------------------
    # STEP 7: Evaluate
    # -------------------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== STEP 7: Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Not Spam','Spam'])}")

    # -------------------------------------------------------------------------
    # STEP 8: Save model
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "naive_bayes_model.pkl")
    joblib.dump(pipeline, model_path)

    print(f"=== STEP 8: Model Saved ===")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type     : Multinomial Naive Bayes (Probabilistic Text Classifier)")
    print(f"Contents : Probability table with {vocab_size} word probabilities per class")
    print(f"           + 2 class prior probabilities (P(Spam), P(NotSpam))")
    print(f"Total    : ~{vocab_size * 2 + 2} probability values")
    print(f"Predict  : Calculate P(Spam|words) vs P(NotSpam|words), pick higher")
    print(f"Speed    : Extremely fast (just multiply probabilities)")

if __name__ == "__main__":
    main()
