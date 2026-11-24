"""
Train the AI Phishing Website Detection Model.

Dataset:
    ml/dataset/processed_dataset.csv
    (columns: URL, label -- 0 = Phishing, 1 = Legitimate)

Features:
    23 URL-based features from feature_extraction.py

Model:
    Random Forest Classifier

Output:
    ml/model/phishing_model.pkl

KNOWN DATASET LIMITATION (documented, not yet fixed):
    In this dataset, every "Legitimate" (label=1) URL is a bare root
    domain with no path (e.g. "https://www.example.com"), while
    "Phishing" (label=0) URLs include a mix of formats, some with
    paths/query strings. This makes the model over-rely on features
    like uses_https / path_length / number_of_slashes, and it can
    misclassify legitimate URLs that happen to have a path (e.g. a
    real article or subpage URL) as phishing. Fixing this properly
    requires adding legitimate URLs *with* realistic paths to the
    training data.
"""

import os
import sys
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml.feature_extraction import extract_features

DATASET_FILE = os.path.join(BASE_DIR, "ml",  "dataset", "processed_dataset_v3.csv")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "model")
MODEL_FILE = os.path.join(MODEL_DIR, "phishing_model.pkl")


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Set to None to use the full dataset. Use a number for faster iteration.
MAX_SAMPLES = None

FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "dots", "hyphens",
    "underscores", "slashes", "question_marks", "equals", "at_symbols",
    "ampersands", "percent_symbols", "uses_https", "uses_ip", "subdomains",
    "suspicious_words", "has_at", "double_slash", "shortener", "digits",
    "digit_ratio", "query_params", "entropy",
]


def train_model():
    print("=" * 60)
    print("AI PHISHING WEBSITE DETECTOR - MODEL TRAINING")
    print("=" * 60)

    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset not found:\n{DATASET_FILE}")

    print("\nLoading dataset...")
    df = pd.read_csv(DATASET_FILE)
    print(f"Dataset shape: {df.shape}")
    print(df["label"].value_counts())

    if MAX_SAMPLES is not None and len(df) > MAX_SAMPLES:
        print(f"\nSampling {MAX_SAMPLES} rows for this run.")
        df = df.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE)

    df = df.reset_index(drop=True)
    print(f"Training data shape: {df.shape}")

    urls = df["URL"].astype(str)
    labels = df["label"].astype(int)

    print("\nExtracting URL features...")
    feature_list = []
    total = len(urls)

    for i, url in enumerate(urls):
        feature_list.append(extract_features(url))
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{total}")

    X = pd.DataFrame(feature_list, columns=FEATURE_NAMES)
    y = labels

    print(f"\nFeature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training samples: {len(X_train)}  |  Testing samples: {len(X_test)}")

    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    print("Training completed.")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["Phishing", "Legitimate"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nTop feature importances:")
    for name, importance in sorted(
        zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1]
    )[:8]:
        print(f"  {name}: {importance:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel saved to:\n{MODEL_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    train_model()