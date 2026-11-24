"""
URL Prediction System
AI Phishing Website Detector

Loads the trained Random Forest model and predicts whether a URL
is phishing or legitimate.
"""

import os
import sys
import pickle
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml.feature_extraction import extract_features

MODEL_FILE = os.path.join(BASE_DIR, "ml", "model", "phishing_model.pkl")

# Must match the column names/order used in train.py's FEATURE_NAMES
FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "dots", "hyphens",
    "underscores", "slashes", "question_marks", "equals", "at_symbols",
    "ampersands", "percent_symbols", "uses_https", "uses_ip", "subdomains",
    "suspicious_words", "has_at", "double_slash", "shortener", "digits",
    "digit_ratio", "query_params", "entropy",
]

# Dataset labels: 0 = Phishing, 1 = Legitimate
_model_cache = None


def load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model not found:\n{MODEL_FILE}")

    with open(MODEL_FILE, "rb") as f:
        _model_cache = pickle.load(f)

    return _model_cache


def predict_url(url):
    model = load_model()

    features = extract_features(url)
    features_df = pd.DataFrame([features], columns=FEATURE_NAMES)

    prediction = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    confidence = max(probabilities) * 100

    result = "PHISHING" if prediction == 0 else "LEGITIMATE"
    return result, confidence


if __name__ == "__main__":
    print("=" * 60)
    print("AI PHISHING WEBSITE DETECTOR")
    print("URL PREDICTION SYSTEM")
    print("=" * 60)

    while True:
        url = input("\nEnter URL (or type 'exit' to quit): ").strip()

        if url.lower() == "exit":
            print("\nExiting...")
            break

        if not url:
            print("Please enter a URL.")
            continue

        try:
            result, confidence = predict_url(url)
            print("\n" + "-" * 60)
            if result == "PHISHING":
                print("RESULT: PHISHING")
            else:
                print("RESULT: LEGITIMATE")
            print(f"Confidence: {confidence:.2f}%")
            print("-" * 60)
        except Exception as error:
            print(f"\nError: {error}")