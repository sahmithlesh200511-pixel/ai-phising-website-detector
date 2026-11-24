"""
Flask Backend for AI Phishing Website Detector

Serves:
    - The frontend (index.html, script.js, style.css) at "/"
    - POST /predict  -> {"url": "..."}  returns {"result": "...", "confidence": ...}
    - GET  /health   -> simple check that the backend is running
"""

import os
import sys
from urllib.parse import urlparse

from flask import Flask, request, jsonify, send_from_directory

# ---------------------------------------------------------
# Make project root available for imports
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml.predict import predict_url
from ml.feature_extraction import extract_features

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# ---------------------------------------------------------
# Extra display-only checks (independent of the trained model's
# feature vector -- these do NOT feed the classifier, so they can
# be added/changed without needing to retrain).
# ---------------------------------------------------------

RISKY_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "work", "click", "link",
    "loan", "download", "review", "science", "party", "gdn", "men",
    "kim", "country", "stream", "bid", "win", "icu", "cam", "cyou",
}


def get_tld(hostname):
    parts = hostname.split(".")
    return parts[-1].lower() if len(parts) >= 2 else ""


def has_non_ascii(hostname):
    return any(ord(ch) > 127 for ch in hostname)



# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "message": "Backend is working!"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "Please provide a 'url' field."}), 400

    try:
        result, confidence = predict_url(url)

        # Feature order matches ml/feature_extraction.py
        feats = extract_features(url)
        (url_length, hostname_length, path_length, dots, hyphens, underscores,
         slashes, question_marks, equals, at_symbols, ampersands, percent_symbols,
         uses_https, uses_ip, subdomains, suspicious_words, has_at, double_slash,
         shortener, digits, digit_ratio, query_params, entropy) = feats

        # Parse hostname for the two extra display-only checks
        raw = url if "://" in url else "https://" + url
        hostname = (urlparse(raw).hostname or "").lower()
        tld = get_tld(hostname)
        tld_risky = tld in RISKY_TLDS
        non_ascii = has_non_ascii(hostname)

        signals = [
            {"label": "HTTPS", "value": "Yes" if uses_https else "No", "risky": not uses_https},
            {"label": "TLD risk", "value": f".{tld}" if tld else "N/A", "risky": tld_risky},
            {"label": "Shortener", "value": "Yes" if shortener else "No", "risky": bool(shortener)},
            {"label": "Non-ASCII chars", "value": "Yes" if non_ascii else "No", "risky": non_ascii},
            {"label": "IP hostname", "value": "Yes" if uses_ip else "No", "risky": bool(uses_ip)},
            {"label": "Subdomain depth", "value": str(subdomains), "risky": subdomains >= 3},
            {"label": "Suspicious keywords", "value": str(suspicious_words), "risky": suspicious_words >= 2},
            {"label": "Domain length", "value": str(hostname_length), "risky": hostname_length > 30},
        ]

        # Trust score: express confidence in the LEGITIMATE direction, 0-100
        trust_score = round(confidence) if result == "LEGITIMATE" else round(100 - confidence)

        return jsonify({
            "url": url,
            "result": result,
            "confidence": round(confidence, 2),
            "trust_score": trust_score,
            "signals": signals,
        })
    except Exception as error:
        return jsonify({"error": str(error)}), 500


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("AI PHISHING WEBSITE DETECTOR - BACKEND")
    print("=" * 60)
    print("\nServer running at: http://127.0.0.1:5000")
    print("Open that link in your browser to use the demo.")
    print("=" * 60)
    app.run(debug=True, port=5000)