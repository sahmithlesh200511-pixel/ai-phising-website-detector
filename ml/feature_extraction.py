"""
Feature Extraction for AI Phishing Website Detector

Extracts a fixed-order list of 23 numerical features from a URL.
This exact function and feature order must be used identically
in both train.py and predict.py.

Usage:
    from feature_extraction import extract_features
    features = extract_features("https://www.google.com")
"""

import re
import math
from urllib.parse import urlparse


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def count_character(url, character):
    """Count how many times a character appears in a URL."""
    return url.count(character)


def has_ip_address(hostname):
    """Check whether the URL uses an IPv4 address instead of a domain."""
    if not hostname:
        return 0
    ipv4_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    return 1 if re.match(ipv4_pattern, hostname) else 0


def calculate_entropy(text):
    """Shannon entropy of a string. Higher = more random-looking."""
    if not text:
        return 0.0
    probabilities = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


SUSPICIOUS_WORDS = [
    "login", "signin", "sign-in", "verify", "verification", "secure",
    "security", "account", "update", "confirm", "confirmation",
    "password", "bank", "banking", "paypal", "payment", "wallet",
    "free", "bonus", "claim", "urgent", "alert", "suspended", "unlock",
]

SHORTENING_SERVICES = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "cutt.ly", "rb.gy", "shorturl.at",
]


# ---------------------------------------------------------
# Main feature extraction function
# ---------------------------------------------------------

def extract_features(url):
    """
    Extract a fixed-order 23-element feature list from a URL.

    IMPORTANT: Do not change the order of the returned list
    without retraining the model — train.py and predict.py both
    depend on this exact order.
    """

    original_url = url.strip()

    if not original_url:
        raise ValueError("URL cannot be empty.")

    # Strip accidental markdown-link wrapping, e.g. "[text](url)"
    markdown_match = re.match(r"^\[.*?\]\((.*?)\)$", original_url)
    if markdown_match:
        original_url = markdown_match.group(1).strip()

    # If no scheme is present, assume https (the realistic modern default
    # for real-world sites, and what most legitimate URLs use today).
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", original_url):
        parsed_url = urlparse("https://" + original_url)
    else:
        parsed_url = urlparse(original_url)

    hostname = (parsed_url.hostname or "").lower()
    path = parsed_url.path or ""
    query = parsed_url.query or ""
    full_url = original_url.lower()

    # Lengths
    url_length = len(original_url)
    hostname_length = len(hostname)
    path_length = len(path)

    # Character counts
    number_of_dots = count_character(original_url, ".")
    number_of_hyphens = count_character(original_url, "-")
    number_of_underscores = count_character(original_url, "_")
    number_of_slashes = count_character(original_url, "/")
    number_of_question_marks = count_character(original_url, "?")
    number_of_equals = count_character(original_url, "=")
    number_of_at_symbols = count_character(original_url, "@")
    number_of_ampersands = count_character(original_url, "&")
    number_of_percent_symbols = count_character(original_url, "%")

    # Protocol
    uses_https = 1 if parsed_url.scheme.lower() == "https" else 0

    # Domain
    uses_ip_address = has_ip_address(hostname)
    domain_parts = hostname.split(".") if hostname else []
    number_of_subdomains = max(len(domain_parts) - 2, 0)

    # Suspicious keywords
    suspicious_word_count = sum(1 for word in SUSPICIOUS_WORDS if word in full_url)

    # Special indicators
    has_at_symbol = 1 if "@" in original_url else 0
    double_slash_in_path = 1 if "//" in path else 0
    uses_url_shortener = 1 if any(s in hostname for s in SHORTENING_SERVICES) else 0

    # Digits
    number_of_digits = sum(1 for c in original_url if c.isdigit())
    digit_ratio = number_of_digits / url_length if url_length > 0 else 0

    # Query
    number_of_query_parameters = len(query.split("&")) if query else 0

    # Entropy
    url_entropy = calculate_entropy(original_url)

    return [
        url_length,
        hostname_length,
        path_length,
        number_of_dots,
        number_of_hyphens,
        number_of_underscores,
        number_of_slashes,
        number_of_question_marks,
        number_of_equals,
        number_of_at_symbols,
        number_of_ampersands,
        number_of_percent_symbols,
        uses_https,
        uses_ip_address,
        number_of_subdomains,
        suspicious_word_count,
        has_at_symbol,
        double_slash_in_path,
        uses_url_shortener,
        number_of_digits,
        digit_ratio,
        number_of_query_parameters,
        url_entropy,
    ]


if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "google.com",
        "www.google.com",
        "https://www.paypal.com/login",
        "http://192.168.1.1/login",
        "http://secure-login-account-verify.com/confirm",
    ]

    print("=" * 60)
    print("FEATURE EXTRACTION TEST")
    print("=" * 60)

    for url in test_urls:
        print("\nURL:", url)
        try:
            features = extract_features(url)
            print("Feature vector:", features)
        except Exception as error:
            print("Error:", error)