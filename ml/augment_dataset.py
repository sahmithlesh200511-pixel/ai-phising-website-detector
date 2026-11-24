"""
Augment the Legitimate (label=1) class with realistic format variants.

Problem being fixed:
    In the original dataset, every Legitimate URL is a bare root
    domain with "www." (e.g. "https://www.example.com"), while
    Phishing URLs are far more diverse in format (with/without www,
    with/without paths, http/https). The model was learning to
    associate "unusual formatting" with phishing, rather than
    learning real phishing signals -- causing false positives on
    legitimate URLs typed without "www." or with a path.

What this script does:
    For a sample of existing Legitimate URLs, generates additional
    Legitimate training rows:
      - the same domain WITHOUT "www."
      - the same domain WITH a plausible path added

    These are appended to the dataset (originals are kept too) and
    saved as a new CSV. Run train.py against this new file to get
    a model that no longer relies on those spurious cues.

Usage:
    python ml/augment_dataset.py
"""

import os
import sys
import random

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

INPUT_FILE = os.path.join(BASE_DIR, "ml", "dataset", "processed_dataset.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "ml", "dataset", "processed_dataset_augmented.csv")

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

# Plausible realistic paths to append to legitimate domains
SAMPLE_PATHS = [
    "/about", "/about-us", "/contact", "/contact-us", "/products",
    "/services", "/blog", "/news", "/en/home", "/support",
    "/help", "/faq", "/team", "/careers", "/index.html",
    "/article/2024/summary", "/docs/getting-started",
]


def augment_legitimate(df, sample_size=None):
    legit = df[df["label"] == 1].copy()

    if sample_size is not None and len(legit) > sample_size:
        legit = legit.sample(n=sample_size, random_state=RANDOM_STATE)

    new_rows = []

    for url in legit["URL"]:
        url = str(url).strip()

        # Variant 1: remove "www." if present
        if "://www." in url:
            no_www = url.replace("://www.", "://", 1)
            new_rows.append({"URL": no_www, "label": 1})

        # Variant 2: add a plausible path
        path = random.choice(SAMPLE_PATHS)
        with_path = url.rstrip("/") + path
        new_rows.append({"URL": with_path, "label": 1})

    return pd.DataFrame(new_rows)


def main():
    print("=" * 60)
    print("DATASET AUGMENTATION - Legitimate class format diversity")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Dataset not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print(f"\nOriginal dataset shape: {df.shape}")
    print(df["label"].value_counts())

    # Augment using up to 40,000 legitimate URLs (adjust as needed)
    augmented = augment_legitimate(df, sample_size=40000)
    print(f"\nGenerated {len(augmented)} new legitimate variant rows.")

    combined = pd.concat([df, augmented], ignore_index=True)
    combined = combined.drop_duplicates(subset="URL", keep="first")

    print(f"\nCombined dataset shape: {combined.shape}")
    print(combined["label"].value_counts())

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved augmented dataset to:\n{OUTPUT_FILE}")
    print("=" * 60)
    print("\nNext step: point train.py's DATASET_FILE at")
    print("processed_dataset_augmented.csv, then retrain.")


if __name__ == "__main__":
    main()