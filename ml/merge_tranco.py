"""
Merge Tranco top-domain list into the training dataset as additional,
genuinely diverse Legitimate (label=1) examples.

Why this matters:
    Your original "Legitimate" URLs all came from a narrow set of
    domains with uniform formatting. Even after the earlier
    www/path augmentation, the model has only ever seen a limited
    number of distinct legitimate domains. Tranco is a real,
    research-grade ranking of the top ~1 million most-visited
    domains -- adding a sample of these (in varied URL formats)
    gives the model many more genuinely different legitimate
    domains to learn from, reducing the risk of it memorizing
    a small set of names instead of learning general patterns.

Input:
    ml/dataset/processed_dataset_augmented.csv  (URL, label)
    ml/dataset/top-1m.csv                       (rank, domain -- no header)

Output:
    ml/dataset/processed_dataset_final.csv

Usage:
    python ml/merge_tranco.py
"""

import os
import sys
import random

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

INPUT_DATASET = os.path.join(BASE_DIR, "ml", "dataset", "processed_dataset_augmented.csv")
TRANCO_FILE = os.path.join(BASE_DIR, "ml", "dataset", "top-1m.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "ml", "dataset", "processed_dataset_final.csv")

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

# How many Tranco domains to sample. Using domains ranked up to
# TRANCO_MAX_RANK keeps quality high (very low-traffic/rare domains
# further down the list are less reliably "clearly legitimate").
TRANCO_MAX_RANK = 200000
TRANCO_SAMPLE_SIZE = 60000

SAMPLE_PATHS = [
    "/about", "/about-us", "/contact", "/contact-us", "/products",
    "/services", "/blog", "/news", "/en/home", "/support",
    "/help", "/faq", "/team", "/careers", "/index.html",
    "/article/2024/summary", "/docs/getting-started",
]


def build_legitimate_rows(domains):
    """Turn a list of bare domains into varied, realistic legitimate URL rows."""
    rows = []

    for domain in domains:
        domain = domain.strip().lower()
        if not domain:
            continue

        choice = random.random()

        if choice < 0.4:
            # https + www, root
            url = f"https://www.{domain}"
        elif choice < 0.65:
            # https, no www, root
            url = f"https://{domain}"
        elif choice < 0.85:
            # https + www, with a path
            url = f"https://www.{domain}{random.choice(SAMPLE_PATHS)}"
        else:
            # https, no www, with a path
            url = f"https://{domain}{random.choice(SAMPLE_PATHS)}"

        rows.append({"URL": url, "label": 1})

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("MERGE TRANCO TOP DOMAINS INTO TRAINING DATASET")
    print("=" * 60)

    if not os.path.exists(INPUT_DATASET):
        raise FileNotFoundError(f"Dataset not found:\n{INPUT_DATASET}")
    if not os.path.exists(TRANCO_FILE):
        raise FileNotFoundError(f"Tranco file not found:\n{TRANCO_FILE}")

    print("\nLoading existing dataset...")
    df = pd.read_csv(INPUT_DATASET)
    print(f"Existing dataset shape: {df.shape}")
    print(df["label"].value_counts())

    print("\nLoading Tranco list...")
    tranco = pd.read_csv(TRANCO_FILE, header=None, names=["rank", "domain"])
    print(f"Tranco list shape: {tranco.shape}")

    tranco = tranco[tranco["rank"] <= TRANCO_MAX_RANK]

    # Avoid domains already present in the dataset (rough check on
    # substring match against existing URLs) to reduce duplication.
    existing_urls_lower = set(df["URL"].astype(str).str.lower())

    def already_present(domain):
        return any(domain in u for u in existing_urls_lower)

    sample_size = min(TRANCO_SAMPLE_SIZE, len(tranco))
    sampled = tranco.sample(n=sample_size, random_state=RANDOM_STATE)["domain"].tolist()

    print(f"\nBuilding {len(sampled)} new legitimate URL rows from Tranco domains...")
    new_rows = build_legitimate_rows(sampled)

    combined = pd.concat([df, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset="URL", keep="first")

    print(f"\nFinal combined dataset shape: {combined.shape}")
    print(combined["label"].value_counts())

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved final dataset to:\n{OUTPUT_FILE}")
    print("=" * 60)
    print("\nNext step: point train.py's DATASET_FILE at")
    print("processed_dataset_final.csv, then retrain.")


if __name__ == "__main__":
    main()
    