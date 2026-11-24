"""
Add hyphenated, slug-style legitimate paths to the training dataset.

Why: Real blog posts, articles, and product pages very commonly use
hyphenated URL slugs (e.g. "/phishing-link-checker/",
"/best-practices-2024/", "/how-to-guide/"). The dataset so far has
very few legitimate examples like this, so the model over-associates
hyphens-in-path with phishing. This script generates many realistic
hyphenated-slug legitimate URLs across the Tranco domains already
sampled, to correct that imbalance.

Input:
    ml/dataset/processed_dataset_final.csv

Output:
    ml/dataset/processed_dataset_v2.csv

Usage:
    python ml/augment_slugs.py
"""

import os
import sys
import random

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

INPUT_FILE = os.path.join(BASE_DIR, "ml", "dataset", "processed_dataset_final.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "ml", "dataset", "processed_dataset_v2.csv")

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

# Word banks to build realistic multi-word hyphenated slugs
TOPIC_WORDS = [
    "guide", "review", "checker", "tool", "tips", "best-practices",
    "how-to", "tutorial", "overview", "comparison", "update", "report",
    "faq", "checklist", "case-study", "insights", "trends", "analysis",
    "strategy", "solutions", "features", "pricing", "benefits",
    "security", "privacy", "setup", "integration", "migration",
]

PREFIX_WORDS = [
    "what-is", "how-to", "top-10", "ultimate", "complete", "beginners",
    "advanced", "free", "best", "new", "2024", "2025", "2026",
]

CATEGORIES = [
    "blog", "articles", "news", "resources", "docs", "help", "learn",
    "insights", "guides", "products", "solutions",
]


def random_slug():
    parts = []
    if random.random() < 0.5:
        parts.append(random.choice(PREFIX_WORDS))
    parts.append(random.choice(TOPIC_WORDS))
    if random.random() < 0.4:
        parts.append(random.choice(TOPIC_WORDS))
    return "-".join(parts)


def build_slug_url(domain):
    category = random.choice(CATEGORIES)
    slug = random_slug()

    use_www = random.random() < 0.5
    host = f"www.{domain}" if use_www else domain

    if random.random() < 0.6:
        path = f"/{category}/{slug}/"
    else:
        path = f"/{category}/{slug}"

    return f"https://{host}{path}"


def main():
    print("=" * 60)
    print("AUGMENT: hyphenated slug-style legitimate paths")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Dataset not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print(f"\nExisting dataset shape: {df.shape}")
    print(df["label"].value_counts())

    # Pull domains from existing legitimate rows to build slug variants
    legit_urls = df[df["label"] == 1]["URL"].astype(str)

    domains = set()
    for url in legit_urls:
        cleaned = url.replace("https://", "").replace("http://", "")
        cleaned = cleaned.replace("www.", "")
        domain = cleaned.split("/")[0]
        if domain:
            domains.add(domain)

    domains = list(domains)
    print(f"\nFound {len(domains)} unique legitimate domains to build slugs from.")

    sample_size = min(60000, len(domains) * 3)
    new_rows = []

    for _ in range(sample_size):
        domain = random.choice(domains)
        url = build_slug_url(domain)
        new_rows.append({"URL": url, "label": 1})

    new_df = pd.DataFrame(new_rows)
    print(f"Generated {len(new_df)} new hyphenated-slug legitimate rows.")

    combined = pd.concat([df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="URL", keep="first")

    print(f"\nFinal dataset shape: {combined.shape}")
    print(combined["label"].value_counts())

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to:\n{OUTPUT_FILE}")
    print("=" * 60)
    print("\nNext step: point train.py's DATASET_FILE at")
    print("processed_dataset_v2.csv, set MAX_SAMPLES = None, then retrain.")


if __name__ == "__main__":
    main()