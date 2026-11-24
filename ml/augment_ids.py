"""
Add legitimate URLs with numeric IDs / high-entropy path segments.

Why: Real legitimate sites very commonly have paths containing numeric
IDs, usernames with digits, hashes, or ticket numbers -- e.g. GitHub
usernames/repos, YouTube video IDs, e-commerce product IDs, blog post
dates. The training data so far has almost none of these, so the
model over-associates digit-heavy / high-entropy paths with phishing.

Input:
    ml/dataset/processed_dataset_v2.csv

Output:
    ml/dataset/processed_dataset_v3.csv

Usage:
    python ml/augment_ids.py
"""

import os
import sys
import random
import string

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

INPUT_FILE = os.path.join(BASE_DIR, "ml", "dataset", "processed_dataset_v2.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "ml", "dataset", "processed_dataset_v3.csv")

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

USERNAME_WORDS = [
    "alex", "sarah", "dev", "code", "team", "user", "admin", "project",
    "sahmithlesh", "priya", "kumar", "smith", "johnson", "tech", "lab",
]

REPO_WORDS = [
    "python", "webapp", "backend", "frontend", "api", "ml-project",
    "dataset", "portfolio", "website", "notes", "scripts", "tools",
]


def random_digits(n):
    return "".join(random.choice(string.digits) for _ in range(n))


def random_hash(n):
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def build_id_path():
    kind = random.random()

    if kind < 0.3:
        # GitHub-style: /username123-suffix/repo-name
        user = random.choice(USERNAME_WORDS) + random_digits(random.randint(3, 6))
        repo = random.choice(REPO_WORDS)
        return f"/{user}-pixel/{repo}"

    elif kind < 0.5:
        # YouTube-style video ID
        return f"/watch?v={random_hash(11)}"

    elif kind < 0.7:
        # E-commerce product ID
        return f"/product/{random_digits(random.randint(4, 8))}"

    elif kind < 0.85:
        # Ticket / order number
        return f"/order/{random_hash(3).upper()}{random_digits(5)}"

    else:
        # Dated blog post
        year = random.choice(["2023", "2024", "2025", "2026"])
        month = f"{random.randint(1,12):02d}"
        day = f"{random.randint(1,28):02d}"
        return f"/{year}/{month}/{day}/{random.choice(REPO_WORDS)}"


def build_url(domain):
    use_www = random.random() < 0.5
    host = f"www.{domain}" if use_www else domain
    path = build_id_path()
    return f"https://{host}{path}"


def main():
    print("=" * 60)
    print("AUGMENT: numeric-ID / high-entropy legitimate paths")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Dataset not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print(f"\nExisting dataset shape: {df.shape}")
    print(df["label"].value_counts())

    legit_urls = df[df["label"] == 1]["URL"].astype(str)
    domains = set()
    for url in legit_urls:
        cleaned = url.replace("https://", "").replace("http://", "").replace("www.", "")
        domain = cleaned.split("/")[0]
        if domain:
            domains.add(domain)

    domains = list(domains)
    print(f"\nFound {len(domains)} unique legitimate domains.")

    sample_size = min(50000, len(domains) * 2)
    new_rows = []
    for _ in range(sample_size):
        domain = random.choice(domains)
        url = build_url(domain)
        new_rows.append({"URL": url, "label": 1})

    new_df = pd.DataFrame(new_rows)
    print(f"Generated {len(new_df)} new ID-pattern legitimate rows.")

    combined = pd.concat([df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="URL", keep="first")

    print(f"\nFinal dataset shape: {combined.shape}")
    print(combined["label"].value_counts())

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to:\n{OUTPUT_FILE}")
    print("=" * 60)
    print("\nNext step: point train.py's DATASET_FILE at")
    print("processed_dataset_v3.csv, then retrain.")


if __name__ == "__main__":
    main()