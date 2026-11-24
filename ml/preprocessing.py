
"""
Preprocessing for AI Phishing Website Detector

Converts the PhiUSIIL dataset into:
    URL + label

The original dataset uses:
    1 = legitimate
    0 = phishing

Our ML model will use the URL itself and extract
23 features using feature_extraction.py.
"""

import os
import pandas as pd


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "ml",
    "dataset",
    "phishing_dataset.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "ml",
    "dataset",
    "processed_dataset.csv"
)


# ---------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------

def preprocess_dataset():

    print("=" * 60)
    print("AI PHISHING WEBSITE DETECTOR")
    print("DATASET PREPROCESSING")
    print("=" * 60)

    # Check whether dataset exists
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}"
        )

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Original dataset shape: {df.shape}")

    # -----------------------------------------------------
    # Check required columns
    # -----------------------------------------------------

    required_columns = ["URL", "label"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' was not found."
            )

    # -----------------------------------------------------
    # Keep only URL and label
    # -----------------------------------------------------

    df = df[["URL", "label"]].copy()

    # -----------------------------------------------------
    # Remove missing URLs
    # -----------------------------------------------------

    before = len(df)

    df = df.dropna(subset=["URL"])

    after = len(df)

    print(f"Removed missing URLs: {before - after}")

    # -----------------------------------------------------
    # Remove duplicate URLs
    # -----------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(subset=["URL"])

    after = len(df)

    print(f"Removed duplicate URLs: {before - after}")

    # -----------------------------------------------------
    # Clean URLs
    # -----------------------------------------------------

    df["URL"] = df["URL"].astype(str).str.strip()

    # Remove empty URLs
    df = df[df["URL"] != ""]

    # -----------------------------------------------------
    # Make sure labels are integers
    # -----------------------------------------------------

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    # Remove rows with invalid labels
    df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    # -----------------------------------------------------
    # Keep only valid labels
    # -----------------------------------------------------

    df = df[df["label"].isin([0, 1])]

    # -----------------------------------------------------
    # Save processed dataset
    # -----------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Display information
    # -----------------------------------------------------

    print("\nPreprocessing completed successfully!")

    print(f"Processed dataset shape: {df.shape}")

    print("\nLabel distribution:")

    print(
        df["label"].value_counts()
        .sort_index()
    )

    print("\nLabel meaning:")
    print("0 = Phishing")
    print("1 = Legitimate")

    print(f"\nSaved processed dataset to:")
    print(OUTPUT_FILE)

    print("=" * 60)


# ---------------------------------------------------------
# Run preprocessing
# ---------------------------------------------------------

if __name__ == "__main__":
    preprocess_dataset()

