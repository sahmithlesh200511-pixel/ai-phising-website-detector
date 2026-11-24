import pandas as pd
df = pd.read_csv("ml/dataset/processed_dataset.csv")

print(df["label"].value_counts())
print()
print("--- Sample label=1 (should be Legitimate) ---")
print(df[df["label"] == 1]["URL"].head(10).to_list())
print()
print("--- Sample label=0 (should be Phishing) ---")
print(df[df["label"] == 0]["URL"].head(10).to_list())
