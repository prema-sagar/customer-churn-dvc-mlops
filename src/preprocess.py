import os
import yaml
import pandas as pd

from sklearn.model_selection import train_test_split

# -----------------------------
# Load Parameters
# -----------------------------
with open("params.yaml") as f:
    params = yaml.safe_load(f)

print("===================================")
print("Loading dataset...")
print("===================================")

df = pd.read_csv("data/raw/churn.csv")

print(f"Dataset shape: {df.shape}")

# Create output directory
os.makedirs("data/processed", exist_ok=True)

print("Splitting dataset...")

train_df, test_df = train_test_split(
    df,
    test_size=params["split"]["test_size"],
    random_state=params["model"]["random_state"]
)

train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("===================================")
print("Preprocessing completed!")
print(f"Training rows : {len(train_df)}")
print(f"Testing rows  : {len(test_df)}")
print("===================================")
