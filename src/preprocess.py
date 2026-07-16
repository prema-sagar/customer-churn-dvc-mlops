import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Create processed directory if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

print("Loading dataset...")
df = pd.read_csv("data/raw/churn.csv")

print(f"Dataset shape: {df.shape}")

# Remove customerID column
df = df.drop("customerID", axis=1)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

print("Splitting dataset...")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("================================")
print("Preprocessing completed!")
print(f"Training rows : {len(train_df)}")
print(f"Testing rows  : {len(test_df)}")
print("================================")
print("Pipeline Version 2")
