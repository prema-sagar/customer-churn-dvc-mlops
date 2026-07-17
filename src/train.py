import os
import joblib
import yaml
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load Parameters
# -----------------------------
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# -----------------------------
# Create models directory
# -----------------------------
os.makedirs("models", exist_ok=True)

print("===================================")
print("Loading training dataset...")
print("===================================")

df = pd.read_csv("data/processed/train.csv")

print(f"Training dataset shape: {df.shape}")

# -----------------------------
# Separate Features and Target
# -----------------------------
y = df["Churn"]
X = df.drop("Churn", axis=1)

# Convert categorical columns
X = pd.get_dummies(X)

# Convert target labels
y = y.map({"Yes": 1, "No": 0})

print("===================================")
print("Training Random Forest model...")
print("===================================")

model = RandomForestClassifier(
    n_estimators=params["model"]["n_estimators"],
    random_state=params["model"]["random_state"]
)

model.fit(X, y)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "models/model.pkl")

# Save feature names for prediction
joblib.dump(X.columns.tolist(), "models/features.pkl")

print("===================================")
print("Model training completed")
print("Model saved to models/model.pkl")
print("===================================")
