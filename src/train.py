import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# Create models directory
os.makedirs("models", exist_ok=True)

print("Loading training dataset...")

df = pd.read_csv("data/processed/train.csv")

print(f"Training dataset shape: {df.shape}")

# Target column
y = df["Churn"]

# Features
X = df.drop("Churn", axis=1)

# Convert categorical columns to numeric
X = pd.get_dummies(X)

# Convert target to numeric
y = y.map({"Yes": 1, "No": 0})

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "models/model.pkl")

# Save feature names (needed during prediction)
joblib.dump(X.columns.tolist(), "models/features.pkl")

print("===================================")
print("Model training completed")
print("Model saved to models/model.pkl")
print("===================================")
