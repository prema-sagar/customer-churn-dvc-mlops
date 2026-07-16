import os
import json
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.makedirs("reports", exist_ok=True)

print("Loading model...")
model = joblib.load("models/model.pkl")

print("Loading feature names...")
features = joblib.load("models/features.pkl")

print("Loading test dataset...")
df = pd.read_csv("data/processed/test.csv")

y = df["Churn"]
X = df.drop("Churn", axis=1)

X = pd.get_dummies(X)

# Make sure test columns match training columns
X = X.reindex(columns=features, fill_value=0)

y = y.map({"Yes": 1, "No": 0})

print("Predicting...")
predictions = model.predict(X)

metrics = {
    "accuracy": round(accuracy_score(y, predictions), 4),
    "precision": round(precision_score(y, predictions), 4),
    "recall": round(recall_score(y, predictions), 4),
    "f1_score": round(f1_score(y, predictions), 4)
}

with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(metrics)
print("Evaluation completed.")
