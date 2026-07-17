import json
import joblib
import pandas as pd

print("=" * 50)
print("Customer Churn Prediction")
print("=" * 50)

# -----------------------------
# Load trained model
# -----------------------------
print("Loading trained model...")
model = joblib.load("models/model.pkl")

# -----------------------------
# Load feature names
# -----------------------------
print("Loading feature list...")
features = joblib.load("models/features.pkl")

# -----------------------------
# Load sample customer
# -----------------------------
print("Loading customer data...")
with open("sample_data/customer.json") as f:
    customer = json.load(f)

# Convert JSON to DataFrame
df = pd.DataFrame([customer])

# Convert categorical columns
df = pd.get_dummies(df)

# Match training columns
df = df.reindex(columns=features, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(df)[0]

# Probability
probability = model.predict_proba(df)[0]

print("\nPrediction Result")
print("-" * 30)

if prediction == 1:
    print("Customer is likely to CHURN")
else:
    print("Customer is NOT likely to churn")

print(f"Probability of staying : {probability[0] * 100:.2f}%")
print(f"Probability of churn   : {probability[1] * 100:.2f}%")

print("=" * 50)
