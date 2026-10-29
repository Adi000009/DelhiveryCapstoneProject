
import json
import os
import joblib
import pandas as pd

from src.shipment_prep import clean_shipments, build_features

os.makedirs("outputs", exist_ok=True)

# Load model configuration
with open("models/model_config.json", "r", encoding="utf-8") as file:
    config = json.load(file)

# Load saved models
regression_model = joblib.load(
    os.path.join("models", config["regression_model"])
)
classification_model = joblib.load(
    os.path.join("models", config["classification_model"])
)

# Load new shipments and hub information
new_shipments = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# Clean the new shipment data using training-data medians
# These medians are calculated from the original labelled training dataset.
training_data = pd.read_csv("data/raw_shipments.csv")
_, training_medians = clean_shipments(training_data)

clean_new, _ = clean_shipments(new_shipments, medians=training_medians)

# Build model features
model_data = build_features(clean_new, hubs)
features = config["features"]

# Validate input features
missing_features = [feature for feature in features if feature not in model_data.columns]
if missing_features:
    raise ValueError(f"Missing required model features: {missing_features}")

if model_data[features].isna().any().any():
    raise ValueError("Missing values remain in model features after preparation.")

# Generate predictions
model_data["predicted_delivery_hours"] = regression_model.predict(
    model_data[features]
)

delay_probabilities = classification_model.predict_proba(
    model_data[features]
)[:, 1]

model_data["delay_probability"] = delay_probabilities
model_data["predicted_is_delayed"] = (
    delay_probabilities >= config["classification_threshold"]
).astype(int)

# Add a human-readable prediction label
model_data["delay_prediction"] = model_data["predicted_is_delayed"].map({
    0: "On time",
    1: "Potential delay",
})

# Save prediction output
output_columns = [
    column for column in [
        "shipment_id",
        "hub_id",
        "service_type",
        "distance_km",
        "promised_hours",
        "predicted_delivery_hours",
        "delay_probability",
        "predicted_is_delayed",
        "delay_prediction",
    ]
    if column in model_data.columns
]

predictions = model_data[output_columns].copy()
predictions["predicted_delivery_hours"] = (
    predictions["predicted_delivery_hours"].round(2)
)
predictions["delay_probability"] = (
    predictions["delay_probability"].round(4)
)

predictions.to_csv("outputs/predictions.csv", index=False)

print("Prediction pipeline completed successfully.")
print("New shipment rows:", len(predictions))
print("Potentially delayed shipments:",
      int(predictions["predicted_is_delayed"].sum()))
print("Potential delay rate:",
      f"{predictions['predicted_is_delayed'].mean():.2%}")
print("Predictions saved to outputs/predictions.csv")
