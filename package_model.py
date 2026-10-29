
import json
import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier

from src.shipment_prep import FEATURES

os.makedirs("models", exist_ok=True)

# Load the prepared modelling dataset
data = pd.read_csv("data/modelling_table.csv")

X = data[FEATURES]
y_regression = data["delivery_hours"]
y_classification = data["is_delayed"].astype(int)

# Train the final regression model on all available labelled data
regression_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])
regression_model.fit(X, y_regression)

# Train the final classification model on all available labelled data
classification_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
)
classification_model.fit(X, y_classification)

# Save models
joblib.dump(regression_model, "models/delivery_time_model.joblib")
joblib.dump(classification_model, "models/delay_classifier.joblib")

# Save model configuration
config = {
    "features": FEATURES,
    "regression_model": "delivery_time_model.joblib",
    "classification_model": "delay_classifier.joblib",
    "classification_threshold": 0.35,
    "delay_target_definition": "delivery_hours > promised_hours",
    "service_promised_hours": {
        "Express": 36,
        "Standard": 60,
        "Economy": 96,
    },
}

with open("models/model_config.json", "w", encoding="utf-8") as file:
    json.dump(config, file, indent=4)

print("Model packaging completed successfully.")
print("Training rows:", len(data))
print("Regression model: models/delivery_time_model.joblib")
print("Classification model: models/delay_classifier.joblib")
print("Configuration: models/model_config.json")
