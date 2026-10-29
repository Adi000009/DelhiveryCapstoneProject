
import pandas as pd

from src.shipment_prep import (
    FEATURES,
    clean_shipments,
    build_features,
)

# Read generated data.
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# Clean historical shipments.
clean, medians = clean_shipments(raw)

# Build model features.
model_data = build_features(clean, hubs)

# Verify the important results.
print("Rows after cleaning:", len(clean))
print("Rows after hub merge:", len(model_data))
print("Imputation medians:", medians)
print("Number of model features:", len(FEATURES))
print("Missing feature values:", model_data[FEATURES].isna().sum().sum())

assert len(clean) == 1200
assert len(model_data) == 1200
assert len(FEATURES) == 17
assert model_data[FEATURES].notna().all().all()

print("\nShipment preparation tests passed!")
