
import os
import numpy as np
import pandas as pd

from src.shipment_prep import (
    PROMISED_HOURS,
    FEATURES,
    clean_shipments,
    build_features,
)


os.makedirs("outputs", exist_ok=True)

# Load the generated data.
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# Capture raw row count and pre-cleaning metrics.
raw_rows = len(raw)
raw_duplicates = int(raw.duplicated().sum())

# Remove duplicates for the "before" checks.
deduplicated = raw.drop_duplicates().copy()

missing_distance = int(deduplicated["distance_km"].isna().sum())
missing_weight = int(deduplicated["parcel_weight_kg"].isna().sum())
missing_value = int(deduplicated["declared_value"].isna().sum())
missing_payment = int(deduplicated["payment_mode"].isna().sum())

sentinel_weight = int(
    (deduplicated["parcel_weight_kg"] > 100).sum()
)

service_spellings_before = int(
    deduplicated["service_type"].nunique()
)

# Clean the data and engineer features.
clean, medians = clean_shipments(raw)
model_data = build_features(clean, hubs)

service_spellings_after = int(
    model_data["service_type"].nunique()
)

# Create the target using the promised delivery hours.
model_data["is_delayed"] = (
    model_data["delivery_hours"]
    > model_data["promised_hours"]
).astype(int)

# Save the complete modelling table.
model_data.to_csv("data/modelling_table.csv", index=False)

# --------------------------------------------------
# 1. Part A checks
# --------------------------------------------------

declared_log = np.log(model_data["declared_value_inr"])

q1 = model_data["delivery_hours"].quantile(0.25)
q3 = model_data["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", raw_duplicates),
    ("rows_after_dedup", len(deduplicated)),
    ("missing_distance_km_before", missing_distance),
    ("missing_parcel_weight_before", missing_weight),
    ("sentinel_weight_rows", sentinel_weight),
    ("missing_declared_value_before", missing_value),
    ("missing_payment_mode_before", missing_payment),
    ("service_type_spellings_before", service_spellings_before),
    ("service_type_spellings_after", service_spellings_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(model_data)),
    (
        "nulls_in_features_and_target",
        int(model_data[FEATURES + ["delivery_hours", "is_delayed"]]
            .isna().sum().sum()),
    ),
    ("delayed_count", int(model_data["is_delayed"].sum())),
    ("delay_rate", model_data["is_delayed"].mean()),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lower_bound),
    ("hours_upper_bound", upper_bound),
    (
        "hours_outlier_count",
        int(
            (
                (model_data["delivery_hours"] < lower_bound)
                | (model_data["delivery_hours"] > upper_bound)
            ).sum()
        ),
    ),
    ("declared_value_mean_inr", model_data["declared_value_inr"].mean()),
    ("declared_value_median_inr", model_data["declared_value_inr"].median()),
    ("log_declared_value_mean", declared_log.mean()),
    ("log_declared_value_median", declared_log.median()),
]

checks_df = pd.DataFrame(checks, columns=["metric", "value"])
checks_df["value"] = pd.to_numeric(checks_df["value"]).round(4)
checks_df.to_csv("outputs/part_a_checks.csv", index=False)

# --------------------------------------------------
# 2. Delay evidence by group
# --------------------------------------------------

model_data["distance_band"] = pd.cut(
    model_data["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True,
)

group_tables = []

for dimension in ["service_type", "hub_tier", "distance_band"]:
    grouped = model_data.groupby(
        dimension, observed=True
    ).agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean"),
    ).reset_index()

    grouped = grouped.rename(columns={dimension: "group"})
    grouped.insert(0, "dimension", dimension)
    group_tables.append(grouped)

delay_by_group = pd.concat(group_tables, ignore_index=True)
delay_by_group["delay_rate"] = delay_by_group["delay_rate"].round(4)
delay_by_group["mean_hours"] = delay_by_group["mean_hours"].round(4)

delay_by_group.to_csv(
    "outputs/part_a_delay_by_group.csv",
    index=False,
)

# --------------------------------------------------
# 3. Numeric correlation matrix
# --------------------------------------------------

correlations = model_data.select_dtypes(
    include="number"
).corr()

correlations.to_csv(
    "outputs/part_a_correlations.csv",
    index=True,
    index_label="column",
)

print("Part A completed.")
print("Rows in modelling table:", len(model_data))
print("Delayed shipments:", int(model_data["is_delayed"].sum()))
print("Delay rate:", round(model_data["is_delayed"].mean(), 4))
print("Files written:")
print("- data/modelling_table.csv")
print("- outputs/part_a_checks.csv")
print("- outputs/part_a_delay_by_group.csv")
print("- outputs/part_a_correlations.csv")
