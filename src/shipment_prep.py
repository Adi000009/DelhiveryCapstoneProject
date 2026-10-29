
import numpy as np
import pandas as pd


PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96,
}

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


def clean_shipments(raw, medians=None):
    """Clean shipment data and return the cleaned DataFrame and medians."""

    clean = raw.copy()

    # 1. Remove exact duplicate rows.
    clean = clean.drop_duplicates().copy()

    # 2. Normalize service type.
    clean["service_type"] = (
        clean["service_type"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    # 3. Replace missing payment modes.
    clean["payment_mode"] = clean["payment_mode"].fillna("Unknown")

    # 4. Convert declared value to numeric INR.
    clean["declared_value_inr"] = pd.to_numeric(
        clean["declared_value"]
        .astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )

    clean = clean.drop(columns=["declared_value"])

    # 5. Convert dispatch time to datetime.
    clean["dispatch_time"] = pd.to_datetime(
        clean["dispatch_time"],
        errors="coerce",
    )

    # 6. Treat scanner-fault weights as missing.
    clean.loc[
        clean["parcel_weight_kg"] > 100,
        "parcel_weight_kg",
    ] = np.nan

    # 7. Compute historical medians or reuse supplied medians.
    median_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr",
    ]

    if medians is None:
        medians = {
            column: float(clean[column].median())
            for column in median_columns
        }
    else:
        medians = {
            column: float(medians[column])
            for column in median_columns
        }

    # Convert to float BEFORE filling missing values.
    # This allows decimal medians such as 1121.5.
    for column in median_columns:
        clean[column] = pd.to_numeric(
            clean[column],
            errors="coerce",
        ).astype("float64")

        clean[column] = clean[column].fillna(medians[column])

    return clean, medians


def build_features(clean, hubs):
    """Attach hub details and create the required model features."""

    hub_columns = [
        "hub_id",
        "hub_city",
        "region",
        "hub_tier",
        "hub_daily_capacity",
    ]

    result = clean.merge(
        hubs[hub_columns],
        on="hub_id",
        how="left",
        validate="many_to_one",
    )

    # Promise duration based on service type.
    result["promised_hours"] = (
        result["service_type"].map(PROMISED_HOURS)
    )

    # Payment mode feature.
    result["is_cod"] = (
        result["payment_mode"].eq("COD").astype(int)
    )

    # Dispatch date/time features.
    result["is_weekend"] = (
        result["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    result["dispatch_hour"] = result["dispatch_time"].dt.hour

    # Log-transformed declared value.
    result["log_declared_value"] = np.log(
        result["declared_value_inr"]
    )

    return result
