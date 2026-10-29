
import os
import pandas as pd

os.makedirs("outputs", exist_ok=True)

# Load the prepared shipment data
data = pd.read_csv("data/modelling_table.csv")

# Create distance bands
data["distance_band"] = pd.cut(
    data["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True,
)

# Confirm the delay target exists
if "is_delayed" not in data.columns:
    data["is_delayed"] = (
        data["delivery_hours"] > data["promised_hours"]
    ).astype(int)

# 1. Delay rate by individual operational dimensions
group_tables = []

for dimension in ["service_type", "hub_tier", "distance_band"]:
    grouped = (
        data.groupby(dimension, observed=True)
        .agg(
            shipments=("is_delayed", "size"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
            mean_delivery_hours=("delivery_hours", "mean"),
        )
        .reset_index()
        .rename(columns={dimension: "segment"})
    )

    grouped.insert(0, "dimension", dimension)
    group_tables.append(grouped)

segment_summary = pd.concat(group_tables, ignore_index=True)
segment_summary["delay_rate"] = segment_summary["delay_rate"].round(4)
segment_summary["mean_delivery_hours"] = (
    segment_summary["mean_delivery_hours"].round(4)
)

segment_summary.to_csv(
    "outputs/part_d_segment_summary.csv", index=False
)

# 2. Combined service type, hub tier and distance segments
combined = (
    data.groupby(
        ["service_type", "hub_tier", "distance_band"],
        observed=True,
    )
    .agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_delivery_hours=("delivery_hours", "mean"),
    )
    .reset_index()
)

# Avoid overinterpreting segments with very few shipments
combined["delay_rate"] = combined["delay_rate"].round(4)
combined["mean_delivery_hours"] = combined["mean_delivery_hours"].round(4)

combined.sort_values(
    ["delay_rate", "shipments"],
    ascending=[False, False],
).to_csv(
    "outputs/part_d_combined_segments.csv", index=False
)

# 3. Rank the individual segments by delay rate
ranked = segment_summary.sort_values(
    ["delay_rate", "shipments"],
    ascending=[False, False],
).reset_index(drop=True)

ranked.insert(0, "risk_rank", range(1, len(ranked) + 1))
ranked.to_csv("outputs/part_d_risk_ranking.csv", index=False)

# 4. Business-facing observations
overall_delay_rate = data["is_delayed"].mean()

service_rates = (
    data.groupby("service_type", observed=True)["is_delayed"]
    .mean()
    .sort_values(ascending=False)
)
hub_rates = (
    data.groupby("hub_tier", observed=True)["is_delayed"]
    .mean()
    .sort_values(ascending=False)
)
distance_rates = (
    data.groupby("distance_band", observed=True)["is_delayed"]
    .mean()
    .sort_values(ascending=False)
)

print("Part D completed successfully.")
print("Total shipments:", len(data))
print(f"Overall delay rate: {overall_delay_rate:.2%}")

print("\nDelay rate by service type:")
print((service_rates * 100).round(2).astype(str) + "%")

print("\nDelay rate by hub tier:")
print((hub_rates * 100).round(2).astype(str) + "%")

print("\nDelay rate by distance band:")
print((distance_rates * 100).round(2).astype(str) + "%")

print("\nHighest-risk combined segments:")
print(
    combined.sort_values("delay_rate", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\nFiles written:")
print("- outputs/part_d_segment_summary.csv")
print("- outputs/part_d_combined_segments.csv")
print("- outputs/part_d_risk_ranking.csv")
