
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from src.shipment_prep import FEATURES

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load modelling data
data = pd.read_csv("data/modelling_table.csv")

X = data[FEATURES]
y = data["delivery_hours"]

# Split: 70% training, 15% validation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# Scale features using training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Helper function for regression metrics
def regression_metrics(model_name, split_name, actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        "model": model_name,
        "split": split_name,
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(actual, predicted),
    }

# Part B checks
checks = [
    ("train_rows", len(X_train)),
    ("val_rows", len(X_val)),
    ("test_rows", len(X_test)),
    ("train_scaled_max_abs_mean",
     float(np.abs(X_train_scaled.mean(axis=0)).max())),
    ("test_scaled_max_abs_mean",
     float(np.abs(X_test_scaled.mean(axis=0)).max())),
]

pd.DataFrame(checks, columns=["metric", "value"]).to_csv(
    "outputs/part_b_checks.csv", index=False
)

# Baseline: predict the mean training delivery time
baseline_prediction = np.full(len(y_val), y_train.mean())
validation_results = [
    regression_metrics("mean_baseline", "validation", y_val, baseline_prediction)
]

# Candidate models
models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "random_forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
}

# Decision tree depth diagnostic
depth_results = []
depth_options = [1, 2, 3, 4, 6, 8, 12, None]

for depth in depth_options:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    train_prediction = tree.predict(X_train)
    val_prediction = tree.predict(X_val)

    depth_results.append({
        "max_depth": "unlimited" if depth is None else depth,
        "train_rmse": np.sqrt(mean_squared_error(y_train, train_prediction)),
        "val_rmse": np.sqrt(mean_squared_error(y_val, val_prediction)),
    })

pd.DataFrame(depth_results).to_csv(
    "outputs/part_b_tree_depth.csv", index=False
)

# Select depth with the lowest validation RMSE
best_depth_row = min(
    depth_results,
    key=lambda row: (row["val_rmse"],
                     float("inf") if row["max_depth"] == "unlimited"
                     else row["max_depth"])
)
best_depth = (
    None if best_depth_row["max_depth"] == "unlimited"
    else int(best_depth_row["max_depth"])
)

models["decision_tree"] = DecisionTreeRegressor(
    max_depth=best_depth, random_state=42
)

# Fit candidates and calculate validation metrics
fitted_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model
    prediction = model.predict(X_val)
    validation_results.append(
        regression_metrics(name, "validation", y_val, prediction)
    )

# Save validation results
metrics_df = pd.DataFrame(validation_results)
metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)

# Gradient boosting learning-rate experiment
learning_rate_results = []

for rate in [0.01, 0.1, 1.0]:
    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=rate, random_state=42
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)

    learning_rate_results.append({
        "learning_rate": rate,
        "val_rmse": np.sqrt(mean_squared_error(y_val, prediction)),
        "val_r2": r2_score(y_val, prediction),
    })

pd.DataFrame(learning_rate_results).to_csv(
    "outputs/part_b_gb_learning_rate.csv", index=False
)

# Select the two best models using validation RMSE
ranked_models = metrics_df[
    metrics_df["model"] != "mean_baseline"
].sort_values("RMSE")

top_two = ranked_models["model"].head(2).tolist()

# Five-fold cross-validation on training + validation rows
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name in top_two:
    model = models[name]
    scores = cross_val_score(
        model, X_train_val, y_train_val, cv=cv, scoring="r2"
    )
    cv_results.append({
        "model": name,
        "mean_r2": scores.mean(),
        "std_r2": scores.std(),
    })

pd.DataFrame(cv_results).to_csv(
    "outputs/part_b_cv.csv", index=False
)

# Final winner selected by validation RMSE
winner_name = ranked_models.iloc[0]["model"]
winner_model = fitted_models[winner_name]

# Evaluate the selected model on the test set exactly once
test_prediction = winner_model.predict(X_test)
test_result = regression_metrics(
    winner_name, "test", y_test, test_prediction
)

# Add the single final test row
final_metrics = pd.concat(
    [metrics_df, pd.DataFrame([test_result])],
    ignore_index=True
)
final_metrics.to_csv("outputs/part_b_metrics.csv", index=False)

print("Part B completed successfully.")
print(f"Training rows: {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Test rows: {len(X_test)}")
print(f"Selected tree depth: {best_depth}")
print(f"Best validation model: {winner_name}")
print(f"Final test RMSE: {test_result['RMSE']:.4f}")
print("\nValidation metrics:")
print(metrics_df.sort_values("RMSE").to_string(index=False))
