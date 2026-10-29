
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)

# Load the prepared dataset
data = pd.read_csv("data/modelling_table.csv")

X = data[FEATURES]
y = data["is_delayed"].astype(int)

# Reproducible 70/15/15 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Candidate classifiers
models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "decision_tree": DecisionTreeClassifier(
        max_depth=6, random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
}

def evaluate_classifier(name, split, actual, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "model": name,
        "split": split,
        "threshold": threshold,
        "accuracy": accuracy_score(actual, predictions),
        "precision": precision_score(actual, predictions, zero_division=0),
        "recall": recall_score(actual, predictions, zero_division=0),
        "f1": f1_score(actual, predictions, zero_division=0),
        "roc_auc": roc_auc_score(actual, probabilities),
    }

# Fit models on training data and evaluate on validation data
fitted_models = {}
validation_results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model
    probabilities = model.predict_proba(X_val)[:, 1]
    validation_results.append(
        evaluate_classifier(name, "validation", y_val, probabilities)
    )

validation_df = pd.DataFrame(validation_results)
validation_df.to_csv("outputs/part_c_metrics.csv", index=False)

# Tune threshold for the best validation F1 model
best_model_name = validation_df.sort_values(
    ["f1", "recall"], ascending=False
).iloc[0]["model"]

best_model = fitted_models[best_model_name]
val_probabilities = best_model.predict_proba(X_val)[:, 1]

threshold_results = []
for threshold in np.arange(0.10, 0.91, 0.05):
    result = evaluate_classifier(
        best_model_name, "validation", y_val, val_probabilities, threshold
    )
    threshold_results.append(result)

threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv("outputs/part_c_thresholds.csv", index=False)

best_threshold = float(
    threshold_df.sort_values(
        ["f1", "recall"], ascending=False
    ).iloc[0]["threshold"]
)

# Evaluate the selected model on test data only after threshold selection
test_probabilities = best_model.predict_proba(X_test)[:, 1]
test_result = evaluate_classifier(
    best_model_name, "test", y_test, test_probabilities, best_threshold
)

all_metrics = pd.concat(
    [validation_df, pd.DataFrame([test_result])],
    ignore_index=True
)
all_metrics.to_csv("outputs/part_c_metrics.csv", index=False)

# Save test confusion matrix
test_predictions = (test_probabilities >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
).ravel()

pd.DataFrame(
    [
        {"actual": "on_time", "predicted": "on_time", "count": tn},
        {"actual": "on_time", "predicted": "delayed", "count": fp},
        {"actual": "delayed", "predicted": "on_time", "count": fn},
        {"actual": "delayed", "predicted": "delayed", "count": tp},
    ]
).to_csv("outputs/part_c_confusion_matrix.csv", index=False)

# Save classification report
report = classification_report(
    y_test, test_predictions,
    labels=[0, 1],
    target_names=["on_time", "delayed"],
    output_dict=True,
    zero_division=0,
)
pd.DataFrame(report).transpose().to_csv(
    "outputs/part_c_classification_report.csv", index=True,
    index_label="class"
)

print("Part C completed successfully.")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))
print("Selected model:", best_model_name)
print("Selected threshold:", round(best_threshold, 2))
print("\nTest metrics:")
print(pd.DataFrame([test_result]).to_string(index=False))
print("\nTest confusion matrix:")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
