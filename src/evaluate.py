# src/evaluate.py
import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def evaluate_model(model_path=None, test_path=None, report_dir=None):
    # Defaults
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, "models", "model.joblib")
    if test_path is None:
        test_path = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
    if report_dir is None:
        report_dir = os.path.join(PROJECT_ROOT, "reports")

    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(os.path.join(report_dir, "figures"), exist_ok=True)

    print(f" Loading model from: {model_path}")
    model = joblib.load(model_path)

    print(f" Loading test data from: {test_path}")
    df = pd.read_csv(test_path)
    X_test, y_test = df["text"], df["label"]

    # Predictions
    y_pred = model.predict(X_test)

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # Save metrics as JSON
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    with open(os.path.join(report_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n Metrics saved to reports/metrics.json")
    print(json.dumps(metrics, indent=4))

    # Classification report (auto-detect labels)
    unique_labels = np.unique(y_test)
    target_names_map = {0: "negative", 1: "neutral", 2: "positive"}
    target_names = [target_names_map[label] for label in unique_labels]

    report = classification_report(
        y_test, y_pred,
        labels=unique_labels,
        target_names=target_names,
        zero_division=0
    )

    with open(os.path.join(report_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    print(" Classification report saved to reports/classification_report.txt")

    # Confusion Matrix (with only available labels)
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    fig_path = os.path.join(report_dir, "figures", "confusion_matrix.png")
    plt.savefig(fig_path)
    plt.close()

    print(f" Confusion matrix saved to {fig_path}")


if __name__ == "__main__":
    print(">>> evaluate.py has started")
    evaluate_model()
    print(">>> Evaluation finished")
