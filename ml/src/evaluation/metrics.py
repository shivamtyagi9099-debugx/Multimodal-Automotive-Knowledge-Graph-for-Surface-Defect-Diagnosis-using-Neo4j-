"""Per-class and aggregate metrics for final classifier evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def _validate_labels(
    true_labels: list[int],
    predicted_labels: list[int],
    number_of_classes: int,
) -> None:
    """Validate label lengths and ranges before metric calculation."""
    if not true_labels:
        raise ValueError("At least one labelled sample is required")
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have equal length")
    valid_labels = set(range(number_of_classes))
    observed = set(true_labels) | set(predicted_labels)
    if not observed.issubset(valid_labels):
        raise ValueError(f"Labels must be in the range 0..{number_of_classes - 1}")


def calculate_per_class_metrics(
    true_labels: list[int],
    predicted_labels: list[int],
    class_names: list[str],
) -> DataFrame:
    """Calculate precision, recall, F1-score, and support for every class."""
    if len(set(class_names)) != len(class_names) or len(class_names) < 2:
        raise ValueError("class_names must contain at least two unique values")
    _validate_labels(true_labels, predicted_labels, len(class_names))

    precision, recall, f1_score, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class_name": class_names,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support.astype(int),
        }
    )


def calculate_overall_accuracy(
    true_labels: list[int],
    predicted_labels: list[int],
) -> float:
    """Calculate overall accuracy as a secondary summary metric."""
    if not true_labels:
        raise ValueError("At least one labelled sample is required")
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have equal length")
    return float(accuracy_score(true_labels, predicted_labels))


def save_metrics_report(
    metrics: DataFrame,
    output_path: Path,
) -> None:
    """Write a per-class metrics table to CSV."""
    required_columns = {"class_name", "precision", "recall", "f1_score", "support"}
    missing = required_columns - set(metrics.columns)
    if missing:
        raise ValueError(f"Metrics table is missing columns: {sorted(missing)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
