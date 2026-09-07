"""Tests for per-class metrics and confusion-matrix output."""

from pathlib import Path

import numpy as np

from ml.src.evaluation.confusion_matrix import (
    build_confusion_matrix,
    plot_confusion_matrix,
)
from ml.src.evaluation.metrics import (
    calculate_overall_accuracy,
    calculate_per_class_metrics,
    save_metrics_report,
)

CLASS_NAMES = ["normal", "scratch", "dent", "rust"]
TRUE_LABELS = [0, 0, 1, 1, 2, 2, 3, 3]
PREDICTED_LABELS = [0, 1, 1, 1, 2, 3, 3, 3]


def test_per_class_metric_generation(tmp_path: Path) -> None:
    """Every configured class should have precision, recall, F1, and support."""
    metrics = calculate_per_class_metrics(
        TRUE_LABELS, PREDICTED_LABELS, CLASS_NAMES
    )
    assert metrics["class_name"].tolist() == CLASS_NAMES
    assert metrics.columns.tolist() == [
        "class_name",
        "precision",
        "recall",
        "f1_score",
        "support",
    ]
    assert metrics["support"].tolist() == [2, 2, 2, 2]

    output_path = tmp_path / "metrics.csv"
    save_metrics_report(metrics, output_path)
    assert output_path.is_file()


def test_overall_accuracy_is_secondary_summary() -> None:
    """Accuracy should still be available as a secondary measure."""
    assert calculate_overall_accuracy(TRUE_LABELS, PREDICTED_LABELS) == 0.75


def test_confusion_matrix_generation(tmp_path: Path) -> None:
    """A fixed four-class matrix and labelled PNG should be generated."""
    matrix = build_confusion_matrix(TRUE_LABELS, PREDICTED_LABELS, 4)
    assert matrix.shape == (4, 4)
    assert int(matrix.sum()) == len(TRUE_LABELS)
    assert np.diag(matrix).tolist() == [1, 2, 1, 2]

    output_path = tmp_path / "confusion_matrix.png"
    plot_confusion_matrix(matrix, CLASS_NAMES, output_path)
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
