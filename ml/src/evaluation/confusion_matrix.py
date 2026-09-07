"""Confusion-matrix calculation and rendering."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import ndarray
from sklearn.metrics import confusion_matrix


def build_confusion_matrix(
    true_labels: list[int],
    predicted_labels: list[int],
    number_of_classes: int,
) -> ndarray:
    """Build a fixed-size class-by-class confusion matrix."""
    if number_of_classes < 2:
        raise ValueError("number_of_classes must be at least 2")
    if not true_labels:
        raise ValueError("At least one labelled sample is required")
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have equal length")
    labels = list(range(number_of_classes))
    observed = set(true_labels) | set(predicted_labels)
    if not observed.issubset(set(labels)):
        raise ValueError(f"Labels must be in the range 0..{number_of_classes - 1}")
    return confusion_matrix(true_labels, predicted_labels, labels=labels)


def plot_confusion_matrix(
    matrix: ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Render and save a labelled confusion matrix as a PNG image."""
    expected_shape = (len(class_names), len(class_names))
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match classes {expected_shape}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
