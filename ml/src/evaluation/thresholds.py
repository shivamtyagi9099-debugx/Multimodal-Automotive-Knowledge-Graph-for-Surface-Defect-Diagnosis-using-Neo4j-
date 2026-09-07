"""Confidence-threshold analysis shared by validation and final evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import device as Device
from torch.nn import Module
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ThresholdAnalysis:
    """Summarize coverage and errors for the manual-review threshold."""

    confidence_threshold: float
    sample_count: int
    accepted_count: int
    rejected_count: int
    accepted_error_count: int
    coverage: float
    manual_review_rate: float
    accepted_accuracy: float | None
    rejected_by_true_class: dict[str, int]
    mean_confidence_correct: float | None
    mean_confidence_incorrect: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


def collect_model_predictions(
    model: Module,
    data_loader: DataLoader,
    device: Device,
) -> tuple[list[int], list[int], list[float]]:
    """Collect true labels, argmax predictions, and maximum probabilities."""
    if len(data_loader.dataset) == 0:
        raise ValueError("Prediction dataset is empty")
    model.to(device)
    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    confidences: list[float] = []
    with torch.inference_mode():
        for images, labels in data_loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    return true_labels, predicted_labels, confidences


def calculate_threshold_analysis(
    true_labels: list[int],
    predicted_labels: list[int],
    confidences: list[float],
    class_names: list[str],
    confidence_threshold: float,
) -> ThresholdAnalysis:
    """Calculate operational review coverage without changing argmax metrics."""
    sample_count = len(true_labels)
    if sample_count == 0:
        raise ValueError("At least one labelled sample is required")
    if len(predicted_labels) != sample_count or len(confidences) != sample_count:
        raise ValueError("Labels, predictions, and confidences must have equal length")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between zero and one")
    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must contain at least two unique values")
    valid_labels = set(range(len(class_names)))
    if not (set(true_labels) | set(predicted_labels)).issubset(valid_labels):
        raise ValueError("Observed labels fall outside the configured class range")
    if any(not 0.0 <= confidence <= 1.0 for confidence in confidences):
        raise ValueError("Confidence values must be between zero and one")

    accepted = [value >= confidence_threshold for value in confidences]
    correct = [truth == prediction for truth, prediction in zip(true_labels, predicted_labels)]
    accepted_count = sum(accepted)
    rejected_count = sample_count - accepted_count
    accepted_correct = sum(
        is_accepted and is_correct
        for is_accepted, is_correct in zip(accepted, correct)
    )
    rejected_by_class = {class_name: 0 for class_name in class_names}
    for truth, is_accepted in zip(true_labels, accepted):
        if not is_accepted:
            rejected_by_class[class_names[truth]] += 1

    correct_confidences = [
        confidence
        for confidence, is_correct in zip(confidences, correct)
        if is_correct
    ]
    incorrect_confidences = [
        confidence
        for confidence, is_correct in zip(confidences, correct)
        if not is_correct
    ]
    return ThresholdAnalysis(
        confidence_threshold=confidence_threshold,
        sample_count=sample_count,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        accepted_error_count=accepted_count - accepted_correct,
        coverage=accepted_count / sample_count,
        manual_review_rate=rejected_count / sample_count,
        accepted_accuracy=(
            accepted_correct / accepted_count if accepted_count else None
        ),
        rejected_by_true_class=rejected_by_class,
        mean_confidence_correct=(
            sum(correct_confidences) / len(correct_confidences)
            if correct_confidences
            else None
        ),
        mean_confidence_incorrect=(
            sum(incorrect_confidences) / len(incorrect_confidences)
            if incorrect_confidences
            else None
        ),
    )


def select_confidence_threshold(
    true_labels: list[int],
    predicted_labels: list[int],
    confidences: list[float],
    class_names: list[str],
    minimum_coverage: float = 0.65,
    candidate_thresholds: list[float] | None = None,
) -> tuple[float, list[ThresholdAnalysis]]:
    """Select a rejection threshold using validation predictions only.

    Among thresholds meeting the configured minimum coverage, the selector
    maximizes accepted accuracy, then coverage, and finally prefers the lower
    threshold. This is an operational review policy rather than probability
    calibration.
    """
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in (0, 1]")
    thresholds = candidate_thresholds or [
        round(value / 100, 2) for value in range(25, 91, 5)
    ]
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("candidate thresholds must be within [0, 1]")
    analyses = [
        calculate_threshold_analysis(
            true_labels,
            predicted_labels,
            confidences,
            class_names,
            threshold,
        )
        for threshold in sorted(set(thresholds))
    ]
    eligible = [item for item in analyses if item.coverage >= minimum_coverage]
    if not eligible:
        eligible = analyses
    selected = max(
        eligible,
        key=lambda item: (
            -1.0 if item.accepted_accuracy is None else item.accepted_accuracy,
            item.coverage,
            -item.confidence_threshold,
        ),
    )
    return selected.confidence_threshold, analyses
