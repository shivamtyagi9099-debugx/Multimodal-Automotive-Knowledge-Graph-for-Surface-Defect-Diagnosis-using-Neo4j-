"""Tests for the combined final evaluation report."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.src.evaluation.evaluate import evaluate_model


def test_evaluation_report_contains_required_outputs() -> None:
    """Evaluation should combine per-class metrics and a confusion matrix."""
    logits = torch.tensor(
        [
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 8.0, 0.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 8.0],
        ]
    )
    labels = torch.tensor([0, 1, 2, 3])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    report = evaluate_model(
        nn.Identity(),
        loader,
        ["normal", "scratch", "dent", "rust"],
        torch.device("cpu"),
        model_identifier="test-model",
        test_manifest_checksum="test-checksum",
    )
    assert report.overall_accuracy == 1.0
    assert report.confusion_matrix.tolist() == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    assert report.per_class_metrics["f1_score"].tolist() == [1.0] * 4
    assert report.macro_f1 == 1.0
    assert report.weighted_f1 == 1.0
    assert report.confidence_analysis["incorrect"]["count"] == 0
    assert report.threshold_analysis.coverage == 1.0
    assert report.threshold_analysis.manual_review_rate == 0.0
    assert report.predictions["requires_manual_review"].tolist() == [False] * 4


def test_low_confidence_predictions_are_marked_for_manual_review() -> None:
    """Predictions below 60% should be retained in metrics but rejected in use."""
    logits = torch.tensor([[0.1, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]])
    labels = torch.tensor([0, 0])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    report = evaluate_model(
        nn.Identity(),
        loader,
        ["normal", "scratch", "dent", "rust"],
        torch.device("cpu"),
        confidence_threshold=0.60,
    )
    assert report.predictions["requires_manual_review"].tolist() == [True, False]
    assert report.threshold_analysis.coverage == 0.5
    assert report.threshold_analysis.manual_review_rate == 0.5
    assert report.threshold_analysis.rejected_by_true_class == {
        "normal": 1,
        "scratch": 0,
        "dent": 0,
        "rust": 0,
    }
