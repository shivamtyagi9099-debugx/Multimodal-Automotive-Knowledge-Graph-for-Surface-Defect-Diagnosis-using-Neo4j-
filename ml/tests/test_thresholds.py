"""Tests for validation-only confidence-threshold selection."""

from ml.src.evaluation.thresholds import select_confidence_threshold


def test_threshold_selection_uses_validation_coverage_policy() -> None:
    """Prefer accepted accuracy without violating minimum coverage."""
    selected, analyses = select_confidence_threshold(
        true_labels=[0, 0, 1, 1],
        predicted_labels=[0, 1, 1, 1],
        confidences=[0.90, 0.40, 0.85, 0.70],
        class_names=["normal", "scratch"],
        minimum_coverage=0.75,
        candidate_thresholds=[0.30, 0.60, 0.80],
    )
    assert selected == 0.60
    assert len(analyses) == 3


def test_threshold_selection_rejects_invalid_policy() -> None:
    """Coverage must be a real fraction."""
    try:
        select_confidence_threshold(
            [0], [0], [0.9], ["a", "b"], minimum_coverage=0.0
        )
    except ValueError as exc:
        assert "minimum_coverage" in str(exc)
        return
    raise AssertionError("Expected invalid minimum coverage to fail")
