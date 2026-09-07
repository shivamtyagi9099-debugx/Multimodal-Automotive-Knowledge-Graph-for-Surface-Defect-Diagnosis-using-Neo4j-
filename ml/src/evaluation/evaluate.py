"""Final evaluation against the checksum-protected test set."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from numpy import ndarray
import numpy as np
from pandas import DataFrame
from torch import device as Device
from torch.nn import Module
from torch.utils.data import DataLoader

from ml.src.data.image_dataset import create_image_loader
from ml.src.data.validate_dataset import (
    calculate_manifest_checksum,
    verify_test_manifest_checksum,
)
from ml.src.evaluation.confusion_matrix import (
    build_confusion_matrix,
    plot_confusion_matrix,
)
from ml.src.evaluation.metrics import (
    calculate_overall_accuracy,
    calculate_per_class_metrics,
    save_metrics_report,
)
from ml.src.evaluation.thresholds import (
    ThresholdAnalysis,
    calculate_threshold_analysis,
    collect_model_predictions,
)
from ml.src.models.classifier import load_classifier_checkpoint
from ml.src.training.train import select_device


@dataclass
class EvaluationReport:
    """Contain all required outputs of the final test-set evaluation."""

    per_class_metrics: DataFrame
    overall_accuracy: float
    confusion_matrix: ndarray
    number_of_test_samples: int
    model_identifier: str
    test_manifest_checksum: str
    threshold_analysis: ThresholdAnalysis
    predictions: DataFrame
    macro_f1: float
    weighted_f1: float
    confidence_analysis: dict[str, object]


def _confidence_summary(
    confidences: list[float],
    correct: list[bool],
) -> dict[str, object]:
    """Return compact confidence evidence without implying calibration."""
    values = np.asarray(confidences, dtype=float)

    def describe(mask: np.ndarray) -> dict[str, float | int | None]:
        selected = values[mask]
        if selected.size == 0:
            return {"count": 0, "mean": None, "min": None, "median": None, "max": None}
        return {
            "count": int(selected.size),
            "mean": float(selected.mean()),
            "min": float(selected.min()),
            "median": float(np.median(selected)),
            "max": float(selected.max()),
        }

    correctness = np.asarray(correct, dtype=bool)
    histogram, edges = np.histogram(values, bins=[0.0, 0.4, 0.6, 0.8, 1.000001])
    return {
        "overall": describe(np.ones(values.shape, dtype=bool)),
        "correct": describe(correctness),
        "incorrect": describe(~correctness),
        "histogram": {
            "bin_edges": [0.0, 0.4, 0.6, 0.8, 1.0],
            "counts": [int(value) for value in histogram.tolist()],
        },
    }


def evaluate_model(
    model: Module,
    test_loader: DataLoader,
    class_names: list[str],
    device: Device,
    model_identifier: str = "unknown",
    test_manifest_checksum: str = "unknown",
    confidence_threshold: float = 0.60,
) -> EvaluationReport:
    """Run deterministic inference and calculate all required test metrics."""
    if len(test_loader.dataset) == 0:
        raise ValueError("Test dataset is empty")

    true_labels, predicted_labels, confidences = collect_model_predictions(
        model,
        test_loader,
        device,
    )

    metrics = calculate_per_class_metrics(
        true_labels, predicted_labels, class_names
    )
    matrix = build_confusion_matrix(
        true_labels, predicted_labels, len(class_names)
    )
    correct = [
        truth == prediction
        for truth, prediction in zip(true_labels, predicted_labels)
    ]
    supports = metrics["support"].astype(float)
    sample_paths = getattr(test_loader.dataset, "samples", None)
    filenames = (
        [Path(str(item[0])).name for item in sample_paths]
        if sample_paths is not None
        else [f"sample-{index}" for index in range(len(true_labels))]
    )
    return EvaluationReport(
        per_class_metrics=metrics,
        overall_accuracy=calculate_overall_accuracy(
            true_labels, predicted_labels
        ),
        confusion_matrix=matrix,
        number_of_test_samples=len(true_labels),
        model_identifier=model_identifier,
        test_manifest_checksum=test_manifest_checksum,
        threshold_analysis=calculate_threshold_analysis(
            true_labels,
            predicted_labels,
            confidences,
            class_names,
            confidence_threshold,
        ),
        predictions=DataFrame(
            {
                "sample_index": list(range(len(true_labels))),
                "filename": filenames,
                "true_class": [class_names[index] for index in true_labels],
                "predicted_class": [
                    class_names[index] for index in predicted_labels
                ],
                "confidence": confidences,
                "requires_manual_review": [
                    confidence < confidence_threshold for confidence in confidences
                ],
                "is_correct": correct,
            }
        ),
        macro_f1=float(metrics["f1_score"].mean()),
        weighted_f1=float(
            (metrics["f1_score"] * supports).sum() / supports.sum()
        ),
        confidence_analysis=_confidence_summary(confidences, correct),
    )


def _file_sha256(path: Path) -> str:
    """Return a model-file identifier derived from its SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_class_names(path: Path) -> list[str]:
    """Load ordered class names from the central JSON configuration."""
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("class_names_path must contain at least two classes")
    return [str(value) for value in values]


def _write_report(report: EvaluationReport, output_path: Path) -> None:
    """Write non-tabular evaluation metadata and the matrix to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "overall_accuracy": report.overall_accuracy,
                "macro_f1": report.macro_f1,
                "weighted_f1": report.weighted_f1,
                "number_of_test_samples": report.number_of_test_samples,
                "model_identifier": report.model_identifier,
                "test_manifest_checksum": report.test_manifest_checksum,
                "confusion_matrix": report.confusion_matrix.tolist(),
                "per_class_metrics": report.per_class_metrics.to_dict(
                    orient="records"
                ),
                "threshold_analysis": report.threshold_analysis.to_dict(),
                "confidence_analysis": report.confidence_analysis,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Verify the frozen test set and generate final evaluation artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("ml/configs/evaluation.yaml")
    )
    args = parser.parse_args()

    configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    test_manifest_path = Path(configuration["test_manifest_path"])
    checksum_path = Path(configuration["test_manifest_checksum_path"])
    if not verify_test_manifest_checksum(test_manifest_path, checksum_path):
        raise SystemExit(
            "Frozen test manifest checksum mismatch; evaluation was stopped"
        )

    class_names = _load_class_names(Path(configuration["class_names_path"]))
    device = select_device()
    model_path = Path(configuration["model_path"])
    model, metadata = load_classifier_checkpoint(model_path, device)
    if metadata["class_names"] != class_names:
        raise ValueError("Checkpoint and project class order do not match")

    test_loader = create_image_loader(
        Path(configuration["frozen_test_path"]),
        class_names,
        int(configuration.get("image_size", metadata["image_size"])),
        int(configuration["batch_size"]),
        training=False,
        num_workers=int(configuration.get("num_workers", 0)),
        shuffle=False,
        preprocessing_geometry=str(
            metadata.get("preprocessing", {}).get(
                "geometry", "resize_stretch"
            )
        ),
    )
    report = evaluate_model(
        model,
        test_loader,
        class_names,
        device,
        model_identifier=_file_sha256(model_path),
        test_manifest_checksum=calculate_manifest_checksum(test_manifest_path),
        confidence_threshold=float(
            configuration.get(
                "confidence_threshold",
                metadata.get("confidence_threshold", 0.60),
            )
        ),
    )
    save_metrics_report(
        report.per_class_metrics, Path(configuration["metrics_output_path"])
    )
    plot_confusion_matrix(
        report.confusion_matrix,
        class_names,
        Path(configuration["confusion_matrix_output_path"]),
    )
    _write_report(report, Path(configuration["report_output_path"]))
    predictions_path = Path(configuration["predictions_output_path"])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    report.predictions.to_csv(predictions_path, index=False)
    confidence_path_value = configuration.get("confidence_analysis_output_path")
    if confidence_path_value:
        confidence_path = Path(confidence_path_value)
        confidence_path.parent.mkdir(parents=True, exist_ok=True)
        confidence_path.write_text(
            json.dumps(report.confidence_analysis, indent=2), encoding="utf-8"
        )
    misclassification_path_value = configuration.get(
        "misclassification_output_path"
    )
    if misclassification_path_value:
        misclassification_path = Path(misclassification_path_value)
        misclassification_path.parent.mkdir(parents=True, exist_ok=True)
        report.predictions.loc[~report.predictions["is_correct"]].to_csv(
            misclassification_path, index=False
        )
    print(report.per_class_metrics.to_string(index=False))
    print(f"Overall accuracy: {report.overall_accuracy:.4f}")
    print(f"Macro F1: {report.macro_f1:.4f}")
    print(f"Weighted F1: {report.weighted_f1:.4f}")


if __name__ == "__main__":
    main()
