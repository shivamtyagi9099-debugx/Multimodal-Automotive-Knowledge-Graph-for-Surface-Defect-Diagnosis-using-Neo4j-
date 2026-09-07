"""Required final evaluation metrics and confusion-matrix utilities.

The package initializer intentionally exports only lightweight helpers.  The
full evaluation runner imports the training device selector, so importing it
here would create a circular dependency while the training module is loading.
Use ``ml.src.evaluation.evaluate`` directly for ``EvaluationReport`` and
``evaluate_model``.
"""

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

__all__ = [
    "build_confusion_matrix",
    "calculate_overall_accuracy",
    "calculate_per_class_metrics",
    "plot_confusion_matrix",
    "save_metrics_report",
    "ThresholdAnalysis",
    "calculate_threshold_analysis",
    "collect_model_predictions",
]
