"""Adapter between the FastAPI backend and the completed ML package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import Tensor
from torch.nn import Module

from backend.app.schemas.prediction import PredictionResult
from ml.src.inference.predict import (
    load_image_for_inference,
    predict_single_image,
)
from ml.src.models.classifier import load_classifier_checkpoint


def load_model_bundle(
    model_path: Path,
) -> tuple[Module, dict[str, Any]]:
    """Load the trained model and the metadata saved with its checkpoint."""
    return load_classifier_checkpoint(model_path, device="cpu")


def load_model(model_path: Path, number_of_classes: int) -> Module:
    """Load a classifier and verify its saved output-class count."""
    model, metadata = load_model_bundle(model_path)
    if len(metadata["class_names"]) != number_of_classes:
        raise ValueError("Checkpoint class count does not match project configuration")
    return model


def preprocess_image(
    image_path: Path,
    image_size: int,
    preprocessing_geometry: str = "square_pad_then_resize",
) -> Tensor:
    """Convert a verified upload into the model's normalized input tensor."""
    return load_image_for_inference(
        image_path, image_size, preprocessing_geometry
    )


def predict_class(
    model: Module,
    image_tensor: Tensor,
    class_names: list[str],
    confidence_threshold: float,
) -> PredictionResult:
    """Run ML inference and map the result into the backend schema."""
    result = predict_single_image(
        model, image_tensor, class_names, confidence_threshold
    )
    return PredictionResult(
        predicted_class=result.predicted_class,
        most_likely_class=result.most_likely_class,
        class_index=result.class_index,
        confidence=result.confidence,
        requires_manual_review=result.requires_manual_review,
        message=result.message,
    )
