"""Single-image PyTorch inference with a transparent confidence threshold."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.nn import Module

from ml.src.data.image_dataset import build_evaluation_transform
from ml.src.data.image_integrity import normalized_rgb
from ml.src.models.classifier import load_classifier_checkpoint
from ml.src.training.train import select_device

UNKNOWN_MESSAGE = "Defect unknown - Please review manually."


@dataclass(frozen=True)
class PredictionResult:
    """Represent one model prediction and its manual-review status."""

    predicted_class: str
    most_likely_class: str
    class_index: int | None
    confidence: float
    requires_manual_review: bool
    message: str


def load_image_for_inference(
    image_path: Path,
    image_size: int,
    preprocessing_geometry: str = "square_pad_then_resize",
) -> Tensor:
    """Load one image as a normalized, batched RGB tensor."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    try:
        with Image.open(image_path) as image:
            image.load()
            rgb_image = normalized_rgb(image)
            tensor = build_evaluation_transform(
                image_size, preprocessing_geometry
            )(rgb_image)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"File is not a readable image: {image_path}") from exc
    return tensor.unsqueeze(0)


def predict_single_image(
    model: Module,
    image_tensor: Tensor,
    class_names: list[str],
    confidence_threshold: float = 0.60,
) -> PredictionResult:
    """Predict one class or request manual review below the threshold."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if image_tensor.ndim != 4 or image_tensor.size(0) != 1:
        raise ValueError("image_tensor must have shape (1, channels, height, width)")
    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must contain at least two unique values")

    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        logits = model(image_tensor.to(device))
        if logits.shape != (1, len(class_names)):
            raise ValueError(
                f"Model output shape {tuple(logits.shape)} does not match "
                f"{len(class_names)} classes"
            )
        probabilities = torch.softmax(logits, dim=1)
        confidence_tensor, index_tensor = probabilities.max(dim=1)

    confidence = float(confidence_tensor.item())
    class_index = int(index_tensor.item())
    most_likely_class = class_names[class_index]
    if confidence < confidence_threshold:
        return PredictionResult(
            predicted_class="unknown",
            most_likely_class=most_likely_class,
            class_index=None,
            confidence=confidence,
            requires_manual_review=True,
            message=UNKNOWN_MESSAGE,
        )
    return PredictionResult(
        predicted_class=most_likely_class,
        most_likely_class=most_likely_class,
        class_index=class_index,
        confidence=confidence,
        requires_manual_review=False,
        message=f"Predicted class: {most_likely_class}",
    )


def predict_from_checkpoint(
    image_path: Path,
    checkpoint_path: Path,
    confidence_threshold: float | None = None,
) -> PredictionResult:
    """Load a saved model and classify one image using checkpoint metadata."""
    device = select_device()
    model, metadata = load_classifier_checkpoint(checkpoint_path, device)
    class_names = [str(name) for name in metadata["class_names"]]
    threshold = (
        float(metadata.get("confidence_threshold", 0.60))
        if confidence_threshold is None
        else confidence_threshold
    )
    image_tensor = load_image_for_inference(
        image_path,
        int(metadata.get("image_size", 224)),
        str(
            metadata.get("preprocessing", {}).get(
                "geometry", "resize_stretch"
            )
        ),
    )
    return predict_single_image(model, image_tensor, class_names, threshold)


def main() -> None:
    """Classify one image from the command line and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    result = predict_from_checkpoint(args.image, args.checkpoint, args.threshold)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
