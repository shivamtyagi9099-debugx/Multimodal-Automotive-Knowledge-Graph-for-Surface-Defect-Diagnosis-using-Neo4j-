"""Tests for image preprocessing and confidence-threshold inference."""

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn

from ml.src.inference.predict import (
    UNKNOWN_MESSAGE,
    load_image_for_inference,
    predict_single_image,
)

CLASS_NAMES = ["normal", "scratch", "dent", "rust"]


class FixedLogitModel(nn.Module):
    """Return the same learned-shape logits for each test image."""

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, images: Tensor) -> Tensor:
        """Repeat fixed logits for the incoming batch size."""
        return self.logits.unsqueeze(0).repeat(images.size(0), 1)


def test_load_single_image_for_inference(tmp_path: Path) -> None:
    """A readable image should become one normalized RGB batch."""
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (40, 30), color="white").save(image_path)
    tensor = load_image_for_inference(image_path, image_size=64)
    assert tensor.shape == (1, 3, 64, 64)


def test_high_confidence_single_image_inference() -> None:
    """A confident class should be returned without manual review."""
    model = FixedLogitModel([8.0, 0.0, 0.0, 0.0])
    result = predict_single_image(
        model, torch.zeros(1, 3, 32, 32), CLASS_NAMES, 0.60
    )
    assert result.predicted_class == "normal"
    assert result.class_index == 0
    assert result.confidence >= 0.60
    assert not result.requires_manual_review


def test_low_confidence_requires_manual_review() -> None:
    """A flat probability distribution should trigger the exact policy."""
    model = FixedLogitModel([0.0, 0.0, 0.0, 0.0])
    result = predict_single_image(
        model, torch.zeros(1, 3, 32, 32), CLASS_NAMES, 0.60
    )
    assert result.predicted_class == "unknown"
    assert result.class_index is None
    assert result.confidence == 0.25
    assert result.requires_manual_review
    assert result.message == UNKNOWN_MESSAGE
