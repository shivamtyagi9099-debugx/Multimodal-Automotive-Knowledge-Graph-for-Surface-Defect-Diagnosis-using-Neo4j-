"""Tests for model construction, freezing, and checkpoint loading."""

from pathlib import Path

import torch

from ml.src.models.classifier import (
    build_classifier,
    freeze_feature_extractor,
    load_classifier_checkpoint,
    unfreeze_feature_extractor,
)
from ml.src.training.train import save_model_checkpoint

CLASS_NAMES = ["normal", "scratch", "dent", "rust"]


def test_classifier_output_shape() -> None:
    """ResNet-18 should return one logit for each configured class."""
    model = build_classifier(len(CLASS_NAMES), pretrained=False)
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, len(CLASS_NAMES))


def test_classifier_rejects_one_class() -> None:
    """Classification requires at least two classes."""
    try:
        build_classifier(1, pretrained=False)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for one output class")


def test_freeze_and_unfreeze_feature_extractor() -> None:
    """Only the final layer should remain trainable while frozen."""
    model = build_classifier(len(CLASS_NAMES), pretrained=False)
    freeze_feature_extractor(model)
    assert all(parameter.requires_grad for parameter in model.fc.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("fc.")
    )

    unfreeze_feature_extractor(model)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    """A saved state dictionary and metadata should load successfully."""
    model = build_classifier(len(CLASS_NAMES), pretrained=False)
    checkpoint_path = tmp_path / "model.pt"
    metadata = {
        "architecture": "resnet18",
        "class_names": CLASS_NAMES,
        "image_size": 224,
        "confidence_threshold": 0.60,
        # Regression coverage for checkpoints created before TorchVersion was
        # normalized to a plain string during training.
        "pytorch_version": torch.__version__,
    }
    save_model_checkpoint(model, checkpoint_path, metadata)
    restored, restored_metadata = load_classifier_checkpoint(checkpoint_path)
    assert restored_metadata == {
        **metadata,
        "pytorch_version": str(torch.__version__),
    }
    assert restored.fc.out_features == len(CLASS_NAMES)
