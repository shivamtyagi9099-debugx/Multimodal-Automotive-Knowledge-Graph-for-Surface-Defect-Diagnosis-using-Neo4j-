"""PyTorch model construction for car-body panel classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import Module
from torch.torch_version import TorchVersion
from torchvision.models import ResNet18_Weights, resnet18

SUPPORTED_ARCHITECTURE = "resnet18"


def build_classifier(
    number_of_classes: int,
    pretrained: bool = True,
) -> Module:
    """Build a ResNet-18 classifier with one output logit per class."""
    if number_of_classes < 2:
        raise ValueError("number_of_classes must be at least 2")

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, number_of_classes)
    return model


def _classification_head(model: Module) -> Module:
    """Return the ResNet classification head or raise a clear error."""
    head = getattr(model, "fc", None)
    if not isinstance(head, Module):
        raise TypeError("Expected a model with an 'fc' classification head")
    return head


def freeze_feature_extractor(model: Module) -> None:
    """Freeze feature layers while keeping the classification head trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in _classification_head(model).parameters():
        parameter.requires_grad = True


def unfreeze_feature_extractor(model: Module) -> None:
    """Make every model parameter trainable for full fine-tuning."""
    for parameter in model.parameters():
        parameter.requires_grad = True


def count_trainable_parameters(model: Module) -> int:
    """Return the number of scalar parameters currently marked trainable."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def load_classifier_checkpoint(
    checkpoint_path: Path,
    device: torch.device | str = "cpu",
) -> tuple[Module, dict[str, Any]]:
    """Rebuild a classifier and load its saved state and metadata."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")
    # PyTorch 2.6+ loads checkpoints in weights-only mode by default. Older
    # project checkpoints stored ``torch.__version__`` as a TorchVersion
    # instance, so allowlist that known PyTorch type without disabling the
    # safer loader.
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")
    metadata = checkpoint.get("metadata")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(metadata, dict) or not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing model_state_dict or metadata")
    if isinstance(metadata.get("pytorch_version"), TorchVersion):
        metadata["pytorch_version"] = str(metadata["pytorch_version"])
    class_names = metadata.get("class_names")
    if not isinstance(class_names, list) or len(class_names) < 2:
        raise ValueError("Checkpoint metadata contains invalid class_names")
    if metadata.get("architecture") != SUPPORTED_ARCHITECTURE:
        raise ValueError(
            f"Unsupported checkpoint architecture: {metadata.get('architecture')}"
        )

    model = build_classifier(len(class_names), pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, metadata
