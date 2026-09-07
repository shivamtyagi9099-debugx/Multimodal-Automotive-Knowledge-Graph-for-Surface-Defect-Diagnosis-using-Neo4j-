"""Reproducible PyTorch training and validation for the scoped classifier."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import device as Device
from torch.nn import CrossEntropyLoss, Module
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ml.src.data.image_dataset import create_image_loader
from ml.src.models.classifier import (
    SUPPORTED_ARCHITECTURE,
    build_classifier,
    freeze_feature_extractor,
    unfreeze_feature_extractor,
)
from ml.src.evaluation.thresholds import (
    calculate_threshold_analysis,
    collect_model_predictions,
    select_confidence_threshold,
)


@dataclass
class TrainingHistory:
    """Record training progress and the selected best validation epoch."""

    training_metrics: list[dict[str, Any]]
    validation_metrics: list[dict[str, Any]]
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int


def set_reproducibility(random_seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device() -> Device:
    """Select CUDA, Apple Metal, or CPU in that priority order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_data_loaders(
    training_directory: Path,
    validation_directory: Path,
    batch_size: int,
    image_size: int,
    class_names: list[str] | None = None,
    num_workers: int = 0,
    preprocessing_geometry: str = "square_pad_then_resize",
) -> tuple[DataLoader, DataLoader]:
    """Create ordered train and validation loaders without test-set access."""
    if class_names is None:
        class_names = sorted(
            path.name for path in training_directory.iterdir() if path.is_dir()
        )
    training_loader = create_image_loader(
        training_directory,
        class_names,
        image_size,
        batch_size,
        training=True,
        num_workers=num_workers,
        preprocessing_geometry=preprocessing_geometry,
    )
    validation_loader = create_image_loader(
        validation_directory,
        class_names,
        image_size,
        batch_size,
        training=False,
        num_workers=num_workers,
        shuffle=False,
        preprocessing_geometry=preprocessing_geometry,
    )
    return training_loader, validation_loader


def train_one_epoch(
    model: Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    loss_function: Module,
    device: Device,
) -> dict[str, float]:
    """Optimize the model for one epoch and return loss and accuracy."""
    if len(data_loader.dataset) == 0:
        raise ValueError("Training dataset is empty")

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def validate_one_epoch(
    model: Module,
    data_loader: DataLoader,
    loss_function: Module,
    device: Device,
) -> dict[str, float]:
    """Evaluate one validation epoch without changing model parameters."""
    if len(data_loader.dataset) == 0:
        raise ValueError("Validation dataset is empty")

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_function(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def calculate_class_weights(
    training_loader: DataLoader,
    number_of_classes: int,
) -> tuple[torch.Tensor, list[int]]:
    """Return inverse-frequency weights and measured training counts."""
    targets = getattr(training_loader.dataset, "targets", None)
    if targets is None:
        raise ValueError("Training dataset does not expose class targets")
    counts = torch.bincount(
        torch.as_tensor(targets, dtype=torch.long), minlength=number_of_classes
    )
    if len(counts) != number_of_classes or bool((counts == 0).any()):
        raise ValueError("Every configured class needs at least one training image")
    weights = counts.sum() / (number_of_classes * counts.float())
    return weights, [int(value) for value in counts.tolist()]


def _training_stages(configuration: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize staged or legacy training configuration."""
    raw_stages = configuration.get("training_stages")
    if raw_stages is None:
        return [
            {
                "name": "legacy",
                "epoch_count": int(configuration["epoch_count"]),
                "learning_rate": float(configuration["learning_rate"]),
                "trainable": (
                    "head"
                    if bool(configuration.get("freeze_feature_extractor", True))
                    else "all"
                ),
            }
        ]
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("training_stages must be a non-empty list")
    stages: list[dict[str, Any]] = []
    for index, raw_stage in enumerate(raw_stages):
        if not isinstance(raw_stage, dict):
            raise ValueError("Every training stage must be a mapping")
        stage = {
            "name": str(raw_stage.get("name", f"stage-{index + 1}")),
            "epoch_count": int(raw_stage.get("epoch_count", 0)),
            "learning_rate": float(raw_stage.get("learning_rate", 0.0)),
            "trainable": str(raw_stage.get("trainable", "all")),
        }
        if stage["epoch_count"] <= 0 or stage["learning_rate"] <= 0:
            raise ValueError("Every stage needs positive epochs and learning rate")
        if stage["trainable"] not in {"head", "all"}:
            raise ValueError("stage trainable must be 'head' or 'all'")
        stages.append(stage)
    return stages


def train_model(
    model: Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    configuration: dict[str, Any],
) -> TrainingHistory:
    """Train with early stopping, restore the best state, and save artifacts."""
    stages = _training_stages(configuration)
    epoch_count = sum(int(stage["epoch_count"]) for stage in stages)
    patience = int(configuration.get("early_stopping_patience", epoch_count))
    if patience <= 0:
        raise ValueError("early_stopping_patience must be positive")

    device = torch.device(configuration.get("device", select_device()))
    model.to(device)
    class_weights, class_counts = calculate_class_weights(
        training_loader, len(configuration["class_names"])
    )
    use_class_weights = bool(configuration.get("use_class_weights", False))
    loss_function = CrossEntropyLoss(
        weight=class_weights.to(device) if use_class_weights else None
    )
    training_metrics: list[dict[str, Any]] = []
    validation_metrics: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    global_epoch = 0

    for stage in stages:
        if stage["trainable"] == "head":
            freeze_feature_extractor(model)
        else:
            unfreeze_feature_extractor(model)
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("Model has no trainable parameters")
        optimizer = AdamW(
            parameters,
            lr=float(stage["learning_rate"]),
            weight_decay=float(configuration.get("weight_decay", 0.0)),
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(configuration.get("scheduler_factor", 0.3)),
            patience=int(configuration.get("scheduler_patience", 2)),
            min_lr=float(configuration.get("minimum_learning_rate", 1e-7)),
        )
        epochs_without_improvement = 0
        for _ in range(int(stage["epoch_count"])):
            global_epoch += 1
            train_metrics = train_one_epoch(
                model, training_loader, optimizer, loss_function, device
            )
            validation_result = validate_one_epoch(
                model, validation_loader, loss_function, device
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            train_metrics.update(
                {"stage": stage["name"], "learning_rate": learning_rate}
            )
            validation_result.update(
                {"stage": stage["name"], "learning_rate": learning_rate}
            )
            training_metrics.append(train_metrics)
            validation_metrics.append(validation_result)
            scheduler.step(float(validation_result["loss"]))

            print(
                f"Epoch {global_epoch}/{epoch_count} [{stage['name']}] - "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"train_accuracy={train_metrics['accuracy']:.4f}, "
                f"validation_loss={validation_result['loss']:.4f}, "
                f"validation_accuracy={validation_result['accuracy']:.4f}, "
                f"lr={learning_rate:.2e}"
            )

            if validation_result["loss"] < best_validation_loss:
                best_validation_loss = float(validation_result["loss"])
                best_epoch = global_epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping stage '{stage['name']}'.")
                    break

    if best_state is None:
        raise RuntimeError("Training completed without a valid model state")
    model.load_state_dict(best_state)

    validation_true, validation_predicted, validation_confidences = (
        collect_model_predictions(model, validation_loader, device)
    )
    threshold_tuning = configuration.get("threshold_tuning", {})
    configured_threshold = float(configuration.get("confidence_threshold", 0.60))
    if isinstance(threshold_tuning, dict) and bool(
        threshold_tuning.get("enabled", False)
    ):
        selected_threshold, threshold_candidates = select_confidence_threshold(
            validation_true,
            validation_predicted,
            validation_confidences,
            list(configuration["class_names"]),
            minimum_coverage=float(
                threshold_tuning.get("minimum_coverage", 0.65)
            ),
            candidate_thresholds=[
                float(value)
                for value in threshold_tuning.get("candidate_thresholds", [])
            ]
            or None,
        )
    else:
        selected_threshold = configured_threshold
        threshold_candidates = []
    validation_threshold_analysis = calculate_threshold_analysis(
        validation_true,
        validation_predicted,
        validation_confidences,
        list(configuration["class_names"]),
        selected_threshold,
    )

    history = TrainingHistory(
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        epochs_completed=len(training_metrics),
    )

    model_output_path = Path(configuration["model_output_path"])
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": configuration.get("model_architecture", SUPPORTED_ARCHITECTURE),
        "class_names": list(configuration["class_names"]),
        "image_size": int(configuration["image_size"]),
        "confidence_threshold": selected_threshold,
        "confidence_threshold_selected_on": "validation",
        "random_seed": int(configuration.get("random_seed", 42)),
        "best_epoch": history.best_epoch,
        "best_validation_loss": history.best_validation_loss,
        "epochs_completed": history.epochs_completed,
        "device": str(device),
        "pytorch_version": str(torch.__version__),
        "pretrained": bool(configuration.get("pretrained", True)),
        "training_stages": stages,
        "class_counts": class_counts,
        "class_weights": (
            [float(value) for value in class_weights.tolist()]
            if use_class_weights
            else None
        ),
        "training_sample_count": len(training_loader.dataset),
        "validation_sample_count": len(validation_loader.dataset),
        "dataset_provenance": dict(configuration.get("dataset_provenance", {})),
        "validation_threshold_analysis": validation_threshold_analysis.to_dict(),
        "preprocessing": {
            "exif_orientation": "applied",
            "colour_mode": "RGB",
            "geometry": str(
                configuration.get(
                    "preprocessing_geometry", "square_pad_then_resize"
                )
            ),
            "image_size": int(configuration["image_size"]),
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
    }
    save_model_checkpoint(model, model_output_path, metadata)

    history_path = Path(configuration["history_output_path"])
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(asdict(history), indent=2), encoding="utf-8"
    )
    threshold_path = Path(configuration["validation_threshold_output_path"])
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.write_text(
        json.dumps(
            {
                "selected_threshold": selected_threshold,
                "selection_dataset": "validation",
                "selected_analysis": validation_threshold_analysis.to_dict(),
                "candidate_analyses": [
                    item.to_dict() for item in threshold_candidates
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    configuration_path_value = configuration.get("training_configuration_output_path")
    if configuration_path_value:
        configuration_path = Path(configuration_path_value)
        configuration_path.parent.mkdir(parents=True, exist_ok=True)
        configuration_path.write_text(
            json.dumps(configuration, indent=2, default=str),
            encoding="utf-8",
        )
    return history


def save_model_checkpoint(
    model: Module,
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    """Save a portable state dictionary together with model metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        output_path,
    )


def _load_json_list(path: Path) -> list[str]:
    """Load an ordered, unique class-name list from JSON."""
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("class_names_path must contain a JSON list of class names")
    class_names = [str(value) for value in values]
    if len(set(class_names)) != len(class_names):
        raise ValueError("Class names must be unique")
    return class_names


def _file_sha256(path: Path) -> str:
    """Return a SHA-256 digest for one experiment input file."""
    if not path.is_file():
        raise FileNotFoundError(f"Required training input does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Train the configured classifier from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("ml/configs/training.yaml")
    )
    args = parser.parse_args()

    configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if configuration.get("model_architecture") != SUPPORTED_ARCHITECTURE:
        raise ValueError(f"Only {SUPPORTED_ARCHITECTURE} is supported in this PoC")
    class_names = _load_json_list(Path(configuration["class_names_path"]))
    configuration["class_names"] = class_names
    provenance_paths = {
        "source_manifest_sha256": Path(configuration["source_manifest_path"]),
        "training_manifest_sha256": Path(configuration["training_manifest_path"]),
        "validation_manifest_sha256": Path(
            configuration["validation_manifest_path"]
        ),
        "class_names_sha256": Path(configuration["class_names_path"]),
    }
    configuration["dataset_provenance"] = {
        key: _file_sha256(path) for key, path in provenance_paths.items()
    }
    set_reproducibility(int(configuration["random_seed"]))

    training_loader, validation_loader = create_data_loaders(
        Path(configuration["training_path"]),
        Path(configuration["validation_path"]),
        int(configuration["batch_size"]),
        int(configuration["image_size"]),
        class_names=class_names,
        num_workers=int(configuration.get("num_workers", 0)),
        preprocessing_geometry=str(
            configuration.get(
                "preprocessing_geometry", "square_pad_then_resize"
            )
        ),
    )
    model = build_classifier(
        len(class_names), pretrained=bool(configuration.get("pretrained", True))
    )
    train_model(model, training_loader, validation_loader, configuration)


if __name__ == "__main__":
    main()
