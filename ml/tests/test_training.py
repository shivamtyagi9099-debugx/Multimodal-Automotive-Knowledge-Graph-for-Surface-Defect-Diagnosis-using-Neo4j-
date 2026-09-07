"""Tests for one-epoch PyTorch training and validation."""

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from ml.src.training.train import train_one_epoch, validate_one_epoch


def _loader() -> DataLoader:
    """Create a tiny two-class tensor dataset."""
    inputs = torch.randn(12, 3, 8, 8)
    labels = torch.tensor([0, 1] * 6)
    return DataLoader(TensorDataset(inputs, labels), batch_size=4)


def test_train_and_validate_one_epoch() -> None:
    """Training and validation should return bounded summary metrics."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
    loss_function = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    training = train_one_epoch(
        model, _loader(), optimizer, loss_function, device
    )
    validation = validate_one_epoch(model, _loader(), loss_function, device)

    for metrics in (training, validation):
        assert metrics["loss"] >= 0.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
