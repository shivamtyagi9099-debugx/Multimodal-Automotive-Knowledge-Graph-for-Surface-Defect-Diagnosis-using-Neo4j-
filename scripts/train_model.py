"""Launch the configured PyTorch training and validation workflow."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _prepare_project_environment() -> None:
    """Make project-relative imports and configuration paths deterministic."""
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.chdir(PROJECT_ROOT)


def main() -> None:
    """Train the classifier using ``ml/configs/training.yaml`` by default."""
    _prepare_project_environment()
    from ml.src.training.train import main as training_main

    training_main()


if __name__ == "__main__":
    main()
