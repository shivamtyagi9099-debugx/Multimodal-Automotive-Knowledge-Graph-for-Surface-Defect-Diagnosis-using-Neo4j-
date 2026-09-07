"""Launch the checksum-protected final test-set evaluation workflow."""

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
    """Evaluate the trained model only after frozen-test checksum validation."""
    _prepare_project_environment()
    from ml.src.evaluation.evaluate import main as evaluation_main

    evaluation_main()


if __name__ == "__main__":
    main()
