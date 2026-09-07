"""Typed configuration loading for the local proof-of-concept API."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PATH_FIELDS = {
    "class_names_path",
    "model_path",
    "catalog_path",
    "feedback_csv_path",
    "upload_directory",
}


class Settings(BaseModel):
    """Validated application settings shared by routes and services."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(min_length=1)
    object_type: str = Field(min_length=1)
    class_names_path: Path
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    unknown_message: str = Field(min_length=1)
    model_path: Path
    catalog_path: Path
    feedback_csv_path: Path
    upload_directory: Path
    maximum_upload_size_mb: int = Field(gt=0)
    allowed_image_types: list[str] = Field(min_length=1)
    inventory_last_updated: str = Field(min_length=1)

    @field_validator("allowed_image_types")
    @classmethod
    def normalize_image_types(cls, values: list[str]) -> list[str]:
        """Normalize configured media types and reject duplicates."""
        normalized = [value.strip().lower() for value in values if value.strip()]
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("allowed_image_types must be non-empty and unique")
        return normalized

    @property
    def maximum_upload_size_bytes(self) -> int:
        """Return the configured upload limit in bytes."""
        return self.maximum_upload_size_mb * 1024 * 1024


def load_settings(settings_path: Path) -> Settings:
    """Load YAML settings and resolve project-relative filesystem paths."""
    settings_path = settings_path.resolve()
    if not settings_path.is_file():
        raise FileNotFoundError(f"Settings file does not exist: {settings_path}")
    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Settings YAML must contain an object at its root")

    project_root = settings_path.parent.parent
    resolved = dict(raw)
    for field_name in PATH_FIELDS:
        if field_name not in resolved:
            continue
        path = Path(str(resolved[field_name])).expanduser()
        resolved[field_name] = path if path.is_absolute() else project_root / path
    return Settings.model_validate(resolved)
