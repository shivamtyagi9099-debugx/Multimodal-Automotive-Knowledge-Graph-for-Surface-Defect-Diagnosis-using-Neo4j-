"""Loading and lookup operations for the local JSON catalogue."""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.schemas.inventory import DefectCatalogEntry, InventoryCatalog


def load_catalog(catalog_path: Path) -> InventoryCatalog:
    """Load, validate, and check class uniqueness in the static catalogue."""
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalogue does not exist: {catalog_path}")
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Catalogue is not valid JSON: {catalog_path}") from exc
    catalog = InventoryCatalog.model_validate(raw)
    class_names = [entry.class_name for entry in catalog.defects]
    if len(class_names) != len(set(class_names)):
        raise ValueError("Catalogue class_name values must be unique")
    return catalog


def get_defect_details(
    class_name: str,
    catalog: InventoryCatalog,
) -> DefectCatalogEntry | None:
    """Return the exact catalogue entry matching a predicted class."""
    normalized = class_name.strip().casefold()
    return next(
        (
            entry
            for entry in catalog.defects
            if entry.class_name.casefold() == normalized
        ),
        None,
    )


def get_inventory_last_updated(catalog: InventoryCatalog) -> str:
    """Return the simulated-inventory update date for UI transparency."""
    return catalog.inventory_last_updated
