"""Append-only CSV persistence for user prediction feedback."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from backend.app.schemas.feedback import FeedbackRequest

FEEDBACK_FIELDS = [
    "timestamp",
    "prediction_id",
    "image_filename",
    "image_sha256",
    "predicted_class",
    "confidence",
    "is_correct",
    "corrected_class",
    "comment",
]


def find_feedback_class(
    image_sha256: str,
    feedback_csv_path: Path,
    class_names: set[str],
) -> str | None:
    """Return the latest valid human label recorded for an exact image."""
    if not feedback_csv_path.is_file() or feedback_csv_path.stat().st_size == 0:
        return None

    latest_class: str | None = None
    with feedback_csv_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("image_sha256") != image_sha256:
                continue
            is_correct = str(row.get("is_correct", "")).strip().lower() == "true"
            candidate = (
                row.get("predicted_class")
                if is_correct
                else row.get("corrected_class")
            )
            if candidate in class_names:
                latest_class = candidate
    return latest_class


def _upgrade_legacy_csv(feedback_csv_path: Path) -> bool:
    """Add new feedback columns without discarding legacy records."""
    if not feedback_csv_path.is_file() or feedback_csv_path.stat().st_size == 0:
        return True
    with feedback_csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames == FEEDBACK_FIELDS:
            return False
        rows = list(reader)

    temporary_path = feedback_csv_path.with_suffix(
        f"{feedback_csv_path.suffix}.tmp"
    )
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FEEDBACK_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FEEDBACK_FIELDS})
    temporary_path.replace(feedback_csv_path)
    return False


def append_feedback(
    feedback: FeedbackRequest,
    feedback_csv_path: Path,
    image_sha256: str | None = None,
) -> None:
    """Append one validated feedback record, creating the CSV header if needed."""
    feedback_csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = _upgrade_legacy_csv(feedback_csv_path)
    record = feedback.model_dump()
    record["image_sha256"] = image_sha256 or feedback.image_sha256 or ""
    record["timestamp"] = datetime.now(UTC).isoformat()

    with feedback_csv_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FEEDBACK_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(record)
