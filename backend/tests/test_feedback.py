"""Endpoint tests for append-only local feedback persistence."""

import csv
from collections.abc import Callable

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _payload() -> dict[str, object]:
    """Return one valid positive-feedback request."""
    return {
        "prediction_id": "prediction-1",
        "image_filename": "panel.png",
        "predicted_class": "scratch",
        "confidence": 0.91,
        "is_correct": True,
        "corrected_class": None,
        "comment": "Looks correct",
    }


def test_feedback_submission_is_written_to_csv(
    app_factory: Callable[[list[float] | None], FastAPI],
) -> None:
    """Accepted feedback should create one local CSV record."""
    app = app_factory([8.0, 0.0, 0.0, 0.0])
    client = TestClient(app)
    response = client.post("/api/v1/feedback", json=_payload())
    assert response.status_code == 200
    assert response.json() == {
        "prediction_id": "prediction-1",
        "accepted": True,
        "message": "Feedback recorded locally for future evaluation.",
    }

    with app.state.settings.feedback_csv_path.open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["prediction_id"] == "prediction-1"
    assert rows[0]["is_correct"] == "True"
    assert rows[0]["timestamp"]


def test_corrected_class_feedback(
    app_factory: Callable[[list[float] | None], FastAPI],
) -> None:
    """An incorrect prediction may include a supported corrected class."""
    app = app_factory([8.0, 0.0, 0.0, 0.0])
    client = TestClient(app)
    payload = _payload()
    payload.update({"is_correct": False, "corrected_class": "dent"})
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 200

    with app.state.settings.feedback_csv_path.open(
        encoding="utf-8", newline=""
    ) as stream:
        row = next(csv.DictReader(stream))
    assert row["corrected_class"] == "dent"
    assert row["is_correct"] == "False"


def test_feedback_correction_is_applied_to_the_same_image(
    app_factory: Callable[[list[float] | None], FastAPI],
    png_bytes: bytes,
) -> None:
    """An exact image should use its saved human label on later uploads."""
    app = app_factory([8.0, 0.0, 0.0, 0.0])
    client = TestClient(app)
    first = client.post(
        "/api/v1/predict",
        files={"file": ("panel.png", png_bytes, "image/png")},
    )
    assert first.status_code == 200
    first_prediction = first.json()
    assert first_prediction["predicted_class"] == "normal"
    assert first_prediction["feedback_applied"] is False

    feedback = {
        "prediction_id": first_prediction["prediction_id"],
        "image_filename": first_prediction["filename"],
        "image_sha256": first_prediction["image_sha256"],
        "predicted_class": first_prediction["predicted_class"],
        "confidence": first_prediction["confidence"],
        "is_correct": False,
        "corrected_class": "scratch",
        "comment": "Visible scratch",
    }
    feedback_response = client.post("/api/v1/feedback", json=feedback)
    assert feedback_response.status_code == 200
    assert "exact image" in feedback_response.json()["message"]

    restarted_client = TestClient(app_factory([8.0, 0.0, 0.0, 0.0]))
    repeated = restarted_client.post(
        "/api/v1/predict",
        files={"file": ("renamed-panel.png", png_bytes, "image/png")},
    )
    assert repeated.status_code == 200
    repeated_prediction = repeated.json()
    assert repeated_prediction["predicted_class"] == "scratch"
    assert repeated_prediction["most_likely_class"] == "normal"
    assert repeated_prediction["feedback_applied"] is True
    assert repeated_prediction["requires_manual_review"] is False
    assert "Saved human correction" in repeated_prediction["message"]


def test_positive_feedback_is_not_presented_as_a_correction(
    app_factory: Callable[[list[float] | None], FastAPI],
    png_bytes: bytes,
) -> None:
    """Confirmation may be stored without falsely claiming it changed output."""
    app = app_factory([8.0, 0.0, 0.0, 0.0])
    client = TestClient(app)
    first = client.post(
        "/api/v1/predict",
        files={"file": ("panel.png", png_bytes, "image/png")},
    ).json()
    feedback = {
        "prediction_id": first["prediction_id"],
        "image_filename": first["filename"],
        "image_sha256": first["image_sha256"],
        "predicted_class": first["predicted_class"],
        "confidence": first["confidence"],
        "is_correct": True,
        "corrected_class": None,
        "comment": "Confirmed",
    }
    response = client.post("/api/v1/feedback", json=feedback)
    assert response.status_code == 200

    repeated = client.post(
        "/api/v1/predict",
        files={"file": ("panel-again.png", png_bytes, "image/png")},
    ).json()
    assert repeated["predicted_class"] == "normal"
    assert repeated["feedback_applied"] is False
    assert "Saved human correction" not in repeated["message"]


def test_incorrect_feedback_requires_a_corrected_class(
    app_factory: Callable[[list[float] | None], FastAPI],
) -> None:
    """A rejection without the correct label cannot teach a later result."""
    client = TestClient(app_factory([8.0, 0.0, 0.0, 0.0]))
    payload = _payload()
    payload.update({"is_correct": False, "corrected_class": None})
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 422


def test_unsupported_corrected_class_is_rejected(
    app_factory: Callable[[list[float] | None], FastAPI],
) -> None:
    """Feedback must not introduce labels outside the project taxonomy."""
    client = TestClient(app_factory([8.0, 0.0, 0.0, 0.0]))
    payload = _payload()
    payload.update({"is_correct": False, "corrected_class": "broken_window"})
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 400
