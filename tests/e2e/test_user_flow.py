"""End-to-end test for the frontend, prediction, catalog, and feedback flow."""

from __future__ import annotations

import csv
from io import BytesIO
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import Tensor, nn

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.services.catalog_service import load_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ["normal", "scratch", "dent", "rust"]


class FixedScratchModel(nn.Module):
    """Return deterministic logits selecting the scratch class."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(
            torch.tensor([0.0, 8.0, 0.0, 0.0], dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return one fixed four-class logit vector for each input image."""
        return self.logits.unsqueeze(0).repeat(images.size(0), 1)


def _valid_png_bytes() -> bytes:
    """Create a small valid RGB image entirely in memory."""
    buffer = BytesIO()
    Image.new("RGB", (48, 48), color=(210, 215, 220)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_complete_upload_to_feedback_user_flow(tmp_path: Path) -> None:
    """Verify the journey from page load through local feedback persistence."""
    settings = Settings(
        project_name="car-body-panel-e2e-test",
        object_type="car_body_panel",
        class_names_path=PROJECT_ROOT / "config" / "class_names.json",
        confidence_threshold=0.60,
        unknown_message="Defect unknown - Please review manually.",
        model_path=tmp_path / "model.pt",
        catalog_path=PROJECT_ROOT / "data" / "static" / "defect_catalog.json",
        feedback_csv_path=tmp_path / "feedback" / "prediction_feedback.csv",
        upload_directory=tmp_path / "uploads",
        maximum_upload_size_mb=1,
        allowed_image_types=["image/jpeg", "image/png"],
        inventory_last_updated="2026-08-19",
    )
    application = create_app(
        settings=settings,
        model=FixedScratchModel(),
        model_metadata={
            "class_names": CLASS_NAMES,
            "image_size": 64,
            "confidence_threshold": 0.60,
        },
        catalog=load_catalog(settings.catalog_path),
        configure_neo4j=False,
    )

    with TestClient(application) as client:
        page_response = client.get("/")
        health_response = client.get("/health")
        prediction_response = client.post(
            "/api/v1/predict",
            files={
                "file": (
                    "sample-panel.png",
                    _valid_png_bytes(),
                    "image/png",
                )
            },
        )

        assert page_response.status_code == 200
        assert "PanelCheck" in page_response.text
        assert health_response.status_code == 200
        health = health_response.json()
        assert health["status"] == "degraded"
        assert health["model_loaded"] is True
        assert health["catalog_loaded"] is True
        assert health["neo4j_connected"] is False
        assert health["knowledge_source"] == "static_catalog_fallback"

        assert prediction_response.status_code == 200
        prediction = prediction_response.json()
        assert prediction["filename"] == "sample-panel.png"
        assert prediction["predicted_class"] == "scratch"
        assert prediction["requires_manual_review"] is False
        assert prediction["confidence"] >= 0.60
        assert len(prediction["repair_steps"]) == 3
        assert len(prediction["parts"]) == 2
        assert prediction["inventory_last_updated"] == "2026-08-19"
        assert prediction["inventory_is_simulated"] is True
        assert prediction["knowledge_source"] == "static_catalog_fallback"
        assert "not a universal vehicle diagnostic tool" in prediction[
            "poc_disclaimer"
        ]

        feedback_response = client.post(
            "/api/v1/feedback",
            json={
                "prediction_id": prediction["prediction_id"],
                "image_filename": prediction["filename"],
                "predicted_class": prediction["predicted_class"],
                "confidence": prediction["confidence"],
                "is_correct": False,
                "corrected_class": "dent",
                "comment": "Confirmed as a dent during manual review.",
            },
        )

    assert feedback_response.status_code == 200
    assert feedback_response.json() == {
        "prediction_id": prediction["prediction_id"],
        "accepted": True,
        "message": (
            "Feedback saved. It will be applied when this exact image is "
            "analysed again."
        ),
    }

    stored_uploads = list(settings.upload_directory.glob("*.png"))
    assert len(stored_uploads) == 1
    assert stored_uploads[0].name != "sample-panel.png"

    with settings.feedback_csv_path.open(encoding="utf-8", newline="") as stream:
        feedback_rows = list(csv.DictReader(stream))
    assert len(feedback_rows) == 1
    assert feedback_rows[0]["prediction_id"] == prediction["prediction_id"]
    assert feedback_rows[0]["predicted_class"] == "scratch"
    assert feedback_rows[0]["is_correct"] == "False"
    assert feedback_rows[0]["corrected_class"] == "dent"
    assert feedback_rows[0]["timestamp"]
