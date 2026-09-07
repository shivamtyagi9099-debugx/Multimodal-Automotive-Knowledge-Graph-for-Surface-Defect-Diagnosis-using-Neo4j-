"""Endpoint tests for multipart upload and classification behavior."""

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

UNKNOWN_MESSAGE = "Defect unknown - Please review manually."


def test_valid_image_upload_and_prediction(
    app_factory: Callable[[list[float] | None], FastAPI],
    png_bytes: bytes,
) -> None:
    """A valid image should be stored, classified, and enriched."""
    app = app_factory([0.0, 8.0, 0.0, 0.0])
    client = TestClient(app)
    response = client.post(
        "/api/v1/predict",
        files={"file": ("panel.png", png_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "panel.png"
    assert body["predicted_class"] == "scratch"
    assert body["most_likely_class"] == "scratch"
    assert body["requires_manual_review"] is False
    assert body["confidence"] >= 0.60
    assert len(body["repair_steps"]) == 2
    assert body["parts"][0]["fake_stock_quantity"] == 5
    assert body["inventory_last_updated"] == "2026-08-19"
    assert body["inventory_is_simulated"] is True
    assert body["knowledge_source"] == "static_catalog_fallback"
    assert "not a universal vehicle diagnostic tool" in body["poc_disclaimer"]
    saved_uploads = list(app.state.settings.upload_directory.glob("*.png"))
    assert len(saved_uploads) == 1


def test_low_confidence_requires_manual_review(
    app_factory: Callable[[list[float] | None], FastAPI],
    fake_neo4j_service: Any,
    png_bytes: bytes,
) -> None:
    """A flat four-class distribution must trigger the exact unknown rule."""
    client = TestClient(
        app_factory([0.0, 0.0, 0.0, 0.0], fake_neo4j_service)
    )
    response = client.post(
        "/api/v1/predict",
        files={"file": ("uncertain.png", png_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "unknown"
    assert body["confidence"] == 0.25
    assert body["requires_manual_review"] is True
    assert body["message"] == UNKNOWN_MESSAGE
    assert body["repair_steps"] == []
    assert body["parts"] == []
    assert body["inventory_is_simulated"] is True
    assert body["knowledge_source"] == "static_catalog_fallback"
    assert fake_neo4j_service.detail_requests == []


def test_prediction_uses_neo4j_when_graph_is_available(
    app_factory: Callable[..., FastAPI],
    fake_neo4j_service: Any,
    png_bytes: bytes,
) -> None:
    """Known predictions should obtain guidance and parts from Neo4j."""
    app = app_factory([0.0, 8.0, 0.0, 0.0], fake_neo4j_service)
    response = TestClient(app).post(
        "/api/v1/predict",
        files={"file": ("graph-panel.png", png_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "scratch"
    assert body["knowledge_source"] == "neo4j"
    assert body["inventory_is_simulated"] is True
    assert len(body["repair_steps"]) == 2
    assert len(body["parts"]) == 1
    assert fake_neo4j_service.detail_requests == ["scratch"]
    assert app.state.neo4j_connected is True


def test_prediction_falls_back_transparently_when_graph_query_fails(
    app_factory: Callable[..., FastAPI],
    fake_neo4j_service: Any,
    png_bytes: bytes,
) -> None:
    """A graph outage should use static data and disclose the fallback."""
    fake_neo4j_service.fail_queries = True
    app = app_factory([0.0, 0.0, 8.0, 0.0], fake_neo4j_service)
    response = TestClient(app).post(
        "/api/v1/predict",
        files={"file": ("fallback-panel.png", png_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "dent"
    assert body["knowledge_source"] == "static_catalog_fallback"
    assert body["inventory_is_simulated"] is True
    assert len(body["repair_steps"]) == 2
    assert app.state.neo4j_connected is False


def test_unsupported_upload_type(
    app_factory: Callable[[list[float] | None], FastAPI],
) -> None:
    """Non-image media types should be rejected before inference."""
    client = TestClient(app_factory([8.0, 0.0, 0.0, 0.0]))
    response = client.post(
        "/api/v1/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_invalid_image_bytes(
    app_factory: Callable[[list[float] | None], FastAPI],
) -> None:
    """Spoofed PNG metadata should not bypass byte-level image verification."""
    client = TestClient(app_factory([8.0, 0.0, 0.0, 0.0]))
    response = client.post(
        "/api/v1/predict",
        files={"file": ("fake.png", b"not a png", "image/png")},
    )
    assert response.status_code == 400


def test_prediction_without_model_returns_service_unavailable(
    app_factory: Callable[[list[float] | None], FastAPI],
    png_bytes: bytes,
) -> None:
    """Prediction should fail transparently before a checkpoint exists."""
    client = TestClient(app_factory(None))
    response = client.post(
        "/api/v1/predict",
        files={"file": ("panel.png", png_bytes, "image/png")},
    )
    assert response.status_code == 503
