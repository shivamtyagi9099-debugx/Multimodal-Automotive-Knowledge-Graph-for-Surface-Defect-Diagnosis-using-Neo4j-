"""Endpoint tests for API readiness reporting."""

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_health_response_with_loaded_model(
    app_factory: Callable[[list[float] | None], FastAPI],
) -> None:
    """A loaded model and catalogue should produce a healthy response."""
    client = TestClient(app_factory([8.0, 0.0, 0.0, 0.0]))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "model_loaded": True,
        "catalog_loaded": True,
        "neo4j_configured": False,
        "neo4j_connected": False,
        "knowledge_source": "static_catalog_fallback",
        "neo4j_message": (
            "Neo4j is not configured; using the static catalog fallback."
        ),
    }


def test_health_response_without_trained_model(
    app_factory: Callable[[list[float] | None], FastAPI],
) -> None:
    """The API should start transparently in degraded mode before training."""
    client = TestClient(app_factory(None))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False
    assert response.json()["catalog_loaded"] is True
    assert response.json()["neo4j_connected"] is False


def test_health_response_with_connected_neo4j(
    app_factory: Callable[..., FastAPI],
    fake_neo4j_service: Any,
) -> None:
    """The API should be healthy when model, catalog, and graph are ready."""
    client = TestClient(
        app_factory([8.0, 0.0, 0.0, 0.0], fake_neo4j_service)
    )
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
        "catalog_loaded": True,
        "neo4j_configured": True,
        "neo4j_connected": True,
        "knowledge_source": "neo4j",
        "neo4j_message": "Neo4j connection verified.",
    }


def test_frontend_and_static_javascript_are_served(
    app_factory: Callable[[list[float] | None], FastAPI],
) -> None:
    """The frontend should share the API origin without requiring CORS."""
    client = TestClient(app_factory([8.0, 0.0, 0.0, 0.0]))
    page = client.get("/")
    script = client.get("/static/js/app.js")
    assert page.status_code == 200
    assert "PanelCheck" in page.text
    assert "Simulated parts inventory" in page.text
    assert "Last Updated:" in page.text
    assert "No live" in page.text
    assert script.status_code == 200
    assert "submitPrediction" in script.text
    assert "Neo4j knowledge graph" in script.text
    assert "Static catalog fallback" in script.text
