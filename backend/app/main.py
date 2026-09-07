"""FastAPI application wiring for the local classification PoC."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from torch.nn import Module

from backend.app.api.routes import feedback, health, prediction
from backend.app.core.config import Settings, load_settings
from backend.app.schemas.inventory import InventoryCatalog
from backend.app.services.catalog_service import load_catalog
from backend.app.services.inference_service import load_model_bundle
from backend.app.services.neo4j_service import (
    DefectKnowledgeService,
    Neo4jConfigurationError,
    Neo4jService,
)

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
DEFAULT_NEO4J_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "neo4j.yaml"
)
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
FRONTEND_ROOT = Path(__file__).resolve().parents[2] / "frontend"


def _load_class_names(path: Path) -> list[str]:
    """Load the central ordered class list and enforce basic integrity."""
    if not path.is_file():
        raise FileNotFoundError(f"Class-name configuration does not exist: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("class_names.json must contain at least two classes")
    class_names = [str(value).strip() for value in values]
    if any(not value for value in class_names) or len(set(class_names)) != len(
        class_names
    ):
        raise ValueError("Configured class names must be non-empty and unique")
    return class_names


def create_app(
    settings: Settings | None = None,
    model: Module | None = None,
    model_metadata: dict[str, Any] | None = None,
    catalog: InventoryCatalog | None = None,
    neo4j_service: DefectKnowledgeService | None = None,
    neo4j_config_path: Path = DEFAULT_NEO4J_CONFIG_PATH,
    configure_neo4j: bool = True,
) -> FastAPI:
    """Create the API and wire inference, graph data, fallback, and UI.

    Optional injected resources keep endpoint tests independent from a trained
    checkpoint and live graph while production uses the configured resources.
    """
    active_settings = settings or load_settings(DEFAULT_SETTINGS_PATH)
    class_names = _load_class_names(active_settings.class_names_path)
    active_catalog = catalog or load_catalog(active_settings.catalog_path)
    catalog_classes = {entry.class_name for entry in active_catalog.defects}
    if set(class_names) != catalog_classes:
        raise ValueError("Class configuration and static catalogue do not match")

    active_model = model
    active_metadata = dict(model_metadata or {})
    model_load_error: str | None = None
    if active_model is None and active_settings.model_path.is_file():
        try:
            active_model, active_metadata = load_model_bundle(
                active_settings.model_path
            )
        except (OSError, RuntimeError, ValueError) as exc:
            model_load_error = str(exc)
    if active_model is not None:
        saved_classes = active_metadata.get("class_names", class_names)
        if list(saved_classes) != class_names:
            raise ValueError("Checkpoint and project class order do not match")
        active_metadata.setdefault("class_names", class_names)
        active_metadata.setdefault("image_size", 224)

    active_neo4j_service = neo4j_service
    neo4j_configuration_error: str | None = None
    if active_neo4j_service is None and configure_neo4j:
        load_dotenv(DEFAULT_ENV_PATH, override=False)
        try:
            active_neo4j_service = Neo4jService.from_config(neo4j_config_path)
        except (Neo4jConfigurationError, OSError, ValueError) as exc:
            neo4j_configuration_error = str(exc)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Close the lazy Neo4j driver when the API process stops."""
        try:
            yield
        finally:
            if active_neo4j_service is not None:
                active_neo4j_service.close()

    application = FastAPI(
        title=active_settings.project_name,
        version="0.1.0",
        description=(
            "Academic PoC for four-class car-body panel image classification "
            "with Neo4j-backed repair and simulated inventory information."
        ),
        lifespan=lifespan,
    )
    application.state.settings = active_settings
    application.state.class_names = class_names
    application.state.catalog = active_catalog
    application.state.model = active_model
    application.state.model_metadata = active_metadata
    application.state.model_load_error = model_load_error
    application.state.prediction_fingerprints = {}
    application.state.neo4j_service = active_neo4j_service
    application.state.neo4j_configured = active_neo4j_service is not None
    application.state.neo4j_connected = False
    application.state.neo4j_message = (
        neo4j_configuration_error
        or (
            "Neo4j connectivity has not been checked yet."
            if active_neo4j_service is not None
            else "Neo4j is not configured; using the static catalog fallback."
        )
    )
    application.include_router(health.router)
    application.include_router(prediction.router)
    application.include_router(feedback.router)
    application.mount(
        "/static",
        StaticFiles(directory=FRONTEND_ROOT / "static"),
        name="static",
    )

    @application.get("/", include_in_schema=False, response_class=FileResponse)
    def frontend_index() -> FileResponse:
        """Serve the single-page frontend from the same API origin."""
        return FileResponse(FRONTEND_ROOT / "index.html")

    return application


app = create_app()
