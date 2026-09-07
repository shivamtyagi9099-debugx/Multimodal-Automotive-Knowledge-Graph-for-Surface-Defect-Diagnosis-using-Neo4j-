"""Shared FastAPI fixtures using small deterministic PyTorch models."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
from fastapi import FastAPI
from PIL import Image
from torch import Tensor, nn

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.schemas.inventory import InventoryCatalog
from backend.app.schemas.neo4j import Neo4jHealthStatus
from backend.app.services.catalog_service import get_defect_details
from backend.app.services.neo4j_service import Neo4jServiceError

CLASS_NAMES = ["normal", "scratch", "dent", "rust"]


class FixedLogitModel(nn.Module):
    """Return deterministic logits independent of uploaded image content."""

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, images: Tensor) -> Tensor:
        """Repeat the configured logits for the input batch size."""
        return self.logits.unsqueeze(0).repeat(images.size(0), 1)


class FakeNeo4jService:
    """Provide deterministic graph behavior without a live database."""

    def __init__(
        self,
        *,
        connected: bool = True,
        fail_queries: bool = False,
    ) -> None:
        self.connected = connected
        self.fail_queries = fail_queries
        self.detail_requests: list[str] = []
        self.closed = False

    def verify_connectivity(self) -> Neo4jHealthStatus:
        """Return the configured fake connection state."""
        return Neo4jHealthStatus(
            connected=self.connected,
            database="neo4j",
            uri="bolt://test-neo4j:7687",
            message=(
                "Neo4j connection verified."
                if self.connected
                else "Neo4j is unavailable: test double"
            ),
        )

    def get_defect_details(self, class_name: str) -> Any:
        """Return catalog-shaped graph data or simulate a query outage."""
        self.detail_requests.append(class_name)
        if self.fail_queries:
            raise Neo4jServiceError("Neo4j query failed: test double")
        return get_defect_details(class_name, _catalog())

    def get_inventory_last_updated(self) -> str:
        """Return the graph's static simulated-inventory date."""
        if self.fail_queries:
            raise Neo4jServiceError("Neo4j query failed: test double")
        return "2026-08-19"

    def close(self) -> None:
        """Record lifecycle cleanup for assertions when needed."""
        self.closed = True


def _catalog() -> InventoryCatalog:
    """Create a complete four-class static catalogue for endpoint tests."""
    defects = []
    for class_name in CLASS_NAMES:
        defects.append(
            {
                "class_name": class_name,
                "display_name": class_name.title(),
                "repair_steps": (
                    []
                    if class_name == "normal"
                    else ["Verified test step 1", "Verified test step 2"]
                ),
                "parts": (
                    []
                    if class_name == "normal"
                    else [
                        {
                            "part_id": f"PART-{class_name.upper()}",
                            "part_name": f"Mock {class_name} repair item",
                            "mock_price": 100.0,
                            "currency": "INR",
                            "fake_stock_quantity": 5,
                            "availability_status": "in_stock",
                        }
                    ]
                ),
            }
        )
    return InventoryCatalog.model_validate(
        {
            "object_type": "car_body_panel",
            "inventory_last_updated": "2026-08-19",
            "confidence_threshold": 0.60,
            "unknown_message": "Defect unknown - Please review manually.",
            "defects": defects,
        }
    )


@pytest.fixture
def png_bytes() -> bytes:
    """Return a small valid PNG upload."""
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def fake_neo4j_service() -> FakeNeo4jService:
    """Return a connected Neo4j-compatible test double."""
    return FakeNeo4jService()


@pytest.fixture
def app_factory(tmp_path: Path) -> Callable[..., FastAPI]:
    """Create isolated applications with optional deterministic model logits."""
    class_names_path = tmp_path / "class_names.json"
    class_names_path.write_text(json.dumps(CLASS_NAMES), encoding="utf-8")

    def factory(
        logits: list[float] | None,
        neo4j_service: FakeNeo4jService | None = None,
    ) -> FastAPI:
        """Build one test application using temporary local storage."""
        settings = Settings(
            project_name="backend-test",
            object_type="car_body_panel",
            class_names_path=class_names_path,
            confidence_threshold=0.60,
            unknown_message="Defect unknown - Please review manually.",
            model_path=tmp_path / "model.pt",
            catalog_path=tmp_path / "catalog.json",
            feedback_csv_path=tmp_path / "feedback.csv",
            upload_directory=tmp_path / "uploads",
            maximum_upload_size_mb=1,
            allowed_image_types=["image/jpeg", "image/png"],
            inventory_last_updated="2026-08-19",
        )
        model = None if logits is None else FixedLogitModel(logits)
        metadata = {
            "class_names": CLASS_NAMES,
            "image_size": 64,
            "confidence_threshold": 0.60,
        }
        return create_app(
            settings=settings,
            model=model,
            model_metadata=metadata,
            catalog=_catalog(),
            neo4j_service=neo4j_service,
            configure_neo4j=False,
        )

    return factory
