"""Unit tests for Neo4j configuration, mapping, and catalog seeding."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from neo4j.exceptions import ServiceUnavailable

from backend.app.services.catalog_service import load_catalog
from backend.app.services.neo4j_service import (
    PLACEHOLDER_PASSWORD,
    Neo4jConfigurationError,
    Neo4jService,
    Neo4jServiceError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeDriver:
    """Return driver-shaped records while capturing parameterized calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.connectivity_checked = False

    def verify_connectivity(self) -> None:
        """Record a successful connectivity check."""
        self.connectivity_checked = True

    def execute_query(
        self,
        query: str,
        *,
        parameters_: dict[str, Any],
        database_: str,
    ) -> SimpleNamespace:
        """Return records selected by the stable query shape."""
        self.calls.append(
            {
                "query": query,
                "parameters": parameters_,
                "database": database_,
            }
        )
        if "defect.display_name AS display_name" in query:
            records = [
                {"class_name": "scratch", "display_name": "Scratch"}
            ]
        elif "step.instruction AS instruction" in query:
            records = [
                {"instruction": "Verified graph repair step 1"},
                {"instruction": "Verified graph repair step 2"},
            ]
        elif "inventory.mock_price AS mock_price" in query:
            records = [
                {
                    "part_id": "SIM-SCR-001",
                    "part_name": "Mock graph polishing kit",
                    "mock_price": 899.0,
                    "currency": "INR",
                    "fake_stock_quantity": 14,
                    "availability_status": "in_stock",
                }
            ]
        elif "max(inventory.last_updated)" in query:
            records = [{"inventory_last_updated": "2026-08-19"}]
        elif "RETURN defect.class_name AS class_name" in query:
            records = [
                {"class_name": "dent"},
                {"class_name": "normal"},
                {"class_name": "rust"},
                {"class_name": "scratch"},
            ]
        else:
            records = []
        return SimpleNamespace(records=records)

    def close(self) -> None:
        """Record driver cleanup."""
        self.closed = True


class UnavailableDriver(FakeDriver):
    """Mimic the Neo4j 6 driver when the container is offline."""

    def verify_connectivity(self) -> None:
        raise ServiceUnavailable("offline")

    def execute_query(
        self,
        query: str,
        *,
        parameters_: dict[str, Any],
        database_: str,
    ) -> SimpleNamespace:
        raise ServiceUnavailable("offline")


def _service(driver: FakeDriver) -> Neo4jService:
    """Build a service through the real YAML and environment loader."""
    service = Neo4jService.from_config(
        PROJECT_ROOT / "config" / "neo4j.yaml",
        environment={
            "NEO4J_URI": "bolt://test-neo4j:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "test-password",
            "NEO4J_DATABASE": "neo4j",
        },
        driver=driver,
    )
    assert service is not None
    return service


def test_configuration_rejects_placeholder_password() -> None:
    """Credentials must come from a non-placeholder environment value."""
    with pytest.raises(Neo4jConfigurationError, match="non-placeholder"):
        Neo4jService.from_config(
            PROJECT_ROOT / "config" / "neo4j.yaml",
            environment={"NEO4J_PASSWORD": PLACEHOLDER_PASSWORD},
            driver=FakeDriver(),
        )


def test_configuration_rejects_negative_retry_time(tmp_path: Path) -> None:
    """Failover timing must not silently accept an invalid negative value."""
    config_path = tmp_path / "neo4j.yaml"
    config_path.write_text(
        """
enabled: true
uri: bolt://127.0.0.1:7687
username: neo4j
database: neo4j
password_environment_variable: NEO4J_PASSWORD
managed_by: test
max_transaction_retry_time_seconds: -1
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(Neo4jConfigurationError, match="retry time"):
        Neo4jService.from_config(
            config_path,
            environment={"NEO4J_PASSWORD": "test-password"},
            driver=FakeDriver(),
        )


def test_driver_unavailability_is_converted_to_safe_service_status() -> None:
    """Neo4j 6 DriverError subclasses must activate the static fallback."""
    service = _service(UnavailableDriver())

    health = service.verify_connectivity()

    assert health.connected is False
    assert health.message == "Neo4j is unavailable: ServiceUnavailable"
    with pytest.raises(Neo4jServiceError, match="ServiceUnavailable"):
        service.get_defect_details("scratch")


def test_graph_records_map_to_existing_api_schemas() -> None:
    """Graph query results should produce validated guidance and parts."""
    driver = FakeDriver()
    service = _service(driver)

    health = service.verify_connectivity()
    details = service.get_defect_details(" Scratch ")
    inventory_date = service.get_inventory_last_updated()

    assert health.connected is True
    assert driver.connectivity_checked is True
    assert details is not None
    assert details.class_name == "scratch"
    assert details.repair_steps == [
        "Verified graph repair step 1",
        "Verified graph repair step 2",
    ]
    assert details.parts[0].fake_stock_quantity == 14
    assert inventory_date == "2026-08-19"
    assert all(call["database"] == "neo4j" for call in driver.calls)
    defect_call = next(
        call
        for call in driver.calls
        if "defect.display_name AS display_name" in call["query"]
    )
    assert "$class_name" in defect_call["query"]
    assert defect_call["parameters"] == {"class_name": "scratch"}


def test_catalog_seed_is_complete_and_idempotent_in_shape() -> None:
    """The seed should submit every scoped class, step, and mock part."""
    driver = FakeDriver()
    service = _service(driver)
    catalog = load_catalog(
        PROJECT_ROOT / "data" / "static" / "defect_catalog.json"
    )

    summary = service.seed_catalog(catalog)

    assert summary.model_dump() == {
        "object_types": 1,
        "defects": 4,
        "repair_steps": 9,
        "parts": 7,
        "inventory_records": 7,
    }
    seed_queries = [call["query"] for call in driver.calls]
    assert any("MERGE (object:ObjectType" in query for query in seed_queries)
    assert any("HAS_REPAIR_STEP" in query for query in seed_queries)
    assert any("inventory.is_simulated = true" in query for query in seed_queries)
