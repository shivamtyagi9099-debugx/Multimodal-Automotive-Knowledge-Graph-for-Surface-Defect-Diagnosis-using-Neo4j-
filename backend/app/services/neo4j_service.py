"""Neo4j access for repair guidance and simulated parts inventory."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import yaml
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError

from backend.app.schemas.inventory import (
    DefectCatalogEntry,
    InventoryCatalog,
    PartRecord,
)
from backend.app.schemas.neo4j import (
    GraphSeedSummary,
    Neo4jConnectionSettings,
    Neo4jHealthStatus,
)

PLACEHOLDER_PASSWORD = "replace-with-a-local-password"


class Neo4jConfigurationError(ValueError):
    """Report an invalid or incomplete local Neo4j configuration."""


class Neo4jServiceError(RuntimeError):
    """Report a database operation that could not be completed."""


class DefectKnowledgeService(Protocol):
    """Define the graph operations used by FastAPI and test doubles."""

    def verify_connectivity(self) -> Neo4jHealthStatus:
        """Return the current graph connection status."""

    def get_defect_details(self, class_name: str) -> DefectCatalogEntry | None:
        """Return graph-backed guidance for one supported class."""

    def get_inventory_last_updated(self) -> str:
        """Return the static date attached to simulated inventory records."""

    def close(self) -> None:
        """Release database resources."""


class Neo4jService:
    """Query and seed the project's small Neo4j knowledge graph."""

    _DEFECT_QUERY = """
        MATCH (defect:Defect {class_name: $class_name})
        RETURN defect.class_name AS class_name,
               defect.display_name AS display_name
    """
    _REPAIR_STEPS_QUERY = """
        MATCH (:Defect {class_name: $class_name})
              -[relationship:HAS_REPAIR_STEP]->(step:RepairStep)
        RETURN step.instruction AS instruction
        ORDER BY relationship.position ASC
    """
    _PARTS_QUERY = """
        MATCH (:Defect {class_name: $class_name})-[:MAY_REQUIRE]->(part:Part)
              -[:HAS_INVENTORY]->(inventory:InventoryRecord)
        RETURN part.part_id AS part_id,
               part.part_name AS part_name,
               inventory.mock_price AS mock_price,
               inventory.currency AS currency,
               inventory.fake_stock_quantity AS fake_stock_quantity,
               inventory.availability_status AS availability_status
        ORDER BY part.part_id ASC
    """
    _INVENTORY_DATE_QUERY = """
        MATCH (:Part)-[:HAS_INVENTORY]->(inventory:InventoryRecord)
        RETURN max(inventory.last_updated) AS inventory_last_updated
    """

    def __init__(
        self,
        settings: Neo4jConnectionSettings,
        *,
        managed_by: str = "panelcheck-seed-v1",
        connection_timeout_seconds: float = 10.0,
        connection_acquisition_timeout_seconds: float = 10.0,
        max_transaction_retry_time_seconds: float = 30.0,
        max_connection_pool_size: int = 20,
        driver: Driver | Any | None = None,
    ) -> None:
        """Create a lazy Neo4j driver without connecting during API import."""
        self.settings = settings
        self.managed_by = managed_by
        self._driver = driver or GraphDatabase.driver(
            settings.uri,
            auth=(settings.username, settings.password.get_secret_value()),
            connection_timeout=connection_timeout_seconds,
            connection_acquisition_timeout=(
                connection_acquisition_timeout_seconds
            ),
            max_transaction_retry_time=max_transaction_retry_time_seconds,
            max_connection_pool_size=max_connection_pool_size,
        )

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        *,
        environment: Mapping[str, str] | None = None,
        driver: Driver | Any | None = None,
    ) -> Neo4jService | None:
        """Build a service from non-secret YAML plus environment credentials."""
        resolved_path = config_path.resolve()
        if not resolved_path.is_file():
            raise Neo4jConfigurationError(
                f"Neo4j configuration does not exist: {resolved_path}"
            )
        raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise Neo4jConfigurationError(
                "Neo4j configuration must contain an object at its root"
            )
        if not bool(raw.get("enabled", True)):
            return None

        values = os.environ if environment is None else environment
        password_variable = str(
            raw.get("password_environment_variable", "NEO4J_PASSWORD")
        ).strip()
        if not password_variable:
            raise Neo4jConfigurationError(
                "password_environment_variable must not be blank"
            )
        password = values.get(password_variable, "").strip()
        if not password or password == PLACEHOLDER_PASSWORD:
            raise Neo4jConfigurationError(
                f"Set a non-placeholder {password_variable} value in the root .env file"
            )

        settings = Neo4jConnectionSettings(
            uri=values.get("NEO4J_URI", str(raw.get("uri", ""))),
            username=values.get(
                "NEO4J_USERNAME", str(raw.get("username", ""))
            ),
            password=password,
            database=values.get(
                "NEO4J_DATABASE", str(raw.get("database", "neo4j"))
            ),
        )
        managed_by = str(raw.get("managed_by", "panelcheck-seed-v1")).strip()
        if not managed_by:
            raise Neo4jConfigurationError("managed_by must not be blank")

        try:
            connection_timeout = float(
                raw.get("connection_timeout_seconds", 10.0)
            )
            acquisition_timeout = float(
                raw.get("connection_acquisition_timeout_seconds", 10.0)
            )
            retry_time = float(
                raw.get("max_transaction_retry_time_seconds", 30.0)
            )
            pool_size = int(raw.get("max_connection_pool_size", 20))
        except (TypeError, ValueError) as exc:
            raise Neo4jConfigurationError(
                "Neo4j connection limits must be numeric"
            ) from exc
        if (
            connection_timeout <= 0
            or acquisition_timeout <= 0
            or retry_time < 0
            or pool_size <= 0
        ):
            raise Neo4jConfigurationError(
                "Neo4j connection limits must be positive and retry time non-negative"
            )

        return cls(
            settings,
            managed_by=managed_by,
            connection_timeout_seconds=connection_timeout,
            connection_acquisition_timeout_seconds=acquisition_timeout,
            max_transaction_retry_time_seconds=retry_time,
            max_connection_pool_size=pool_size,
            driver=driver,
        )

    def verify_connectivity(self) -> Neo4jHealthStatus:
        """Check Neo4j without exposing credentials in the status response."""
        try:
            self._driver.verify_connectivity()
        except (DriverError, Neo4jError, OSError, RuntimeError) as exc:
            return Neo4jHealthStatus(
                connected=False,
                database=self.settings.database,
                uri=self.settings.uri,
                message=f"Neo4j is unavailable: {type(exc).__name__}",
            )
        return Neo4jHealthStatus(
            connected=True,
            database=self.settings.database,
            uri=self.settings.uri,
            message="Neo4j connection verified.",
        )

    def get_defect_details(self, class_name: str) -> DefectCatalogEntry | None:
        """Build one defect record from graph nodes and relationships."""
        normalized = class_name.strip().casefold()
        if not normalized:
            raise ValueError("class_name must not be blank")

        defect_records = self._execute(
            self._DEFECT_QUERY,
            class_name=normalized,
        )
        if not defect_records:
            return None
        try:
            defect_record = defect_records[0]
            repair_steps = [
                str(record["instruction"])
                for record in self._execute(
                    self._REPAIR_STEPS_QUERY,
                    class_name=normalized,
                )
            ]
            parts = [
                PartRecord.model_validate(dict(record))
                for record in self._execute(
                    self._PARTS_QUERY,
                    class_name=normalized,
                )
            ]
            return DefectCatalogEntry(
                class_name=str(defect_record["class_name"]),
                display_name=str(defect_record["display_name"]),
                repair_steps=repair_steps,
                parts=parts,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Neo4jServiceError(
                "Neo4j returned incomplete or invalid defect data"
            ) from exc

    def list_defect_classes(self) -> list[str]:
        """Return all seeded defect class names in a stable order."""
        records = self._execute(
            """
            MATCH (defect:Defect)
            RETURN defect.class_name AS class_name
            ORDER BY defect.class_name ASC
            """
        )
        try:
            return [str(record["class_name"]) for record in records]
        except (KeyError, TypeError) as exc:
            raise Neo4jServiceError(
                "Neo4j returned an invalid class-name record"
            ) from exc

    def get_inventory_last_updated(self) -> str:
        """Return the latest static date stored on graph inventory records."""
        records = self._execute(self._INVENTORY_DATE_QUERY)
        if not records or not records[0]["inventory_last_updated"]:
            raise Neo4jServiceError(
                "Neo4j does not contain an inventory last-updated date"
            )
        return str(records[0]["inventory_last_updated"])

    def seed_catalog(self, catalog: InventoryCatalog) -> GraphSeedSummary:
        """Idempotently create graph nodes and relationships from the catalog."""
        constraints = (
            "CREATE CONSTRAINT object_type_name IF NOT EXISTS "
            "FOR (node:ObjectType) REQUIRE node.name IS UNIQUE",
            "CREATE CONSTRAINT defect_class_name IF NOT EXISTS "
            "FOR (node:Defect) REQUIRE node.class_name IS UNIQUE",
            "CREATE CONSTRAINT repair_step_id IF NOT EXISTS "
            "FOR (node:RepairStep) REQUIRE node.step_id IS UNIQUE",
            "CREATE CONSTRAINT part_id IF NOT EXISTS "
            "FOR (node:Part) REQUIRE node.part_id IS UNIQUE",
            "CREATE CONSTRAINT inventory_id IF NOT EXISTS "
            "FOR (node:InventoryRecord) REQUIRE node.inventory_id IS UNIQUE",
        )
        for query in constraints:
            self._execute(query)

        defects = [
            {
                "class_name": entry.class_name,
                "display_name": entry.display_name,
            }
            for entry in catalog.defects
        ]
        self._execute(
            """
            MERGE (object:ObjectType {name: $object_type})
            SET object.managed_by = $managed_by
            WITH object
            UNWIND $defects AS defect
            MERGE (node:Defect {class_name: defect.class_name})
            SET node.display_name = defect.display_name,
                node.managed_by = $managed_by
            MERGE (object)-[:CAN_HAVE]->(node)
            """,
            object_type=catalog.object_type,
            defects=defects,
            managed_by=self.managed_by,
        )

        repair_steps: list[dict[str, Any]] = []
        parts: list[dict[str, Any]] = []
        for entry in catalog.defects:
            for position, instruction in enumerate(entry.repair_steps, start=1):
                repair_steps.append(
                    {
                        "class_name": entry.class_name,
                        "step_id": f"{entry.class_name}-step-{position:02d}",
                        "position": position,
                        "instruction": instruction,
                    }
                )
            for part in entry.parts:
                parts.append(
                    {
                        "class_name": entry.class_name,
                        "inventory_id": part.part_id,
                        "last_updated": catalog.inventory_last_updated,
                        **part.model_dump(),
                    }
                )

        if repair_steps:
            self._execute(
                """
                UNWIND $repair_steps AS repair
                MATCH (defect:Defect {class_name: repair.class_name})
                MERGE (step:RepairStep {step_id: repair.step_id})
                SET step.instruction = repair.instruction,
                    step.managed_by = $managed_by
                MERGE (defect)-[relationship:HAS_REPAIR_STEP]->(step)
                SET relationship.position = repair.position
                """,
                repair_steps=repair_steps,
                managed_by=self.managed_by,
            )
        if parts:
            self._execute(
                """
                UNWIND $parts AS item
                MATCH (defect:Defect {class_name: item.class_name})
                MERGE (part:Part {part_id: item.part_id})
                SET part.part_name = item.part_name,
                    part.managed_by = $managed_by
                MERGE (defect)-[:MAY_REQUIRE]->(part)
                MERGE (inventory:InventoryRecord {
                    inventory_id: item.inventory_id
                })
                SET inventory.mock_price = item.mock_price,
                    inventory.currency = item.currency,
                    inventory.fake_stock_quantity = item.fake_stock_quantity,
                    inventory.availability_status = item.availability_status,
                    inventory.last_updated = item.last_updated,
                    inventory.is_simulated = true,
                    inventory.managed_by = $managed_by
                MERGE (part)-[:HAS_INVENTORY]->(inventory)
                """,
                parts=parts,
                managed_by=self.managed_by,
            )

        return GraphSeedSummary(
            object_types=1,
            defects=len(defects),
            repair_steps=len(repair_steps),
            parts=len(parts),
            inventory_records=len(parts),
        )

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        self._driver.close()

    def _execute(self, query: str, **parameters: Any) -> list[Any]:
        """Execute one parameterized query against the configured database."""
        try:
            result = self._driver.execute_query(
                query,
                parameters_=parameters,
                database_=self.settings.database,
            )
        except (DriverError, Neo4jError, OSError, RuntimeError) as exc:
            raise Neo4jServiceError(
                f"Neo4j query failed: {type(exc).__name__}"
            ) from exc
        records = getattr(result, "records", None)
        if records is None and isinstance(result, tuple) and result:
            records = result[0]
        return list(records or [])
