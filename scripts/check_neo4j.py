"""Verify that Neo4j contains the complete scoped PoC knowledge graph."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.neo4j_service import (
    Neo4jConfigurationError,
    Neo4jService,
    Neo4jServiceError,
)


def build_parser() -> argparse.ArgumentParser:
    """Create command-line options for graph and taxonomy configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "neo4j.yaml",
        help="Neo4j YAML configuration path.",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=PROJECT_ROOT / "config" / "class_names.json",
        help="Ordered supported-class JSON path.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    """Check connectivity, classes, guidance, parts, and inventory metadata."""
    options = build_parser().parse_args(arguments)
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    try:
        expected_classes = json.loads(options.classes.read_text(encoding="utf-8"))
        if not isinstance(expected_classes, list) or not expected_classes:
            raise ValueError("Class configuration must be a non-empty JSON list")
        service = Neo4jService.from_config(options.config)
        if service is None:
            raise Neo4jConfigurationError("Neo4j is disabled in its configuration")
        health = service.verify_connectivity()
        if not health.connected:
            raise Neo4jServiceError(health.message)

        actual_classes = service.list_defect_classes()
        if set(actual_classes) != set(expected_classes):
            raise Neo4jServiceError(
                "Graph classes do not match config/class_names.json"
            )
        for class_name in expected_classes:
            details = service.get_defect_details(str(class_name))
            if details is None:
                raise Neo4jServiceError(
                    f"Graph record is missing for class '{class_name}'"
                )
            if class_name == "normal":
                if details.repair_steps or details.parts:
                    raise Neo4jServiceError(
                        "The normal class must not recommend repairs or parts"
                    )
            elif not (2 <= len(details.repair_steps) <= 3) or not details.parts:
                raise Neo4jServiceError(
                    f"Class '{class_name}' has incomplete guidance or parts"
                )
        inventory_date = service.get_inventory_last_updated()
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        Neo4jConfigurationError,
        Neo4jServiceError,
        ValueError,
    ) as exc:
        raise SystemExit(f"Neo4j verification failed: {exc}") from exc
    finally:
        if "service" in locals() and service is not None:
            service.close()

    print("Neo4j verification passed.")
    print(
        json.dumps(
            {
                "classes": actual_classes,
                "inventory_last_updated": inventory_date,
                "inventory_is_simulated": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
