"""Seed Neo4j from the project's verified static defect catalog."""

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

from backend.app.services.catalog_service import load_catalog
from backend.app.services.neo4j_service import (
    Neo4jConfigurationError,
    Neo4jService,
    Neo4jServiceError,
)


def build_parser() -> argparse.ArgumentParser:
    """Create command-line options for explicit local seed paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "neo4j.yaml",
        help="Neo4j YAML configuration path.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "data" / "static" / "defect_catalog.json",
        help="Verified JSON catalog used as the graph seed source.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    """Verify connectivity and idempotently seed the academic graph."""
    options = build_parser().parse_args(arguments)
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    try:
        service = Neo4jService.from_config(options.config)
        if service is None:
            raise Neo4jConfigurationError("Neo4j is disabled in its configuration")
        catalog = load_catalog(options.catalog)
        health = service.verify_connectivity()
        if not health.connected:
            raise Neo4jServiceError(health.message)
        summary = service.seed_catalog(catalog)
    except (FileNotFoundError, Neo4jConfigurationError, Neo4jServiceError, ValueError) as exc:
        raise SystemExit(f"Neo4j seed failed: {exc}") from exc
    finally:
        if "service" in locals() and service is not None:
            service.close()

    print("Neo4j seed completed successfully.")
    print(json.dumps(summary.model_dump(), indent=2))


if __name__ == "__main__":
    main()
