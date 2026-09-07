"""Start the local FastAPI server for the academic proof-of-concept."""

from __future__ import annotations

import argparse
import os
import socket
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
LOG_LEVELS = ("critical", "error", "warning", "info", "debug", "trace")


def _prepare_project_environment() -> None:
    """Make application imports and project-relative paths deterministic."""
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.chdir(PROJECT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for local API server options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Interface to bind (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"TCP port to bind (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload the development server when source files change.",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="info",
        help="Server logging verbosity (default: info).",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    """Parse server options and run the same-origin API and frontend app."""
    parser = build_parser()
    options = parser.parse_args(arguments)
    if not 1 <= options.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    probe_host = options.host if options.host not in {"0.0.0.0", "::"} else "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        if probe.connect_ex((probe_host, options.port)) == 0:
            parser.error(
                f"port {options.port} is already in use; stop the existing server "
                "or choose another port with --port"
            )

    _prepare_project_environment()
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=options.host,
        port=options.port,
        reload=options.reload,
        log_level=options.log_level,
    )


if __name__ == "__main__":
    main()
