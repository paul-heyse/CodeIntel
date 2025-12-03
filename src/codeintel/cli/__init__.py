"""CodeIntel application CLI entry point.

This module provides the Typer-based CLI for CodeIntel, offering command groups
for pipeline orchestration, operation invocation, dataset inspection, and
serving HTTP/MCP endpoints.

Command Groups
--------------
- **pipeline**: Run full or operation-targeted pipelines, check status
- **op**: List and invoke serving operations
- **dataset**: List, describe, and verify dataset contracts
- **serve**: Start HTTP or MCP servers

Example Usage
-------------
.. code-block:: bash

    # Run full pipeline
    codeintel-app pipeline run-full

    # Run minimal pipeline for an operation
    codeintel-app pipeline run-op function.summary

    # List available operations
    codeintel-app op list

    # Start HTTP server
    codeintel-app serve http --port 8000
"""

from __future__ import annotations

import typer

from codeintel.cli.main import dataset_app, op_app, pipeline_app, serve_app

app = typer.Typer(
    name="codeintel-app",
    help="CodeIntel application CLI for pipeline and serving operations.",
    no_args_is_help=True,
)

app.add_typer(pipeline_app, name="pipeline")
app.add_typer(op_app, name="op")
app.add_typer(dataset_app, name="dataset")
app.add_typer(serve_app, name="serve")


def main() -> None:
    """Entry point for the codeintel-app CLI."""
    app()


__all__ = [
    "app",
    "dataset_app",
    "main",
    "op_app",
    "pipeline_app",
    "serve_app",
]

