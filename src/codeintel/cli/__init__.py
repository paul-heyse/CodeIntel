"""CodeIntel unified CLI entry point.

This module provides the Typer-based CLI for CodeIntel, exposing all functional
areas (pipeline, ingest, graph, docs, datasets, etc.) under a single coherent
interface.

Command Groups
--------------
- **pipeline**: Pipeline orchestration (run-full, run-op, run, list-steps, deps, status)
- **build**: Build system commands (run, status) for minimal-work target computation
- **op**: List and invoke serving operations
- **dataset**: Dataset inspection and verification
- **datasets**: Extended dataset management (lint, diff, snapshot, scaffold, catalog)
- **serve**: HTTP and MCP server startup
- **ingest**: Ingestion plugin orchestration
- **graph**: Graph analytics plugin management
- **docs**: Document export utilities
- **storage**: Storage validation utilities
- **history**: Historical timeseries aggregation
- **ide**: IDE integration helpers
- **subsystem**: Subsystem exploration

Example Usage
-------------
.. code-block:: bash

    # Run full pipeline (using project file)
    codeintel pipeline run-full

    # Run full pipeline (using explicit options)
    codeintel pipeline run --repo my-org/repo --commit abc123

    # Run minimal pipeline for an operation
    codeintel pipeline run-op function.summary

    # List available operations
    codeintel op list

    # Start HTTP server
    codeintel serve http --port 8000

    # Run ingestion with a recipe
    codeintel ingest run --recipe full_python

    # List graph plugins
    codeintel graph plugins --plan

    # Export documents
    codeintel docs export

    # Generate dataset catalog
    codeintel datasets catalog

    # Build specific targets with dependency resolution
    codeintel build run function_metrics

    # Show build target status
    codeintel build status
"""

from __future__ import annotations

import typer

from codeintel.cli.commands.build import build_app
from codeintel.cli.commands.datasets import datasets_ext_app
from codeintel.cli.commands.docs import docs_app
from codeintel.cli.commands.graphs import graphs_app
from codeintel.cli.commands.history import history_app
from codeintel.cli.commands.ide import ide_app
from codeintel.cli.commands.ingest import ingest_app
from codeintel.cli.commands.pipeline import (
    pipeline_deps,
    pipeline_list_steps,
    pipeline_run,
)
from codeintel.cli.commands.storage import storage_app
from codeintel.cli.commands.subsystem import subsystem_app
from codeintel.cli.main import dataset_app, op_app, pipeline_app, serve_app

app = typer.Typer(
    name="codeintel",
    help="CodeIntel unified CLI for pipeline, analytics, and serving operations.",
    no_args_is_help=True,
)

# Add extended pipeline commands to the existing pipeline_app
# These add list-steps, deps, and run (with scope filtering)
pipeline_app.command("run")(pipeline_run)
pipeline_app.command("list-steps")(pipeline_list_steps)
pipeline_app.command("deps")(pipeline_deps)

# Core application commands (from existing Typer CLI)
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(op_app, name="op")
app.add_typer(dataset_app, name="dataset")
app.add_typer(serve_app, name="serve")
app.add_typer(build_app, name="build")

# Migrated commands from legacy CLI
app.add_typer(ingest_app, name="ingest")
app.add_typer(graphs_app, name="graph")
app.add_typer(docs_app, name="docs")
app.add_typer(storage_app, name="storage")
app.add_typer(history_app, name="history")
app.add_typer(ide_app, name="ide")
app.add_typer(subsystem_app, name="subsystem")
app.add_typer(datasets_ext_app, name="datasets")


def main() -> None:
    """Entry point for the codeintel CLI."""
    app()


__all__ = [
    "app",
    "build_app",
    "dataset_app",
    "datasets_ext_app",
    "docs_app",
    "graphs_app",
    "history_app",
    "ide_app",
    "ingest_app",
    "main",
    "op_app",
    "pipeline_app",
    "pipeline_deps",
    "pipeline_list_steps",
    "pipeline_run",
    "serve_app",
    "storage_app",
    "subsystem_app",
]
