"""Docs export commands.

Note: Docs commands require runtime/gateway access via handler pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.docs import docs_export_handler

docs_app = App(
    name="docs",
    help="Document export utilities.",
)

_CYCLOPTS_PATH_TYPE = Path


class NxBackend(StrEnum):
    """NetworkX backend selection."""

    AUTO = "auto"
    CPU = "cpu"
    NX_CUGRAPH = "nx-cugraph"


_DOCS_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)


@cli_command("docs.export", handler=docs_export_handler, config=_DOCS_CONFIG)
@docs_app.command(name="export")
@dataclass
class DocsExportCommand:
    """Export datasets to Document Output/."""

    repo: Annotated[
        str | None,
        Parameter(
            name="--repo",
            help="Repository slug.",
        ),
    ] = None
    commit: Annotated[
        str | None,
        Parameter(
            name="--commit",
            help="Commit SHA.",
        ),
    ] = None
    db_path: Annotated[
        Path | None,
        Parameter(
            name="--db-path",
            help="Path to DuckDB database.",
        ),
    ] = None
    build_dir: Annotated[
        Path | None,
        Parameter(
            name="--build-dir",
            help="Build directory for docs export.",
        ),
    ] = None
    repo_root: Annotated[
        Path | None,
        Parameter(
            name="--repo-root",
            help="Repository root directory.",
        ),
    ] = None
    document_output_dir: Annotated[
        Path | None,
        Parameter(
            name="--document-output-dir",
            help="Document Output directory for emitted artifacts.",
        ),
    ] = None

    nx_backend: Annotated[
        NxBackend,
        Parameter(
            name="--nx-backend",
            help="NetworkX backend selection: auto, cpu, or nx-cugraph.",
            show_choices=True,
        ),
    ] = NxBackend.AUTO
    nx_gpu_mode: Annotated[
        str,
        Parameter(
            name="--nx-gpu-mode",
            help="GPU backend preference: disabled, enabled, or strict.",
        ),
    ] = "disabled"

    validation_mode: Annotated[
        str,
        Parameter(
            name="--validation-mode",
            help="Validation strategy: required or skip.",
            show_choices=True,
        ),
    ] = "skip"
    validate: Annotated[
        bool,
        Parameter(
            name="--validate",
            help="Enable export validation.",
            negative=("--no-validate",),
        ),
    ] = False
    skip_prereqs: Annotated[
        bool,
        Parameter(
            name="--skip-prereqs",
            help="Skip prerequisite ingestion or build steps.",
            negative=("--run-prereqs",),
        ),
    ] = False
    schemas: Annotated[
        list[str] | None,
        Parameter(
            name="--schema",
            help="Table key to validate (repeatable).",
        ),
    ] = None
    datasets: Annotated[
        list[str] | None,
        Parameter(
            name="--dataset",
            help="Dataset name to export (repeatable).",
        ),
    ] = None
    run_mode: Annotated[
        str,
        Parameter(
            name="--run-mode",
            help="Execution mode for docs export.",
            show_choices=True,
        ),
    ] = "execute"
    dry_run: Annotated[
        bool,
        Parameter(
            name="--dry-run",
            help="Preview export without writing files.",
            negative=("--no-dry-run",),
        ),
    ] = False
    prereq_mode: Annotated[
        str,
        Parameter(
            name="--prereq-mode",
            help="Prerequisite execution mode.",
            show_choices=True,
        ),
    ] = "run"

    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


__all__ = ["docs_app"]
