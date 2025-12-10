"""Cyclopts wiring for docs export commands.

This module wires Cyclopts command classes to unified ExecutionContext handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.docs_handlers import (
    DryRunMode,
    ExportValidationMode,
    MacroRequirement,
    NxGpuMode,
    PrereqMode,
    docs_export_ctx,
)
from codeintel.cli.execution.adapter import CycloptsAdapter

docs_app = App(
    name="docs",
    help="Document export utilities.",
)


class NxBackend(Enum):
    """NetworkX backend selection."""

    AUTO = "auto"
    CPU = "cpu"
    NX_CUGRAPH = "nx-cugraph"


@docs_app.command(name="export")
@dataclass
class DocsExportCommand:
    """Export datasets to Document Output/."""

    # Project options
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
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

    # Backend options
    nx_backend: Annotated[
        NxBackend,
        Parameter(
            name="--nx-backend",
            help="NetworkX backend selection: auto, cpu, or nx-cugraph.",
            show_choices=True,
        ),
    ] = NxBackend.AUTO
    nx_gpu_mode: Annotated[
        NxGpuMode,
        Parameter(
            name="--nx-gpu-mode",
            help="GPU backend preference: disabled, enabled, or strict.",
        ),
    ] = NxGpuMode.DISABLED

    # Export options
    validation_mode: Annotated[
        ExportValidationMode,
        Parameter(
            name="--validation-mode",
            help="Validation strategy: required or skip.",
            show_choices=True,
        ),
    ] = ExportValidationMode.SKIP
    macro_requirement: Annotated[
        MacroRequirement,
        Parameter(
            name="--macro-requirement",
            help="Normalized macro requirement policy.",
            show_choices=True,
        ),
    ] = MacroRequirement.ALLOW_PARTIAL
    schemas: Annotated[
        list[str] | None,
        Parameter(
            name="--schema",
            help="Schema name to validate (repeatable).",
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
        DryRunMode,
        Parameter(
            name="--run-mode",
            help="Execution mode for docs export.",
            show_choices=True,
        ),
    ] = DryRunMode.EXECUTE
    prereq_mode: Annotated[
        PrereqMode,
        Parameter(
            name="--prereq-mode",
            help="Prerequisite execution mode.",
            show_choices=True,
        ),
    ] = PrereqMode.RUN

    # Output options
    output_format: Annotated[
        OutputFormat,
        Parameter(
            name="--output-format",
            help="Output format (text or json).",
        ),
    ] = OutputFormat.TEXT
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the docs export command."""
        CycloptsAdapter("docs.export", docs_export_ctx)(self)


__all__ = ["docs_app"]
