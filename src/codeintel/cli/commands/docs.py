"""Cyclopts wiring for docs export commands.

This module wires Cyclopts command classes to unified handlers via @cli_command.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.docs import docs_export_handler
from codeintel.cli.rendering.types import OutputFormat

docs_app = App(
    name="docs",
    help="Document export utilities.",
)


class NxBackend(Enum):
    """NetworkX backend selection."""

    AUTO = "auto"
    CPU = "cpu"
    NX_CUGRAPH = "nx-cugraph"


class NxGpuMode(Enum):
    """GPU backend mode for NetworkX."""

    DISABLED = "disabled"
    ENABLED = "enabled"
    STRICT = "strict"


class ExportValidationMode(Enum):
    """Validation strategy for docs exports."""

    REQUIRED = "required"
    SKIP = "skip"


class MacroRequirement(Enum):
    """Requirement policy for normalized macros."""

    REQUIRE_NORMALIZED = "require_normalized"
    ALLOW_PARTIAL = "allow_partial"


class DryRunMode(Enum):
    """Execution mode for docs exports."""

    EXECUTE = "execute"
    DRY_RUN = "dry_run"


class PrereqMode(Enum):
    """Prerequisite execution strategy."""

    RUN = "run"
    SKIP = "skip"


# Config for docs commands - requires runtime and gateway
_DOCS_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)


@cli_command("docs.export", handler=docs_export_handler, config=_DOCS_CONFIG)
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


__all__ = ["docs_app"]
