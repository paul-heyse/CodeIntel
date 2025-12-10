"""Cyclopts wiring for docs export commands.

This module wires Cyclopts command classes to unified handlers via command_context.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.command_context import command_context
from codeintel.cli.cyclopts_common import OutputFormatCLI, RuntimeCLI
from codeintel.cli.handlers.docs import docs_export_handler

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
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            repo=self.repo,
            commit=self.commit,
            db_path=self.db_path,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {
            "validation": self.validation_mode.value,
            "macro_requirement": self.macro_requirement.value,
            "schemas": self.schemas,
            "datasets": self.datasets,
            "dry_run": self.run_mode == DryRunMode.DRY_RUN,
            "skip_prereqs": self.prereq_mode == PrereqMode.SKIP,
            "nx_backend": self.nx_backend.value,
            "nx_gpu_mode": self.nx_gpu_mode.value,
        }

        with command_context(
            "docs.export",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = docs_export_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


__all__ = ["docs_app"]
