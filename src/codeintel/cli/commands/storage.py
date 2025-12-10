"""Cyclopts wiring for storage commands.

This module wires Cyclopts command classes to unified handler functions
via command_context().
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import OutputFormatCLI, RuntimeCLI
from codeintel.cli.commands.context import command_context
from codeintel.cli.handlers.storage import (
    MacroRequirement,
    generate_macros_handler,
    profile_storage_handler,
    validate_macros_handler,
)
from codeintel.cli.rendering.types import OutputFormat

storage_app = App(
    name="storage",
    help="Storage validation utilities.",
)


@storage_app.command(name="validate-macros")
@dataclass
class ValidateMacrosCommand:
    """Validate macro registry hashes and normalized macro schemas."""

    db_path: Annotated[
        Path | None,
        Parameter(
            name="--db-path",
            help="Path to DuckDB database.",
        ),
    ] = None
    macro_requirement: Annotated[
        MacroRequirement,
        Parameter(
            name="--macros",
            help="Ingest macro requirement policy.",
            show_choices=True,
        ),
    ] = MacroRequirement.REQUIRE
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
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
        """Execute the storage validate-macros command."""
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            db_path=self.db_path,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {
            "db_path": str(self.db_path) if self.db_path else None,
            "macro_requirement": self.macro_requirement,
        }

        with command_context(
            "storage.validate_macros",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = validate_macros_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@storage_app.command(name="generate-macros")
@dataclass
class GenerateMacrosCommand:
    """Generate macros for tables."""

    tables: Annotated[
        list[str] | None,
        Parameter(
            name="--table",
            help="Tables to generate macros for (repeatable).",
        ),
    ] = None
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
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
        """Execute the storage generate-macros command."""
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {
            "tables": self.tables,
        }

        with command_context(
            "storage.generate_macros",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = generate_macros_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@storage_app.command(name="profile")
@dataclass
class ProfileStorageCommand:
    """Profile storage paths and sizes."""

    db_path: Annotated[
        Path | None,
        Parameter(
            name="--db-path",
            help="Path to DuckDB database.",
        ),
    ] = None
    output_dir: Annotated[
        Path,
        Parameter(
            name="--output-dir",
            help="Output directory for profile report.",
        ),
    ] = field(default_factory=lambda: Path("build/storage_profile"))
    include_views: Annotated[
        bool,
        Parameter(
            name="--include-views",
            help="Include views in profiling.",
            negative=(),
        ),
    ] = False
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
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
        """Execute the storage profile command."""
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            db_path=self.db_path,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {
            "db_path": str(self.db_path) if self.db_path else None,
            "output_dir": str(self.output_dir),
            "include_views": self.include_views,
        }

        with command_context(
            "storage.profile",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = profile_storage_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


__all__ = ["storage_app"]
