"""Cyclopts wiring for storage commands.

This module wires Cyclopts command classes to unified handlers via @cli_command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands.decorators import CommandConfig, cli_command
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

# Config for storage commands - requires runtime and gateway
_STORAGE_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)


@cli_command("storage.validate_macros", handler=validate_macros_handler, config=_STORAGE_CONFIG)
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


@cli_command("storage.generate_macros", handler=generate_macros_handler, config=_STORAGE_CONFIG)
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


@cli_command("storage.profile", handler=profile_storage_handler, config=_STORAGE_CONFIG)
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


__all__ = ["storage_app"]
