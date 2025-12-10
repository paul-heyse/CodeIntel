"""Cyclopts wiring for storage commands.

This module wires Cyclopts command classes to unified ExecutionContext handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.execution.adapter import CycloptsAdapter
from codeintel.cli.storage_handlers import (
    MacroRequirement,
    generate_macros_structured,
    profile_storage_structured,
    storage_validate_macros_structured,
)

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
        CycloptsAdapter("storage.validate_macros", storage_validate_macros_structured)(self)


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
        CycloptsAdapter("storage.generate_macros", generate_macros_structured)(self)


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
    ] = Path("build/storage_profile")
    format: Annotated[
        str,
        Parameter(
            name="--format",
            help="Output format (text, json, csv).",
        ),
    ] = "text"
    include_samples: Annotated[
        bool,
        Parameter(
            name="--include-samples",
            help="Include sample data in profile.",
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
        CycloptsAdapter("storage.profile", profile_storage_structured)(self)


__all__ = ["storage_app"]
