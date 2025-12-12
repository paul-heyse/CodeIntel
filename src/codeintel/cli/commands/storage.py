"""Storage commands for macro validation, generation, and profiling.

Note: Storage commands require runtime/gateway access via handler pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.storage import (
    MacroRequirement,
    generate_macros_handler,
    profile_storage_handler,
    validate_macros_handler,
)

storage_app = App(
    name="storage",
    help="Storage validation utilities.",
)


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
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


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
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


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
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


__all__ = ["storage_app"]
