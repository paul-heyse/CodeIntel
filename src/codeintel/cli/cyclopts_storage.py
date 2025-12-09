"""Cyclopts wiring for storage commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_errors import run_handler
from codeintel.cli.cyclopts_common import (
    RUNTIME_PARAM_FIELD,
    ExistingPath,
    RuntimeParam,
    get_verbose,
    runtime_cli_to_options,
)
from codeintel.cli.storage_handlers import MacroRequirement, storage_validate_macros

storage_app = App(
    name="storage",
    help="Storage validation utilities.",
)


@dataclass
class ValidateMacrosCli:
    """CLI surface for `codeintel storage validate-macros`."""

    runtime: RuntimeParam = RUNTIME_PARAM_FIELD
    macros: Annotated[
        MacroRequirement,
        Parameter(
            name="--macros",
            help="Ingest macro requirement policy.",
            show_choices=True,
        ),
    ] = MacroRequirement.REQUIRE
    db_path: Annotated[
        ExistingPath,
        Parameter(
            name="--db-path",
            help="Path to DuckDB database.",
        ),
    ] = Path("build/db/codeintel.duckdb")


@storage_app.command(name="validate-macros")
def validate_macros(
    cfg: Annotated[ValidateMacrosCli, Parameter(name="*")] | None = None,
) -> None:
    """Validate macro registry hashes and normalized macro schemas."""
    cfg = cfg or ValidateMacrosCli()
    runtime_options = runtime_cli_to_options(cfg.runtime)
    db_path = cfg.db_path or runtime_options.db_path or Path("build/db/codeintel.duckdb")
    verbose = get_verbose(cfg.runtime)
    run_handler(
        storage_validate_macros,
        db_path,
        cfg.macros,
        verbose,
    )


__all__ = ["storage_app"]
