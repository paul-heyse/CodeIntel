"""Cyclopts wiring for storage commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from cyclopts import App, Parameter

from codeintel.cli.commands.storage import MacroRequirement, storage_validate_macros
from codeintel.cli.cyclopts_common import Verbose

storage_app = App(
    name="storage",
    help="Storage validation utilities.",
)


@dataclass
class ValidateMacrosCli:
    """CLI surface for `codeintel storage validate-macros`."""

    db_path: Annotated[
        Path,
        Parameter(
            name="--db-path",
            help="Path to the DuckDB database to validate.",
        ),
    ] = Path("build/db/codeintel.duckdb")
    macros: Annotated[
        MacroRequirement,
        Parameter(
            name="--macros",
            help="Ingest macro requirement policy.",
        ),
    ] = MacroRequirement.REQUIRE
    verbose: Verbose = 0


@storage_app.command(name="validate-macros")
def validate_macros(
    cfg: Annotated[ValidateMacrosCli, Parameter(name="*")] | None = None,
) -> None:
    """Validate macro registry hashes and normalized macro schemas.

    Raises
    ------
    SystemExit
        When validation fails or the handler requests exit.
    """
    cfg = cfg or ValidateMacrosCli()
    try:
        storage_validate_macros(cfg.db_path, cfg.macros, cfg.verbose)
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


__all__ = ["storage_app"]
