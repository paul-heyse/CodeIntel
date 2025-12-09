"""Cyclopts wiring for storage commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_errors import invoke_with_typer_translation
from codeintel.cli.commands.storage import MacroRequirement, storage_validate_macros
from codeintel.cli.cyclopts_common import RuntimeCLI, RuntimeParam, runtime_cli_to_options

storage_app = App(
    name="storage",
    help="Storage validation utilities.",
)


@dataclass
class ValidateMacrosCli:
    """CLI surface for `codeintel storage validate-macros`."""

    runtime: RuntimeParam = field(default_factory=RuntimeCLI)
    macros: Annotated[
        MacroRequirement,
        Parameter(
            name="--macros",
            help="Ingest macro requirement policy.",
            show_choices=True,
        ),
    ] = MacroRequirement.REQUIRE


@storage_app.command(name="validate-macros")
def validate_macros(
    cfg: Annotated[ValidateMacrosCli, Parameter(name="*")] | None = None,
) -> None:
    """Validate macro registry hashes and normalized macro schemas."""
    cfg = cfg or ValidateMacrosCli()
    runtime_options = runtime_cli_to_options(cfg.runtime)
    db_path = runtime_options.db_path or Path("build/db/codeintel.duckdb")
    invoke_with_typer_translation(
        storage_validate_macros,
        db_path,
        cfg.macros,
        cfg.runtime.verbose,
    )


__all__ = ["storage_app"]
