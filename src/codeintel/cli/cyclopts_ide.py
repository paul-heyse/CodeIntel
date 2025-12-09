"""Cyclopts wiring for IDE helper commands."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_errors import invoke_with_typer_translation
from codeintel.cli.commands.ide import IdeHintsOptions, ide_hints_handler
from codeintel.cli.cyclopts_common import RuntimeCLI, runtime_cli_to_options

ide_app = App(
    name="ide",
    help="IDE helper commands.",
)


@ide_app.command(name="hints")
def hints(
    rel_path: Annotated[
        str,
        Parameter(
            name=None,
            help="File path relative to repo root (e.g., pkg/module.py).",
        ),
    ],
    runtime: Annotated[RuntimeCLI, Parameter(name="*")] | None = None,
) -> None:
    """Emit IDE hints (module + subsystem context) for a relative file path."""
    cfg = runtime or RuntimeCLI()
    runtime_options = runtime_cli_to_options(cfg)
    options = IdeHintsOptions(
        rel_path=rel_path,
        runtime_options=runtime_options,
        verbose=cfg.verbose,
    )
    invoke_with_typer_translation(ide_hints_handler, options)


__all__ = ["ide_app"]
