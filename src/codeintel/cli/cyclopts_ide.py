"""Cyclopts wiring for IDE helper commands."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_errors import run_handler
from codeintel.cli.cyclopts_common import RuntimeCLI, RuntimeParam, runtime_cli_to_options
from codeintel.cli.ide_handlers import IdeHintsOptions, RuntimeCliOptions, ide_hints_handler

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
    runtime: RuntimeParam | None = None,
) -> None:
    """Emit IDE hints (module + subsystem context) for a relative file path."""
    cfg = runtime or RuntimeCLI()
    cli_opts = runtime_cli_to_options(cfg)
    options = IdeHintsOptions(
        rel_path=rel_path,
        runtime_options=RuntimeCliOptions(project_root=cli_opts.project_root),
        verbose=cfg.verbose,
    )
    run_handler(ide_hints_handler, options)


__all__ = ["ide_app"]
