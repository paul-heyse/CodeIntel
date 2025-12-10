"""Cyclopts wiring for IDE helper commands.

This module wires Cyclopts command classes to unified EnhancedHandlerContext handlers.
Commands use the command_context() helper for standardized infrastructure:

- Configuration loading via ConfigService
- Runtime resolution
- Logging setup based on verbosity
- Unified rendering via UnifiedRenderer
- Automatic resource cleanup
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cyclopts_common import (
    OutputFormatCLI,
    RuntimeCLI,
    command_context,
)
from codeintel.cli.handlers.ide import ide_hints_handler

ide_app = App(
    name="ide",
    help="IDE helper commands.",
)


@ide_app.command(name="hints")
@dataclass
class IdeHintsCommand:
    """Emit IDE hints (module + subsystem context) for a relative file path."""

    rel_path: Annotated[
        str,
        Parameter(
            name=None,
            help="File path relative to repo root (e.g., pkg/module.py).",
        ),
    ] = ""
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
    output_format: Annotated[
        OutputFormatCLI,
        Parameter(name="*"),
    ] = field(default_factory=OutputFormatCLI)

    def __call__(self) -> None:
        """Execute the IDE hints command."""
        # Build RuntimeCLI from individual params
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            verbose=self.verbose,
        )

        params: dict[str, object] = {"rel_path": self.rel_path}

        with command_context("ide.hints", runtime_cli, self.output_format, params=params) as (
            ctx,
            renderer,
        ):
            result = ide_hints_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


__all__ = ["ide_app"]
