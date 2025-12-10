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
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import OutputFormatCLI, RuntimeCLI
from codeintel.cli.commands.context import command_context
from codeintel.cli.handlers.ide import ide_hints_handler
from codeintel.cli.rendering.types import OutputFormat

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
        """Execute the IDE hints command."""
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {"rel_path": self.rel_path}

        with command_context(
            "ide.hints",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = ide_hints_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


__all__ = ["ide_app"]
