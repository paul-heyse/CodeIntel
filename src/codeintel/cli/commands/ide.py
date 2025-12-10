"""Cyclopts wiring for IDE helper commands.

This module wires Cyclopts command classes to unified handlers via @cli_command.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.ide import ide_hints_handler
from codeintel.cli.rendering.types import OutputFormat

ide_app = App(
    name="ide",
    help="IDE helper commands.",
)

# Config for IDE commands - requires runtime and gateway
_IDE_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)


@cli_command("ide.hints", handler=ide_hints_handler, config=_IDE_CONFIG)
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


__all__ = ["ide_app"]
