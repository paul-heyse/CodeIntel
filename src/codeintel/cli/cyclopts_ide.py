"""Cyclopts wiring for IDE helper commands.

This module wires Cyclopts command classes to unified ExecutionContext handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.execution.adapter import CycloptsAdapter
from codeintel.cli.ide_handlers import ide_hints_ctx

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

    def __call__(self) -> None:
        """Execute the IDE hints command."""
        CycloptsAdapter("ide.hints", ide_hints_ctx)(self)


__all__ = ["ide_app"]
