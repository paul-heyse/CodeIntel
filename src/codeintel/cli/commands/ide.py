"""Cyclopts wiring for IDE helper commands.

This module wires Cyclopts command classes to unified handlers via @cli_command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.ide import ide_hints_handler

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
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


__all__ = ["ide_app"]
