"""Semantic registry commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import App

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.semantic import semantic_compile_handler
from codeintel.cli.options.registry import (
    SEMANTIC_REGISTRY_OUTPUT,
    SEMANTIC_REGISTRY_VERSION,
)
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param

semantic_app = App(
    name="semantic",
    help="Semantic registry commands.",
)

_SEMANTIC_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)

SEMANTIC_COMPILE_PATH: CommandPath = ("semantic", "compile")
_SEMANTIC_COMPILE_FLAGS_FIELD = shared_flags_field(SEMANTIC_COMPILE_PATH)


@cli_command("semantic.compile", handler=semantic_compile_handler, config=_SEMANTIC_CONFIG)
@semantic_app.command(name="compile")
@dataclass(frozen=True)
class SemanticCompileCommand:
    """Compile semantic registry JSON from the Hamilton DAG."""

    output_file: Annotated[
        str | None,
        option_param(SEMANTIC_REGISTRY_OUTPUT, command_path=SEMANTIC_COMPILE_PATH),
    ] = None
    version: Annotated[
        str,
        option_param(SEMANTIC_REGISTRY_VERSION, command_path=SEMANTIC_COMPILE_PATH),
    ] = "v1"
    flags: SharedFlagsProtocol = _SEMANTIC_COMPILE_FLAGS_FIELD


__all__ = [
    "SemanticCompileCommand",
    "semantic_app",
]
