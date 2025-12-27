"""BuildSpec commands.

BuildSpec is a deterministic compiled contract derived from the Hamilton DAG.
These CLI commands expose compilation for CI gating and human inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import App

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.build_spec import build_spec_compile_handler
from codeintel.cli.options.registry import (
    BUILD_SPEC_FORMAT,
    BUILD_SPEC_INCLUDE_COLUMNS,
    BUILD_SPEC_OUTPUT,
)
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param

build_spec_app = App(
    name="spec",
    help="BuildSpec product commands (compile, show, diff, etc.).",
)

_SPEC_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)

BUILD_SPEC_COMPILE_PATH: CommandPath = ("build", "spec", "compile")

_BUILD_SPEC_COMPILE_FLAGS_FIELD = shared_flags_field(BUILD_SPEC_COMPILE_PATH)


@cli_command("build.spec.compile", handler=build_spec_compile_handler, config=_SPEC_CONFIG)
@build_spec_app.command(name="compile")
@dataclass
class BuildSpecCompileCommand:
    """Compile a deterministic BuildSpec from the Hamilton DAG."""

    include_columns: Annotated[
        bool,
        option_param(BUILD_SPEC_INCLUDE_COLUMNS, command_path=BUILD_SPEC_COMPILE_PATH),
    ] = False
    output_format: Annotated[
        str,
        option_param(BUILD_SPEC_FORMAT, command_path=BUILD_SPEC_COMPILE_PATH),
    ] = "json"
    output_file: Annotated[
        str | None,
        option_param(BUILD_SPEC_OUTPUT, command_path=BUILD_SPEC_COMPILE_PATH),
    ] = None
    flags: SharedFlagsProtocol = _BUILD_SPEC_COMPILE_FLAGS_FIELD


__all__ = [
    "BuildSpecCompileCommand",
    "build_spec_app",
]
