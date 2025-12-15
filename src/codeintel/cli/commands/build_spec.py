"""BuildSpec commands.

BuildSpec is a deterministic compiled contract derived from the Hamilton DAG.
These CLI commands expose compilation for CI gating and human inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.build_spec import build_spec_compile_handler

build_spec_app = App(
    name="spec",
    help="BuildSpec product commands (compile, show, diff, etc.).",
)

_SPEC_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)


@cli_command("build.spec.compile", handler=build_spec_compile_handler, config=_SPEC_CONFIG)
@build_spec_app.command(name="compile")
@dataclass
class BuildSpecCompileCommand:
    """Compile a deterministic BuildSpec from the Hamilton DAG."""

    include_columns: Annotated[
        bool,
        Parameter(
            name=["--include-columns"],
            help="Include dataset column names in the compiled spec.",
            negative=["--no-include-columns"],
        ),
    ] = False
    output_format: Annotated[
        str,
        Parameter(
            name=["--format"],
            help="Output format: json (default).",
        ),
    ] = "json"
    output_file: Annotated[
        str | None,
        Parameter(
            name=["--output", "-o"],
            help="Output file path (stdout if not specified).",
        ),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


__all__ = [
    "BuildSpecCompileCommand",
    "build_spec_app",
]
