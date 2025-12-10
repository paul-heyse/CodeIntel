"""Cyclopts wiring for serve command group."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import OutputFormatCLI, RuntimeCLI
from codeintel.cli.commands.context import command_context
from codeintel.cli.handlers.ops import serve_http_handler, serve_mcp_handler

serve_app = App(
    name="serve",
    help="HTTP and MCP server commands.",
)


@serve_app.command(name="http")
@dataclass
class ServeHttpCommand:
    """Start the HTTP server."""

    host: Annotated[
        str,
        Parameter(
            name=["--host", "-h"],
            help="Host to bind to.",
        ),
    ] = "127.0.0.1"
    port: Annotated[
        int,
        Parameter(
            name=["--port", "-p"],
            help="Port to bind to.",
        ),
    ] = 8000
    auto_pipeline: Annotated[
        bool,
        Parameter(
            name="--auto-pipeline",
            help="Enable automatic prerequisite pipeline execution.",
            negative=(),
        ),
    ] = False
    reload: Annotated[
        bool,
        Parameter(
            name="--reload",
            help="Enable auto-reload for development.",
            negative=(),
        ),
    ] = False
    root: Annotated[
        Path | None,
        Parameter(
            name=["--root", "-r"],
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
        """Execute the serve http command."""
        runtime_cli = RuntimeCLI(
            project_root=self.root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI()

        params: dict[str, object] = {
            "host": self.host,
            "port": self.port,
            "auto_pipeline": self.auto_pipeline,
            "reload": self.reload,
        }

        with command_context(
            "serve.http",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = serve_http_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@serve_app.command(name="mcp")
@dataclass
class ServeMcpCommand:
    """Start the MCP server."""

    auto_pipeline: Annotated[
        bool,
        Parameter(
            name="--auto-pipeline",
            help="Enable automatic prerequisite pipeline execution.",
            negative=(),
        ),
    ] = False
    root: Annotated[
        Path | None,
        Parameter(
            name=["--root", "-r"],
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
        """Execute the serve mcp command."""
        runtime_cli = RuntimeCLI(
            project_root=self.root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI()

        params: dict[str, object] = {
            "auto_pipeline": self.auto_pipeline,
        }

        with command_context(
            "serve.mcp",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = serve_mcp_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


__all__ = [
    "ServeHttpCommand",
    "ServeMcpCommand",
    "serve_app",
]
