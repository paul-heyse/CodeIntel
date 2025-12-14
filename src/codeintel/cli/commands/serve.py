"""Serve command group for HTTP and MCP servers.

Note: Serve commands require runtime/gateway access via handler pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import cli_command
from codeintel.cli.handlers.ops import serve_http_handler, serve_mcp_handler

serve_app = App(
    name="serve",
    help="HTTP and MCP server commands.",
)


@serve_app.command(name="http")
@cli_command("serve.http", handler=serve_http_handler)
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
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


@serve_app.command(name="mcp")
@cli_command("serve.mcp", handler=serve_mcp_handler)
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
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


__all__ = [
    "ServeHttpCommand",
    "ServeMcpCommand",
    "serve_app",
]
