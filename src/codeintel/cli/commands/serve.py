"""Serve command group for HTTP and MCP servers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.ops import serve_http_handler, serve_mcp_handler

serve_app = App(
    name="serve",
    help="HTTP and MCP server commands.",
)

_SERVE_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)


@serve_app.command(name="http")
@cli_command("serve.http", handler=serve_http_handler, config=_SERVE_CONFIG)
@dataclass
class ServeHttpCommand:
    """Start the HTTP server."""

    host: Annotated[
        str | None,
        Parameter(
            name=["--host", "-h"],
            help="Host to bind to.",
        ),
    ] = None
    port: Annotated[
        int | None,
        Parameter(
            name=["--port", "-p"],
            help="Port to bind to.",
        ),
    ] = None
    reload: Annotated[
        bool,
        Parameter(
            name="--reload",
            help="Enable auto-reload for development.",
            negative=(),
        ),
    ] = False
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@serve_app.command(name="mcp")
@cli_command("serve.mcp", handler=serve_mcp_handler, config=_SERVE_CONFIG)
@dataclass
class ServeMcpCommand:
    """Start the MCP server."""

    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


__all__ = [
    "ServeHttpCommand",
    "ServeMcpCommand",
    "serve_app",
]
