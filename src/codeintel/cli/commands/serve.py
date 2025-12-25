"""Serve command group for HTTP and MCP servers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import App

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.ops import serve_http_handler, serve_mcp_handler
from codeintel.cli.options.registry import SERVE_HOST, SERVE_PORT, SERVE_RELOAD
from codeintel.cli.options.shared_flags import SharedFlags, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param

serve_app = App(
    name="serve",
    help="HTTP and MCP server commands.",
)

_SERVE_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)
SERVE_HTTP_PATH: CommandPath = ("serve", "http")
SERVE_MCP_PATH: CommandPath = ("serve", "mcp")

_SERVE_HTTP_FLAGS_FIELD = shared_flags_field(SERVE_HTTP_PATH)
_SERVE_MCP_FLAGS_FIELD = shared_flags_field(SERVE_MCP_PATH)


@serve_app.command(name="http")
@cli_command("serve.http", handler=serve_http_handler, config=_SERVE_CONFIG)
@dataclass
class ServeHttpCommand:
    """Start the HTTP server."""

    host: Annotated[
        str | None,
        option_param(SERVE_HOST, command_path=SERVE_HTTP_PATH),
    ] = None
    port: Annotated[
        int | None,
        option_param(SERVE_PORT, command_path=SERVE_HTTP_PATH),
    ] = None
    reload: Annotated[
        bool,
        option_param(SERVE_RELOAD, command_path=SERVE_HTTP_PATH),
    ] = False
    flags: SharedFlags = _SERVE_HTTP_FLAGS_FIELD


@serve_app.command(name="mcp")
@cli_command("serve.mcp", handler=serve_mcp_handler, config=_SERVE_CONFIG)
@dataclass
class ServeMcpCommand:
    """Start the MCP server."""

    flags: SharedFlags = _SERVE_MCP_FLAGS_FIELD


__all__ = [
    "ServeHttpCommand",
    "ServeMcpCommand",
    "serve_app",
]
