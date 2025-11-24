"""MCP server exposing CodeIntel datasets and tools."""

from __future__ import annotations

from collections.abc import Callable

from mcp.server.fastmcp import FastMCP

from codeintel.config.serving_models import ServingConfig
from codeintel.mcp.registry import register_tools
from codeintel.services.factory import BackendResource, build_backend_resource


def create_mcp_server(cfg: ServingConfig | None = None) -> tuple[FastMCP, Callable[[], None]]:
    """
    Create the MCP server instance plus shutdown hook.

    Parameters
    ----------
    cfg:
        Optional pre-loaded ServingConfig. When omitted, environment variables are used.

    Returns
    -------
    tuple[FastMCP, Callable[[], None]]
        Configured MCP server and shutdown callback.
    """
    config = cfg or ServingConfig.from_env()
    resource: BackendResource = build_backend_resource(config)
    backend = resource.backend
    close = resource.close
    server = FastMCP("CodeIntel", json_response=True)
    service = getattr(backend, "service", None)
    register_tools(server, service or backend)
    return server, close


def main() -> None:
    """
    Run the CodeIntel MCP server.

    By default this uses stdio transport, which is what Cursor and the
    OpenAI CLI expect for local MCP servers. :contentReference[oaicite:14]{index=14}
    """
    server, close = create_mcp_server()
    try:
        server.run()  # stdio by default
    finally:
        close()


if __name__ == "__main__":
    main()
