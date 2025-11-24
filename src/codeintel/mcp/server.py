"""MCP server exposing CodeIntel datasets and tools."""

from __future__ import annotations

from collections.abc import Callable

from mcp.server.fastmcp import FastMCP

from codeintel.config.serving_models import ServingConfig
from codeintel.mcp.registry import register_tools
from codeintel.services.factory import BackendResource, build_backend_resource
from codeintel.storage.gateway import StorageGateway


def create_mcp_server(
    cfg: ServingConfig | None = None,
    *,
    backend_factory: Callable[[ServingConfig], BackendResource] | None = None,
    gateway: StorageGateway | None = None,
    register_tools_fn: Callable[[FastMCP, object], None] | None = None,
) -> tuple[FastMCP, Callable[[], None]]:
    """
    Create the MCP server instance plus shutdown hook.

    Parameters
    ----------
    cfg:
        Optional pre-loaded ServingConfig. When omitted, environment variables are used.
    backend_factory:
        Optional factory for producing BackendResource (defaults to build_backend_resource).
    gateway:
        Optional StorageGateway supplying the DuckDB connection and registry.
    register_tools_fn:
        Optional function to register tools against the MCP server (defaults to registry helper).

    Returns
    -------
    tuple[FastMCP, Callable[[], None]]
        Configured MCP server and shutdown callback.
    """
    config = cfg or ServingConfig.from_env()
    if gateway is not None and backend_factory is None:
        resource = build_backend_resource(config, gateway=gateway)
    else:
        resource: BackendResource = (backend_factory or build_backend_resource)(config)
    backend = resource.backend
    close = resource.close
    server = FastMCP("CodeIntel", json_response=True)
    service = getattr(backend, "service", None)
    (register_tools_fn or register_tools)(server, service or backend)
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
