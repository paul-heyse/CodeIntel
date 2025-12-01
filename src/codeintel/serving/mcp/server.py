"""MCP server exposing CodeIntel datasets and tools."""

from __future__ import annotations

from collections.abc import Callable

import httpx
from mcp.server.fastmcp import FastMCP

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.mcp.registry import register_tools
from codeintel.serving.services.factory import BackendResource, build_backend_resource
from codeintel.serving.services.wiring import BackendResourceOptions
from codeintel.storage.gateway import StorageGateway

BackendFactory = Callable[..., BackendResource]


def create_mcp_server(
    cfg: ServingConfig | None = None,
    *,
    backend_factory: BackendFactory | None = None,
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
        StorageGateway supplying the DuckDB connection and registry.
    register_tools_fn:
        Optional function to register tools against the MCP server (defaults to registry helper).

    Returns
    -------
    tuple[FastMCP, Callable[[], None]]
        Configured MCP server and shutdown callback.

    Raises
    ------
    ValueError
        If a gateway is not provided for local_db mode.
    """
    config = cfg or ServingConfig.from_env()
    if gateway is None and config.mode == "local_db":
        message = "StorageGateway is required for MCP server in local_db mode"
        raise ValueError(message)

    def _adapt_factory(factory: BackendFactory) -> BackendFactory:
        def _wrapped(
            wrapped_cfg: ServingConfig,
            *,
            gateway: StorageGateway | None = None,
            http_client: httpx.Client | httpx.AsyncClient | None = None,
            options: BackendResourceOptions | None = None,
        ) -> BackendResource:
            try:
                return factory(
                    wrapped_cfg,
                    gateway=gateway,
                    http_client=http_client,
                    options=options,
                )
            except TypeError:
                return factory(wrapped_cfg)

        return _wrapped

    factory: BackendFactory = _adapt_factory(backend_factory or build_backend_resource)
    resource: BackendResource = factory(config, gateway=gateway)
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
