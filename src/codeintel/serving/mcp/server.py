"""MCP server exposing CodeIntel datasets and tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import httpx
from mcp.server.fastmcp import FastMCP

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.bootstrap import (
    BackendResource,
    BackendResourceOptions,
    build_backend_resource,
)
from codeintel.serving.mcp.registry import register_tools
from codeintel.serving.mcp.tools_base import as_registrar, expose_tools
from codeintel.storage.gateway import StorageGateway

BackendFactory = Callable[..., BackendResource]
McpFactory = Callable[[str], object]


def create_mcp_server(
    cfg: ServingConfig | None = None,
    *,
    backend_factory: BackendFactory | None = None,
    gateway: StorageGateway | None = None,
    register_tools_fn: Callable[[object, object, ServingConfig | None], None] | None = None,
    mcp_factory: McpFactory | None = None,
) -> tuple[object, Callable[[], None]]:
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
        It receives (mcp, service_or_backend, config).
    mcp_factory:
        Optional factory to construct the MCP registrar/server (defaults to FastMCP).

    Returns
    -------
    tuple[object, Callable[[], None]]
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
    mcp_ctor = mcp_factory or (lambda name: FastMCP(name, json_response=True))
    mcp_instance = mcp_ctor("CodeIntel")
    service = getattr(backend, "service", None)
    # Pass config to enable auto-pipeline support
    registrar = as_registrar(mcp_instance)
    if register_tools_fn is not None:
        register_tools_fn(registrar, service or backend, config)
    else:
        register_tools(registrar, service or backend, config)
    expose_tools(mcp_instance)
    return mcp_instance, close


def main() -> None:
    """
    Run the CodeIntel MCP server.

    By default this uses stdio transport, which is what Cursor and the
    OpenAI CLI expect for local MCP servers. :contentReference[oaicite:14]{index=14}
    """
    server_obj, close = create_mcp_server()
    mcp_server = cast("FastMCP", server_obj)
    try:
        mcp_server.run()  # stdio by default
    finally:
        close()


if __name__ == "__main__":
    main()
