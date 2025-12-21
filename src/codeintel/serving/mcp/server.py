"""MCP server exposing the semantic serving kernel."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.runtime import build_db_manager, build_kernel
from codeintel.serving.settings import ServingSettings, get_serving_settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fastmcp import FastMCP


def create_mcp_server(
    settings: ServingSettings,
    *,
    db_manager: ServingDBManager | None = None,
) -> FastMCP:
    """Create an MCP server bound to the current serving snapshot.

    Parameters
    ----------
    settings
        Serving settings for runtime configuration.
    db_manager
        Optional pre-configured database manager. If provided, the MCP server
        will not manage its lifecycle (caller is responsible for start/stop).
        If None, creates and manages its own db_manager.

    Returns
    -------
    FastMCP
        Configured MCP server.
    """
    cfg = settings

    # Fail-fast security check
    cfg.validate_auth_for_host()

    # Use injected or create new
    if db_manager is None:
        db_manager = build_db_manager(cfg)
        owns_db_manager = True
    else:
        owns_db_manager = False

    kernel = build_kernel(db_manager, cfg)

    @asynccontextmanager
    async def lifespan(_mcp: FastMCP) -> AsyncGenerator[object]:
        if owns_db_manager:
            await db_manager.start()
        try:
            yield object()
        finally:
            if owns_db_manager:
                await db_manager.stop()

    return build_mcp_app(
        kernel=kernel,
        settings=cfg,
        lifespan=lifespan,
    )


def main(*, settings: ServingSettings | None = None) -> None:
    """Run the CodeIntel MCP server with stdio or HTTP transport."""
    cfg = settings or get_serving_settings()
    mcp = create_mcp_server(cfg)
    transport: Literal["stdio", "http"]
    transport = "stdio" if cfg.mcp_transport == "stdio" else "http"
    if transport == "stdio":
        mcp.run(transport="stdio")
        return
    mcp.run(
        transport="streamable-http",
        host=cfg.host,
        port=cfg.port,
        json_response=True,
        stateless_http=False,
    )


__all__ = ["create_mcp_server", "main"]
