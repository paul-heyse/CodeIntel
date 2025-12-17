"""MCP server exposing the semantic serving kernel."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from codeintel.serving.mcp._compat import FastMCP


def create_mcp_server(settings: ServingSettings | None = None) -> FastMCP:
    """Create an MCP server bound to the current serving snapshot.

    Parameters
    ----------
    settings
        Serving settings (defaults to environment).

    Returns
    -------
    FastMCP
        Configured MCP server.
    """
    cfg = settings or ServingSettings.from_env()
    db_manager = ServingDBManager(
        pointer_path=cfg.serve_dir / "current.json",
        pool_cfg=PoolConfig(size=cfg.pool_size),
        poll_interval_s=cfg.poll_interval_s,
        hot_swap=cfg.hot_swap,
    )
    kernel = SemanticQueryKernel(db=db_manager, settings=cfg)

    @asynccontextmanager
    async def lifespan(_mcp: FastMCP) -> AsyncGenerator[object]:
        await db_manager.start()
        try:
            yield object()
        finally:
            await db_manager.stop()

    return build_mcp_app(
        kernel=kernel,
        settings=cfg,
        lifespan=lifespan,
    )


def main() -> None:
    """Run the CodeIntel MCP server with stdio or HTTP transport."""
    cfg = ServingSettings.from_env()
    mcp = create_mcp_server(cfg)
    transport: Literal["stdio", "http"]
    transport = "stdio" if cfg.mcp_transport == "stdio" else "http"
    mcp.run(transport=transport)


__all__ = ["create_mcp_server", "main"]
