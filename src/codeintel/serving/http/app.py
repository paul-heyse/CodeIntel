"""FastAPI application factory for semantic serving."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.db.pool import DuckDBPoolConfig
from codeintel.serving.http.routes import search, semantic
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


def create_serving_app(
    settings: ServingSettings | None = None,
    *,
    mount_mcp: bool = True,
) -> FastAPI:
    """Create FastAPI serving application.

    Parameters
    ----------
    settings
        Serving settings (defaults to environment).
    mount_mcp
        Whether to mount an MCP server under `/mcp`.

    Returns
    -------
    FastAPI
        Configured application.
    """
    cfg = settings or ServingSettings.from_env()

    db_manager = ServingDBManager(
        pointer_path=cfg.serve_dir / "current.json",
        pool_cfg=DuckDBPoolConfig(size=cfg.pool_size),
        poll_interval_s=cfg.poll_interval_s,
    )
    kernel = SemanticQueryKernel(db=db_manager, settings=cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        await db_manager.start()
        try:
            yield
        finally:
            await db_manager.stop()

    app = FastAPI(
        title="CodeIntel Serving",
        description="Semantic layer API for CodeIntel",
        lifespan=lifespan,
    )

    app.state.kernel = kernel
    app.state.db_manager = db_manager

    def get_kernel() -> SemanticQueryKernel:
        return kernel

    app.dependency_overrides[semantic.get_kernel] = get_kernel
    app.dependency_overrides[search.get_kernel] = get_kernel
    app.include_router(semantic.router)
    app.include_router(search.router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        pointer = db_manager.current_pointer()
        return {
            "status": "ok",
            "repo": pointer.repo,
            "commit": pointer.commit,
            "run_id": pointer.run_id,
        }

    @app.get("/meta")
    async def meta() -> dict[str, object]:
        return kernel.meta()

    if mount_mcp:
        mcp = build_mcp_app(kernel=kernel, streamable_http_path="/")
        app.mount("/mcp", mcp.streamable_http_app())

    return app


__all__ = ["create_serving_app"]
