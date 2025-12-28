"""FastMCP application builder for CodeIntel serving."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import anyio
from anyio import lowlevel
from fastmcp import FastMCP
from starlette.responses import JSONResponse, PlainTextResponse

from codeintel.observability.runtime import get_observability
from codeintel.serving.auth.policy import mcp_auth_provider
from codeintel.serving.features import ServingFeatureSet
from codeintel.serving.mcp.middleware_stack import build_mcp_middleware
from codeintel.serving.mcp.prompts import register_prompts
from codeintel.serving.mcp.protocols import SemanticKernelProtocol
from codeintel.serving.mcp.resource_store import ResourceStore
from codeintel.serving.mcp.resources import register_resources
from codeintel.serving.mcp.runtime import QueryLimiter
from codeintel.serving.mcp.tools import (
    register_catalog_tool,
    register_describe_tool,
    register_explain_tool,
    register_export_tool,
    register_meta_tool,
    register_query_tool,
    register_search_tool,
)
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.settings import ServingSettings

try:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"
    generate_latest = None

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from contextlib import AbstractAsyncContextManager

    from starlette.requests import Request
    from starlette.responses import Response


LOG = logging.getLogger(__name__)

_SERVER_STARTED_AT = datetime.now(UTC)
_HEALTH_READY_TIMEOUT_S = 0.25


def build_mcp_app(
    *,
    kernel: SemanticKernelProtocol,
    settings: ServingSettings,
    lifespan: Callable[[FastMCP], AbstractAsyncContextManager[object]] | None = None,
) -> FastMCP:
    """Build FastMCP application with semantic tools.

    Returns
    -------
    FastMCP
        Configured FastMCP application instance.
    """
    ops = ServingOperations(kernel=kernel, settings=settings)
    features = ServingFeatureSet.from_settings(settings)
    store = ResourceStore(
        settings.serve_dir / "exports",
        ttl_seconds=settings.mcp_export_ttl_seconds,
    )

    @asynccontextmanager
    async def composed_lifespan(server: FastMCP) -> AsyncIterator[object]:
        async with _optional_lifespan(lifespan, server):
            store.cleanup_expired()
            async with anyio.create_task_group() as tg:
                if settings.mcp_export_ttl_seconds is not None:
                    interval = max(settings.mcp_export_cleanup_interval_seconds, 1)
                    tg.start_soon(_periodic_store_cleanup, store, interval)
                yield object()

    mcp = FastMCP(
        "CodeIntel",
        mask_error_details=settings.mcp_mask_errors,
        lifespan=composed_lifespan,
        auth=mcp_auth_provider(settings),
        middleware=build_mcp_middleware(settings),
        on_duplicate_tools="error",
        on_duplicate_resources="error",
        on_duplicate_prompts="error",
        strict_input_validation=True,
    )

    query_limiter = QueryLimiter(max_concurrent=settings.mcp_max_concurrent_queries)
    export_limiter = QueryLimiter(max_concurrent=settings.mcp_max_concurrent_exports)

    register_catalog_tool(mcp, ops, query_limiter, settings=settings)
    register_describe_tool(mcp, ops, query_limiter, settings=settings)
    register_query_tool(
        mcp,
        ops,
        query_limiter,
        settings=settings,
        store=store,
        export_limiter=export_limiter,
    )

    if features.enable_mcp_explain:
        register_explain_tool(mcp, ops, query_limiter, settings=settings)
    if features.enable_mcp_meta:
        register_meta_tool(
            mcp, ops, query_limiter, settings=settings, started_at=_SERVER_STARTED_AT
        )
    if features.enable_mcp_search:
        register_search_tool(mcp, ops, query_limiter, settings=settings)
    if features.enable_mcp_export:
        register_export_tool(mcp, ops, export_limiter, store, settings=settings)

    register_resources(mcp, ops, store, settings=settings)
    _register_health_routes(mcp, ops)
    _register_metrics_routes(mcp)
    register_prompts(mcp, settings=settings, kernel=ops)

    return mcp


@asynccontextmanager
async def _optional_lifespan(
    lifespan: Callable[[FastMCP], AbstractAsyncContextManager[object]] | None,
    server: FastMCP,
) -> AsyncIterator[object]:
    if lifespan is None:
        yield object()
        return
    async with lifespan(server):
        yield object()


async def _periodic_store_cleanup(store: ResourceStore, interval_seconds: int) -> None:
    while True:
        store.cleanup_expired()
        await anyio.sleep(interval_seconds)


def _register_health_routes(mcp: FastMCP, ops: ServingOperations) -> None:
    """Register health check routes for load balancers and orchestrators."""

    @mcp.custom_route("/health", methods=["GET"])
    async def mcp_health(_request: Request) -> Response:
        ready = await ops.db.wait_ready(timeout_s=_HEALTH_READY_TIMEOUT_S)
        if not ready:
            return JSONResponse(
                {"status": "error", "detail": "No active snapshot"},
                status_code=503,
            )
        try:
            summary = ops.db.current_summary()
        except RuntimeError:
            return JSONResponse(
                {"status": "error", "detail": "No active snapshot"},
                status_code=503,
            )
        return JSONResponse(
            {
                "status": "ok",
                "repo": summary.get("repo"),
                "commit": str(summary.get("commit", ""))[:12],
                "run_id": summary.get("run_id"),
            }
        )

    @mcp.custom_route("/ready", methods=["GET"])
    async def mcp_ready(_request: Request) -> Response:
        ready = await ops.db.wait_ready(timeout_s=_HEALTH_READY_TIMEOUT_S)
        if not ready:
            return PlainTextResponse("not ready", status_code=503)
        return PlainTextResponse("ready")


def _register_metrics_routes(mcp: FastMCP) -> None:
    obs = get_observability()
    if not obs.prometheus_enabled or not _PROMETHEUS_AVAILABLE or generate_latest is None:
        return

    generate_latest_fn = generate_latest

    @mcp.custom_route("/metrics", methods=["GET"])
    async def mcp_metrics(_request: Request) -> Response:
        await lowlevel.checkpoint()
        payload = generate_latest_fn()
        return PlainTextResponse(payload, media_type=CONTENT_TYPE_LATEST)


__all__ = ["build_mcp_app"]
