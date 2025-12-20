"""FastMCP tool: semantic_catalog."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from codeintel.serving.mcp._compat import Context, FastMCP
from codeintel.serving.mcp.runtime import QueryLimiter
from codeintel.serving.mcp.tools.shared import (
    READ_ONLY_LOCAL_ANNOTATIONS,
    TAG_READ,
    TAG_SEMANTIC,
    maybe_report_progress,
    mcp_correlation_id,
)
from codeintel.serving.metrics import QueryMetrics, log_query_metrics
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.models import SemanticCatalogResponse

if TYPE_CHECKING:
    from codeintel.serving.settings import ServingSettings


def register_catalog_tool(
    mcp: FastMCP,
    ops: ServingOperations,
    limiter: QueryLimiter,
    *,
    settings: ServingSettings,
) -> None:
    """Register semantic_catalog tool."""

    @mcp.tool(
        name="semantic_catalog",
        description="List available semantic views in the CodeIntel database",
        annotations=READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_SEMANTIC, TAG_READ},
    )
    async def semantic_catalog(*, ctx: Context) -> SemanticCatalogResponse:
        start = time.perf_counter()
        catalog = await limiter.run(ops.catalog)
        row_count = len(catalog.views)
        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(
            QueryMetrics(
                endpoint="mcp:semantic_catalog",
                view_id=None,
                query=None,
                row_count=row_count,
                truncated=False,
                duration_ms=duration_ms,
                correlation_id=mcp_correlation_id(ctx),
            )
        )
        await ctx.info("Retrieved semantic catalog")
        await maybe_report_progress(ctx, settings=settings, progress=100, total=100)
        return catalog


__all__ = ["register_catalog_tool"]
