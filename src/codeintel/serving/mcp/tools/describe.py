"""FastMCP tool: semantic_describe."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

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
from codeintel.serving.semantic.models import SemanticViewDescriptionResponse

if TYPE_CHECKING:
    from codeintel.serving.settings import ServingSettings

_CURRENT_CONTEXT = CurrentContext()


def register_describe_tool(
    mcp: FastMCP,
    ops: ServingOperations,
    limiter: QueryLimiter,
    *,
    settings: ServingSettings,
) -> None:
    """Register semantic_describe tool."""

    @mcp.tool(
        name="semantic_describe",
        description="Describe a semantic view's schema and metadata",
        annotations=READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_SEMANTIC, TAG_READ},
    )
    async def semantic_describe(
        view_id: str,
        *,
        ctx: Context = _CURRENT_CONTEXT,
    ) -> SemanticViewDescriptionResponse:
        start = time.perf_counter()
        await ctx.info(f"Describing view: {view_id}")
        await maybe_report_progress(ctx, settings=settings, progress=10, total=100)
        result = await limiter.run(ops.describe, view_id)
        await maybe_report_progress(ctx, settings=settings, progress=100, total=100)
        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(
            QueryMetrics(
                endpoint="mcp:semantic_describe",
                view_id=view_id,
                query=None,
                row_count=0,
                truncated=False,
                duration_ms=duration_ms,
                correlation_id=mcp_correlation_id(ctx),
            )
        )
        return result


__all__ = ["register_describe_tool"]
