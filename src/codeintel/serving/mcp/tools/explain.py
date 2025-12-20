"""FastMCP tool: semantic_explain."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from codeintel.serving.mcp._compat import Context, FastMCP
from codeintel.serving.mcp.runtime import QueryLimiter
from codeintel.serving.mcp.tools.shared import (
    READ_ONLY_LOCAL_ANNOTATIONS,
    TAG_READ,
    TAG_SEMANTIC,
    build_semantic_request,
    maybe_report_progress,
    mcp_correlation_id,
)
from codeintel.serving.metrics import QueryMetrics, log_query_metrics
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.models import SemanticExplainResponse

if TYPE_CHECKING:
    from codeintel.serving.settings import ServingSettings


def register_explain_tool(
    mcp: FastMCP,
    ops: ServingOperations,
    limiter: QueryLimiter,
    *,
    settings: ServingSettings,
) -> None:
    """Register semantic_explain tool."""

    @mcp.tool(
        name="semantic_explain",
        description="Return compiled SQL and DuckDB plan for a semantic query",
        annotations=READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_SEMANTIC, TAG_READ},
    )
    async def semantic_explain(  # noqa: PLR0913 - MCP tool signature requires these params
        view_id: str,
        filters: list[dict[str, object]] | None = None,
        select: list[str] | None = None,
        order_by: list[str] | None = None,
        pagination: dict[str, int] | None = None,
        *,
        ctx: Context,
    ) -> SemanticExplainResponse:
        start = time.perf_counter()
        await ctx.info(f"Explaining query for view: {view_id}")
        await maybe_report_progress(ctx, settings=settings, progress=10, total=100)
        request = build_semantic_request(view_id, filters, select, order_by, pagination)
        await maybe_report_progress(ctx, settings=settings, progress=20, total=100)
        result = await limiter.run(ops.explain, request)
        await maybe_report_progress(ctx, settings=settings, progress=100, total=100)

        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(
            QueryMetrics(
                endpoint="mcp:semantic_explain",
                view_id=view_id,
                query=None,
                row_count=0,
                truncated=False,
                duration_ms=duration_ms,
                correlation_id=mcp_correlation_id(ctx),
            )
        )
        return result


__all__ = ["register_explain_tool"]
