"""FastMCP tool: semantic_explain."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, get_type_hints

from fastmcp import Context, FastMCP

from codeintel.serving.mcp.models import SemanticExplainToolRequest
from codeintel.serving.mcp.runtime import QueryLimiter
from codeintel.serving.mcp.tools.shared import (
    READ_ONLY_LOCAL_ANNOTATIONS,
    TAG_READ,
    TAG_SEMANTIC,
    McpMetricsInput,
    log_mcp_query_metrics,
    maybe_report_progress,
    validate_semantic_query_request,
)
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.models import SemanticExplainResponse, SemanticQueryRequest

if TYPE_CHECKING:
    from codeintel.serving.settings import ServingSettings


@dataclass(frozen=True, slots=True)
class SemanticExplainHandler:
    """Handle semantic_explain tool execution."""

    ops: ServingOperations
    limiter: QueryLimiter
    settings: ServingSettings

    async def handle(
        self,
        request: SemanticQueryRequest,
        *,
        ctx: Context,
    ) -> SemanticExplainResponse:
        start = time.perf_counter()
        await ctx.info(f"Explaining query for view: {request.view_id}")
        await maybe_report_progress(ctx, settings=self.settings, progress=10, total=100)
        result = await self.limiter.run(self.ops.explain, request)
        await maybe_report_progress(ctx, settings=self.settings, progress=100, total=100)

        duration_ms = (time.perf_counter() - start) * 1000
        metrics = McpMetricsInput(
            endpoint="mcp:semantic_explain",
            view_id=request.view_id,
            query=None,
            row_count=0,
            truncated=False,
            duration_ms=duration_ms,
        )
        log_mcp_query_metrics(metrics, ctx=ctx)
        return result


def register_explain_tool(
    mcp: FastMCP,
    ops: ServingOperations,
    limiter: QueryLimiter,
    *,
    settings: ServingSettings,
) -> None:
    """Register semantic_explain tool."""
    handler = SemanticExplainHandler(ops=ops, limiter=limiter, settings=settings)

    async def semantic_explain(
        request: SemanticExplainToolRequest,
        *,
        ctx: Context,
    ) -> SemanticExplainResponse:
        validated = validate_semantic_query_request(request)
        return await handler.handle(validated, ctx=ctx)

    semantic_explain.__annotations__ = get_type_hints(
        semantic_explain, include_extras=True
    )
    mcp.tool(
        name="semantic_explain",
        description="Return compiled SQL and DuckDB plan for a semantic query",
        annotations=READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_SEMANTIC, TAG_READ},
    )(semantic_explain)


__all__ = ["register_explain_tool"]
