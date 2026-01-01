"""FastMCP tool: code_search."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from codeintel.serving.mcp.runtime import QueryLimiter
from codeintel.serving.mcp.tools.shared import (
    READ_ONLY_LOCAL_ANNOTATIONS,
    TAG_READ,
    TAG_SEARCH,
    maybe_report_progress,
    mcp_correlation_id,
)
from codeintel.serving.metrics import QueryMetrics, log_query_metrics
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.search.models import SearchQueryRequest, SearchQueryResponse

if TYPE_CHECKING:
    from codeintel.serving.settings import ServingSettings

_CURRENT_CONTEXT = CurrentContext()


def register_search_tool(
    mcp: FastMCP,
    ops: ServingOperations,
    limiter: QueryLimiter,
    *,
    settings: ServingSettings,
) -> None:
    """Register code_search tool."""

    @mcp.tool(
        name="code_search",
        description="Search code metadata using BM25 full-text search",
        annotations=READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_SEARCH, TAG_READ},
    )
    async def code_search(
        query: str,
        kinds: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
        *,
        ctx: Context = _CURRENT_CONTEXT,
    ) -> SearchQueryResponse:
        start = time.perf_counter()
        await ctx.info(f"Searching: {query}")
        await maybe_report_progress(ctx, settings=settings, progress=10, total=100)
        request = SearchQueryRequest(
            query=query,
            kinds=kinds,
            limit=limit,
            offset=offset,
        )
        await maybe_report_progress(ctx, settings=settings, progress=20, total=100)
        result = await limiter.run(ops.search, request)
        await maybe_report_progress(ctx, settings=settings, progress=100, total=100)

        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(
            QueryMetrics(
                endpoint="mcp:code_search",
                view_id=None,
                query=query,
                row_count=len(result.results),
                truncated=False,
                duration_ms=duration_ms,
                correlation_id=mcp_correlation_id(ctx),
                query_hash=result.query_hash,
            )
        )
        return result


__all__ = ["register_search_tool"]
