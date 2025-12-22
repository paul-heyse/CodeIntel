"""FastMCP tool: semantic_query."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, get_type_hints

from fastmcp import Context, FastMCP

from codeintel.serving.features import ServingFeatureSet
from codeintel.serving.mcp.models import (
    QueryPreview,
    SemanticQueryToolRequest,
    SemanticQueryToolResponse,
)
from codeintel.serving.mcp.runtime import QueryLimiter
from codeintel.serving.mcp.tools.shared import (
    PREVIEW_ROW_COUNT,
    READ_ONLY_LOCAL_ANNOTATIONS,
    TAG_READ,
    TAG_SEMANTIC,
    McpMetricsInput,
    log_mcp_query_metrics,
    maybe_report_progress,
    try_sample_summary,
    validate_semantic_query_request,
)
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.models import SemanticQueryRequest

if TYPE_CHECKING:
    from codeintel.serving.settings import ServingSettings


@dataclass(frozen=True, slots=True)
class SemanticQueryHandler:
    """Handle semantic_query tool execution."""

    ops: ServingOperations
    limiter: QueryLimiter
    settings: ServingSettings
    feature_set: ServingFeatureSet

    async def handle(
        self,
        request: SemanticQueryRequest,
        *,
        ctx: Context,
    ) -> SemanticQueryToolResponse:
        start = time.perf_counter()
        await ctx.info(f"Querying view: {request.view_id}")
        await maybe_report_progress(ctx, settings=self.settings, progress=10, total=100)
        result = await self.limiter.run(self.ops.query, request)
        await maybe_report_progress(ctx, settings=self.settings, progress=100, total=100)

        row_count = len(result.rows)
        truncated = result.truncated
        query_hash = result.query_hash
        schema_hash = result.schema_hash

        sql_fingerprint = result.sql_fingerprint

        preview: QueryPreview | None = None
        if truncated or row_count > PREVIEW_ROW_COUNT:
            preview = QueryPreview(
                columns=tuple(result.columns),
                rows=tuple(result.rows[:PREVIEW_ROW_COUNT]),
                truncated=row_count > PREVIEW_ROW_COUNT or truncated,
            )

        summary: str | None = None
        if self.feature_set.enable_mcp_sampling and preview is not None:
            should_sample = truncated or row_count >= self.settings.mcp_sample_threshold
            if should_sample:
                summary = await try_sample_summary(
                    ctx,
                    view_id=request.view_id,
                    preview=preview,
                    query_hash=query_hash,
                )

        note = None
        if truncated:
            note = "Result truncated; use semantic_export for full dataset."

        duration_ms = (time.perf_counter() - start) * 1000
        metrics = McpMetricsInput(
            endpoint="mcp:semantic_query",
            view_id=request.view_id,
            query=None,
            row_count=row_count,
            truncated=truncated,
            duration_ms=duration_ms,
            query_hash=query_hash,
            schema_hash=schema_hash,
        )
        log_mcp_query_metrics(metrics, ctx=ctx)

        return SemanticQueryToolResponse(
            result=result,
            preview=preview,
            summary=summary,
            sql_fingerprint=sql_fingerprint,
            note=note,
        )


def register_query_tool(
    mcp: FastMCP,
    ops: ServingOperations,
    limiter: QueryLimiter,
    *,
    settings: ServingSettings,
) -> None:
    """Register semantic_query tool."""
    feature_set = ServingFeatureSet.from_settings(settings)
    handler = SemanticQueryHandler(
        ops=ops,
        limiter=limiter,
        settings=settings,
        feature_set=feature_set,
    )

    async def semantic_query(
        request: SemanticQueryToolRequest,
        *,
        ctx: Context,
    ) -> SemanticQueryToolResponse:
        validated = validate_semantic_query_request(request)
        return await handler.handle(validated, ctx=ctx)

    semantic_query.__annotations__ = get_type_hints(semantic_query, include_extras=True)
    mcp.tool(
        name="semantic_query",
        description="Query a semantic view with structured filters",
        annotations=READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_SEMANTIC, TAG_READ},
    )(semantic_query)


__all__ = ["register_query_tool"]
