"""FastMCP tool: semantic_query."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from fastmcp import Context, FastMCP

from codeintel.serving.features import ServingFeatureSet
from codeintel.serving.mcp.models import QueryPreview, SemanticQueryToolResponse
from codeintel.serving.mcp.runtime import QueryLimiter
from codeintel.serving.mcp.tools.shared import (
    PREVIEW_ROW_COUNT,
    READ_ONLY_LOCAL_ANNOTATIONS,
    TAG_READ,
    TAG_SEMANTIC,
    build_semantic_request,
    maybe_report_progress,
    mcp_correlation_id,
    try_sample_summary,
)
from codeintel.serving.metrics import QueryMetrics, log_query_metrics
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.fingerprints import sqlglot_canonical_sha256

if TYPE_CHECKING:
    from codeintel.serving.settings import ServingSettings


def register_query_tool(
    mcp: FastMCP,
    ops: ServingOperations,
    limiter: QueryLimiter,
    *,
    settings: ServingSettings,
) -> None:
    """Register semantic_query tool."""
    feature_set = ServingFeatureSet.from_settings(settings)

    @mcp.tool(
        name="semantic_query",
        description="Query a semantic view with structured filters",
        annotations=READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_SEMANTIC, TAG_READ},
    )
    async def semantic_query(  # noqa: PLR0913 - MCP tool signature requires these params
        view_id: str,
        filters: list[dict[str, object]] | None = None,
        select: list[str] | None = None,
        order_by: list[str] | None = None,
        pagination: dict[str, int] | None = None,
        *,
        ctx: Context,
    ) -> SemanticQueryToolResponse:
        start = time.perf_counter()
        await ctx.info(f"Querying view: {view_id}")
        await maybe_report_progress(ctx, settings=settings, progress=10, total=100)
        request = build_semantic_request(view_id, filters, select, order_by, pagination)
        await maybe_report_progress(ctx, settings=settings, progress=20, total=100)
        result = await limiter.run(ops.query, request)
        await maybe_report_progress(ctx, settings=settings, progress=100, total=100)

        row_count = len(result.rows)
        truncated = result.truncated
        query_hash = result.query_hash
        schema_hash = result.schema_hash

        sql_fingerprint: str | None = None
        try:
            compiled_sql = await limiter.run(ops.compile_query_sql, request)
        except (KeyError, TypeError, ValueError):
            compiled_sql = None
        if isinstance(compiled_sql, str) and compiled_sql:
            sql_fingerprint = sqlglot_canonical_sha256(compiled_sql)

        preview: QueryPreview | None = None
        if truncated or row_count > PREVIEW_ROW_COUNT:
            preview = QueryPreview(
                columns=tuple(result.columns),
                rows=tuple(result.rows[:PREVIEW_ROW_COUNT]),
                truncated=row_count > PREVIEW_ROW_COUNT or truncated,
            )

        summary: str | None = None
        if feature_set.enable_mcp_sampling and preview is not None:
            should_sample = truncated or row_count >= settings.mcp_sample_threshold
            if should_sample:
                summary = await try_sample_summary(
                    ctx,
                    view_id=view_id,
                    preview=preview,
                    query_hash=query_hash,
                )

        note = None
        if truncated:
            note = "Result truncated; use semantic_export for full dataset."

        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(
            QueryMetrics(
                endpoint="mcp:semantic_query",
                view_id=view_id,
                query=None,
                row_count=row_count,
                truncated=truncated,
                duration_ms=duration_ms,
                correlation_id=mcp_correlation_id(ctx),
                query_hash=query_hash,
                schema_hash=schema_hash,
            )
        )

        return SemanticQueryToolResponse(
            result=result,
            preview=preview,
            summary=summary,
            sql_fingerprint=sql_fingerprint,
            note=note,
        )


__all__ = ["register_query_tool"]
