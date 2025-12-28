"""FastMCP tool: semantic_query."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, get_type_hints

from fastmcp import Context, FastMCP

from codeintel.serving.features import ServingFeatureSet
from codeintel.serving.mcp.models import (
    ExportHandleResponse,
    QueryPreview,
    SemanticQueryToolRequest,
    SemanticQueryToolResponse,
)
from codeintel.serving.mcp.runtime import QueryLimiter
from codeintel.serving.mcp.tools.export import ExportWorkflow
from codeintel.serving.mcp.tools.shared import (
    PREVIEW_ROW_COUNT,
    READ_ONLY_LOCAL_ANNOTATIONS,
    TAG_READ,
    TAG_SEMANTIC,
    McpMetricsInput,
    log_mcp_query_metrics,
    maybe_report_progress,
    try_sample_summary,
)
from codeintel.serving.operations.cancellation import CancelToken
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.models import SemanticExportRequest

if TYPE_CHECKING:
    from codeintel.serving.export.formats import ExportFormat
    from codeintel.serving.mcp.resource_store import ResourceStore
    from codeintel.serving.settings import ServingSettings


@dataclass(frozen=True, slots=True)
class SemanticQueryHandler:
    """Handle semantic_query tool execution."""

    ops: ServingOperations
    limiter: QueryLimiter
    settings: ServingSettings
    feature_set: ServingFeatureSet
    export_workflow: ExportWorkflow | None = None

    async def handle(
        self,
        request: SemanticQueryToolRequest,
        *,
        ctx: Context,
    ) -> SemanticQueryToolResponse:
        start = time.perf_counter()
        await ctx.info(f"Querying view: {request.view_id}")
        await maybe_report_progress(ctx, settings=self.settings, progress=10, total=100)
        cancel_token = CancelToken.from_timeout(self.settings.query_timeout_s)
        semantic_request = request.to_semantic_request()
        result = await self.limiter.run_with_timeout(
            self.ops.query,
            self.settings.query_timeout_s,
            semantic_request,
            cancel_check=cancel_token.raise_if_cancelled,
        )
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

        export_handle = await self._maybe_export_handle(request, ctx=ctx)
        export_uri = export_handle.uri if export_handle is not None else None
        export_meta_uri = export_handle.meta_uri if export_handle is not None else None

        note = self._build_note(
            truncated=truncated,
            export_requested=request.export_format is not None,
            export_handle=export_handle,
        )

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
            export=export_handle,
            export_uri=export_uri,
            export_meta_uri=export_meta_uri,
            summary=summary,
            sql_fingerprint=sql_fingerprint,
            note=note,
        )

    async def _maybe_export_handle(
        self,
        request: SemanticQueryToolRequest,
        *,
        ctx: Context,
    ) -> ExportHandleResponse | None:
        export_format = request.export_format
        if export_format is None:
            return None
        if self.export_workflow is None:
            return None
        export_request = _export_request_from_query(
            request,
            export_format=export_format,
            settings=self.settings,
        )
        return await self.export_workflow.run(export_request, ctx=ctx)

    @staticmethod
    def _build_note(
        *,
        truncated: bool,
        export_requested: bool,
        export_handle: ExportHandleResponse | None,
    ) -> str | None:
        if export_requested and export_handle is None:
            return "Export requested but export support is unavailable."
        if truncated and export_handle is None:
            return "Result truncated; use semantic_export for full dataset."
        return None


def _export_request_from_query(
    request: SemanticQueryToolRequest,
    *,
    export_format: ExportFormat,
    settings: ServingSettings,
) -> SemanticExportRequest:
    pagination = request.pagination
    limit = pagination.limit if pagination is not None else settings.export_max_rows
    offset = pagination.offset if pagination is not None else 0
    return SemanticExportRequest(
        view_id=request.view_id,
        select=request.select,
        filters=request.filters,
        order_by=request.order_by,
        format=export_format,
        limit=limit,
        offset=offset,
    )


def register_query_tool(
    mcp: FastMCP,
    ops: ServingOperations,
    limiter: QueryLimiter,
    *,
    settings: ServingSettings,
    store: ResourceStore | None = None,
    export_limiter: QueryLimiter | None = None,
) -> None:
    """Register semantic_query tool."""
    feature_set = ServingFeatureSet.from_settings(settings)
    export_workflow = None
    if store is not None and export_limiter is not None and feature_set.enable_mcp_export:
        export_workflow = ExportWorkflow(
            ops=ops,
            limiter=export_limiter,
            store=store,
            settings=settings,
        )
    handler = SemanticQueryHandler(
        ops=ops,
        limiter=limiter,
        settings=settings,
        feature_set=feature_set,
        export_workflow=export_workflow,
    )

    async def semantic_query(
        request: SemanticQueryToolRequest,
        *,
        ctx: Context,
    ) -> SemanticQueryToolResponse:
        return await handler.handle(request, ctx=ctx)

    semantic_query.__annotations__ = get_type_hints(semantic_query, include_extras=True)
    mcp.tool(
        name="semantic_query",
        description="Query a semantic view with structured filters",
        annotations=READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_SEMANTIC, TAG_READ},
    )(semantic_query)


__all__ = ["register_query_tool"]
