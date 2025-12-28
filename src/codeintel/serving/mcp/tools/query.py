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
from codeintel.serving.semantic.models import SemanticExportRequest, SemanticQueryResponse

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
        result = await self.limiter.run_with_timeout(
            self.ops.query,
            self.settings.query_timeout_s,
            request.to_semantic_request(),
            cancel_check=cancel_token.raise_if_cancelled,
        )
        await maybe_report_progress(ctx, settings=self.settings, progress=100, total=100)

        preview = self._preview_from_result(result)
        summary = await self._maybe_sample_summary(
            ctx=ctx,
            request=request,
            preview=preview,
            result=result,
        )
        export_handle = await self._maybe_export_handle(request, ctx=ctx)
        note = self._build_note(
            truncated=result.truncated,
            export_requested=request.export_format is not None,
            export_handle=export_handle,
        )

        duration_ms = (time.perf_counter() - start) * 1000
        row_count = len(result.rows)
        metrics = McpMetricsInput(
            endpoint="mcp:semantic_query",
            view_id=request.view_id,
            query=None,
            row_count=row_count,
            truncated=result.truncated,
            duration_ms=duration_ms,
            engine=result.engine,
            engine_preference=self.settings.query_engine,
            query_hash=result.query_hash,
            schema_hash=result.schema_hash,
            batch_size=result.batch_size,
            scan_rows=result.scan_metrics.row_count if result.scan_metrics else None,
            scan_files=result.scan_metrics.file_count if result.scan_metrics else None,
            scan_bytes=result.scan_metrics.total_bytes if result.scan_metrics else None,
        )
        log_mcp_query_metrics(metrics, ctx=ctx)

        return SemanticQueryToolResponse(
            result=result,
            preview=preview,
            export=export_handle,
            export_uri=export_handle.uri if export_handle is not None else None,
            export_meta_uri=export_handle.meta_uri if export_handle is not None else None,
            summary=summary,
            sql_fingerprint=result.sql_fingerprint,
            note=note,
        )

    @staticmethod
    def _preview_from_result(result: SemanticQueryResponse) -> QueryPreview | None:
        row_count = len(result.rows)
        if not result.truncated and row_count <= PREVIEW_ROW_COUNT:
            return None
        return QueryPreview(
            columns=tuple(result.columns),
            rows=tuple(result.rows[:PREVIEW_ROW_COUNT]),
            truncated=row_count > PREVIEW_ROW_COUNT or result.truncated,
        )

    async def _maybe_sample_summary(
        self,
        *,
        ctx: Context,
        request: SemanticQueryToolRequest,
        preview: QueryPreview | None,
        result: SemanticQueryResponse,
    ) -> str | None:
        if not self.feature_set.enable_mcp_sampling or preview is None:
            return None
        should_sample = result.truncated or len(result.rows) >= self.settings.mcp_sample_threshold
        if not should_sample:
            return None
        return await try_sample_summary(
            ctx,
            view_id=request.view_id,
            preview=preview,
            query_hash=result.query_hash,
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


@dataclass(frozen=True, slots=True)
class ExportWorkflowConfig:
    """Export workflow configuration for semantic_query."""

    store: ResourceStore
    limiter: QueryLimiter


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
    export_config: ExportWorkflowConfig | None = None,
) -> None:
    """Register semantic_query tool."""
    feature_set = ServingFeatureSet.from_settings(settings)
    export_workflow = None
    if export_config is not None and feature_set.enable_mcp_export:
        export_workflow = ExportWorkflow(
            ops=ops,
            limiter=export_config.limiter,
            store=export_config.store,
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


__all__ = ["ExportWorkflowConfig", "register_query_tool"]
