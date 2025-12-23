"""FastMCP tool: semantic_export."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, get_type_hints

import anyio
from fastmcp import Context, FastMCP

from codeintel.serving.export.formats import suffix_for_export_format, supports_preview
from codeintel.serving.export.meta import (
    ExportArtifactInputs,
    build_export_artifact_spec,
    build_export_snapshot_dict,
)
from codeintel.serving.features import ServingFeatureSet
from codeintel.serving.mcp.export_dispatch import write_export_to_store
from codeintel.serving.mcp.models import (
    ExportHandleResponse,
    ExportSnapshot,
    SemanticExportToolRequest,
    SnapshotRef,
)
from codeintel.serving.mcp.resource_store import ResourceStore
from codeintel.serving.mcp.runtime import QueryLimiter
from codeintel.serving.mcp.tools.shared import (
    READ_ONLY_LOCAL_ANNOTATIONS,
    TAG_EXPORT,
    TAG_SEMANTIC,
    InvalidExportFormatError,
    McpMetricsInput,
    log_mcp_query_metrics,
    maybe_report_progress,
    normalize_export_format_for_tool,
    validate_semantic_export_request,
)
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.models import SemanticExportRequest, SemanticViewDescriptionResponse
from codeintel.serving.uris import (
    EXPORT_META_URI_TEMPLATE,
    export_meta_uri,
    export_preview_uri,
    export_sql_uri,
    export_uri,
)

if TYPE_CHECKING:
    from codeintel.serving.export.formats import ExportFormat
    from codeintel.serving.export.models import ExportArtifactSpec
    from codeintel.serving.mcp.resource_store import StoredArtifact, StoredMetadata
    from codeintel.serving.settings import ServingSettings


@dataclass(frozen=True, slots=True)
class ExportPreparation:
    """Intermediate export context for response and metrics."""

    format_type: ExportFormat
    export_snapshot: ExportSnapshot
    spec: ExportArtifactSpec
    query_hash: str
    schema_hash: str | None


@dataclass(frozen=True, slots=True)
class ExportOutcome:
    """Materialized export output from ResourceStore."""

    token: str
    artifact: StoredArtifact
    stored_meta: StoredMetadata


def _safe_view_description(
    ops: ServingOperations, view_id: str
) -> SemanticViewDescriptionResponse | None:
    try:
        return ops.describe(view_id)
    except (KeyError, TypeError, ValueError):
        return None


def _column_types_from_description(
    description: SemanticViewDescriptionResponse | None,
) -> dict[str, str]:
    if description is None:
        return {}
    return {str(k): str(v) for k, v in description.column_types.items()}


@dataclass(frozen=True, slots=True)
class ExportWorkflow:
    """Coordinate export request execution and response assembly."""

    ops: ServingOperations
    limiter: QueryLimiter
    store: ResourceStore
    settings: ServingSettings

    @staticmethod
    def _normalize_request(
        request: SemanticExportRequest,
    ) -> tuple[SemanticExportRequest, ExportFormat]:
        format_type = normalize_export_format_for_tool(str(request.format))
        if request.format == format_type:
            return request, format_type
        return request.model_copy(update={"format": format_type}), format_type

    @staticmethod
    def _build_export_snapshot(
        snapshot: SnapshotRef,
        *,
        semantic_layer_hash: str,
        buildspec_hash: str,
    ) -> ExportSnapshot:
        return ExportSnapshot(
            snapshot=snapshot,
            semantic_layer_hash=semantic_layer_hash,
            buildspec_hash=buildspec_hash,
        )

    async def _build_export_spec(
        self,
        request: SemanticExportRequest,
        *,
        format_type: ExportFormat,
        snapshot_dict: dict[str, str],
    ) -> tuple[ExportArtifactSpec, str, str | None]:
        description = _safe_view_description(self.ops, request.view_id)
        column_types = _column_types_from_description(description)
        columns = tuple(column_types)
        compiled_sql = await self.limiter.run(self.ops.export_sql, request)
        query_hash, schema_hash = await self.limiter.run(self.ops.export_fingerprint, request)
        spec = build_export_artifact_spec(
            ExportArtifactInputs(
                view_id=request.view_id,
                columns=columns,
                column_types=column_types,
                compiled_sql=compiled_sql,
                snapshot=snapshot_dict,
                export_format=format_type,
                query_hash=query_hash,
                schema_hash=schema_hash,
            )
        )
        return spec, query_hash, schema_hash

    async def _prepare_export(
        self, request: SemanticExportRequest, *, format_type: ExportFormat
    ) -> ExportPreparation:
        ptr = self.ops.db.current_pointer()
        meta_result = await self.limiter.run(self.ops.meta)
        meta_payload = meta_result.model_dump(mode="json")
        buildspec_value = meta_payload.get("buildspec_hash")
        buildspec_hash = str(buildspec_value) if buildspec_value is not None else "unknown"
        snapshot_dict = build_export_snapshot_dict(ptr, buildspec_hash=buildspec_hash)
        snapshot_ref = SnapshotRef.from_pointer(ptr)
        export_snapshot = self._build_export_snapshot(
            snapshot_ref,
            semantic_layer_hash=ptr.semantic_layer_version,
            buildspec_hash=buildspec_hash,
        )
        spec, query_hash, schema_hash = await self._build_export_spec(
            request,
            format_type=format_type,
            snapshot_dict=snapshot_dict,
        )
        return ExportPreparation(
            format_type=format_type,
            export_snapshot=export_snapshot,
            spec=spec,
            query_hash=query_hash,
            schema_hash=schema_hash,
        )

    async def _store_export(
        self,
        request: SemanticExportRequest,
        *,
        spec: ExportArtifactSpec,
        ctx: Context | None,
    ) -> ExportOutcome:
        export_id = secrets.token_urlsafe(16)
        cancel_exc = anyio.get_cancelled_exc_class()
        try:
            token, artifact, stored_meta = await self.limiter.run(
                lambda: write_export_to_store(
                    ops=self.ops,
                    store=self.store,
                    request=request,
                    spec=spec,
                    export_id=export_id,
                )
            )
        except cancel_exc:
            if ctx is not None:
                await ctx.info("Export cancelled; cleaning up partial artifacts")
            self.store.mark_cancelled(export_id)
            self.store.delete(export_id, include_cancel_marker=False)
            raise
        except ValueError as exc:
            raise InvalidExportFormatError(str(request.format)) from exc
        return ExportOutcome(token=token, artifact=artifact, stored_meta=stored_meta)

    async def run(
        self,
        request: SemanticExportRequest,
        *,
        ctx: Context | None,
    ) -> ExportHandleResponse:
        start = time.perf_counter()
        if ctx is not None:
            await ctx.info(f"Exporting view: {request.view_id} (format={request.format})")
        await maybe_report_progress(ctx, settings=self.settings, progress=10, total=100)

        normalized_request, format_type = self._normalize_request(request)
        preparation = await self._prepare_export(normalized_request, format_type=format_type)

        await maybe_report_progress(ctx, settings=self.settings, progress=20, total=100)
        outcome = await self._store_export(normalized_request, spec=preparation.spec, ctx=ctx)

        await maybe_report_progress(ctx, settings=self.settings, progress=100, total=100)
        if ctx is not None:
            await ctx.info(f"Export complete: {outcome.stored_meta.row_count} rows")

        duration_ms = (time.perf_counter() - start) * 1000
        metrics = McpMetricsInput(
            endpoint="mcp:semantic_export",
            view_id=normalized_request.view_id,
            query=None,
            row_count=outcome.stored_meta.row_count,
            truncated=False,
            duration_ms=duration_ms,
            query_hash=preparation.query_hash,
            schema_hash=preparation.schema_hash,
        )
        log_mcp_query_metrics(metrics, ctx=ctx)

        return ExportHandleResponse(
            export_id=outcome.token,
            format=preparation.format_type,
            mime_type=outcome.artifact.mime_type,
            filename=f"{normalized_request.view_id}{suffix_for_export_format(format_type)}",
            uri=export_uri(outcome.token),
            meta_uri=export_meta_uri(outcome.token),
            preview_uri=export_preview_uri(outcome.token)
            if supports_preview(preparation.format_type)
            else None,
            sql_uri=export_sql_uri(outcome.token),
            created_at=outcome.stored_meta.created_at,
            expires_at=outcome.stored_meta.expires_at,
            row_count=outcome.stored_meta.row_count,
            byte_size=outcome.artifact.size_bytes,
            snapshot=preparation.export_snapshot,
            note=f"Use {EXPORT_META_URI_TEMPLATE} to discover safe retrieval URIs.",
        )


def register_export_tool(
    mcp: FastMCP,
    ops: ServingOperations,
    limiter: QueryLimiter,
    store: ResourceStore,
    *,
    settings: ServingSettings,
) -> None:
    """Register semantic_export tool."""
    feature_set = ServingFeatureSet.from_settings(settings)
    workflow = ExportWorkflow(
        ops=ops,
        limiter=limiter,
        store=store,
        settings=settings,
    )

    async def semantic_export(
        request: SemanticExportToolRequest,
        ctx: Context | None = None,
    ) -> ExportHandleResponse:
        validated = validate_semantic_export_request(request)
        return await workflow.run(validated, ctx=ctx)

    semantic_export.__annotations__ = get_type_hints(semantic_export, include_extras=True)
    mcp.tool(
        name="semantic_export",
        description="Export semantic view data and return a resource URI",
        annotations=READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_SEMANTIC, TAG_EXPORT},
        task=feature_set.enable_mcp_export_tasks,
    )(semantic_export)


__all__ = ["register_export_tool"]
