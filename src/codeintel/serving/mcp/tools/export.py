"""FastMCP tool: semantic_export."""

from __future__ import annotations

import secrets
import time
from typing import TYPE_CHECKING

import anyio

from codeintel.serving.export.formats import suffix_for_export_format, supports_preview
from codeintel.serving.export.meta import (
    ExportArtifactInputs,
    build_export_artifact_spec,
    build_export_snapshot_dict,
)
from codeintel.serving.features import ServingFeatureSet
from codeintel.serving.mcp._compat import Context, FastMCP
from codeintel.serving.mcp.export_dispatch import write_export_to_store
from codeintel.serving.mcp.models import ExportHandleResponse, ExportSnapshot, SnapshotRef
from codeintel.serving.mcp.resource_store import ResourceStore
from codeintel.serving.mcp.runtime import QueryLimiter
from codeintel.serving.mcp.tools.shared import (
    READ_ONLY_LOCAL_ANNOTATIONS,
    TAG_EXPORT,
    TAG_SEMANTIC,
    InvalidExportFormatError,
    maybe_report_progress,
    mcp_correlation_id,
    normalize_export_format_for_tool,
)
from codeintel.serving.metrics import QueryMetrics, log_query_metrics
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.models import FilterSpec, SemanticExportRequest
from codeintel.serving.uris import (
    EXPORT_META_URI_TEMPLATE,
    export_meta_uri,
    export_preview_uri,
    export_sql_uri,
    export_uri,
)

if TYPE_CHECKING:
    from codeintel.serving.settings import ServingSettings


def _safe_column_types(ops: ServingOperations, view_id: str) -> dict[str, str]:
    try:
        describe = ops.describe(view_id)
    except (KeyError, TypeError, ValueError):
        return {}
    return {str(k): str(v) for k, v in describe.column_types.items()}


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

    @mcp.tool(
        name="semantic_export",
        description="Export semantic view data and return a resource URI",
        annotations=READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_SEMANTIC, TAG_EXPORT},
        task=feature_set.enable_mcp_export_tasks,
    )
    async def semantic_export(  # noqa: PLR0914
        view_id: str,
        filters: list[dict[str, object]] | None = None,
        export_format: str = "ndjson",
        limit: int = 100_000,
        ctx: Context | None = None,
    ) -> ExportHandleResponse:
        start = time.perf_counter()
        if ctx is not None:
            await ctx.info(f"Exporting view: {view_id} (format={export_format})")
        await maybe_report_progress(ctx, settings=settings, progress=10, total=100)

        format_type = normalize_export_format_for_tool(export_format)

        request = SemanticExportRequest(
            view_id=view_id,
            filters=[FilterSpec.model_validate(f) for f in (filters or [])],
            format=format_type,
            limit=limit,
        )

        ptr = ops.db.current_pointer()
        meta_result = await limiter.run(ops.meta)
        meta_payload = meta_result if isinstance(meta_result, dict) else {}
        snapshot_dict = build_export_snapshot_dict(
            ptr, buildspec_hash=str(meta_payload.get("buildspec_hash", "unknown"))
        )

        await maybe_report_progress(ctx, settings=settings, progress=20, total=100)
        column_types = _safe_column_types(ops, view_id)
        columns = tuple(column_types)
        compiled_sql = await limiter.run(ops.export_sql, request)
        query_hash, schema_hash = await limiter.run(ops.export_fingerprint, request)
        spec = build_export_artifact_spec(
            ExportArtifactInputs(
                view_id=view_id,
                columns=columns,
                column_types=column_types,
                compiled_sql=compiled_sql,
                snapshot=snapshot_dict,
                export_format=format_type,
                query_hash=query_hash,
                schema_hash=schema_hash,
            )
        )

        cancel_exc = anyio.get_cancelled_exc_class()
        export_id = secrets.token_urlsafe(16)
        try:
            token, artifact, stored_meta = await limiter.run(
                lambda: write_export_to_store(
                    ops=ops,
                    store=store,
                    request=request,
                    spec=spec,
                    export_id=export_id,
                )
            )
        except cancel_exc:
            if ctx is not None:
                await ctx.info("Export cancelled; cleaning up partial artifacts")
            store.mark_cancelled(export_id)
            store.delete(export_id, include_cancel_marker=False)
            raise
        except ValueError as exc:
            raise InvalidExportFormatError(export_format) from exc

        await maybe_report_progress(ctx, settings=settings, progress=100, total=100)
        if ctx is not None:
            await ctx.info(f"Export complete: {stored_meta.row_count} rows")

        snapshot_ref = SnapshotRef(
            repo=ptr.repo,
            commit=ptr.commit,
            run_id=ptr.run_id,
            published_at=ptr.published_at,
        )
        export_snapshot = ExportSnapshot(
            snapshot=snapshot_ref,
            semantic_layer_hash=ptr.semantic_layer_version,
            buildspec_hash=str(meta_payload.get("buildspec_hash", "unknown")),
        )

        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(
            QueryMetrics(
                endpoint="mcp:semantic_export",
                view_id=view_id,
                query=None,
                row_count=stored_meta.row_count,
                truncated=False,
                duration_ms=duration_ms,
                correlation_id=mcp_correlation_id(ctx),
                query_hash=query_hash,
                schema_hash=schema_hash,
            )
        )

        return ExportHandleResponse(
            export_id=token,
            format=format_type,
            mime_type=artifact.mime_type,
            filename=f"{view_id}{suffix_for_export_format(format_type)}",
            uri=export_uri(token),
            meta_uri=export_meta_uri(token),
            preview_uri=export_preview_uri(token) if supports_preview(format_type) else None,
            sql_uri=export_sql_uri(token),
            created_at=stored_meta.created_at,
            expires_at=stored_meta.expires_at,
            row_count=stored_meta.row_count,
            byte_size=artifact.size_bytes,
            snapshot=export_snapshot,
            note=f"Use {EXPORT_META_URI_TEMPLATE} to discover safe retrieval URIs.",
        )


__all__ = ["register_export_tool"]
