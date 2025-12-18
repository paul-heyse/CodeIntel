"""FastMCP application builder for semantic tools."""

from __future__ import annotations

import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import anyio
from mcp import McpError
from starlette.responses import JSONResponse, PlainTextResponse

from codeintel.serving.export.formats import (
    is_text_export_format,
    suffix_for_export_format,
    supports_preview,
)
from codeintel.serving.export.formats import (
    normalize_export_format as normalize_export_format_value,
)
from codeintel.serving.http.metrics import QueryMetrics, log_query_metrics
from codeintel.serving.mcp._compat import Context, FastMCP, create_bearer_auth
from codeintel.serving.mcp.export_dispatch import write_export_to_store
from codeintel.serving.mcp.middleware_stack import build_mcp_middleware
from codeintel.serving.mcp.prompts import register_prompts
from codeintel.serving.mcp.protocols import SemanticKernelProtocol, ServingSnapshotPointerProtocol
from codeintel.serving.mcp.resource_store import ExportArtifactSpec, ResourceStore
from codeintel.serving.mcp.resources import register_resources
from codeintel.serving.mcp.response_models import (
    DEFAULT_RESOURCE_TEMPLATES,
    BuildSpecInfo,
    ExportHandleResponse,
    ExportSnapshot,
    QueryPreview,
    SemanticLayerInfo,
    SemanticQueryToolResponse,
    ServingMetaResponse,
    SnapshotRef,
)
from codeintel.serving.mcp.runtime import QueryLimiter
from codeintel.serving.mcp.sql_fingerprint import sqlglot_canonical_sha256
from codeintel.serving.meta.service import build_serving_meta_payload
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.search.models import SearchQueryRequest, SearchQueryResponse
from codeintel.serving.semantic.models import (
    FilterSpec,
    SemanticCatalogResponse,
    SemanticExplainResponse,
    SemanticExportRequest,
    SemanticQueryRequest,
    SemanticViewDescriptionResponse,
)
from codeintel.serving.snapshot.identity import export_snapshot_dict, snapshot_ref_dict

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from contextlib import AbstractAsyncContextManager

    from starlette.requests import Request
    from starlette.responses import Response

    from codeintel.serving.export.formats import ExportFormat
    from codeintel.serving.mcp.resource_store import StoredArtifact, StoredMetadata
    from codeintel.serving.settings import ServingSettings

LOG = logging.getLogger(__name__)

# Server start time captured at module load (used in serving_meta)
_SERVER_STARTED_AT = datetime.now(UTC)


# Reusable annotation sets for tool categories
_READ_ONLY_LOCAL_ANNOTATIONS = {
    "readOnlyHint": True,  # No data modification
    "idempotentHint": True,  # Safe to retry
    "openWorldHint": False,  # Local database only
}

# Tag constants for tool organization
TAG_SEMANTIC = "semantic"
TAG_SEARCH = "search"
TAG_META = "meta"
TAG_READ = "read"
TAG_EXPORT = "export"

# Preview row count constant for QueryPreview
_PREVIEW_ROW_COUNT = 5


def _mcp_correlation_id(ctx: Context | None) -> str:
    if ctx is None:
        return "mcp-unknown"
    session_id_obj = getattr(ctx, "session_id", None)
    if isinstance(session_id_obj, str) and session_id_obj:
        return session_id_obj
    return "mcp-unknown"


async def _maybe_report_progress(
    ctx: Context | None,
    *,
    settings: ServingSettings,
    progress: float,
    total: float | None = None,
    message: str | None = None,
) -> None:
    if ctx is None:
        return
    if not settings.mcp_progress_reporting:
        return
    await ctx.report_progress(progress, total, message)


async def _try_sample_summary(
    ctx: Context,
    *,
    view_id: str,
    preview: QueryPreview,
    query_hash: str | None,
) -> str | None:
    payload = {
        "view_id": view_id,
        "query_hash": query_hash,
        "columns": list(preview.columns),
        "rows": list(preview.rows),
        "truncated": preview.truncated,
    }
    prompt = json.dumps(payload, indent=2, sort_keys=True, default=str)
    try:
        result = await ctx.sample(
            f"Summarize this query preview in 5 bullets (be precise, no speculation):\n\n{prompt}",
            system_prompt=(
                "You are summarizing a database query preview for an agent. "
                "Prefer actionable observations and call out truncation."
            ),
            max_tokens=250,
        )
    except (McpError, RuntimeError, ValueError):
        return None
    if isinstance(result.result, str):
        return result.result
    return result.text


class InvalidExportFormatError(ValueError):
    """Raised when an export_format value is unsupported."""

    def __init__(self, export_format: str) -> None:
        msg = f"Unsupported export_format: {export_format}"
        super().__init__(msg)
        self.export_format = export_format


def _normalize_export_format(export_format: str) -> ExportFormat:
    """Normalize and validate export_format values.

    Parameters
    ----------
    export_format
        Raw export format string from tool inputs.

    Returns
    -------
    str
        Normalized export format ("ndjson", "json", "parquet", or "arrow").

    Raises
    ------
    InvalidExportFormatError
        If export_format is not supported.
    """
    try:
        return normalize_export_format_value(export_format)
    except ValueError as exc:
        raise InvalidExportFormatError(export_format) from exc


def _export_snapshot_dict(ptr: ServingSnapshotPointerProtocol) -> dict[str, str]:
    return export_snapshot_dict(ptr) | {"buildspec_hash": "unknown"}


def _safe_column_types(ops: ServingOperations, view_id: str) -> dict[str, str]:
    try:
        describe = ops.describe(view_id)
    except (KeyError, TypeError, ValueError):
        return {}

    raw_types = describe.get("column_types")
    if not isinstance(raw_types, dict):
        return {}
    return {str(k): str(v) for k, v in raw_types.items()}


async def _catalog_view_count(ops: ServingOperations, limiter: QueryLimiter) -> int:
    catalog_data = await limiter.run(ops.catalog)
    catalog_dict = catalog_data if isinstance(catalog_data, dict) else {}
    views_obj = catalog_dict.get("views")
    views = views_obj if isinstance(views_obj, list) else []
    return len(views)


def _semantic_layer_info(meta: dict[str, object], *, view_count: int) -> SemanticLayerInfo:
    version_raw = meta.get("semantic_layer_version", "unknown")
    hash_raw = meta.get("semantic_layer_hash", "unknown")
    schema_manifest_hash_raw = meta.get("schema_manifest_hash")
    return SemanticLayerInfo(
        version=str(version_raw),
        hash=str(hash_raw),
        view_count=view_count,
        schema_manifest_hash=str(schema_manifest_hash_raw) if schema_manifest_hash_raw is not None else None,
    )


def _buildspec_info(meta: dict[str, object], *, compiled_at: datetime) -> BuildSpecInfo:
    raw_version = meta.get("buildspec_version", "unknown")
    buildspec_hash_raw = meta.get("buildspec_hash", "unknown")
    return BuildSpecInfo(
        version=str(raw_version) if raw_version is not None else "unknown",
        hash=str(buildspec_hash_raw),
        compiled_at=compiled_at,
    )


def _build_semantic_request(
    view_id: str,
    filters: list[dict[str, object]] | None,
    select: list[str] | None,
    order_by: list[str] | None,
    pagination: dict[str, int] | None,
) -> SemanticQueryRequest:
    """Build a SemanticQueryRequest from tool parameters.

    Returns
    -------
    SemanticQueryRequest
        Constructed request object.
    """
    page = pagination or {}
    return SemanticQueryRequest(
        view_id=view_id,
        select=select,
        filters=[FilterSpec.model_validate(f) for f in (filters or [])],
        order_by=order_by or [],
        limit=page.get("limit", 200),
        offset=page.get("offset", 0),
    )


def _invalid_params_msg(err: object) -> str:
    """Create a consistent 'invalid parameters' error message.

    Returns
    -------
    str
        Error message for user display.
    """
    return f"Invalid parameters: {err}"


def build_mcp_app(
    *,
    kernel: SemanticKernelProtocol,
    settings: ServingSettings,
    lifespan: Callable[[FastMCP], AbstractAsyncContextManager[object]] | None = None,
) -> FastMCP:
    """Build FastMCP application with semantic tools.

    Parameters
    ----------
    kernel
        Semantic query kernel.
    settings
        Serving settings for MCP configuration.
    lifespan
        Optional FastMCP lifespan factory.

    Returns
    -------
    FastMCP
        Configured MCP server.
    """
    ops = ServingOperations(kernel=kernel, settings=settings)
    store = ResourceStore(
        settings.serve_dir / "exports",
        ttl_seconds=settings.mcp_export_ttl_seconds,
    )

    @asynccontextmanager
    async def composed_lifespan(server: FastMCP) -> AsyncIterator[object]:
        async with _optional_lifespan(lifespan, server):
            store.cleanup_expired()
            async with anyio.create_task_group() as tg:
                if settings.mcp_export_ttl_seconds is not None:
                    interval = max(settings.mcp_export_cleanup_interval_seconds, 1)
                    tg.start_soon(_periodic_store_cleanup, store, interval)
                yield object()

    mcp = FastMCP(
        "CodeIntel",
        mask_error_details=settings.mcp_mask_errors,
        lifespan=composed_lifespan,
        auth=create_bearer_auth(settings.auth_token),
        middleware=build_mcp_middleware(settings),
        on_duplicate_tools="error",
        on_duplicate_resources="error",
        on_duplicate_prompts="error",
        strict_input_validation=True,
    )

    query_limiter = QueryLimiter(max_concurrent=settings.mcp_max_concurrent_queries)
    export_limiter = QueryLimiter(max_concurrent=settings.mcp_max_concurrent_exports)

    # Register core tools (always enabled)
    _register_catalog_tool(mcp, ops, query_limiter, settings)
    _register_describe_tool(mcp, ops, query_limiter, settings)
    _register_query_tool(mcp, ops, query_limiter, settings)

    # Register optional tools (feature-flagged)
    if settings.mcp_enable_explain:
        _register_explain_tool(mcp, ops, query_limiter, settings)
    if settings.mcp_enable_meta:
        _register_meta_tool(mcp, ops, query_limiter, settings)
    if settings.mcp_enable_search:
        _register_search_tool(mcp, ops, query_limiter, settings)
    if settings.mcp_enable_export:
        _register_export_tool(mcp, ops, export_limiter, store, settings)

    # Register MCP resources
    register_resources(mcp, ops, store, settings=settings)

    # Register health check routes for load balancers
    _register_health_routes(mcp, ops)

    # Register guided prompts for LLM workflows
    register_prompts(mcp, settings=settings, kernel=ops)

    return mcp


@asynccontextmanager
async def _optional_lifespan(
    lifespan: Callable[[FastMCP], AbstractAsyncContextManager[object]] | None,
    server: FastMCP,
) -> AsyncIterator[object]:
    if lifespan is None:
        yield object()
        return
    async with lifespan(server):
        yield object()


async def _periodic_store_cleanup(store: ResourceStore, interval_seconds: int) -> None:
    while True:
        store.cleanup_expired()
        await anyio.sleep(interval_seconds)


def _register_health_routes(mcp: FastMCP, ops: ServingOperations) -> None:
    """Register health check routes for load balancers and orchestrators.

    Parameters
    ----------
    mcp
        FastMCP server instance.
    ops
        Serving operations façade providing snapshot status.
    """

    @mcp.custom_route("/health", methods=["GET"])
    async def mcp_health(_request: Request) -> Response:  # noqa: RUF029
        """Health check for load balancers targeting MCP endpoint.

        Parameters
        ----------
        _request
            Starlette request (unused).

        Returns
        -------
        Response
            JSON response with status and snapshot info.
        """
        # async required by FastMCP custom_route decorator
        try:
            pointer = ops.db.current_pointer()
            return JSONResponse(
                {
                    "status": "ok",
                    "repo": pointer.repo,
                    "commit": pointer.commit[:12],
                    "run_id": pointer.run_id,
                }
            )
        except RuntimeError:
            return JSONResponse(
                {"status": "error", "detail": "No active snapshot"},
                status_code=503,
            )

    @mcp.custom_route("/ready", methods=["GET"])
    async def mcp_ready(_request: Request) -> Response:  # noqa: RUF029
        """Readiness probe for Kubernetes/orchestrators.

        Parameters
        ----------
        _request
            Starlette request (unused).

        Returns
        -------
        Response
            Plain text response indicating readiness.
        """
        # async required by FastMCP custom_route decorator
        try:
            ops.db.current_pointer()
            return PlainTextResponse("ready")
        except RuntimeError:
            return PlainTextResponse("not ready", status_code=503)


def _register_catalog_tool(
    mcp: FastMCP, ops: ServingOperations, limiter: QueryLimiter, settings: ServingSettings
) -> None:
    """Register semantic_catalog tool."""

    @mcp.tool(
        name="semantic_catalog",
        description="List available semantic views in the CodeIntel database",
        annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_SEMANTIC, TAG_READ},
    )
    async def semantic_catalog(*, ctx: Context) -> SemanticCatalogResponse:
        """List available semantic views in the CodeIntel database.

        Parameters
        ----------
        ctx
            MCP execution context for progress and logging.

        Returns
        -------
        SemanticCatalogResponse
            Typed catalog response with views and snapshot metadata.
        """
        start = time.perf_counter()
        result = await limiter.run(ops.catalog)
        data = result if isinstance(result, dict) else {}
        views_obj = data.get("views")
        views = views_obj if isinstance(views_obj, list) else []
        row_count = len(views)
        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(
            QueryMetrics(
                endpoint="mcp:semantic_catalog",
                view_id=None,
                query=None,
                row_count=row_count,
                truncated=False,
                duration_ms=duration_ms,
                correlation_id=_mcp_correlation_id(ctx),
            )
        )
        await ctx.info("Retrieved semantic catalog")
        await _maybe_report_progress(ctx, settings=settings, progress=100, total=100)
        return SemanticCatalogResponse.model_validate(data)


def _register_describe_tool(
    mcp: FastMCP, ops: ServingOperations, limiter: QueryLimiter, settings: ServingSettings
) -> None:
    """Register semantic_describe tool."""

    @mcp.tool(
        name="semantic_describe",
        description="Describe a semantic view's schema and metadata",
        annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_SEMANTIC, TAG_READ},
    )
    async def semantic_describe(view_id: str, *, ctx: Context) -> SemanticViewDescriptionResponse:
        """Describe a semantic view's schema and metadata.

        Parameters
        ----------
        view_id
            Semantic view identifier.
        ctx
            MCP execution context for progress and logging.

        Returns
        -------
        SemanticViewDescriptionResponse
            Typed view description with schema and metadata.
        """
        start = time.perf_counter()
        await ctx.info(f"Describing view: {view_id}")
        await _maybe_report_progress(ctx, settings=settings, progress=10, total=100)
        result = await limiter.run(ops.describe, view_id)
        await _maybe_report_progress(ctx, settings=settings, progress=100, total=100)
        data = result if isinstance(result, dict) else {}
        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(
            QueryMetrics(
                endpoint="mcp:semantic_describe",
                view_id=view_id,
                query=None,
                row_count=0,
                truncated=False,
                duration_ms=duration_ms,
                correlation_id=_mcp_correlation_id(ctx),
            )
        )
        return SemanticViewDescriptionResponse.model_validate(data)


def _register_query_tool(
    mcp: FastMCP, ops: ServingOperations, limiter: QueryLimiter, settings: ServingSettings
) -> None:
    """Register semantic_query tool."""

    @mcp.tool(
        name="semantic_query",
        description="Query a semantic view with structured filters",
        annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
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
        """Query a semantic view with structured filters.

        Parameters
        ----------
        view_id
            Semantic view identifier.
        filters
            Optional list of filter specifications.
        select
            Optional column selection.
        order_by
            Optional ordering specification.
        pagination
            Optional pagination with limit/offset.
        ctx
            MCP execution context for progress and logging.

        Returns
        -------
        SemanticQueryToolResponse
            Typed query response with result, optional preview, and notes.
        """
        start = time.perf_counter()
        await ctx.info(f"Querying view: {view_id}")
        await _maybe_report_progress(ctx, settings=settings, progress=10, total=100)
        request = _build_semantic_request(view_id, filters, select, order_by, pagination)
        await _maybe_report_progress(ctx, settings=settings, progress=20, total=100)
        result = await limiter.run(ops.query, request)
        await _maybe_report_progress(ctx, settings=settings, progress=100, total=100)

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
        if truncated or row_count > _PREVIEW_ROW_COUNT:
            preview = QueryPreview(
                columns=tuple(result.columns),
                rows=tuple(result.rows[:_PREVIEW_ROW_COUNT]),
                truncated=row_count > _PREVIEW_ROW_COUNT or truncated,
            )

        summary: str | None = None
        if settings.mcp_enable_sampling and preview is not None:
            should_sample = truncated or row_count >= settings.mcp_sample_threshold
            if should_sample:
                summary = await _try_sample_summary(
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
                correlation_id=_mcp_correlation_id(ctx),
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


def _register_explain_tool(
    mcp: FastMCP, ops: ServingOperations, limiter: QueryLimiter, settings: ServingSettings
) -> None:
    """Register semantic_explain tool."""

    @mcp.tool(
        name="semantic_explain",
        description="Return compiled SQL and DuckDB plan for a semantic query",
        annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
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
        """Return compiled SQL and DuckDB plan for a semantic query.

        Parameters
        ----------
        view_id
            Semantic view identifier.
        filters
            Optional list of filter specifications.
        select
            Optional column selection.
        order_by
            Optional ordering specification.
        pagination
            Optional pagination with limit/offset.
        ctx
            MCP execution context for progress and logging.

        Returns
        -------
        SemanticExplainResponse
            Typed explain response with SQL and query plan.
        """
        start = time.perf_counter()
        await ctx.info(f"Explaining query for view: {view_id}")
        await _maybe_report_progress(ctx, settings=settings, progress=10, total=100)
        request = _build_semantic_request(view_id, filters, select, order_by, pagination)
        await _maybe_report_progress(ctx, settings=settings, progress=20, total=100)
        result = await limiter.run(ops.explain, request)
        await _maybe_report_progress(ctx, settings=settings, progress=100, total=100)

        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(
            QueryMetrics(
                endpoint="mcp:semantic_explain",
                view_id=view_id,
                query=None,
                row_count=0,
                truncated=False,
                duration_ms=duration_ms,
                correlation_id=_mcp_correlation_id(ctx),
            )
        )
        return result


def _register_meta_tool(
    mcp: FastMCP, ops: ServingOperations, limiter: QueryLimiter, settings: ServingSettings
) -> None:
    """Register serving_meta tool."""

    @mcp.tool(
        name="serving_meta",
        description="Get serving layer metadata including snapshot info",
        annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_META, TAG_READ},
    )
    async def serving_meta(*, ctx: Context) -> ServingMetaResponse:
        """Get serving layer metadata.

        Parameters
        ----------
        ctx
            MCP execution context for progress and logging.

        Returns
        -------
        ServingMetaResponse
            Typed serving metadata with snapshot, limits, and features.
        """
        start = time.perf_counter()
        await ctx.info("Retrieving serving metadata")

        view_count = await _catalog_view_count(ops, limiter)
        features = {
            "supports_explain": settings.mcp_enable_explain,
            "supports_export": settings.mcp_enable_export,
            "supports_export_tasks": settings.mcp_export_enable_tasks,
            "supports_search": settings.mcp_enable_search,
            "supports_resources": True,
            "supports_sampling": settings.mcp_enable_sampling,
        }
        payload = build_serving_meta_payload(
            ops,
            settings=settings,
            started_at=_SERVER_STARTED_AT,
            features=features,
            inventories={"views": view_count},
            resource_templates=DEFAULT_RESOURCE_TEMPLATES,
        )

        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(
            QueryMetrics(
                endpoint="mcp:serving_meta",
                view_id=None,
                query=None,
                row_count=0,
                truncated=False,
                duration_ms=duration_ms,
                correlation_id=_mcp_correlation_id(ctx),
            )
        )

        return ServingMetaResponse.model_validate(payload)


def _register_search_tool(
    mcp: FastMCP, ops: ServingOperations, limiter: QueryLimiter, settings: ServingSettings
) -> None:
    """Register code_search tool."""

    @mcp.tool(
        name="code_search",
        description="Search code metadata using BM25 full-text search",
        annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_SEARCH, TAG_READ},
    )
    async def code_search(
        query: str,
        kinds: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
        *,
        ctx: Context,
    ) -> SearchQueryResponse:
        """Search code metadata using the serving snapshot search index.

        Parameters
        ----------
        query
            Search query string.
        kinds
            Optional filter by symbol kinds.
        limit
            Maximum results to return.
        offset
            Result offset for pagination.
        ctx
            MCP execution context for progress and logging.

        Returns
        -------
        SearchQueryResponse
            Typed search response with matching results.
        """
        start = time.perf_counter()
        await ctx.info(f"Searching: {query}")
        await _maybe_report_progress(ctx, settings=settings, progress=10, total=100)
        request = SearchQueryRequest(
            query=query,
            kinds=kinds,
            limit=limit,
            offset=offset,
        )
        await _maybe_report_progress(ctx, settings=settings, progress=20, total=100)
        result = await limiter.run(ops.search, request)
        await _maybe_report_progress(ctx, settings=settings, progress=100, total=100)

        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(
            QueryMetrics(
                endpoint="mcp:code_search",
                view_id=None,
                query=query,
                row_count=len(result.results),
                truncated=False,
                duration_ms=duration_ms,
                correlation_id=_mcp_correlation_id(ctx),
                query_hash=result.query_hash,
            )
        )
        return result


def _register_export_tool(
    mcp: FastMCP,
    ops: ServingOperations,
    limiter: QueryLimiter,
    store: ResourceStore,
    settings: ServingSettings,
) -> None:
    """Register semantic_export tool."""

    @mcp.tool(
        name="semantic_export",
        description="Export semantic view data and return a resource URI",
        annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_SEMANTIC, TAG_EXPORT},
        task=settings.mcp_export_enable_tasks,
    )
    async def semantic_export(  # noqa: PLR0914
        view_id: str,
        filters: list[dict[str, object]] | None = None,
        export_format: str = "ndjson",
        limit: int = 100_000,
        ctx: Context | None = None,
    ) -> ExportHandleResponse:
        """Export semantic view data and return a resource URI.

        For large datasets, this tool returns a resource URI that can be fetched
        separately, avoiding OOM from materializing large result sets in JSON.

        Parameters
        ----------
        view_id
            Semantic view identifier.
        filters
            Optional filter specifications.
        export_format
            Export format: "ndjson", "json", "parquet", or "arrow".
        limit
            Maximum rows to export.
        ctx
            MCP execution context.

        Returns
        -------
        ExportHandleResponse
            Typed export handle with URIs, metadata, and snapshot info.

        Raises
        ------
        InvalidExportFormatError
            If ``export_format`` is not supported.
        """
        start = time.perf_counter()
        if ctx is not None:
            await ctx.info(f"Exporting view: {view_id} (format={export_format})")
        await _maybe_report_progress(ctx, settings=settings, progress=10, total=100)

        format_type = _normalize_export_format(export_format)

        request = SemanticExportRequest(
            view_id=view_id,
            filters=[FilterSpec.model_validate(f) for f in (filters or [])],
            format=format_type,
            limit=limit,
        )

        ptr = ops.db.current_pointer()
        meta_result = await limiter.run(ops.meta)
        meta_payload = meta_result if isinstance(meta_result, dict) else {}
        snapshot_dict = _export_snapshot_dict(ptr) | {
            "buildspec_hash": str(meta_payload.get("buildspec_hash", "unknown")),
        }

        await _maybe_report_progress(ctx, settings=settings, progress=20, total=100)
        column_types = _safe_column_types(ops, view_id)
        columns = tuple(column_types)
        compiled_sql = ops.export_sql(request)
        query_hash, schema_hash = ops.export_fingerprint(request)
        spec = ExportArtifactSpec(
            view_id=view_id,
            columns=columns,
            column_types=column_types,
            compiled_sql=compiled_sql,
            snapshot=snapshot_dict,
            format=format_type,
            query_hash=query_hash,
            schema_hash=schema_hash,
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

        await _maybe_report_progress(ctx, settings=settings, progress=100, total=100)
        if ctx is not None:
            await ctx.info(f"Export complete: {stored_meta.row_count} rows")

        snapshot_ref = SnapshotRef(
            **snapshot_ref_dict(ptr),
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
                correlation_id=_mcp_correlation_id(ctx),
                query_hash=query_hash,
                schema_hash=schema_hash,
            )
        )

        return ExportHandleResponse(
            export_id=token,
            format=format_type,
            mime_type=artifact.mime_type,
            filename=f"{view_id}{suffix_for_export_format(format_type)}",
            uri=f"codeintel://exports/{token}",
            meta_uri=f"codeintel://exports/{token}/meta",
            preview_uri=f"codeintel://exports/{token}/preview" if supports_preview(format_type) else None,
            sql_uri=f"codeintel://exports/{token}/sql",
            created_at=stored_meta.created_at,
            expires_at=stored_meta.expires_at,
            row_count=stored_meta.row_count,
            byte_size=artifact.size_bytes,
            snapshot=export_snapshot,
            note="Use codeintel://exports/{export_id}/meta to discover safe retrieval URIs.",
        )
__all__ = ["build_mcp_app"]
