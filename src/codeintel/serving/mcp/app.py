"""FastMCP application builder for semantic tools."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Protocol, cast

from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from codeintel.serving.http.metrics import QueryMetrics, log_query_metrics
from codeintel.serving.mcp._compat import Context, FastMCP, ToolError, create_bearer_auth
from codeintel.serving.mcp.resource_store import ResourceStore
from codeintel.serving.mcp.resources import register_resources
from codeintel.serving.mcp.response import build_envelope
from codeintel.serving.mcp.runtime import QueryLimiter
from codeintel.serving.search.models import SearchQueryRequest
from codeintel.serving.semantic.models import (
    FilterSpec,
    SemanticExportRequest,
    SemanticQueryRequest,
)
from codeintel.serving.settings import ServingSettings

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from contextlib import AbstractAsyncContextManager

    from codeintel.serving.db.manager import ServingDBManager
    from codeintel.serving.search.models import SearchQueryResponse
    from codeintel.serving.semantic.models import SemanticExplainResponse, SemanticQueryResponse

LOG = logging.getLogger(__name__)

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

# Error message constants for consistent, clear messages
_ERR_CATALOG_FAILED = "Failed to retrieve catalog. Check server logs."
_ERR_DESCRIBE_FAILED = "Failed to describe view. Check server logs."
_ERR_QUERY_FAILED = "Query execution failed. Check server logs."
_ERR_EXPLAIN_FAILED = "Explain execution failed. Check server logs."
_ERR_META_FAILED = "Failed to retrieve metadata. Check server logs."
_ERR_SEARCH_FAILED = "Search execution failed. Check server logs."
_ERR_EXPORT_FAILED = "Export execution failed. Check server logs."


class SemanticKernel(Protocol):
    """Protocol for the kernel interface used by MCP tools."""

    @property
    def db(self) -> ServingDBManager:
        """Return the serving database manager."""
        ...

    def catalog(self) -> dict[str, object]: ...

    def describe(self, view_id: str) -> dict[str, object]: ...

    def query(self, request: SemanticQueryRequest) -> SemanticQueryResponse: ...

    def explain(self, request: SemanticQueryRequest) -> SemanticExplainResponse: ...

    def search(self, request: SearchQueryRequest) -> SearchQueryResponse: ...

    def meta(self) -> dict[str, object]: ...

    def export_rows(self, request: SemanticExportRequest) -> Iterator[dict[str, object]]: ...


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


def _view_not_found_msg(view_id: str) -> str:
    """Create a consistent 'view not found' error message.

    Returns
    -------
    str
        Error message for user display.
    """
    return f"View '{view_id}' not found in semantic registry"


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
    kernel: SemanticKernel,
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
    mcp = FastMCP(
        "CodeIntel",
        json_response=True,
        mask_error_details=settings.mcp_mask_errors,
        lifespan=lifespan,
        auth=create_bearer_auth(settings.auth_token),  # Bearer token auth when set
    )

    # Initialize query limiter for concurrency control
    limiter = QueryLimiter(max_concurrent=settings.mcp_max_concurrent_queries)

    # Initialize resource store for exports
    store = ResourceStore(settings.serve_dir / "exports")

    # Register core tools (always enabled)
    _register_catalog_tool(mcp, kernel, limiter)
    _register_describe_tool(mcp, kernel, limiter)
    _register_query_tool(mcp, kernel, limiter)

    # Register optional tools (feature-flagged)
    if settings.mcp_enable_explain:
        _register_explain_tool(mcp, kernel, limiter)
    if settings.mcp_enable_meta:
        _register_meta_tool(mcp, kernel, limiter)
    if settings.mcp_enable_search:
        _register_search_tool(mcp, kernel, limiter)
    if settings.mcp_enable_export:
        _register_export_tool(mcp, kernel, limiter, store, settings)

    # Register MCP resources
    register_resources(mcp, kernel, store)

    # Register health check routes for load balancers
    _register_health_routes(mcp, kernel)

    return mcp


def _register_health_routes(mcp: FastMCP, kernel: SemanticKernel) -> None:
    """Register health check routes for load balancers and orchestrators.

    Parameters
    ----------
    mcp
        FastMCP server instance.
    kernel
        Semantic query kernel for health status.
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
            pointer = kernel.db.current_pointer()
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
            kernel.db.current_pointer()
            return PlainTextResponse("ready")
        except RuntimeError:
            return PlainTextResponse("not ready", status_code=503)


def _register_catalog_tool(mcp: FastMCP, kernel: SemanticKernel, limiter: QueryLimiter) -> None:
    """Register semantic_catalog tool."""

    @mcp.tool(
        name="semantic_catalog",
        description="List available semantic views in the CodeIntel database",
        annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_SEMANTIC, TAG_READ},
    )
    async def semantic_catalog(*, ctx: Context) -> dict[str, object]:
        """List available semantic views in the CodeIntel database.

        Parameters
        ----------
        ctx
            MCP execution context for progress and logging.

        Returns
        -------
        dict[str, object]
            Catalog response payload wrapped in envelope.

        Raises
        ------
        ToolError
            If catalog retrieval fails.
        """
        start = time.perf_counter()
        row_count = 0
        try:
            await ctx.info("Retrieving semantic catalog")
            result = await limiter.run(kernel.catalog)
            query_ms = int((time.perf_counter() - start) * 1000)
            data = result if isinstance(result, dict) else {}
            row_count = len(data.get("views", [])) if isinstance(data, dict) else 0
            return build_envelope(kernel, data, query_ms=query_ms).model_dump(mode="json")
        except Exception as e:
            LOG.exception("Catalog retrieval failed")
            raise ToolError(_ERR_CATALOG_FAILED) from e
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            log_query_metrics(
                QueryMetrics(
                    endpoint="mcp:semantic_catalog",
                    view_id=None,
                    query=None,
                    row_count=row_count,
                    truncated=False,
                    duration_ms=duration_ms,
                    correlation_id=getattr(ctx, "session_id", None) or "mcp-unknown",
                )
            )


def _register_describe_tool(mcp: FastMCP, kernel: SemanticKernel, limiter: QueryLimiter) -> None:
    """Register semantic_describe tool."""

    @mcp.tool(
        name="semantic_describe",
        description="Describe a semantic view's schema and metadata",
        annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_SEMANTIC, TAG_READ},
    )
    async def semantic_describe(view_id: str, *, ctx: Context) -> dict[str, object]:
        """Describe a semantic view's schema and metadata.

        Parameters
        ----------
        view_id
            Semantic view identifier.
        ctx
            MCP execution context for progress and logging.

        Returns
        -------
        dict[str, object]
            View description payload wrapped in envelope.

        Raises
        ------
        ToolError
            If view not found or description fails.
        """
        start = time.perf_counter()
        view_id_for_metrics = view_id
        try:
            await ctx.info(f"Describing view: {view_id}")
            result = await limiter.run(kernel.describe, view_id)
            query_ms = int((time.perf_counter() - start) * 1000)
            data = result if isinstance(result, dict) else {}
            return build_envelope(kernel, data, query_ms=query_ms).model_dump(mode="json")
        except KeyError as e:
            raise ToolError(_view_not_found_msg(view_id)) from e
        except Exception as e:
            LOG.exception("View description failed for %s", view_id)
            raise ToolError(_ERR_DESCRIBE_FAILED) from e
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            log_query_metrics(
                QueryMetrics(
                    endpoint="mcp:semantic_describe",
                    view_id=view_id_for_metrics,
                    query=None,
                    row_count=0,
                    truncated=False,
                    duration_ms=duration_ms,
                    correlation_id=getattr(ctx, "session_id", None) or "mcp-unknown",
                )
            )


def _register_query_tool(mcp: FastMCP, kernel: SemanticKernel, limiter: QueryLimiter) -> None:
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
    ) -> dict[str, object]:
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
        dict[str, object]
            Query response payload with rows wrapped in envelope.

        Raises
        ------
        ToolError
            If view not found or query parameters invalid.
        """
        start = time.perf_counter()
        view_id_for_metrics = view_id
        row_count = 0
        truncated = False
        try:
            await ctx.info(f"Querying view: {view_id}")
            await ctx.report_progress(10, 100)
            request = _build_semantic_request(view_id, filters, select, order_by, pagination)
            await ctx.report_progress(20, 100)
            result = cast("SemanticQueryResponse", await limiter.run(kernel.query, request))
            await ctx.report_progress(100, 100)
            query_ms = int((time.perf_counter() - start) * 1000)
            data = result.model_dump(mode="json")
            row_count = len(result.rows)
            truncated = result.truncated
            return build_envelope(
                kernel,
                data,
                truncated=truncated,
                query_ms=query_ms,
                row_count=row_count,
            ).model_dump(mode="json")
        except KeyError as e:
            raise ToolError(_view_not_found_msg(view_id)) from e
        except ValueError as e:
            raise ToolError(_invalid_params_msg(e)) from e
        except Exception as e:
            LOG.exception("Query failed for view %s", view_id)
            raise ToolError(_ERR_QUERY_FAILED) from e
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            log_query_metrics(
                QueryMetrics(
                    endpoint="mcp:semantic_query",
                    view_id=view_id_for_metrics,
                    query=None,
                    row_count=row_count,
                    truncated=truncated,
                    duration_ms=duration_ms,
                    correlation_id=getattr(ctx, "session_id", None) or "mcp-unknown",
                )
            )


def _register_explain_tool(mcp: FastMCP, kernel: SemanticKernel, limiter: QueryLimiter) -> None:
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
    ) -> dict[str, object]:
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
        dict[str, object]
            Explain response payload with SQL and plan wrapped in envelope.

        Raises
        ------
        ToolError
            If view not found or explain fails.
        """
        start = time.perf_counter()
        view_id_for_metrics = view_id
        try:
            await ctx.info(f"Explaining query for view: {view_id}")
            await ctx.report_progress(10, 100)
            request = _build_semantic_request(view_id, filters, select, order_by, pagination)
            await ctx.report_progress(20, 100)
            result = cast("SemanticExplainResponse", await limiter.run(kernel.explain, request))
            await ctx.report_progress(100, 100)
            query_ms = int((time.perf_counter() - start) * 1000)
            data = result.model_dump(mode="json")
            return build_envelope(kernel, data, query_ms=query_ms).model_dump(mode="json")
        except KeyError as e:
            raise ToolError(_view_not_found_msg(view_id)) from e
        except ValueError as e:
            raise ToolError(_invalid_params_msg(e)) from e
        except Exception as e:
            LOG.exception("Explain failed for view %s", view_id)
            raise ToolError(_ERR_EXPLAIN_FAILED) from e
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            log_query_metrics(
                QueryMetrics(
                    endpoint="mcp:semantic_explain",
                    view_id=view_id_for_metrics,
                    query=None,
                    row_count=0,
                    truncated=False,
                    duration_ms=duration_ms,
                    correlation_id=getattr(ctx, "session_id", None) or "mcp-unknown",
                )
            )


def _register_meta_tool(mcp: FastMCP, kernel: SemanticKernel, limiter: QueryLimiter) -> None:
    """Register serving_meta tool."""

    @mcp.tool(
        name="serving_meta",
        description="Get serving layer metadata including snapshot info",
        annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_META, TAG_READ},
    )
    async def serving_meta(*, ctx: Context) -> dict[str, object]:
        """Get serving layer metadata.

        Parameters
        ----------
        ctx
            MCP execution context for progress and logging.

        Returns
        -------
        dict[str, object]
            Serving metadata payload wrapped in envelope.

        Raises
        ------
        ToolError
            If metadata retrieval fails.
        """
        start = time.perf_counter()
        try:
            await ctx.info("Retrieving serving metadata")
            result = await limiter.run(kernel.meta)
            query_ms = int((time.perf_counter() - start) * 1000)
            data = result if isinstance(result, dict) else {}
            return build_envelope(kernel, data, query_ms=query_ms).model_dump(mode="json")
        except Exception as e:
            LOG.exception("Metadata retrieval failed")
            raise ToolError(_ERR_META_FAILED) from e
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            log_query_metrics(
                QueryMetrics(
                    endpoint="mcp:serving_meta",
                    view_id=None,
                    query=None,
                    row_count=0,
                    truncated=False,
                    duration_ms=duration_ms,
                    correlation_id=getattr(ctx, "session_id", None) or "mcp-unknown",
                )
            )


def _register_search_tool(mcp: FastMCP, kernel: SemanticKernel, limiter: QueryLimiter) -> None:
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
    ) -> dict[str, object]:
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
        dict[str, object]
            Search response payload wrapped in envelope.

        Raises
        ------
        ToolError
            If search fails.
        """
        start = time.perf_counter()
        query_for_metrics = query
        row_count = 0
        try:
            await ctx.info(f"Searching: {query}")
            await ctx.report_progress(10, 100)
            request = SearchQueryRequest(
                query=query,
                kinds=kinds,
                limit=limit,
                offset=offset,
            )
            await ctx.report_progress(20, 100)
            result = cast("SearchQueryResponse", await limiter.run(kernel.search, request))
            await ctx.report_progress(100, 100)
            query_ms = int((time.perf_counter() - start) * 1000)
            data = result.model_dump(mode="json")
            row_count = len(result.results)
            return build_envelope(
                kernel,
                data,
                query_ms=query_ms,
                row_count=row_count,
            ).model_dump(mode="json")
        except ValueError as e:
            raise ToolError(_invalid_params_msg(e)) from e
        except Exception as e:
            LOG.exception("Search failed for query: %s", query)
            raise ToolError(_ERR_SEARCH_FAILED) from e
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            log_query_metrics(
                QueryMetrics(
                    endpoint="mcp:code_search",
                    view_id=None,
                    query=query_for_metrics,
                    row_count=row_count,
                    truncated=False,
                    duration_ms=duration_ms,
                    correlation_id=getattr(ctx, "session_id", None) or "mcp-unknown",
                )
            )


def _register_export_tool(
    mcp: FastMCP,
    kernel: SemanticKernel,
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
    )
    async def semantic_export(
        view_id: str,
        filters: list[dict[str, object]] | None = None,
        export_format: str = "ndjson",
        limit: int = 100_000,
        *,
        ctx: Context,
    ) -> dict[str, object]:
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
            Export format: "json" or "ndjson".
        limit
            Maximum rows to export.
        ctx
            MCP execution context.

        Returns
        -------
        dict[str, object]
            Envelope with export_uri and metadata.

        Raises
        ------
        ToolError
            If view not found, parameters invalid, or export fails.
        """
        start = time.perf_counter()
        view_id_for_metrics = view_id
        row_count = 0
        try:
            await ctx.info(f"Exporting view: {view_id} (format={export_format})")
            await ctx.report_progress(10, 100)

            request = SemanticExportRequest(
                view_id=view_id,
                filters=[FilterSpec.model_validate(f) for f in (filters or [])],
                format=export_format,  # type: ignore[arg-type]
                limit=min(limit, settings.export_max_rows),
            )

            # Export rows via limiter
            await ctx.report_progress(20, 100)
            rows = cast(
                "list[dict[str, object]]",
                await limiter.run(lambda: list(kernel.export_rows(request))),
            )
            await ctx.report_progress(80, 100)

            # Store artifact
            if export_format == "ndjson":
                token, artifact = store.put_ndjson(rows)
            else:
                token, artifact = store.put_json({"rows": rows}, row_count=len(rows))

            await ctx.info(f"Export complete: {artifact.row_count} rows")
            await ctx.report_progress(100, 100)

            row_count = artifact.row_count
            query_ms = int((time.perf_counter() - start) * 1000)
            return build_envelope(
                kernel,
                {
                    "export_uri": f"codeintel://exports/{token}",
                    "format": export_format,
                    "row_count": row_count,
                    "size_bytes": artifact.size_bytes,
                },
                query_ms=query_ms,
                row_count=row_count,
            ).model_dump(mode="json")
        except KeyError as e:
            raise ToolError(_view_not_found_msg(view_id)) from e
        except ValueError as e:
            raise ToolError(_invalid_params_msg(e)) from e
        except Exception as e:
            LOG.exception("Export failed for view %s", view_id)
            raise ToolError(_ERR_EXPORT_FAILED) from e
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            log_query_metrics(
                QueryMetrics(
                    endpoint="mcp:semantic_export",
                    view_id=view_id_for_metrics,
                    query=None,
                    row_count=row_count,
                    truncated=False,
                    duration_ms=duration_ms,
                    correlation_id=getattr(ctx, "session_id", None) or "mcp-unknown",
                )
            )


__all__ = ["build_mcp_app"]
