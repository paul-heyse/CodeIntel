"""FastMCP application builder for semantic tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from mcp.server.fastmcp import FastMCP

from codeintel.serving.search.models import SearchQueryRequest
from codeintel.serving.semantic.models import FilterSpec, SemanticQueryRequest

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from codeintel.serving.search.models import SearchQueryResponse
    from codeintel.serving.semantic.models import SemanticExplainResponse, SemanticQueryResponse


class SemanticKernel(Protocol):
    """Protocol for the kernel interface used by MCP tools."""

    def catalog(self) -> dict[str, object]: ...

    def describe(self, view_id: str) -> dict[str, object]: ...

    def query(self, request: SemanticQueryRequest) -> SemanticQueryResponse: ...

    def explain(self, request: SemanticQueryRequest) -> SemanticExplainResponse: ...

    def search(self, request: SearchQueryRequest) -> SearchQueryResponse: ...

    def meta(self) -> dict[str, object]: ...


def build_mcp_app(
    *,
    kernel: SemanticKernel,
    host: str = "127.0.0.1",
    port: int = 8000,
    streamable_http_path: str = "/mcp",
    lifespan: Callable[[FastMCP], AbstractAsyncContextManager[object]] | None = None,
) -> FastMCP:
    """Build FastMCP application with semantic tools.

    Parameters
    ----------
    kernel
        Semantic query kernel.
    host
        Host for HTTP transports.
    port
        Port for HTTP transports.
    streamable_http_path
        Route path for the streamable HTTP transport.
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
        host=host,
        port=port,
        streamable_http_path=streamable_http_path,
        lifespan=lifespan,
    )

    @mcp.tool()
    def semantic_catalog() -> dict[str, object]:
        """List available semantic views in the CodeIntel database.

        Returns
        -------
        dict[str, object]
            Catalog response payload.
        """
        return kernel.catalog()

    @mcp.tool()
    def semantic_describe(view_id: str) -> dict[str, object]:
        """Describe a semantic view's schema and metadata.

        Returns
        -------
        dict[str, object]
            View description payload.
        """
        return kernel.describe(view_id)

    @mcp.tool()
    def semantic_query(
        view_id: str,
        filters: list[dict[str, object]] | None = None,
        select: list[str] | None = None,
        order_by: list[str] | None = None,
        pagination: dict[str, int] | None = None,
    ) -> dict[str, object]:
        """Query a semantic view with structured filters.

        Returns
        -------
        dict[str, object]
            Query response payload.
        """
        page = pagination or {}
        request = SemanticQueryRequest(
            view_id=view_id,
            select=select,
            filters=[FilterSpec.model_validate(f) for f in (filters or [])],
            order_by=order_by or [],
            limit=page.get("limit", 200),
            offset=page.get("offset", 0),
        )
        result = kernel.query(request)
        return result.model_dump(mode="json")

    @mcp.tool()
    def semantic_explain(
        view_id: str,
        filters: list[dict[str, object]] | None = None,
        select: list[str] | None = None,
        order_by: list[str] | None = None,
        pagination: dict[str, int] | None = None,
    ) -> dict[str, object]:
        """Return compiled SQL and DuckDB plan for a semantic query.

        Returns
        -------
        dict[str, object]
            Explain response payload.
        """
        page = pagination or {}
        request = SemanticQueryRequest(
            view_id=view_id,
            select=select,
            filters=[FilterSpec.model_validate(f) for f in (filters or [])],
            order_by=order_by or [],
            limit=page.get("limit", 200),
            offset=page.get("offset", 0),
        )
        result = kernel.explain(request)
        return result.model_dump(mode="json")

    @mcp.tool()
    def serving_meta() -> dict[str, object]:
        """Get serving layer metadata.

        Returns
        -------
        dict[str, object]
            Serving metadata payload.
        """
        return kernel.meta()

    @mcp.tool()
    def code_search(
        query: str,
        kinds: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, object]:
        """Search code metadata using the serving snapshot search index.

        Returns
        -------
        dict[str, object]
            Search response payload.
        """
        request = SearchQueryRequest(
            query=query,
            kinds=kinds,
            limit=limit,
            offset=offset,
        )
        result = kernel.search(request)
        return result.model_dump(mode="json")

    return mcp


__all__ = ["build_mcp_app"]
