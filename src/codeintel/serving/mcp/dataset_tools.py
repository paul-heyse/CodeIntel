"""Dataset MCP tools."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.models import DatasetRowsResponse, ProblemDetail
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap


def register_dataset_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """Register dataset browsing MCP tools."""

    @mcp.tool()
    @_wrap
    def list_datasets() -> list[dict[str, object]]:
        return [descriptor.model_dump() for descriptor in backend.list_datasets()]

    @mcp.tool()
    @_wrap
    def read_dataset_rows(
        dataset_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: DatasetRowsResponse = backend.read_dataset_rows(
            dataset_name=dataset_name,
            limit=limit,
            offset=offset,
        )
        return resp.model_dump()


__all__ = ["register_dataset_tools"]
