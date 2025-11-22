"""MCP tool registration and error-to-problem mapping."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP

from codeintel.mcp import errors
from codeintel.mcp.backend import QueryBackend
from codeintel.mcp.models import (
    CallGraphNeighborsResponse,
    DatasetRowsResponse,
    FileSummaryResponse,
    FunctionSummaryResponse,
    HighRiskFunctionsResponse,
    ProblemDetail,
    TestsForFunctionResponse,
)


def _wrap(tool: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wrap a backend-facing tool to normalize McpError into ProblemDetail payloads.

    Returns
    -------
    Callable[..., object]
        Wrapped tool function that emits dict payloads.
    """

    def _inner(*args: object, **kwargs: object) -> object:
        try:
            return tool(*args, **kwargs)
        except errors.McpError as exc:
            return {"error": exc.detail.model_dump()}

    return _inner


def register_tools(mcp: FastMCP, backend: QueryBackend) -> None:
    """Register all MCP tools on the given FastMCP instance."""

    @mcp.tool()
    @_wrap
    def get_function_summary(
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> dict[str, Any] | dict[str, ProblemDetail]:
        resp: FunctionSummaryResponse = backend.get_function_summary(
            urn=urn,
            goid_h128=goid_h128,
            rel_path=rel_path,
            qualname=qualname,
        )
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def list_high_risk_functions(
        min_risk: float = 0.7,
        limit: int = 50,
        *,
        tested_only: bool = False,
    ) -> dict[str, Any] | dict[str, ProblemDetail]:
        resp: HighRiskFunctionsResponse = backend.list_high_risk_functions(
            min_risk=min_risk,
            limit=limit,
            tested_only=tested_only,
        )
        return {"functions": resp.functions, "truncated": resp.truncated}

    @mcp.tool()
    @_wrap
    def get_callgraph_neighbors(
        goid_h128: int,
        direction: str = "both",
        limit: int = 50,
    ) -> dict[str, Any] | dict[str, ProblemDetail]:
        resp: CallGraphNeighborsResponse = backend.get_callgraph_neighbors(
            goid_h128=goid_h128,
            direction=direction,
            limit=limit,
        )
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def get_tests_for_function(
        goid_h128: int | None = None,
        urn: str | None = None,
    ) -> dict[str, Any] | dict[str, ProblemDetail]:
        resp: TestsForFunctionResponse = backend.get_tests_for_function(
            goid_h128=goid_h128,
            urn=urn,
        )
        return {"tests": resp.tests}

    @mcp.tool()
    @_wrap
    def get_file_summary(rel_path: str) -> dict[str, Any] | dict[str, ProblemDetail]:
        resp: FileSummaryResponse = backend.get_file_summary(rel_path=rel_path)
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def list_datasets() -> list[dict[str, Any]]:
        return [descriptor.model_dump() for descriptor in backend.list_datasets()]

    @mcp.tool()
    @_wrap
    def read_dataset_rows(
        dataset_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any] | dict[str, ProblemDetail]:
        resp: DatasetRowsResponse = backend.read_dataset_rows(
            dataset_name=dataset_name,
            limit=limit,
            offset=offset,
        )
        return resp.model_dump()
