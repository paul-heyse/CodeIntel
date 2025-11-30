"""Function-oriented MCP tools for CodeIntel."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    FileSummaryResponse,
    FunctionSummaryResponse,
    HighRiskFunctionsResponse,
    ProblemDetail,
    TestsForFunctionResponse,
)
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap


def register_function_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """Register function-related MCP tools on the given FastMCP instance."""

    @mcp.tool()
    @_wrap
    def get_function_summary(
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> dict[str, object] | dict[str, ProblemDetail]:
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
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: HighRiskFunctionsResponse = backend.list_high_risk_functions(
            min_risk=min_risk,
            limit=limit,
            tested_only=tested_only,
        )
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def get_callgraph_neighbors(
        goid_h128: int,
        direction: str = "both",
        limit: int = 50,
    ) -> dict[str, object] | dict[str, ProblemDetail]:
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
        limit: int | None = None,
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: TestsForFunctionResponse = backend.get_tests_for_function(
            goid_h128=goid_h128,
            urn=urn,
            limit=limit,
        )
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def get_file_summary(rel_path: str) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: FileSummaryResponse = backend.get_file_summary(rel_path=rel_path)
        return resp.model_dump()


__all__ = ["register_function_tools"]
