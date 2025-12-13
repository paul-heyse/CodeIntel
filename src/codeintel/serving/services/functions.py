"""Function-centric delegates for query services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    FileSummaryResponse,
    FunctionSummaryResponse,
    GraphNeighborhoodResponse,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    TestsForFunctionResponse,
    parse_graph_scope,
)
from codeintel.serving.services.conversion import to_domain_result
from codeintel.serving.services.transport import _HttpTransportMixin

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.serving.backend.query_api import DuckDBQueryApi
    from codeintel.serving.mcp.models import (
        GraphScopePayload,
    )


class _FunctionQueryDelegates:
    """Local delegates that call DuckDBQueryService for function-related APIs."""

    query: DuckDBQueryApi
    _call: Callable[..., Any]

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.FunctionSummaryResult:
        raw = self._call(
            "get_function_summary",
            lambda: self.query.functions.get_function_summary(
                urn=urn,
                goid_h128=goid_h128,
                rel_path=rel_path,
                qualname=qualname,
                scope=parse_graph_scope(scope),
            ),
        )
        return to_domain_result(raw, dm.FunctionSummaryResult, FunctionSummaryResponse)

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> dm.HighRiskFunctionsResult:
        raw = self._call(
            "list_high_risk_functions",
            lambda: self.query.functions.list_high_risk_functions(
                min_risk=min_risk,
                limit=limit,
                tested_only=tested_only,
                scope=parse_graph_scope(scope),
            ),
        )
        return to_domain_result(raw, dm.HighRiskFunctionsResult, HighRiskFunctionsResponse)

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.CallGraphNeighbors:
        raw = self._call(
            "get_callgraph_neighbors",
            lambda: self.query.functions.get_callgraph_neighbors(
                goid_h128=goid_h128,
                direction=direction,
                limit=limit,
                scope=parse_graph_scope(scope),
            ),
        )
        return to_domain_result(raw, dm.CallGraphNeighbors, CallGraphNeighborsResponse)

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.TestsForFunctionResult:
        raw = self._call(
            "get_tests_for_function",
            lambda: self.query.functions.get_tests_for_function(
                goid_h128=goid_h128,
                urn=urn,
                limit=limit,
                scope=parse_graph_scope(scope),
            ),
        )
        return to_domain_result(raw, dm.TestsForFunctionResult, TestsForFunctionResponse)

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> dm.GraphNeighborhood:
        raw = self._call(
            "get_callgraph_neighborhood",
            lambda: self.query.functions.get_callgraph_neighborhood(
                goid_h128=goid_h128, radius=radius, max_nodes=max_nodes
            ),
            dataset="call_graph_nodes",
        )
        return to_domain_result(raw, dm.GraphNeighborhood, GraphNeighborhoodResponse)

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> dm.ImportBoundary:
        raw = self._call(
            "get_import_boundary",
            lambda: self.query.functions.get_import_boundary(
                subsystem_id=subsystem_id, max_edges=max_edges
            ),
            dataset="import_graph_edges",
        )
        return to_domain_result(raw, dm.ImportBoundary, ImportBoundaryResponse)

    def get_file_summary(
        self, *, rel_path: str, scope: GraphScopePayload | None = None
    ) -> dm.FileSummaryResult:
        raw = self._call(
            "get_file_summary",
            lambda: self.query.modules.get_file_summary(
                rel_path=rel_path,
                scope=parse_graph_scope(scope),
            ),
        )
        return to_domain_result(raw, dm.FileSummaryResult, FileSummaryResponse)


def _serialize_scope(scope: GraphScopePayload | None) -> dict[str, object] | None:
    """Serialize a GraphScopePayload for HTTP requests."""
    return scope.model_dump() if scope is not None else None


class _HttpFunctionQueryMixin(_HttpTransportMixin):
    """HTTP-based implementation of the function query API.

    Architecture Note
    -----------------
    This mixin implements the **HTTP transport path** for ``HttpQueryService``.
    It uses ``_http_query()`` for methods with limit clamping and the standard
    pattern for methods without limits.

    See ``codeintel.serving.domain_models`` for the full architecture contract.
    """

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> dm.HighRiskFunctionsResult:
        return self._http_query(
            "list_high_risk_functions",
            "/functions/high-risk",
            {
                "min_risk": min_risk,
                "tested_only": tested_only,
                "scope": _serialize_scope(scope),
            },
            HighRiskFunctionsResponse,
            dm.HighRiskFunctionsResult,
            empty_data=HighRiskFunctionsResponse(functions=[], truncated=False),
            limit=limit,
        )

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.FunctionSummaryResult:
        def _run() -> FunctionSummaryResponse:
            payload = self.request_json(
                "/function/summary",
                {
                    "urn": urn,
                    "goid_h128": goid_h128,
                    "rel_path": rel_path,
                    "qualname": qualname,
                    "scope": _serialize_scope(scope),
                },
            )
            if isinstance(payload, dm.FunctionSummaryResult):
                return FunctionSummaryResponse.from_domain(payload)
            if isinstance(payload, FunctionSummaryResponse):
                return payload
            return FunctionSummaryResponse.model_validate(payload)

        pydantic_resp: FunctionSummaryResponse = self._http_call("get_function_summary", _run)
        return pydantic_resp.to_domain()

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.CallGraphNeighbors:
        return self._http_query(
            "get_callgraph_neighbors",
            "/function/callgraph",
            {
                "goid_h128": goid_h128,
                "direction": direction,
                "scope": _serialize_scope(scope),
            },
            CallGraphNeighborsResponse,
            dm.CallGraphNeighbors,
            empty_data=CallGraphNeighborsResponse(outgoing=[], incoming=[]),
            limit=limit,
        )

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> dm.GraphNeighborhood:
        return self._http_query(
            "get_callgraph_neighborhood",
            "/graph/call/neighborhood",
            {"goid_h128": goid_h128, "radius": radius},
            GraphNeighborhoodResponse,
            dm.GraphNeighborhood,
            empty_data=GraphNeighborhoodResponse(nodes=[], edges=[]),
            limit=max_nodes,
            limit_param="max_nodes",
            dataset="call_graph_nodes",
        )

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> dm.ImportBoundary:
        return self._http_query(
            "get_import_boundary",
            "/graph/import/boundary",
            {"subsystem_id": subsystem_id},
            ImportBoundaryResponse,
            dm.ImportBoundary,
            empty_data=ImportBoundaryResponse(nodes=[], edges=[]),
            limit=max_edges,
            limit_param="max_edges",
            dataset="import_graph_edges",
        )

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.TestsForFunctionResult:
        return self._http_query(
            "get_tests_for_function",
            "/function/tests",
            {
                "goid_h128": goid_h128,
                "urn": urn,
                "scope": _serialize_scope(scope),
            },
            TestsForFunctionResponse,
            dm.TestsForFunctionResult,
            empty_data=TestsForFunctionResponse(tests=[]),
            limit=limit,
        )

    def get_file_summary(
        self, *, rel_path: str, scope: GraphScopePayload | None = None
    ) -> dm.FileSummaryResult:
        def _run() -> FileSummaryResponse:
            payload = self.request_json(
                "/file/summary",
                {"rel_path": rel_path, "scope": _serialize_scope(scope)},
            )
            if isinstance(payload, dm.FileSummaryResult):
                return FileSummaryResponse.from_domain(payload)
            if isinstance(payload, FileSummaryResponse):
                return payload
            return FileSummaryResponse.model_validate(payload)

        pydantic_resp: FileSummaryResponse = self._http_call("get_file_summary", _run)
        return pydantic_resp.to_domain()


__all__ = ["_FunctionQueryDelegates", "_HttpFunctionQueryMixin"]
