"""Function-centric delegates for query services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from codeintel.serving import domain_models as dm
from codeintel.serving.backend import clamp_limit_value
from codeintel.serving.backend.query_api import DuckDBQueryApi
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    FileSummaryResponse,
    FunctionSummaryResponse,
    GraphNeighborhoodResponse,
    GraphScopePayload,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    ResponseMeta,
    TestsForFunctionResponse,
    parse_graph_scope,
)
from codeintel.serving.services.http_transport import _HttpTransportMixin


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
        raw_resp = self._call(
            "get_function_summary",
            lambda: self.query.functions.get_function_summary(
                urn=urn,
                goid_h128=goid_h128,
                rel_path=rel_path,
                qualname=qualname,
                scope=parse_graph_scope(scope),
            ),
        )
        pydantic_resp = (
            raw_resp
            if isinstance(raw_resp, FunctionSummaryResponse)
            else FunctionSummaryResponse.model_validate(raw_resp)
        )
        return pydantic_resp.to_domain()

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> dm.HighRiskFunctionsResult:
        pydantic_resp: HighRiskFunctionsResponse = self._call(
            "list_high_risk_functions",
            lambda: self.query.functions.list_high_risk_functions(
                min_risk=min_risk,
                limit=limit,
                tested_only=tested_only,
                scope=parse_graph_scope(scope),
            ),
        )
        return pydantic_resp.to_domain()

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.CallGraphNeighbors:
        pydantic_resp: CallGraphNeighborsResponse = self._call(
            "get_callgraph_neighbors",
            lambda: self.query.functions.get_callgraph_neighbors(
                goid_h128=goid_h128,
                direction=direction,
                limit=limit,
                scope=parse_graph_scope(scope),
            ),
        )
        return pydantic_resp.to_domain()

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.TestsForFunctionResult:
        pydantic_resp: TestsForFunctionResponse = self._call(
            "get_tests_for_function",
            lambda: self.query.functions.get_tests_for_function(
                goid_h128=goid_h128,
                urn=urn,
                limit=limit,
                scope=parse_graph_scope(scope),
            ),
        )
        return pydantic_resp.to_domain()

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> dm.GraphNeighborhood:
        pydantic_resp: GraphNeighborhoodResponse = self._call(
            "get_callgraph_neighborhood",
            lambda: self.query.functions.get_callgraph_neighborhood(
                goid_h128=goid_h128, radius=radius, max_nodes=max_nodes
            ),
            dataset="call_graph_nodes",
        )
        return pydantic_resp.to_domain()

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> dm.ImportBoundary:
        pydantic_resp: ImportBoundaryResponse = self._call(
            "get_import_boundary",
            lambda: self.query.functions.get_import_boundary(
                subsystem_id=subsystem_id, max_edges=max_edges
            ),
            dataset="import_graph_edges",
        )
        return pydantic_resp.to_domain()

    def get_file_summary(
        self, *, rel_path: str, scope: GraphScopePayload | None = None
    ) -> dm.FileSummaryResult:
        pydantic_resp: FileSummaryResponse = self._call(
            "get_file_summary",
            lambda: self.query.modules.get_file_summary(
                rel_path=rel_path,
                scope=parse_graph_scope(scope),
            ),
        )
        return pydantic_resp.to_domain()


class _HttpFunctionQueryMixin(_HttpTransportMixin):
    """HTTP-based implementation of the function query API."""

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> dm.HighRiskFunctionsResult:
        def _run() -> HighRiskFunctionsResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return HighRiskFunctionsResponse(
                    functions=[],
                    truncated=False,
                    meta=ResponseMeta(),
                )
            payload = self.request_json(
                "/functions/high-risk",
                {
                    "min_risk": min_risk,
                    "limit": clamp.applied,
                    "tested_only": tested_only,
                    "scope": scope.model_dump() if scope is not None else None,
                },
            )
            if isinstance(payload, dm.HighRiskFunctionsResult):
                return HighRiskFunctionsResponse.from_domain(payload)
            if isinstance(payload, HighRiskFunctionsResponse):
                return payload
            return HighRiskFunctionsResponse.model_validate(payload)

        pydantic_resp: HighRiskFunctionsResponse = self._http_call("list_high_risk_functions", _run)
        return pydantic_resp.to_domain()

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
                    "scope": scope.model_dump() if scope is not None else None,
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
        def _run() -> CallGraphNeighborsResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return CallGraphNeighborsResponse(outgoing=[], incoming=[], meta=ResponseMeta())
            payload = self.request_json(
                "/function/callgraph",
                {
                    "goid_h128": goid_h128,
                    "direction": direction,
                    "limit": clamp.applied,
                    "scope": scope.model_dump() if scope is not None else None,
                },
            )
            if isinstance(payload, dm.CallGraphNeighbors):
                return CallGraphNeighborsResponse.from_domain(payload)
            if isinstance(payload, CallGraphNeighborsResponse):
                return payload
            return CallGraphNeighborsResponse.model_validate(payload)

        pydantic_resp: CallGraphNeighborsResponse = self._http_call("get_callgraph_neighbors", _run)
        return pydantic_resp.to_domain()

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> dm.GraphNeighborhood:
        def _run() -> GraphNeighborhoodResponse:
            applied_limit = self.limits.default_limit if max_nodes is None else max_nodes
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return GraphNeighborhoodResponse(nodes=[], edges=[], meta=ResponseMeta())
            payload = self.request_json(
                "/graph/call/neighborhood",
                {
                    "goid_h128": goid_h128,
                    "radius": radius,
                    "max_nodes": clamp.applied,
                },
            )
            if isinstance(payload, dm.GraphNeighborhood):
                return GraphNeighborhoodResponse.from_domain(payload)
            if isinstance(payload, GraphNeighborhoodResponse):
                return payload
            return GraphNeighborhoodResponse.model_validate(payload)

        pydantic_resp: GraphNeighborhoodResponse = self._http_call(
            "get_callgraph_neighborhood",
            _run,
            dataset="call_graph_nodes",
        )
        return pydantic_resp.to_domain()

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> dm.ImportBoundary:
        def _run() -> ImportBoundaryResponse:
            applied_limit = self.limits.default_limit if max_edges is None else max_edges
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return ImportBoundaryResponse(nodes=[], edges=[], meta=ResponseMeta())
            payload = self.request_json(
                "/graph/import/boundary",
                {"subsystem_id": subsystem_id, "max_edges": clamp.applied},
            )
            if isinstance(payload, dm.ImportBoundary):
                return ImportBoundaryResponse.from_domain(payload)
            if isinstance(payload, ImportBoundaryResponse):
                return payload
            return ImportBoundaryResponse.model_validate(payload)

        pydantic_resp: ImportBoundaryResponse = self._http_call(
            "get_import_boundary", _run, dataset="import_graph_edges"
        )
        return pydantic_resp.to_domain()

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.TestsForFunctionResult:
        def _run() -> TestsForFunctionResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return TestsForFunctionResponse(tests=[], meta=ResponseMeta())
            payload = self.request_json(
                "/function/tests",
                {
                    "goid_h128": goid_h128,
                    "urn": urn,
                    "limit": clamp.applied,
                    "scope": scope.model_dump() if scope is not None else None,
                },
            )
            if isinstance(payload, dm.TestsForFunctionResult):
                return TestsForFunctionResponse.from_domain(payload)
            if isinstance(payload, TestsForFunctionResponse):
                return payload
            return TestsForFunctionResponse.model_validate(payload)

        pydantic_resp: TestsForFunctionResponse = self._http_call("get_tests_for_function", _run)
        return pydantic_resp.to_domain()

    def get_file_summary(
        self, *, rel_path: str, scope: GraphScopePayload | None = None
    ) -> dm.FileSummaryResult:
        def _run() -> FileSummaryResponse:
            payload = self.request_json(
                "/file/summary",
                {
                    "rel_path": rel_path,
                    "scope": scope.model_dump() if scope is not None else None,
                },
            )
            if isinstance(payload, dm.FileSummaryResult):
                return FileSummaryResponse.from_domain(payload)
            if isinstance(payload, FileSummaryResponse):
                return payload
            return FileSummaryResponse.model_validate(payload)

        pydantic_resp: FileSummaryResponse = self._http_call("get_file_summary", _run)
        return pydantic_resp.to_domain()


__all__ = ["_FunctionQueryDelegates", "_HttpFunctionQueryMixin"]
