"""Function-centric delegates for query services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from codeintel.serving.backend import DuckDBQueryService, clamp_limit_value
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

    query: DuckDBQueryService
    _call: Callable[..., Any]

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphScopePayload | None = None,
    ) -> FunctionSummaryResponse:
        return self._call(
            "get_function_summary",
            lambda: self.query.get_function_summary(
                urn=urn,
                goid_h128=goid_h128,
                rel_path=rel_path,
                qualname=qualname,
                scope=parse_graph_scope(scope),
            ),
        )

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> HighRiskFunctionsResponse:
        return self._call(
            "list_high_risk_functions",
            lambda: self.query.list_high_risk_functions(
                min_risk=min_risk,
                limit=limit,
                tested_only=tested_only,
                scope=parse_graph_scope(scope),
            ),
        )

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> CallGraphNeighborsResponse:
        return self._call(
            "get_callgraph_neighbors",
            lambda: self.query.get_callgraph_neighbors(
                goid_h128=goid_h128,
                direction=direction,
                limit=limit,
                scope=parse_graph_scope(scope),
            ),
        )

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> TestsForFunctionResponse:
        return self._call(
            "get_tests_for_function",
            lambda: self.query.get_tests_for_function(
                goid_h128=goid_h128,
                urn=urn,
                limit=limit,
                scope=parse_graph_scope(scope),
            ),
        )

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> GraphNeighborhoodResponse:
        return self._call(
            "get_callgraph_neighborhood",
            lambda: self.query.get_callgraph_neighborhood(
                goid_h128=goid_h128, radius=radius, max_nodes=max_nodes
            ),
            dataset="call_graph_nodes",
        )

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> ImportBoundaryResponse:
        return self._call(
            "get_import_boundary",
            lambda: self.query.get_import_boundary(subsystem_id=subsystem_id, max_edges=max_edges),
            dataset="import_graph_edges",
        )

    def get_file_summary(
        self, *, rel_path: str, scope: GraphScopePayload | None = None
    ) -> FileSummaryResponse:
        return self._call(
            "get_file_summary",
            lambda: self.query.get_file_summary(
                rel_path=rel_path,
                scope=parse_graph_scope(scope),
            ),
        )


class _HttpFunctionQueryMixin(_HttpTransportMixin):
    """HTTP-based implementation of the function query API."""

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> HighRiskFunctionsResponse:
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
            return HighRiskFunctionsResponse.model_validate(
                self.request_json(
                    "/functions/high-risk",
                    {
                        "min_risk": min_risk,
                        "limit": clamp.applied,
                        "tested_only": tested_only,
                        "scope": scope.model_dump() if scope is not None else None,
                    },
                )
            )

        return self._http_call("list_high_risk_functions", _run)

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphScopePayload | None = None,
    ) -> FunctionSummaryResponse:
        def _run() -> FunctionSummaryResponse:
            return FunctionSummaryResponse.model_validate(
                self.request_json(
                    "/function/summary",
                    {
                        "urn": urn,
                        "goid_h128": goid_h128,
                        "rel_path": rel_path,
                        "qualname": qualname,
                        "scope": scope.model_dump() if scope is not None else None,
                    },
                )
            )

        return self._http_call("get_function_summary", _run)

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> CallGraphNeighborsResponse:
        def _run() -> CallGraphNeighborsResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return CallGraphNeighborsResponse(outgoing=[], incoming=[], meta=ResponseMeta())
            return CallGraphNeighborsResponse.model_validate(
                self.request_json(
                    "/function/callgraph",
                    {
                        "goid_h128": goid_h128,
                        "direction": direction,
                        "limit": clamp.applied,
                        "scope": scope.model_dump() if scope is not None else None,
                    },
                )
            )

        return self._http_call("get_callgraph_neighbors", _run)

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> GraphNeighborhoodResponse:
        def _run() -> GraphNeighborhoodResponse:
            applied_limit = self.limits.default_limit if max_nodes is None else max_nodes
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return GraphNeighborhoodResponse(nodes=[], edges=[], meta=ResponseMeta())
            return GraphNeighborhoodResponse.model_validate(
                self.request_json(
                    "/graph/call/neighborhood",
                    {
                        "goid_h128": goid_h128,
                        "radius": radius,
                        "max_nodes": clamp.applied,
                    },
                )
            )

        return self._http_call(
            "get_callgraph_neighborhood",
            _run,
            dataset="call_graph_nodes",
        )

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> ImportBoundaryResponse:
        def _run() -> ImportBoundaryResponse:
            applied_limit = self.limits.default_limit if max_edges is None else max_edges
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return ImportBoundaryResponse(nodes=[], edges=[], meta=ResponseMeta())
            return ImportBoundaryResponse.model_validate(
                self.request_json(
                    "/graph/import/boundary",
                    {"subsystem_id": subsystem_id, "max_edges": clamp.applied},
                )
            )

        return self._http_call("get_import_boundary", _run, dataset="import_graph_edges")

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> TestsForFunctionResponse:
        def _run() -> TestsForFunctionResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return TestsForFunctionResponse(tests=[], meta=ResponseMeta())
            return TestsForFunctionResponse.model_validate(
                self.request_json(
                    "/function/tests",
                    {
                        "goid_h128": goid_h128,
                        "urn": urn,
                        "limit": clamp.applied,
                        "scope": scope.model_dump() if scope is not None else None,
                    },
                )
            )

        return self._http_call("get_tests_for_function", _run)

    def get_file_summary(
        self, *, rel_path: str, scope: GraphScopePayload | None = None
    ) -> FileSummaryResponse:
        def _run() -> FileSummaryResponse:
            return FileSummaryResponse.model_validate(
                self.request_json(
                    "/file/summary",
                    {
                        "rel_path": rel_path,
                        "scope": scope.model_dump() if scope is not None else None,
                    },
                )
            )

        return self._http_call("get_file_summary", _run)


__all__ = ["_FunctionQueryDelegates", "_HttpFunctionQueryMixin"]
