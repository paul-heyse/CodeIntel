"""Function-centric HTTP routes."""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from codeintel.serving.http.dependencies import ServiceDep
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    FileSummaryResponse,
    FunctionSummaryResponse,
    GraphNeighborhoodResponse,
    GraphScopePayload,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    TestsForFunctionResponse,
)
from codeintel.serving.registry import OperationSpec, get_operation_spec

LOG = logging.getLogger("codeintel.serving.http.routes.functions")


class FunctionSummaryParams(BaseModel):
    """Function summary query parameters."""

    model_config = ConfigDict(extra="ignore")

    urn: str | None = None
    goid_h128: int | None = None
    rel_path: str | None = None
    qualname: str | None = None
    scope: GraphScopePayload | None = None


def _require_spec(op_id: str) -> OperationSpec:
    spec = get_operation_spec(op_id)
    if spec is None:
        message = f"OperationSpec {op_id} is not registered"
        raise ValueError(message)
    return spec


def _function_summary_params(
    urn: str | None = None,
    goid_h128: int | None = None,
    rel_path: str | None = None,
    qualname: str | None = None,
    scope: GraphScopePayload | None = None,
) -> FunctionSummaryParams:
    return FunctionSummaryParams(
        urn=urn,
        goid_h128=goid_h128,
        rel_path=rel_path,
        qualname=qualname,
        scope=scope,
    )


def _load_function_specs() -> tuple[dict[str, OperationSpec], dict[str, str]]:
    ids = [
        "function.summary",
        "functions.high_risk",
        "functions.tests",
        "graph.call_neighbors",
        "graph.call_neighborhood",
        "graph.import_boundary",
        "file.summary",
    ]
    specs: dict[str, OperationSpec] = {}
    paths: dict[str, str] = {}
    missing: list[str] = []
    missing_paths: list[str] = []
    for op_id in ids:
        spec = get_operation_spec(op_id)
        if spec is None:
            missing.append(op_id)
            continue
        specs[op_id] = spec
        if spec.http_path is None:
            missing_paths.append(op_id)
        else:
            paths[op_id] = spec.http_path
    if missing or missing_paths:
        message = f"Missing OperationSpec entries: {missing or 'ok'}; paths: {missing_paths or 'ok'}"
        raise ValueError(message)
    return specs, paths


def build_functions_router() -> APIRouter:
    """
    Construct the router for function-centric endpoints.

    Raises
    ------
    ValueError
        If required OperationSpec entries are missing or incomplete.

    Returns
    -------
    APIRouter
        Router exposing function metadata endpoints.
    """
    router = APIRouter()
    specs, paths = _load_function_specs()
    spec_summary = specs["function.summary"]
    spec_high_risk = specs["functions.high_risk"]
    spec_tests = specs["functions.tests"]
    spec_neighbors = specs["graph.call_neighbors"]
    spec_neighborhood = specs["graph.call_neighborhood"]
    spec_import_boundary = specs["graph.import_boundary"]
    spec_file_summary = specs["file.summary"]
    summary_path = paths["function.summary"]
    high_risk_path = paths["functions.high_risk"]
    tests_path = paths["functions.tests"]
    neighbors_path = paths["graph.call_neighbors"]
    neighborhood_path = paths["graph.call_neighborhood"]
    import_boundary_path = paths["graph.import_boundary"]
    file_summary_path = paths["file.summary"]

    @router.get(
        summary_path,
        response_model=FunctionSummaryResponse,
        summary=spec_summary.summary,
        tags=[spec_summary.category],
    )
    def function_summary(
        *,
        service: ServiceDep,
        params: Annotated[FunctionSummaryParams, Depends(_function_summary_params)],
    ) -> FunctionSummaryResponse:
        """
        Return a function summary identified by GOID, URN, or path.

        Returns
        -------
        FunctionSummaryResponse
            Summary payload describing the requested function.

        Raises
        ------
        errors.not_found
            If the function cannot be located.
        """
        summary = service.get_function_summary(
            urn=params.urn,
            goid_h128=params.goid_h128,
            rel_path=params.rel_path,
            qualname=params.qualname,
            scope=params.scope,
        )
        if not summary.found or summary.summary is None:
            message = "Function not found"
            raise errors.not_found(message)
        return summary

    @router.get(
        high_risk_path,
        response_model=HighRiskFunctionsResponse,
        summary=spec_high_risk.summary,
        tags=[spec_high_risk.category],
    )
    def list_high_risk_functions(
        *,
        service: ServiceDep,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> HighRiskFunctionsResponse:
        """
        List high-risk functions with optional tested-only filtering.

        Returns
        -------
        HighRiskFunctionsResponse
            High-risk functions and truncation flag.
        """
        return service.list_high_risk_functions(
            min_risk=min_risk,
            limit=limit,
            tested_only=tested_only,
            scope=scope,
        )

    @router.get(
        neighbors_path,
        response_model=CallGraphNeighborsResponse,
        summary=spec_neighbors.summary,
        tags=[spec_neighbors.category],
    )
    def function_callgraph(
        *,
        service: ServiceDep,
        goid_h128: int,
        direction: Literal["in", "out", "both"] = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> CallGraphNeighborsResponse:
        """
        Return incoming and outgoing neighbors for a function.

        Returns
        -------
        CallGraphNeighborsResponse
            Incoming and outgoing edges adjacent to the function.
        """
        return service.get_callgraph_neighbors(
            goid_h128=goid_h128,
            direction=direction,
            limit=limit,
            scope=scope,
        )

    @router.get(
        tests_path,
        response_model=TestsForFunctionResponse,
        summary=spec_tests.summary,
        tags=[spec_tests.category],
    )
    def tests_for_function(
        *,
        service: ServiceDep,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> TestsForFunctionResponse:
        """
        List tests that exercised the requested function.

        Returns
        -------
        TestsForFunctionResponse
            Tests linked to the requested function.
        """
        return service.get_tests_for_function(
            goid_h128=goid_h128,
            urn=urn,
            limit=limit,
            scope=scope,
        )

    @router.get(
        neighborhood_path,
        response_model=GraphNeighborhoodResponse,
        summary=spec_neighborhood.summary,
        tags=[spec_neighborhood.category],
    )
    def callgraph_neighborhood(
        *,
        service: ServiceDep,
        goid_h128: int,
        radius: Annotated[int, Field(ge=1)] = 1,
        max_nodes: int | None = None,
    ) -> GraphNeighborhoodResponse:
        """
        Return a bounded ego neighborhood in the call graph.

        Parameters
        ----------
        service : ServiceDep
            Query service providing backend access.
        goid_h128 : int
            GOID of the function to center the neighborhood on.
        radius : int
            Hop radius (>=1).
        max_nodes : int, optional
            Optional node cap; defaults to service max_rows_per_call when omitted.

        Returns
        -------
        GraphNeighborhoodResponse
            Ego subgraph with truncation metadata.
        """
        response = service.get_callgraph_neighborhood(
            goid_h128=goid_h128, radius=radius, max_nodes=max_nodes
        )
        LOG.info(
            "callgraph_neighborhood repo=%s commit=%s goid=%s radius=%s applied_limit=%s "
            "truncated=%s",
            getattr(service, "repo", "unknown"),
            getattr(service, "commit", "unknown"),
            goid_h128,
            radius,
            response.meta.applied_limit,
            response.meta.truncated,
        )
        return response

    @router.get(
        import_boundary_path,
        response_model=ImportBoundaryResponse,
        summary=spec_import_boundary.summary,
        tags=[spec_import_boundary.category],
    )
    def import_boundary(
        *,
        service: ServiceDep,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> ImportBoundaryResponse:
        """
        Return import edges crossing the given subsystem boundary.

        Parameters
        ----------
        service : ServiceDep
            Query service providing backend access.
        subsystem_id : str
            Subsystem identifier to inspect.
        max_edges : int, optional
            Optional edge cap; defaults to service max_rows_per_call.

        Returns
        -------
        ImportBoundaryResponse
            Boundary edges plus metadata describing truncation.
        """
        response = service.get_import_boundary(subsystem_id=subsystem_id, max_edges=max_edges)
        LOG.info(
            "import_boundary repo=%s commit=%s subsystem=%s applied_limit=%s truncated=%s",
            getattr(service, "repo", "unknown"),
            getattr(service, "commit", "unknown"),
            subsystem_id,
            response.meta.applied_limit,
            response.meta.truncated,
        )
        return response

    @router.get(
        file_summary_path,
        response_model=FileSummaryResponse,
        summary=spec_file_summary.summary,
        tags=[spec_file_summary.category],
    )
    def file_summary(
        *,
        service: ServiceDep,
        rel_path: str,
        scope: GraphScopePayload | None = None,
    ) -> FileSummaryResponse:
        """
        Return file-level metrics plus function summaries.

        Returns
        -------
        FileSummaryResponse
            File summary and nested function details.

        Raises
        ------
        errors.not_found
            If the file cannot be located in metadata tables.
        """
        summary = service.get_file_summary(rel_path=rel_path, scope=scope)
        if not summary.found or summary.file is None:
            message = "File not found"
            raise errors.not_found(message)
        return summary

    return router


__all__ = ["build_functions_router"]
