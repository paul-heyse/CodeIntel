"""Function-centric HTTP routes."""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from fastapi import APIRouter
from pydantic import Field

from codeintel.serving.http.dependencies import ServiceDep
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    FileSummaryResponse,
    FunctionSummaryResponse,
    GraphNeighborhoodResponse,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    TestsForFunctionResponse,
)

LOG = logging.getLogger("codeintel.serving.http.routes.functions")


def build_functions_router() -> APIRouter:
    """
    Construct the router for function-centric endpoints.

    Returns
    -------
    APIRouter
        Router exposing function metadata endpoints.
    """
    router = APIRouter()

    @router.get(
        "/function/summary",
        response_model=FunctionSummaryResponse,
        summary="Get function summary",
    )
    def function_summary(
        *,
        service: ServiceDep,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
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
            urn=urn,
            goid_h128=goid_h128,
            rel_path=rel_path,
            qualname=qualname,
        )
        if not summary.found or summary.summary is None:
            message = "Function not found"
            raise errors.not_found(message)
        return summary

    @router.get(
        "/functions/high-risk",
        response_model=HighRiskFunctionsResponse,
        summary="List high-risk functions",
    )
    def list_high_risk_functions(
        *,
        service: ServiceDep,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
    ) -> HighRiskFunctionsResponse:
        """
        List high-risk functions with optional tested-only filtering.

        Returns
        -------
        HighRiskFunctionsResponse
            High-risk functions and truncation flag.
        """
        result = service.list_high_risk_functions(
            min_risk=min_risk,
            limit=limit,
            tested_only=tested_only,
        )
        return HighRiskFunctionsResponse(functions=result.functions, truncated=result.truncated)

    @router.get(
        "/function/callgraph",
        response_model=CallGraphNeighborsResponse,
        summary="Get call graph neighbors for a function",
    )
    def function_callgraph(
        *,
        service: ServiceDep,
        goid_h128: int,
        direction: Literal["in", "out", "both"] = "both",
        limit: int | None = None,
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
        )

    @router.get(
        "/function/tests",
        response_model=TestsForFunctionResponse,
        summary="List tests that exercise a function",
    )
    def tests_for_function(
        *,
        service: ServiceDep,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
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
        )

    @router.get(
        "/graph/call/neighborhood",
        response_model=GraphNeighborhoodResponse,
        summary="Call graph ego neighborhood",
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
        "/graph/import/boundary",
        response_model=ImportBoundaryResponse,
        summary="Import graph edges crossing a subsystem",
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
        "/file/summary",
        response_model=FileSummaryResponse,
        summary="Get file summary with function details",
    )
    def file_summary(
        *,
        service: ServiceDep,
        rel_path: str,
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
        summary = service.get_file_summary(rel_path=rel_path)
        if not summary.found or summary.file is None:
            message = "File not found"
            raise errors.not_found(message)
        return summary

    return router


__all__ = ["build_functions_router"]
