"""Function-centric HTTP routes."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from codeintel.serving.http import dependencies as http_deps
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
from codeintel.serving.operations import get_operation

if TYPE_CHECKING:
    from codeintel.serving.operations import Operation

RouteDeps = Sequence[Any]


@dataclass(frozen=True)
class RouterOptions:
    """Options for building HTTP routers.

    Parameters
    ----------
    auto_pipeline
        When True, attach auto-pipeline dependencies to routes.
    """

    auto_pipeline: bool = False


LOG = logging.getLogger("codeintel.serving.http.routes.functions")
_GRAPH_SCOPE_PAYLOAD = GraphScopePayload


class FunctionSummaryParams(BaseModel):
    """Function summary query parameters."""

    model_config = ConfigDict(extra="ignore")

    urn: str | None = None
    goid_h128: int | None = None
    rel_path: str | None = None
    qualname: str | None = None
    scope: GraphScopePayload | None = None


def _require_spec(op_id: str) -> Operation:
    spec = get_operation(op_id)
    if spec is None:
        message = f"Operation {op_id} is not registered"
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


def _load_function_specs() -> tuple[dict[str, Operation], dict[str, str]]:
    ids = [
        "function.summary",
        "functions.high_risk",
        "functions.tests",
        "graph.call_neighbors",
        "graph.call_neighborhood",
        "graph.import_boundary",
        "file.summary",
    ]
    specs: dict[str, Operation] = {}
    paths: dict[str, str] = {}
    missing: list[str] = []
    missing_paths: list[str] = []
    for op_id in ids:
        spec = get_operation(op_id)
        if spec is None:
            missing.append(op_id)
            continue
        specs[op_id] = spec
        if spec.http_path is None:
            missing_paths.append(op_id)
        else:
            paths[op_id] = spec.http_path
    if missing or missing_paths:
        message = f"Missing Operation entries: {missing or 'ok'}; paths: {missing_paths or 'ok'}"
        raise ValueError(message)
    return specs, paths


def _register_summary_and_risk_routes(
    router: APIRouter,
    specs: dict[str, Operation],
    paths: dict[str, str],
    deps: dict[str, RouteDeps],
) -> None:
    summary_spec = specs["function.summary"]
    risk_spec = specs["functions.high_risk"]

    @router.get(
        paths["function.summary"],
        response_model=FunctionSummaryResponse,
        summary=summary_spec.summary,
        tags=[summary_spec.category],
        dependencies=list(deps.get("function.summary", [])),
    )
    def function_summary(
        *,
        service: http_deps.ServiceDep,
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
        domain_summary = service.get_function_summary(
            urn=params.urn,
            goid_h128=params.goid_h128,
            rel_path=params.rel_path,
            qualname=params.qualname,
            scope=params.scope,
        )
        summary = FunctionSummaryResponse.from_domain(domain_summary)
        if not summary.found or summary.summary is None:
            message = "Function not found"
            raise errors.not_found(message)
        return summary

    @router.get(
        paths["functions.high_risk"],
        response_model=HighRiskFunctionsResponse,
        summary=risk_spec.summary,
        tags=[risk_spec.category],
        dependencies=list(deps.get("functions.high_risk", [])),
    )
    def list_high_risk_functions(
        *,
        service: http_deps.ServiceDep,
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
        domain_response = service.list_high_risk_functions(
            min_risk=min_risk,
            limit=limit,
            tested_only=tested_only,
            scope=scope,
        )
        return HighRiskFunctionsResponse.from_domain(domain_response)


def _register_graph_and_tests_routes(
    router: APIRouter,
    specs: dict[str, Operation],
    paths: dict[str, str],
    deps: dict[str, RouteDeps],
) -> None:
    neighbors_spec = specs["graph.call_neighbors"]
    neighborhood_spec = specs["graph.call_neighborhood"]
    import_boundary_spec = specs["graph.import_boundary"]
    tests_spec = specs["functions.tests"]

    @router.get(
        paths["graph.call_neighbors"],
        response_model=CallGraphNeighborsResponse,
        summary=neighbors_spec.summary,
        tags=[neighbors_spec.category],
        dependencies=list(deps.get("graph.call_neighbors", [])),
    )
    def function_callgraph(
        *,
        service: http_deps.ServiceDep,
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
        domain_neighbors = service.get_callgraph_neighbors(
            goid_h128=goid_h128,
            direction=direction,
            limit=limit,
            scope=scope,
        )
        return CallGraphNeighborsResponse.from_domain(domain_neighbors)

    @router.get(
        paths["functions.tests"],
        response_model=TestsForFunctionResponse,
        summary=tests_spec.summary,
        tags=[tests_spec.category],
        dependencies=list(deps.get("functions.tests", [])),
    )
    def tests_for_function(
        *,
        service: http_deps.ServiceDep,
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

        Raises
        ------
        errors.invalid_argument
            When no identifier is supplied.
        """
        if goid_h128 is None and urn is None:
            message = "At least one identifier (goid_h128 or urn) is required"
            raise errors.invalid_argument(message)
        domain_tests = service.get_tests_for_function(
            goid_h128=goid_h128,
            urn=urn,
            limit=limit,
            scope=scope,
        )
        return TestsForFunctionResponse.from_domain(domain_tests)

    @router.get(
        paths["graph.call_neighborhood"],
        response_model=GraphNeighborhoodResponse,
        summary=neighborhood_spec.summary,
        tags=[neighborhood_spec.category],
        dependencies=list(deps.get("graph.call_neighborhood", [])),
    )
    def callgraph_neighborhood(
        *,
        service: http_deps.ServiceDep,
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

        Raises
        ------
        errors.invalid_argument
            When max_nodes is negative.
        """
        if max_nodes is not None and max_nodes < 0:
            message = "max_nodes must be non-negative"
            raise errors.invalid_argument(message)
        domain_response = service.get_callgraph_neighborhood(
            goid_h128=goid_h128, radius=radius, max_nodes=max_nodes
        )
        response = GraphNeighborhoodResponse.from_domain(domain_response)
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
        paths["graph.import_boundary"],
        response_model=ImportBoundaryResponse,
        summary=import_boundary_spec.summary,
        tags=[import_boundary_spec.category],
        dependencies=list(deps.get("graph.import_boundary", [])),
    )
    def import_boundary(
        *,
        service: http_deps.ServiceDep,
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

        Raises
        ------
        errors.invalid_argument
            When max_edges is negative.
        """
        if max_edges is not None and max_edges < 0:
            message = "max_edges must be non-negative"
            raise errors.invalid_argument(message)
        domain_response = service.get_import_boundary(
            subsystem_id=subsystem_id, max_edges=max_edges
        )
        response = ImportBoundaryResponse.from_domain(domain_response)
        LOG.info(
            "import_boundary repo=%s commit=%s subsystem=%s applied_limit=%s truncated=%s",
            getattr(service, "repo", "unknown"),
            getattr(service, "commit", "unknown"),
            subsystem_id,
            response.meta.applied_limit,
            response.meta.truncated,
        )
        return response


def _register_file_summary_route(
    router: APIRouter,
    specs: dict[str, Operation],
    paths: dict[str, str],
    deps: dict[str, RouteDeps],
) -> None:
    file_spec = specs["file.summary"]

    @router.get(
        paths["file.summary"],
        response_model=FileSummaryResponse,
        summary=file_spec.summary,
        tags=[file_spec.category],
        dependencies=list(deps.get("file.summary", [])),
    )
    def file_summary(
        *,
        service: http_deps.ServiceDep,
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
        domain_summary = service.get_file_summary(rel_path=rel_path, scope=scope)
        summary = FileSummaryResponse.from_domain(domain_summary)
        if not summary.found or summary.file is None:
            message = "File not found"
            raise errors.not_found(message)
        return summary


def _build_prereq_deps(
    specs: dict[str, Operation],
    options: RouterOptions | None,
) -> dict[str, RouteDeps]:
    """Build auto-pipeline dependencies for each operation.

    Parameters
    ----------
    specs
        Operation specifications keyed by operation ID.
    options
        Router options; when auto_pipeline is enabled, creates dependencies.

    Returns
    -------
    dict[str, RouteDeps]
        Mapping of operation ID to list of FastAPI dependencies.
    """
    if options is None or not options.auto_pipeline:
        return {}
    return {op_id: [Depends(http_deps.make_op_prereq_dependency(op_id))] for op_id in specs}


def build_functions_router(options: RouterOptions | None = None) -> APIRouter:
    """Construct the router for function-centric endpoints.

    Parameters
    ----------
    options
        Router configuration options. When auto_pipeline is enabled,
        dependencies are attached that automatically run prerequisites
        before the operation executes.

    Raises
    ------
    ValueError
        If required Operation entries are missing or incomplete.

    Returns
    -------
    APIRouter
        Router exposing function metadata endpoints.
    """
    router = APIRouter()
    try:
        specs, paths = _load_function_specs()
    except ValueError as exc:
        message = "Failed to load function Operation entries"
        raise ValueError(message) from exc

    deps = _build_prereq_deps(specs, options)
    _register_summary_and_risk_routes(router, specs, paths, deps)
    _register_graph_and_tests_routes(router, specs, paths, deps)
    _register_file_summary_route(router, specs, paths, deps)

    return router


__all__ = ["RouterOptions", "build_functions_router"]
