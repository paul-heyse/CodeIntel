"""Response builder functions for transforming repository data to typed responses.

This module provides single-responsibility functions that transform RowDict
results from repositories into typed response models. Each builder handles:
1. None/not-found cases
2. Row validation via Pydantic models
3. Consistent ResponseMeta construction

These functions should be pure transformations with no side effects or
database access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.serving.backend.pagination import PaginatedFetch
from codeintel.serving.mcp.models import (
    CallGraphEdgeRow,
    CallGraphNeighborsResponse,
    FileHintsResponse,
    FileProfileResponse,
    FileProfileRow,
    FileSummaryResponse,
    FileSummaryRow,
    FunctionArchitectureResponse,
    FunctionProfileResponse,
    FunctionProfileRow,
    FunctionSummaryResponse,
    FunctionSummaryRow,
    GraphNeighborhoodResponse,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    Message,
    ModuleArchitectureResponse,
    ModuleArchitectureRow,
    ModuleProfileResponse,
    ModuleProfileRow,
    ModuleSubsystemResponse,
    ModuleWithSubsystemRow,
    ResponseMeta,
    SubsystemCoverageResponse,
    SubsystemCoverageRow,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemProfileRow,
    SubsystemSummaryResponse,
    SubsystemSummaryRow,
    TestsForFunctionResponse,
    ViewRow,
)

if TYPE_CHECKING:
    from codeintel.storage.repositories.base import RowDict

# -----------------------------------------------------------------------------
# Function Response Builders
# -----------------------------------------------------------------------------


def build_function_summary_response(
    row: RowDict | None,
    *,
    meta: ResponseMeta | None = None,
) -> FunctionSummaryResponse:
    """
    Build a function summary response from a repository row.

    Parameters
    ----------
    row
        Function summary row from repository, or None if not found.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    FunctionSummaryResponse
        Response with found=True and validated summary, or found=False.
    """
    if row is None:
        return FunctionSummaryResponse(
            found=False,
            summary=None,
            meta=meta or ResponseMeta(),
        )
    return FunctionSummaryResponse(
        found=True,
        summary=FunctionSummaryRow.model_validate(row),
        meta=meta or ResponseMeta(),
    )


def build_function_profile_response(
    row: RowDict | None,
    *,
    meta: ResponseMeta | None = None,
) -> FunctionProfileResponse:
    """
    Build a function profile response from a repository row.

    Parameters
    ----------
    row
        Function profile row from repository, or None if not found.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    FunctionProfileResponse
        Response with found=True and validated profile, or found=False.
    """
    if row is None:
        return FunctionProfileResponse(
            found=False,
            profile=None,
            meta=meta or ResponseMeta(),
        )
    return FunctionProfileResponse(
        found=True,
        profile=FunctionProfileRow.model_validate(row),
        meta=meta or ResponseMeta(),
    )


def build_function_architecture_response(
    row: RowDict | None,
    *,
    meta: ResponseMeta | None = None,
) -> FunctionArchitectureResponse:
    """
    Build a function architecture response from a repository row.

    Parameters
    ----------
    row
        Function architecture row from repository, or None if not found.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    FunctionArchitectureResponse
        Response with found=True and validated architecture, or found=False.
    """
    if row is None:
        return FunctionArchitectureResponse(
            found=False,
            architecture=None,
            meta=meta or ResponseMeta(),
        )
    return FunctionArchitectureResponse(
        found=True,
        architecture=ViewRow.model_validate(row),
        meta=meta or ResponseMeta(),
    )


def build_high_risk_functions_response(
    rows: list[RowDict],
    *,
    meta: ResponseMeta | None = None,
) -> HighRiskFunctionsResponse:
    """
    Build a high-risk functions response from repository rows.

    Parameters
    ----------
    rows
        List of high-risk function rows from repository.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    HighRiskFunctionsResponse
        Response with validated function rows.
    """
    return HighRiskFunctionsResponse(
        functions=[ViewRow.model_validate(r) for r in rows],
        meta=meta or ResponseMeta(),
    )


def build_callgraph_neighbors_response(
    incoming: list[RowDict],
    outgoing: list[RowDict],
    *,
    meta: ResponseMeta | None = None,
) -> CallGraphNeighborsResponse:
    """
    Build a call graph neighbors response from repository rows.

    Parameters
    ----------
    incoming
        List of incoming edge rows.
    outgoing
        List of outgoing edge rows.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    CallGraphNeighborsResponse
        Response with validated edge rows for both directions.
    """
    return CallGraphNeighborsResponse(
        incoming=[CallGraphEdgeRow.model_validate(r) for r in incoming],
        outgoing=[CallGraphEdgeRow.model_validate(r) for r in outgoing],
        meta=meta or ResponseMeta(),
    )


def build_tests_for_function_response(
    rows: list[RowDict],
    *,
    meta: ResponseMeta | None = None,
) -> TestsForFunctionResponse:
    """
    Build a tests-for-function response from repository rows.

    Parameters
    ----------
    rows
        List of test rows covering the function.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    TestsForFunctionResponse
        Response with validated test rows.
    """
    return TestsForFunctionResponse(
        tests=[ViewRow.model_validate(r) for r in rows],
        meta=meta or ResponseMeta(),
    )


def build_graph_neighborhood_response(
    nodes: list[RowDict],
    edges: list[dict[str, object]],
    *,
    meta: ResponseMeta | None = None,
) -> GraphNeighborhoodResponse:
    """
    Build a graph neighborhood response from node and edge data.

    Parameters
    ----------
    nodes
        List of node summary rows.
    edges
        List of edge dictionaries with caller/callee info.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    GraphNeighborhoodResponse
        Response with validated node and edge rows.
    """
    return GraphNeighborhoodResponse(
        nodes=[ViewRow.model_validate(n) for n in nodes],
        edges=[ViewRow.model_validate(e) for e in edges],
        meta=meta or ResponseMeta(),
    )


def build_import_boundary_response(
    nodes: list[str],
    edges: list[dict[str, object]],
    *,
    meta: ResponseMeta | None = None,
) -> ImportBoundaryResponse:
    """
    Build an import boundary response from node IDs and edge data.

    Parameters
    ----------
    nodes
        List of node ID strings.
    edges
        List of edge dictionaries with source/target/weight.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    ImportBoundaryResponse
        Response with validated node and edge rows.
    """
    return ImportBoundaryResponse(
        nodes=[ViewRow.model_validate({"id": node}) for node in sorted(nodes)],
        edges=[ViewRow.model_validate(e) for e in edges],
        meta=meta or ResponseMeta(),
    )


# -----------------------------------------------------------------------------
# File/Module Response Builders
# -----------------------------------------------------------------------------


def build_file_summary_response(
    row: RowDict | None,
    *,
    meta: ResponseMeta | None = None,
) -> FileSummaryResponse:
    """
    Build a file summary response from a repository row.

    Parameters
    ----------
    row
        File summary row from repository, or None if not found.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    FileSummaryResponse
        Response with found=True and validated file summary, or found=False.
    """
    if row is None:
        return FileSummaryResponse(
            found=False,
            file=None,
            meta=meta or ResponseMeta(),
        )
    return FileSummaryResponse(
        found=True,
        file=FileSummaryRow.model_validate(row),
        meta=meta or ResponseMeta(),
    )


def build_file_profile_response(
    row: RowDict | None,
    *,
    meta: ResponseMeta | None = None,
    not_found_message: str | None = None,
) -> FileProfileResponse:
    """
    Build a file profile response from a repository row.

    Parameters
    ----------
    row
        File profile row from repository, or None if not found.
    meta
        Optional response metadata; defaults to empty ResponseMeta.
    not_found_message
        Optional message to include when row is None.

    Returns
    -------
    FileProfileResponse
        Response with found=True and validated profile, or found=False.
    """
    if row is None:
        messages = []
        if not_found_message:
            messages.append(
                Message(code="not_found", severity="warning", detail=not_found_message)
            )
        return FileProfileResponse(
            found=False,
            profile=None,
            meta=ResponseMeta(messages=messages) if messages else (meta or ResponseMeta()),
        )
    return FileProfileResponse(
        found=True,
        profile=FileProfileRow.model_validate(row),
        meta=meta or ResponseMeta(),
    )


def build_file_hints_response(
    rows: list[RowDict],
    *,
    meta: ResponseMeta | None = None,
) -> FileHintsResponse:
    """
    Build a file hints response from repository rows.

    Parameters
    ----------
    rows
        List of IDE hint rows from repository.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    FileHintsResponse
        Response with found=True and validated hint rows.
    """
    return FileHintsResponse(
        found=True,
        hints=[ViewRow.model_validate(r) for r in rows],
        meta=meta or ResponseMeta(),
    )


def build_module_profile_response(
    row: RowDict | None,
    *,
    meta: ResponseMeta | None = None,
) -> ModuleProfileResponse:
    """
    Build a module profile response from a repository row.

    Parameters
    ----------
    row
        Module profile row from repository, or None if not found.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    ModuleProfileResponse
        Response with found=True and validated profile, or found=False.
    """
    if row is None:
        return ModuleProfileResponse(
            found=False,
            profile=None,
            meta=meta or ResponseMeta(),
        )
    return ModuleProfileResponse(
        found=True,
        profile=ModuleProfileRow.model_validate(row),
        meta=meta or ResponseMeta(),
    )


def build_module_architecture_response(
    row: RowDict | None,
    *,
    meta: ResponseMeta | None = None,
) -> ModuleArchitectureResponse:
    """
    Build a module architecture response from a repository row.

    Parameters
    ----------
    row
        Module architecture row from repository, or None if not found.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    ModuleArchitectureResponse
        Response with found=True and validated architecture, or found=False.
    """
    if row is None:
        return ModuleArchitectureResponse(
            found=False,
            architecture=None,
            meta=meta or ResponseMeta(),
        )
    return ModuleArchitectureResponse(
        found=True,
        architecture=ModuleArchitectureRow.model_validate(row),
        meta=meta or ResponseMeta(),
    )


# -----------------------------------------------------------------------------
# Subsystem Response Builders
# -----------------------------------------------------------------------------


def build_subsystem_summary_response(
    rows: list[RowDict],
    *,
    meta: ResponseMeta | None = None,
) -> SubsystemSummaryResponse:
    """
    Build a subsystem summary response from repository rows.

    Parameters
    ----------
    rows
        List of subsystem summary rows from repository.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    SubsystemSummaryResponse
        Response with validated subsystem rows.
    """
    return SubsystemSummaryResponse(
        subsystems=[SubsystemSummaryRow.model_validate(r) for r in rows],
        meta=meta or ResponseMeta(),
    )


def build_subsystem_modules_response(
    subsystem_row: RowDict | None,
    module_rows: list[RowDict],
    *,
    meta: ResponseMeta | None = None,
) -> SubsystemModulesResponse:
    """
    Build a subsystem modules response from repository rows.

    Parameters
    ----------
    subsystem_row
        Subsystem summary row, or None if not found.
    module_rows
        List of module membership rows.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    SubsystemModulesResponse
        Response with subsystem detail and module list.
    """
    if subsystem_row is None:
        return SubsystemModulesResponse(
            found=False,
            subsystem=None,
            modules=[],
            meta=meta or ResponseMeta(),
        )
    return SubsystemModulesResponse(
        found=True,
        subsystem=SubsystemSummaryRow.model_validate(subsystem_row),
        modules=[ModuleWithSubsystemRow.model_validate(r) for r in module_rows],
        meta=meta or ResponseMeta(),
    )


def build_module_subsystem_response(
    rows: list[RowDict],
    *,
    meta: ResponseMeta | None = None,
) -> ModuleSubsystemResponse:
    """
    Build a module subsystem response from repository rows.

    Parameters
    ----------
    rows
        List of subsystem membership rows for a module.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    ModuleSubsystemResponse
        Response with found=True and membership list.
    """
    return ModuleSubsystemResponse(
        found=True,
        memberships=[ModuleWithSubsystemRow.model_validate(r) for r in rows],
        meta=meta or ResponseMeta(),
    )


def build_subsystem_profile_response(
    rows: list[RowDict],
    *,
    meta: ResponseMeta | None = None,
) -> SubsystemProfileResponse:
    """
    Build a subsystem profile response from repository rows.

    Parameters
    ----------
    rows
        List of subsystem profile rows from repository.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    SubsystemProfileResponse
        Response with validated profile rows.
    """
    return SubsystemProfileResponse(
        profiles=[SubsystemProfileRow.model_validate(r) for r in rows],
        meta=meta or ResponseMeta(),
    )


def build_subsystem_coverage_response(
    rows: list[RowDict],
    *,
    meta: ResponseMeta | None = None,
) -> SubsystemCoverageResponse:
    """
    Build a subsystem coverage response from repository rows.

    Parameters
    ----------
    rows
        List of subsystem coverage rows from repository.
    meta
        Optional response metadata; defaults to empty ResponseMeta.

    Returns
    -------
    SubsystemCoverageResponse
        Response with validated coverage rows.
    """
    return SubsystemCoverageResponse(
        coverage=[SubsystemCoverageRow.model_validate(r) for r in rows],
        meta=meta or ResponseMeta(),
    )


# -----------------------------------------------------------------------------
# Paginated Response Builders
# -----------------------------------------------------------------------------


def build_paginated_functions_response(
    result: PaginatedFetch[RowDict],
) -> HighRiskFunctionsResponse:
    """
    Build a high-risk functions response from a paginated fetch result.

    Parameters
    ----------
    result
        Paginated fetch result with function rows.

    Returns
    -------
    HighRiskFunctionsResponse
        Response with validated function rows and pagination metadata.
    """
    return HighRiskFunctionsResponse(
        functions=[ViewRow.model_validate(r) for r in result.items],
        meta=result.to_response_meta(),
    )


def build_paginated_subsystems_response(
    result: PaginatedFetch[RowDict],
) -> SubsystemSummaryResponse:
    """
    Build a subsystem summary response from a paginated fetch result.

    Parameters
    ----------
    result
        Paginated fetch result with subsystem rows.

    Returns
    -------
    SubsystemSummaryResponse
        Response with validated subsystem rows and pagination metadata.
    """
    return SubsystemSummaryResponse(
        subsystems=[SubsystemSummaryRow.model_validate(r) for r in result.items],
        meta=result.to_response_meta(),
    )


__all__ = [
    "build_callgraph_neighbors_response",
    "build_file_hints_response",
    "build_file_profile_response",
    "build_file_summary_response",
    "build_function_architecture_response",
    "build_function_profile_response",
    "build_function_summary_response",
    "build_graph_neighborhood_response",
    "build_high_risk_functions_response",
    "build_import_boundary_response",
    "build_module_architecture_response",
    "build_module_profile_response",
    "build_module_subsystem_response",
    "build_paginated_functions_response",
    "build_paginated_subsystems_response",
    "build_subsystem_coverage_response",
    "build_subsystem_modules_response",
    "build_subsystem_profile_response",
    "build_subsystem_summary_response",
    "build_tests_for_function_response",
]
