"""Unified registry for serving datasets and operations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.models import DatasetSpecDescriptor
from codeintel.serving.services.query_service import QueryService


@dataclass(frozen=True)
class DatasetMeta:
    """Dataset metadata enriched with serving limits and flags."""

    id: str
    name: str
    table_key: str
    description: str
    schema_version: str | None
    family: str | None
    is_docs_view: bool
    is_read_only: bool
    default_limit: int
    max_limit: int


@dataclass(frozen=True)
class OperationSpec:
    """Cross-transport description of a single serving operation."""

    id: str
    category: str
    summary: str
    description: str | None
    http_method: Literal["GET", "POST"] | None
    http_path: str | None
    tool_name: str | None
    output_model_name: str
    backend_method: str
    required_datasets: Sequence[str]
    required_graphs: Sequence[str]
    default_limit: int | None
    max_limit: int | None


def build_dataset_meta(service: QueryService, limits: BackendLimits) -> list[DatasetMeta]:
    """
    Build dataset metadata entries using dataset_specs and serving limits.

    Parameters
    ----------
    service
        QueryService instance (local or HTTP).
    limits
        Backend limits derived from ServingConfig.

    Returns
    -------
    list[DatasetMeta]
        One entry per dataset in the registry.
    """
    specs: list[DatasetSpecDescriptor] = service.dataset_specs()
    metas: list[DatasetMeta] = []

    for spec in specs:
        family = getattr(spec, "family", None)
        is_docs_view = bool(family == "docs" or spec.table_key.startswith("docs."))
        capabilities = getattr(spec, "capabilities", {}) or {}
        is_read_only = bool(capabilities.get("read_only", False))
        description = spec.description or f"{spec.name} ({spec.table_key})"
        metas.append(
            DatasetMeta(
                id=spec.name,
                name=spec.name,
                table_key=spec.table_key,
                description=description,
                schema_version=spec.schema_version,
                family=family,
                is_docs_view=is_docs_view,
                is_read_only=is_read_only,
                default_limit=limits.default_limit,
                max_limit=limits.max_rows_per_call,
            )
        )

    return metas


_OPERATION_SPECS: dict[str, OperationSpec] = {
    "function.summary": OperationSpec(
        id="function.summary",
        category="functions",
        summary="Summarize a function by GOID, URN, or source location.",
        description=(
            "Summarize a function using Docs and analytics views, identified by GOID, URN, "
            "qualified name, or file path."
        ),
        http_method="GET",
        http_path="/function/summary",
        tool_name="get_function_summary",
        output_model_name="FunctionSummaryResponse",
        backend_method="get_function_summary",
        required_datasets=(),
        required_graphs=("callgraph",),
        default_limit=1,
        max_limit=1,
    ),
    "functions.high_risk": OperationSpec(
        id="functions.high_risk",
        category="functions",
        summary="List high-risk functions, optionally restricted to tested ones.",
        description=(
            "Rank functions by risk using analytics and docs views with optional thresholds "
            "and tested-only filters."
        ),
        http_method="GET",
        http_path="/functions/high-risk",
        tool_name="list_high_risk_functions",
        output_model_name="HighRiskFunctionsResponse",
        backend_method="list_high_risk_functions",
        required_datasets=(),
        required_graphs=(),
        default_limit=None,
        max_limit=None,
    ),
    "functions.tests": OperationSpec(
        id="functions.tests",
        category="functions",
        summary="List tests that exercise a specific function.",
        description=(
            "Return tests linked to a function via coverage and mapping tables to understand "
            "blast radius."
        ),
        http_method="GET",
        http_path="/function/tests",
        tool_name="get_tests_for_function",
        output_model_name="TestsForFunctionResponse",
        backend_method="get_tests_for_function",
        required_datasets=(),
        required_graphs=(),
        default_limit=None,
        max_limit=None,
    ),
    "graph.call_neighbors": OperationSpec(
        id="graph.call_neighbors",
        category="graph",
        summary="Get call graph neighbors for a function.",
        description=(
            "Return incoming and outgoing neighbors in the call graph for a single function "
            "with optional direction and limit control."
        ),
        http_method="GET",
        http_path="/function/callgraph",
        tool_name="get_callgraph_neighbors",
        output_model_name="CallGraphNeighborsResponse",
        backend_method="get_callgraph_neighbors",
        required_datasets=("call_graph_nodes",),
        required_graphs=("callgraph",),
        default_limit=None,
        max_limit=None,
    ),
    "graph.call_neighborhood": OperationSpec(
        id="graph.call_neighborhood",
        category="graph",
        summary="Compute a bounded ego neighborhood in the call graph.",
        description=(
            "Return nodes and edges in a radius-bounded ego neighborhood around a function "
            "in the call graph."
        ),
        http_method="GET",
        http_path="/graph/call/neighborhood",
        tool_name="get_callgraph_neighborhood",
        output_model_name="GraphNeighborhoodResponse",
        backend_method="get_callgraph_neighborhood",
        required_datasets=("call_graph_nodes",),
        required_graphs=("callgraph",),
        default_limit=None,
        max_limit=None,
    ),
    "graph.import_boundary": OperationSpec(
        id="graph.import_boundary",
        category="graph",
        summary="List import graph edges crossing a subsystem boundary.",
        description=(
            "Return edges in the import graph that cross the boundary of a subsystem for "
            "dependency analysis."
        ),
        http_method="GET",
        http_path="/graph/import/boundary",
        tool_name="get_import_boundary",
        output_model_name="ImportBoundaryResponse",
        backend_method="get_import_boundary",
        required_datasets=("import_graph_edges",),
        required_graphs=("importgraph",),
        default_limit=None,
        max_limit=None,
    ),
    "file.summary": OperationSpec(
        id="file.summary",
        category="files",
        summary="Get a file summary with function details.",
        description=(
            "Return file-level metrics plus nested function summaries for all functions "
            "defined in a file."
        ),
        http_method="GET",
        http_path="/file/summary",
        tool_name="get_file_summary",
        output_model_name="FileSummaryResponse",
        backend_method="get_file_summary",
        required_datasets=(),
        required_graphs=(),
        default_limit=1,
        max_limit=1,
    ),
    "profiles.function": OperationSpec(
        id="profiles.function",
        category="profiles",
        summary="Get a function profile.",
        description="Return a rich profile for a single function identified by GOID.",
        http_method="GET",
        http_path="/profiles/function",
        tool_name="get_function_profile",
        output_model_name="FunctionProfileResponse",
        backend_method="get_function_profile",
        required_datasets=(),
        required_graphs=("callgraph",),
        default_limit=1,
        max_limit=1,
    ),
    "profiles.file": OperationSpec(
        id="profiles.file",
        category="profiles",
        summary="Get a file profile.",
        description="Return a profile rollup for a file and its functions.",
        http_method="GET",
        http_path="/profiles/file",
        tool_name="get_file_profile",
        output_model_name="FileProfileResponse",
        backend_method="get_file_profile",
        required_datasets=(),
        required_graphs=(),
        default_limit=1,
        max_limit=1,
    ),
    "profiles.module": OperationSpec(
        id="profiles.module",
        category="profiles",
        summary="Get a module profile.",
        description="Return metrics and rollups for a Python module.",
        http_method="GET",
        http_path="/profiles/module",
        tool_name="get_module_profile",
        output_model_name="ModuleProfileResponse",
        backend_method="get_module_profile",
        required_datasets=(),
        required_graphs=(),
        default_limit=1,
        max_limit=1,
    ),
    "architecture.function": OperationSpec(
        id="architecture.function",
        category="architecture",
        summary="Get architecture metrics for a function.",
        description=(
            "Return architecture metrics for a function including fan-in, fan-out, and "
            "layer violations."
        ),
        http_method="GET",
        http_path="/architecture/function",
        tool_name="get_function_architecture",
        output_model_name="FunctionArchitectureResponse",
        backend_method="get_function_architecture",
        required_datasets=(),
        required_graphs=("callgraph", "importgraph"),
        default_limit=1,
        max_limit=1,
    ),
    "architecture.module": OperationSpec(
        id="architecture.module",
        category="architecture",
        summary="Get architecture metrics for a module.",
        description=(
            "Return module-level architecture metrics including dependencies, subsystems, "
            "and cross-layer violations."
        ),
        http_method="GET",
        http_path="/architecture/module",
        tool_name="get_module_architecture",
        output_model_name="ModuleArchitectureResponse",
        backend_method="get_module_architecture",
        required_datasets=(),
        required_graphs=("callgraph", "importgraph"),
        default_limit=1,
        max_limit=1,
    ),
    "subsystems.list": OperationSpec(
        id="subsystems.list",
        category="subsystems",
        summary="List inferred subsystems.",
        description=(
            "List inferred subsystems with optional filtering by role or search term backed "
            "by docs views."
        ),
        http_method="GET",
        http_path="/architecture/subsystems",
        tool_name="list_subsystems",
        output_model_name="SubsystemSummaryResponse",
        backend_method="list_subsystems",
        required_datasets=("docs.v_subsystem_summary",),
        required_graphs=(),
        default_limit=None,
        max_limit=None,
    ),
    "subsystems.profiles": OperationSpec(
        id="subsystems.profiles",
        category="subsystems",
        summary="List subsystem profiles.",
        description=(
            "List profile rows that aggregate metrics per subsystem such as size, risk, "
            "and ownership."
        ),
        http_method="GET",
        http_path="/architecture/subsystem-profiles",
        tool_name=None,
        output_model_name="SubsystemProfileResponse",
        backend_method="list_subsystem_profiles",
        required_datasets=("docs.v_subsystem_profile",),
        required_graphs=(),
        default_limit=None,
        max_limit=None,
    ),
    "subsystems.coverage": OperationSpec(
        id="subsystems.coverage",
        category="subsystems",
        summary="List subsystem coverage rollups.",
        description="Summarize coverage metrics for each subsystem.",
        http_method="GET",
        http_path="/architecture/subsystem-coverage",
        tool_name=None,
        output_model_name="SubsystemCoverageResponse",
        backend_method="list_subsystem_coverage",
        required_datasets=("docs.v_subsystem_coverage",),
        required_graphs=(),
        default_limit=None,
        max_limit=None,
    ),
    "subsystems.module_memberships": OperationSpec(
        id="subsystems.module_memberships",
        category="subsystems",
        summary="List subsystem memberships for a module.",
        description="Return which subsystems a given module belongs to.",
        http_method="GET",
        http_path="/architecture/module-subsystems",
        tool_name="get_module_subsystems",
        output_model_name="ModuleSubsystemResponse",
        backend_method="get_module_subsystems",
        required_datasets=("docs.v_subsystem_memberships",),
        required_graphs=(),
        default_limit=None,
        max_limit=None,
    ),
    "subsystems.detail": OperationSpec(
        id="subsystems.detail",
        category="subsystems",
        summary="Get modules and detail for a subsystem.",
        description="Return detailed membership and metrics for a single subsystem.",
        http_method="GET",
        http_path="/architecture/subsystem",
        tool_name="get_subsystem_modules",
        output_model_name="SubsystemModulesResponse",
        backend_method="get_subsystem_modules",
        required_datasets=("docs.v_subsystem_profile",),
        required_graphs=(),
        default_limit=None,
        max_limit=None,
    ),
    "subsystems.search": OperationSpec(
        id="subsystems.search",
        category="subsystems",
        summary="Search subsystems by name or role.",
        description="Search subsystems by label or role; currently MCP-only.",
        http_method=None,
        http_path=None,
        tool_name="search_subsystems",
        output_model_name="SubsystemSearchResponse",
        backend_method="search_subsystems",
        required_datasets=("docs.v_subsystem_summary",),
        required_graphs=(),
        default_limit=None,
        max_limit=None,
    ),
    "subsystems.summarize": OperationSpec(
        id="subsystems.summarize",
        category="subsystems",
        summary="Summarize a subsystem with module details.",
        description=(
            "Summarize a subsystem and return its member modules, limited by an optional "
            "module limit parameter."
        ),
        http_method=None,
        http_path=None,
        tool_name="summarize_subsystem",
        output_model_name="SubsystemModulesResponse",
        backend_method="summarize_subsystem",
        required_datasets=("docs.v_subsystem_profile",),
        required_graphs=(),
        default_limit=None,
        max_limit=None,
    ),
    "ide.hints": OperationSpec(
        id="ide.hints",
        category="ide",
        summary="Get IDE hints for a file.",
        description=(
            "Return IDE-friendly hints for a file such as hotspots, missing tests, and "
            "subsystem membership hints."
        ),
        http_method="GET",
        http_path="/ide/hints",
        tool_name="get_file_hints",
        output_model_name="FileHintsResponse",
        backend_method="get_file_hints",
        required_datasets=(),
        required_graphs=(),
        default_limit=1,
        max_limit=1,
    ),
    "datasets.list": OperationSpec(
        id="datasets.list",
        category="datasets",
        summary="List datasets available through the backend.",
        description="List datasets from the registry with basic metadata.",
        http_method="GET",
        http_path="/datasets",
        tool_name="list_datasets",
        output_model_name="DatasetDescriptor",
        backend_method="list_datasets",
        required_datasets=(),
        required_graphs=(),
        default_limit=None,
        max_limit=None,
    ),
    "datasets.specs": OperationSpec(
        id="datasets.specs",
        category="datasets",
        summary="Expose dataset contract and registry metadata.",
        description="Return DatasetSpecDescriptor entries describing dataset contracts.",
        http_method="GET",
        http_path="/datasets/specs",
        tool_name="dataset_specs",
        output_model_name="DatasetSpecDescriptor",
        backend_method="dataset_specs",
        required_datasets=(),
        required_graphs=(),
        default_limit=None,
        max_limit=None,
    ),
    "datasets.rows": OperationSpec(
        id="datasets.rows",
        category="datasets",
        summary="Read rows from a dataset with limits and messaging.",
        description=(
            "Read rows from a named dataset, applying BackendLimits and returning clamping "
            "and truncation details."
        ),
        http_method="GET",
        http_path="/datasets/{dataset_name}",
        tool_name="read_dataset_rows",
        output_model_name="DatasetRowsResponse",
        backend_method="read_dataset_rows",
        required_datasets=(),
        required_graphs=(),
        default_limit=None,
        max_limit=None,
    ),
    "datasets.schema": OperationSpec(
        id="datasets.schema",
        category="datasets",
        summary="Describe dataset schema and sample rows.",
        description=(
            "Return a composite schema description for a dataset combining DuckDB catalog, "
            "JSON Schema, and sample rows."
        ),
        http_method="GET",
        http_path="/datasets/{dataset_name}/schema",
        tool_name="dataset_schema",
        output_model_name="DatasetSchemaResponse",
        backend_method="dataset_schema",
        required_datasets=(),
        required_graphs=(),
        default_limit=None,
        max_limit=None,
    ),
    "graph.plugins.plan": OperationSpec(
        id="graph.plugins.plan",
        category="graph_plugins",
        summary="Compute graph metric plugin execution plan.",
        description=(
            "Compute a graph plugin execution plan with ordering, dependencies, and skips "
            "using the configured analytics graph plugin registry."
        ),
        http_method=None,
        http_path=None,
        tool_name="graph_plugin_plan",
        output_model_name="GraphPlanResponse",
        backend_method="graph_plugin_plan",
        required_datasets=(),
        required_graphs=(),
        default_limit=None,
        max_limit=None,
    ),
    "health.status": OperationSpec(
        id="health.status",
        category="health",
        summary="Health check for CodeIntel API.",
        description=(
            "Return a health payload including status, repo, commit, read-only flag, and "
            "optional limits."
        ),
        http_method="GET",
        http_path="/health",
        tool_name=None,
        output_model_name="HealthPayload",
        backend_method="health",
        required_datasets=(),
        required_graphs=(),
        default_limit=None,
        max_limit=None,
    ),
}


def iter_operation_specs() -> list[OperationSpec]:
    """
    Return all registered OperationSpec instances.

    Returns
    -------
    list[OperationSpec]
        Operation specifications defined in the registry.
    """
    return list(_OPERATION_SPECS.values())


def get_operation_spec(op_id: str) -> OperationSpec | None:
    """
    Return a single OperationSpec by id, or None when unknown.

    Parameters
    ----------
    op_id
        Operation identifier to look up.

    Returns
    -------
    OperationSpec | None
        Matching specification when present.
    """
    return _OPERATION_SPECS.get(op_id)


__all__ = [
    "DatasetMeta",
    "OperationSpec",
    "build_dataset_meta",
    "get_operation_spec",
    "iter_operation_specs",
]
