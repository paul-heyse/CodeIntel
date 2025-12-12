"""Canonical operation catalog unifying HTTP, MCP, and backend contracts.

This module provides a single source of truth for all serving operations,
providing a unified view of all serving operations with HTTP/MCP surface
metadata and data source contracts in a single definition.

It also provides dataflow graph building and dataset metadata utilities
for the serving layer.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import StrEnum
from itertools import chain
from typing import TYPE_CHECKING, Literal

from codeintel.config.datasets import (
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
)
from codeintel.config.datasets.dataflow import (
    DataflowEdge,
    DataflowNode,
    build_contract_dataflow_graph,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.config.datasets.dataflow import (
        NodeKind,
    )
    from codeintel.serving.backend import BackendLimits
    from codeintel.serving.mcp.models import DatasetSpecDescriptor
    from codeintel.serving.services.query_service import QueryService


class DataSourceType(StrEnum):
    """Classification of data sources for serving operations."""

    VIEW = "view"
    """Data is fetched from a docs.* or analytics.* view."""

    TABLE = "table"
    """Data is fetched directly from a core.* or analytics.* table."""

    GRAPH_ENGINE = "graph_engine"
    """Data is computed from the in-memory graph engine (NetworkX)."""

    COMPUTED = "computed"
    """Data is computed at runtime from multiple sources."""


@dataclass(frozen=True)
class Operation:
    """Canonical description of a serving operation across HTTP, MCP, and backend.

    This unified model provides HTTP/MCP surface metadata
    and data source contracts in a single source of truth.

    Parameters
    ----------
    id
        Unique operation identifier (e.g. "function.summary", "datasets.list").
    category
        Operation grouping for routing and tool registration
        (e.g. "functions", "datasets", "graph", "profiles").
    summary
        Short human-readable description for API docs and tool hints.
    description
        Detailed description explaining the operation's behavior and use cases.
    http_method
        HTTP method for the operation (GET or POST), or None if not HTTP-exposed.
    http_path
        URL path for the HTTP endpoint, or None if not HTTP-exposed.
    tool_name
        MCP tool name, or None if not exposed as an MCP tool.
    output_model_name
        Name of the Pydantic response model in serving.mcp.models.
    backend_method
        Method name on QueryService or backend that implements this operation.
    data_source
        Type of data source backing this operation.
    source_name
        Specific source name (e.g. "docs.v_function_summary", "analytics.goid_risk_factors").
    repository_method
        Repository method that fetches the raw data, if applicable.
    required_datasets
        Dataset table_keys that must exist for this operation to work.
    required_graphs
        Graph runtimes required (e.g. "callgraph", "importgraph").
    exposed_datasets
        Datasets exposed through this operation (for datasets.rows).
    supports_pagination
        Whether the operation supports limit/offset pagination.
    default_limit
        Default result limit when none specified, or None for unlimited.
    max_limit
        Maximum allowed limit, or None for unlimited.
    """

    id: str
    category: str

    summary: str
    description: str | None

    http_method: Literal["GET", "POST"] | None
    http_path: str | None

    tool_name: str | None
    output_model_name: str

    backend_method: str

    data_source: DataSourceType
    source_name: str | None
    repository_method: str | None

    required_datasets: tuple[str, ...]
    required_graphs: tuple[str, ...]
    exposed_datasets: tuple[str, ...] = ()

    supports_pagination: bool = False
    default_limit: int | None = None
    max_limit: int | None = None


_FUNCTION_SUMMARY = Operation(
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
    data_source=DataSourceType.VIEW,
    source_name="docs.v_function_summary",
    repository_method="FunctionRepository.get_function_summary_by_goid",
    required_datasets=(),
    required_graphs=("callgraph",),
    supports_pagination=False,
    default_limit=1,
    max_limit=1,
)

_FUNCTIONS_HIGH_RISK = Operation(
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
    data_source=DataSourceType.TABLE,
    source_name="analytics.goid_risk_factors",
    repository_method="FunctionRepository.list_high_risk_functions",
    required_datasets=(),
    required_graphs=(),
    supports_pagination=True,
    default_limit=None,
    max_limit=None,
)

_FUNCTIONS_TESTS = Operation(
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
    data_source=DataSourceType.VIEW,
    source_name="docs.v_tests_for_function",
    repository_method="TestRepository.list_tests_for_function",
    required_datasets=(),
    required_graphs=(),
    supports_pagination=True,
    default_limit=None,
    max_limit=None,
)


_GRAPH_CALL_NEIGHBORS = Operation(
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
    data_source=DataSourceType.TABLE,
    source_name="graph.call_graph_edges",
    repository_method="GraphRepository.list_call_graph_edges",
    required_datasets=("call_graph_nodes",),
    required_graphs=("callgraph",),
    supports_pagination=True,
    default_limit=None,
    max_limit=None,
)

_GRAPH_CALL_NEIGHBORHOOD = Operation(
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
    data_source=DataSourceType.GRAPH_ENGINE,
    source_name="GraphEngine.call_graph().ego_graph",
    repository_method=None,
    required_datasets=("call_graph_nodes",),
    required_graphs=("callgraph",),
    supports_pagination=True,
    default_limit=None,
    max_limit=None,
)

_GRAPH_IMPORT_BOUNDARY = Operation(
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
    data_source=DataSourceType.GRAPH_ENGINE,
    source_name="GraphEngine.import_graph().edges",
    repository_method=None,
    required_datasets=("import_graph_edges",),
    required_graphs=("importgraph",),
    supports_pagination=True,
    default_limit=None,
    max_limit=None,
)


_FILE_SUMMARY = Operation(
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
    data_source=DataSourceType.VIEW,
    source_name="docs.v_file_summary",
    repository_method="ModuleRepository.get_file_summary",
    required_datasets=(),
    required_graphs=(),
    supports_pagination=False,
    default_limit=1,
    max_limit=1,
)


_PROFILES_FUNCTION = Operation(
    id="profiles.function",
    category="profiles",
    summary="Get a function profile.",
    description="Return a rich profile for a single function identified by GOID.",
    http_method="GET",
    http_path="/profiles/function",
    tool_name="get_function_profile",
    output_model_name="FunctionProfileResponse",
    backend_method="get_function_profile",
    data_source=DataSourceType.TABLE,
    source_name="analytics.function_profile",
    repository_method="FunctionRepository.get_function_profile",
    required_datasets=(),
    required_graphs=("callgraph",),
    supports_pagination=False,
    default_limit=1,
    max_limit=1,
)

_PROFILES_FILE = Operation(
    id="profiles.file",
    category="profiles",
    summary="Get a file profile.",
    description="Return a profile rollup for a file and its functions.",
    http_method="GET",
    http_path="/profiles/file",
    tool_name="get_file_profile",
    output_model_name="FileProfileResponse",
    backend_method="get_file_profile",
    data_source=DataSourceType.VIEW,
    source_name="docs.v_file_profile",
    repository_method="ModuleRepository.get_file_profile",
    required_datasets=(),
    required_graphs=(),
    supports_pagination=False,
    default_limit=1,
    max_limit=1,
)

_PROFILES_MODULE = Operation(
    id="profiles.module",
    category="profiles",
    summary="Get a module profile.",
    description="Return metrics and rollups for a Python module.",
    http_method="GET",
    http_path="/profiles/module",
    tool_name="get_module_profile",
    output_model_name="ModuleProfileResponse",
    backend_method="get_module_profile",
    data_source=DataSourceType.TABLE,
    source_name="analytics.module_profile",
    repository_method="ModuleRepository.get_module_profile",
    required_datasets=(),
    required_graphs=(),
    supports_pagination=False,
    default_limit=1,
    max_limit=1,
)


_ARCHITECTURE_FUNCTION = Operation(
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
    data_source=DataSourceType.VIEW,
    source_name="docs.v_function_architecture",
    repository_method="FunctionRepository.get_function_architecture",
    required_datasets=(),
    required_graphs=("callgraph", "importgraph"),
    supports_pagination=False,
    default_limit=1,
    max_limit=1,
)

_ARCHITECTURE_MODULE = Operation(
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
    data_source=DataSourceType.VIEW,
    source_name="docs.v_module_architecture",
    repository_method="ModuleRepository.get_module_architecture",
    required_datasets=(),
    required_graphs=("callgraph", "importgraph"),
    supports_pagination=False,
    default_limit=1,
    max_limit=1,
)


_SUBSYSTEMS_LIST = Operation(
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
    data_source=DataSourceType.VIEW,
    source_name="docs.v_subsystem_summary",
    repository_method="SubsystemRepository.list_subsystems",
    required_datasets=("docs.v_subsystem_summary",),
    required_graphs=(),
    supports_pagination=True,
    default_limit=None,
    max_limit=None,
)

_SUBSYSTEMS_PROFILES = Operation(
    id="subsystems.profiles",
    category="subsystems",
    summary="List subsystem profiles.",
    description=(
        "List profile rows that aggregate metrics per subsystem such as size, risk, and ownership."
    ),
    http_method="GET",
    http_path="/architecture/subsystem-profiles",
    tool_name=None,
    output_model_name="SubsystemProfileResponse",
    backend_method="list_subsystem_profiles",
    data_source=DataSourceType.VIEW,
    source_name="docs.v_subsystem_profile",
    repository_method="SubsystemRepository.list_subsystem_profiles",
    required_datasets=("docs.v_subsystem_profile",),
    required_graphs=(),
    supports_pagination=True,
    default_limit=None,
    max_limit=None,
)

_SUBSYSTEMS_COVERAGE = Operation(
    id="subsystems.coverage",
    category="subsystems",
    summary="List subsystem coverage rollups.",
    description="Summarize coverage metrics for each subsystem.",
    http_method="GET",
    http_path="/architecture/subsystem-coverage",
    tool_name=None,
    output_model_name="SubsystemCoverageResponse",
    backend_method="list_subsystem_coverage",
    data_source=DataSourceType.VIEW,
    source_name="docs.v_subsystem_coverage",
    repository_method="SubsystemRepository.list_subsystem_coverage",
    required_datasets=("docs.v_subsystem_coverage",),
    required_graphs=(),
    supports_pagination=True,
    default_limit=None,
    max_limit=None,
)

_SUBSYSTEMS_MODULE_MEMBERSHIPS = Operation(
    id="subsystems.module_memberships",
    category="subsystems",
    summary="List subsystem memberships for a module.",
    description="Return which subsystems a given module belongs to.",
    http_method="GET",
    http_path="/architecture/module-subsystems",
    tool_name="get_module_subsystems",
    output_model_name="ModuleSubsystemResponse",
    backend_method="get_module_subsystems",
    data_source=DataSourceType.TABLE,
    source_name="analytics.subsystem_modules",
    repository_method="SubsystemRepository.list_subsystems_for_module",
    required_datasets=("docs.v_module_with_subsystem",),
    required_graphs=(),
    supports_pagination=False,
    default_limit=None,
    max_limit=None,
)

_SUBSYSTEMS_DETAIL = Operation(
    id="subsystems.detail",
    category="subsystems",
    summary="Get modules and detail for a subsystem.",
    description="Return detailed membership and metrics for a single subsystem.",
    http_method="GET",
    http_path="/architecture/subsystem",
    tool_name="get_subsystem_modules",
    output_model_name="SubsystemModulesResponse",
    backend_method="get_subsystem_modules",
    data_source=DataSourceType.VIEW,
    source_name="docs.v_subsystem_modules",
    repository_method="SubsystemRepository.list_subsystem_modules",
    required_datasets=("docs.v_subsystem_profile",),
    required_graphs=(),
    supports_pagination=True,
    default_limit=None,
    max_limit=None,
)

_SUBSYSTEMS_SEARCH = Operation(
    id="subsystems.search",
    category="subsystems",
    summary="Search subsystems by name or role.",
    description="Search subsystems by label or role; currently MCP-only.",
    http_method=None,
    http_path=None,
    tool_name="search_subsystems",
    output_model_name="SubsystemSearchResponse",
    backend_method="search_subsystems",
    data_source=DataSourceType.VIEW,
    source_name="docs.v_subsystem_summary",
    repository_method="SubsystemRepository.list_subsystems",
    required_datasets=("docs.v_subsystem_summary",),
    required_graphs=(),
    supports_pagination=True,
    default_limit=None,
    max_limit=None,
)

_SUBSYSTEMS_SUMMARIZE = Operation(
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
    data_source=DataSourceType.VIEW,
    source_name="docs.v_subsystem_profile",
    repository_method="SubsystemRepository.list_subsystem_modules",
    required_datasets=("docs.v_subsystem_profile",),
    required_graphs=(),
    supports_pagination=True,
    default_limit=None,
    max_limit=None,
)


_IDE_HINTS = Operation(
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
    data_source=DataSourceType.VIEW,
    source_name="docs.v_ide_hints",
    repository_method="ModuleRepository.get_file_hints",
    required_datasets=(),
    required_graphs=(),
    supports_pagination=False,
    default_limit=1,
    max_limit=1,
)


_DATASETS_LIST = Operation(
    id="datasets.list",
    category="datasets",
    summary="List datasets available through the backend.",
    description="List datasets from the registry with basic metadata.",
    http_method="GET",
    http_path="/datasets",
    tool_name="list_datasets",
    output_model_name="DatasetDescriptor",
    backend_method="list_datasets",
    data_source=DataSourceType.COMPUTED,
    source_name="dataset_registry",
    repository_method=None,
    required_datasets=(),
    required_graphs=(),
    supports_pagination=False,
    default_limit=None,
    max_limit=None,
)

_DATASETS_SPECS = Operation(
    id="datasets.specs",
    category="datasets",
    summary="Expose dataset contract and registry metadata.",
    description="Return DatasetSpecDescriptor entries describing dataset contracts.",
    http_method="GET",
    http_path="/datasets/specs",
    tool_name="dataset_specs",
    output_model_name="DatasetSpecDescriptor",
    backend_method="dataset_specs",
    data_source=DataSourceType.COMPUTED,
    source_name="dataset_registry",
    repository_method=None,
    required_datasets=(),
    required_graphs=(),
    supports_pagination=False,
    default_limit=None,
    max_limit=None,
)

_DATASETS_ROWS = Operation(
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
    data_source=DataSourceType.TABLE,
    source_name="<dynamic>",
    repository_method="DatasetReadRepository.read_rows",
    required_datasets=(),
    required_graphs=(),
    supports_pagination=True,
    default_limit=None,
    max_limit=None,
)

_DATASETS_SCHEMA = Operation(
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
    data_source=DataSourceType.COMPUTED,
    source_name="information_schema.columns",
    repository_method=None,
    required_datasets=(),
    required_graphs=(),
    supports_pagination=False,
    default_limit=None,
    max_limit=None,
)


_GRAPH_PLUGINS_PLAN = Operation(
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
    data_source=DataSourceType.COMPUTED,
    source_name=None,
    repository_method=None,
    required_datasets=(),
    required_graphs=(),
    supports_pagination=False,
    default_limit=None,
    max_limit=None,
)


_HEALTH_STATUS = Operation(
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
    data_source=DataSourceType.COMPUTED,
    source_name=None,
    repository_method=None,
    required_datasets=(),
    required_graphs=(),
    supports_pagination=False,
    default_limit=None,
    max_limit=None,
)


OPERATIONS_BY_ID: dict[str, Operation] = {
    op.id: op
    for op in (
        _FUNCTION_SUMMARY,
        _FUNCTIONS_HIGH_RISK,
        _FUNCTIONS_TESTS,
        _GRAPH_CALL_NEIGHBORS,
        _GRAPH_CALL_NEIGHBORHOOD,
        _GRAPH_IMPORT_BOUNDARY,
        _FILE_SUMMARY,
        _PROFILES_FUNCTION,
        _PROFILES_FILE,
        _PROFILES_MODULE,
        _ARCHITECTURE_FUNCTION,
        _ARCHITECTURE_MODULE,
        _SUBSYSTEMS_LIST,
        _SUBSYSTEMS_PROFILES,
        _SUBSYSTEMS_COVERAGE,
        _SUBSYSTEMS_MODULE_MEMBERSHIPS,
        _SUBSYSTEMS_DETAIL,
        _SUBSYSTEMS_SEARCH,
        _SUBSYSTEMS_SUMMARIZE,
        _IDE_HINTS,
        _DATASETS_LIST,
        _DATASETS_SPECS,
        _DATASETS_ROWS,
        _DATASETS_SCHEMA,
        _GRAPH_PLUGINS_PLAN,
        _HEALTH_STATUS,
    )
}


_TEST_OPERATIONS: dict[str, Operation] = {}


def get_operation(op_id: str) -> Operation | None:
    """Look up a single operation by id.

    Check both the canonical catalog and the test operation registry.

    Parameters
    ----------
    op_id
        Operation identifier to look up.

    Returns
    -------
    Operation | None
        The operation if found, otherwise None.
    """
    if op_id in _TEST_OPERATIONS:
        return _TEST_OPERATIONS[op_id]
    return OPERATIONS_BY_ID.get(op_id)


def register_test_operation(operation: Operation) -> None:
    """Register a test operation in the test registry.

    Use this to register synthetic operations for CLI or integration tests.
    The registered operation will be discoverable via `get_operation()`.

    Parameters
    ----------
    operation
        Operation instance to register. Its `id` field is used as the key.

    Notes
    -----
    Test operations are stored in a separate registry and do not pollute
    the canonical operation catalog. Always call `unregister_test_operation()`
    to clean up after tests.
    """
    _TEST_OPERATIONS[operation.id] = operation


def unregister_test_operation(op_id: str) -> bool:
    """Remove a test operation from the test registry.

    Parameters
    ----------
    op_id
        Operation identifier to remove.

    Returns
    -------
    bool
        True if the operation was found and removed, False otherwise.
    """
    if op_id in _TEST_OPERATIONS:
        del _TEST_OPERATIONS[op_id]
        return True
    return False


def clear_test_operations() -> int:
    """Remove all test operations from the test registry.

    Returns
    -------
    int
        Number of test operations that were removed.
    """
    count = len(_TEST_OPERATIONS)
    _TEST_OPERATIONS.clear()
    return count


def iter_operations() -> Iterable[Operation]:
    """Iterate over all registered operations.

    Include both canonical and test operations.

    Returns
    -------
    Iterable[Operation]
        All operations in the catalog and test registry.
    """
    combined = {**OPERATIONS_BY_ID, **_TEST_OPERATIONS}
    return combined.values()


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
    owner: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    validation_profile: str | None = None


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
    dataset_specs = getattr(service, "dataset_specs", None)
    if dataset_specs is None:
        return []

    specs: list[DatasetSpecDescriptor] = dataset_specs()
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
                owner=spec.owner,
                freshness_sla=spec.freshness_sla,
                retention_policy=spec.retention_policy,
                validation_profile=spec.validation_profile,
            )
        )

    return metas


def _resolve_dataset_identifier(identifier: str) -> str | None:
    """Resolve a dataset identifier used in Operation into a canonical table_key.

    Returns
    -------
    str | None
        Canonical table_key when found, otherwise None.
    """
    contract = DATASET_CONTRACTS.get(identifier)
    if contract is not None:
        return contract.table_key

    contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(identifier)
    if contract is not None:
        return contract.table_key

    return None


def _build_operations_with_patches() -> dict[str, Operation]:
    """Build operations from the canonical catalog with dynamic patching.

    Patches datasets.rows with exposed_datasets from DATASET_CONTRACTS_BY_TABLE_KEY.

    Returns
    -------
    dict[str, Operation]
        Mapping from operation ID to Operation.
    """
    operations: dict[str, Operation] = {}
    exposed_datasets_keys = tuple(DATASET_CONTRACTS_BY_TABLE_KEY.keys())

    for operation in iter_operations():
        if operation.id == "datasets.rows":
            patched_op = dataclasses.replace(operation, exposed_datasets=exposed_datasets_keys)
            operations[patched_op.id] = patched_op
        else:
            operations[operation.id] = operation

    return operations


_PATCHED_OPERATIONS: dict[str, Operation] = _build_operations_with_patches()


def iter_registry_operations() -> list[Operation]:
    """
    Return all registered Operation instances with dynamic patching.

    Returns
    -------
    list[Operation]
        Operations in the registry with exposed_datasets patched.
    """
    return list(_PATCHED_OPERATIONS.values())


def get_registry_operation(op_id: str) -> Operation | None:
    """
    Return a single Operation by id with dynamic patching, or None when unknown.

    Parameters
    ----------
    op_id
        Operation identifier to look up.

    Returns
    -------
    Operation | None
        Matching operation when present.
    """
    return _PATCHED_OPERATIONS.get(op_id)


def iter_operation_nodes() -> list[DataflowNode]:
    """Return DataflowNode entries for all serving operations.

    Returns
    -------
    list[DataflowNode]
        Operation nodes keyed by Operation.id.
    """
    return [
        DataflowNode(
            id=op.id,
            kind="operation",
            family="serving",
            owner_package=None,
            description=op.summary,
        )
        for op in _PATCHED_OPERATIONS.values()
    ]


def iter_graph_nodes() -> list[DataflowNode]:
    """Return DataflowNode entries for logical graph runtimes.

    Returns
    -------
    list[DataflowNode]
        Graph nodes keyed as graph.<name> for required Operation graphs.
    """
    names: set[str] = set()
    for op in _PATCHED_OPERATIONS.values():
        for graph_name in op.required_graphs:
            names.add(graph_name)

    return [
        DataflowNode(
            id=f"graph.{graph_name}",
            kind="graph",
            family="graph",
            owner_package="graphs",
            description=f"Logical {graph_name} graph runtime",
        )
        for graph_name in sorted(names)
    ]


def iter_operation_dataset_edges() -> list[DataflowEdge]:
    """Build edges from datasets to operations based on required/exposed datasets.

    Returns
    -------
    list[DataflowEdge]
        Reads and exposes edges from datasets/views to operations.
    """
    edges: list[DataflowEdge] = []

    for op in _PATCHED_OPERATIONS.values():
        edges.extend(
            DataflowEdge(src=table_key, dst=op.id, edge_type="reads")
            for table_key in (
                _resolve_dataset_identifier(ds_identifier) for ds_identifier in op.required_datasets
            )
            if table_key is not None
        )
        edges.extend(
            DataflowEdge(src=table_key, dst=op.id, edge_type="exposes")
            for table_key in (
                _resolve_dataset_identifier(ds_identifier) for ds_identifier in op.exposed_datasets
            )
            if table_key is not None
        )

    return edges


def iter_operation_graph_edges() -> list[DataflowEdge]:
    """Build edges from logical graph runtimes to operations (depends_on).

    Returns
    -------
    list[DataflowEdge]
        Edges indicating graph dependencies for each operation.
    """
    return [
        DataflowEdge(
            src=f"graph.{graph_name}",
            dst=op.id,
            edge_type="depends_on",
        )
        for op in _PATCHED_OPERATIONS.values()
        for graph_name in op.required_graphs
    ]


def build_serving_dataflow_graph() -> tuple[list[DataflowNode], list[DataflowEdge]]:
    """Build a combined dataflow graph for datasets/docs/views, operations, and graphs.

    Returns
    -------
    tuple[list[DataflowNode], list[DataflowEdge]]
        Nodes and deduplicated edges across datasets, operations, and graph runtimes.
    """
    ds_nodes, ds_edges = build_contract_dataflow_graph()
    op_nodes = iter_operation_nodes()
    graph_nodes = iter_graph_nodes()

    op_ds_edges = iter_operation_dataset_edges()
    op_graph_edges = iter_operation_graph_edges()

    node_map: dict[tuple[str, NodeKind], DataflowNode] = {}
    for node in chain(ds_nodes, op_nodes, graph_nodes):
        node_map[node.id, node.kind] = node

    nodes = list(node_map.values())

    seen_edges: set[tuple[str, str, str]] = set()
    edges: list[DataflowEdge] = []
    for edge in chain(ds_edges, op_ds_edges, op_graph_edges):
        key = (edge.src, edge.dst, edge.edge_type)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append(edge)

    return nodes, edges


__all__ = [
    "OPERATIONS_BY_ID",
    "DataSourceType",
    "DatasetMeta",
    "Operation",
    "build_dataset_meta",
    "build_serving_dataflow_graph",
    "clear_test_operations",
    "get_operation",
    "get_registry_operation",
    "iter_graph_nodes",
    "iter_operation_dataset_edges",
    "iter_operation_graph_edges",
    "iter_operation_nodes",
    "iter_operations",
    "iter_registry_operations",
    "register_test_operation",
    "unregister_test_operation",
]
