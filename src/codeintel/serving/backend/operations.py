"""Operation contracts registry documenting data sources for serving operations.

This module provides a declarative registry of all serving operations and their
data source contracts. Each operation declares:
- Where its data comes from (view, table, graph_engine, computed)
- What the source name is (e.g., "docs.v_function_architecture")
- Whether it supports pagination

This serves two purposes:
1. Documentation: Makes the data flow explicit and traceable
2. Validation: Tests can verify that implementations follow their contracts
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


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
class OperationContract:
    """
    Declares the data source and transformation contract for an operation.

    This contract makes explicit where each operation's data comes from,
    enabling:
    - Clear documentation of data flow
    - Test validation that implementations follow contracts
    - Easier debugging when data is missing or incorrect

    Parameters
    ----------
    name
        Unique operation identifier (e.g., "function.architecture").
    data_source
        Type of data source (view, table, graph_engine, computed).
    source_name
        Specific source name (e.g., "docs.v_function_architecture").
    supports_pagination
        Whether the operation supports limit/offset pagination.
    description
        Human-readable description of what the operation does.
    repository_method
        Name of the repository method that fetches the data (if applicable).
    """

    name: str
    data_source: DataSourceType
    source_name: str
    supports_pagination: bool = False
    description: str = ""
    repository_method: str | None = None


# =============================================================================
# Function Operations
# =============================================================================

FUNCTION_SUMMARY = OperationContract(
    name="function.summary",
    data_source=DataSourceType.VIEW,
    source_name="docs.v_function_summary",
    supports_pagination=False,
    description="Fetch function summary metadata by GOID or URN.",
    repository_method="FunctionRepository.get_function_summary_by_goid",
)

FUNCTION_PROFILE = OperationContract(
    name="function.profile",
    data_source=DataSourceType.TABLE,
    source_name="analytics.function_profile",
    supports_pagination=False,
    description="Fetch detailed function profile with risk and purity analysis.",
    repository_method="FunctionRepository.get_function_profile",
)

FUNCTION_ARCHITECTURE = OperationContract(
    name="function.architecture",
    data_source=DataSourceType.VIEW,
    source_name="docs.v_function_architecture",
    supports_pagination=False,
    description="Fetch function architecture metrics including graph centrality measures.",
    repository_method="FunctionRepository.get_function_architecture",
)

HIGH_RISK_FUNCTIONS = OperationContract(
    name="function.high_risk",
    data_source=DataSourceType.TABLE,
    source_name="analytics.goid_risk_factors",
    supports_pagination=True,
    description="List functions with risk score above threshold.",
    repository_method="FunctionRepository.list_high_risk_functions",
)

TESTS_FOR_FUNCTION = OperationContract(
    name="function.tests",
    data_source=DataSourceType.VIEW,
    source_name="docs.v_tests_for_function",
    supports_pagination=True,
    description="List tests covering a specific function.",
    repository_method="TestRepository.list_tests_for_function",
)

# =============================================================================
# Call Graph Operations
# =============================================================================

CALLGRAPH_NEIGHBORS = OperationContract(
    name="callgraph.neighbors",
    data_source=DataSourceType.TABLE,
    source_name="graph.call_graph_edges",
    supports_pagination=True,
    description="Fetch direct callers and callees for a function.",
    repository_method="GraphRepository.list_call_graph_edges",
)

CALLGRAPH_NEIGHBORHOOD = OperationContract(
    name="callgraph.neighborhood",
    data_source=DataSourceType.GRAPH_ENGINE,
    source_name="GraphEngine.call_graph().ego_graph",
    supports_pagination=True,
    description="Compute bounded ego neighborhood in the call graph.",
    repository_method=None,
)

IMPORT_BOUNDARY = OperationContract(
    name="graph.import_boundary",
    data_source=DataSourceType.GRAPH_ENGINE,
    source_name="GraphEngine.import_graph().edges",
    supports_pagination=True,
    description="Fetch import edges crossing a subsystem boundary.",
    repository_method=None,
)

# =============================================================================
# File/Module Operations
# =============================================================================

FILE_SUMMARY = OperationContract(
    name="file.summary",
    data_source=DataSourceType.VIEW,
    source_name="docs.v_file_summary",
    supports_pagination=False,
    description="Fetch file summary with aggregate metrics.",
    repository_method="ModuleRepository.get_file_summary",
)

FILE_PROFILE = OperationContract(
    name="file.profile",
    data_source=DataSourceType.VIEW,
    source_name="docs.v_file_profile",
    supports_pagination=False,
    description="Fetch detailed file profile.",
    repository_method="ModuleRepository.get_file_profile",
)

FILE_HINTS = OperationContract(
    name="file.hints",
    data_source=DataSourceType.VIEW,
    source_name="docs.v_ide_hints",
    supports_pagination=False,
    description="Fetch IDE-focused hints for a file.",
    repository_method="ModuleRepository.get_file_hints",
)

MODULE_PROFILE = OperationContract(
    name="module.profile",
    data_source=DataSourceType.TABLE,
    source_name="analytics.module_profile",
    supports_pagination=False,
    description="Fetch module profile with risk and coverage metrics.",
    repository_method="ModuleRepository.get_module_profile",
)

MODULE_ARCHITECTURE = OperationContract(
    name="module.architecture",
    data_source=DataSourceType.VIEW,
    source_name="docs.v_module_architecture",
    supports_pagination=False,
    description="Fetch module architecture with import graph metrics.",
    repository_method="ModuleRepository.get_module_architecture",
)

# =============================================================================
# Subsystem Operations
# =============================================================================

SUBSYSTEMS_LIST = OperationContract(
    name="subsystems.list",
    data_source=DataSourceType.VIEW,
    source_name="docs.v_subsystem_summary",
    supports_pagination=True,
    description="List inferred subsystems with summary metrics.",
    repository_method="SubsystemRepository.list_subsystems",
)

SUBSYSTEM_MODULES = OperationContract(
    name="subsystems.modules",
    data_source=DataSourceType.VIEW,
    source_name="docs.v_subsystem_modules",
    supports_pagination=True,
    description="Fetch subsystem detail and member modules.",
    repository_method="SubsystemRepository.list_subsystem_modules",
)

MODULE_SUBSYSTEMS = OperationContract(
    name="module.subsystems",
    data_source=DataSourceType.TABLE,
    source_name="analytics.subsystem_modules",
    supports_pagination=False,
    description="Fetch subsystem memberships for a module.",
    repository_method="SubsystemRepository.list_subsystems_for_module",
)

SUBSYSTEM_PROFILES = OperationContract(
    name="subsystems.profiles",
    data_source=DataSourceType.VIEW,
    source_name="docs.v_subsystem_profile",
    supports_pagination=True,
    description="List subsystem profiles with detailed metrics.",
    repository_method="SubsystemRepository.list_subsystem_profiles",
)

SUBSYSTEM_COVERAGE = OperationContract(
    name="subsystems.coverage",
    data_source=DataSourceType.VIEW,
    source_name="docs.v_subsystem_coverage",
    supports_pagination=True,
    description="List subsystem coverage rollups from test data.",
    repository_method="SubsystemRepository.list_subsystem_coverage",
)

# =============================================================================
# Dataset Operations
# =============================================================================

DATASETS_LIST = OperationContract(
    name="datasets.list",
    data_source=DataSourceType.COMPUTED,
    source_name="dataset_registry",
    supports_pagination=False,
    description="List available datasets in the registry.",
    repository_method=None,
)

DATASET_ROWS = OperationContract(
    name="datasets.rows",
    data_source=DataSourceType.TABLE,
    source_name="<dynamic>",
    supports_pagination=True,
    description="Read rows from a dataset table.",
    repository_method="DatasetReadRepository.read_rows",
)

DATASET_SCHEMA = OperationContract(
    name="datasets.schema",
    data_source=DataSourceType.COMPUTED,
    source_name="information_schema.columns",
    supports_pagination=False,
    description="Fetch schema metadata for a dataset.",
    repository_method=None,
)

# =============================================================================
# Registry
# =============================================================================

OPERATION_CONTRACTS: dict[str, OperationContract] = {
    # Functions
    FUNCTION_SUMMARY.name: FUNCTION_SUMMARY,
    FUNCTION_PROFILE.name: FUNCTION_PROFILE,
    FUNCTION_ARCHITECTURE.name: FUNCTION_ARCHITECTURE,
    HIGH_RISK_FUNCTIONS.name: HIGH_RISK_FUNCTIONS,
    TESTS_FOR_FUNCTION.name: TESTS_FOR_FUNCTION,
    # Call Graph
    CALLGRAPH_NEIGHBORS.name: CALLGRAPH_NEIGHBORS,
    CALLGRAPH_NEIGHBORHOOD.name: CALLGRAPH_NEIGHBORHOOD,
    IMPORT_BOUNDARY.name: IMPORT_BOUNDARY,
    # Files/Modules
    FILE_SUMMARY.name: FILE_SUMMARY,
    FILE_PROFILE.name: FILE_PROFILE,
    FILE_HINTS.name: FILE_HINTS,
    MODULE_PROFILE.name: MODULE_PROFILE,
    MODULE_ARCHITECTURE.name: MODULE_ARCHITECTURE,
    # Subsystems
    SUBSYSTEMS_LIST.name: SUBSYSTEMS_LIST,
    SUBSYSTEM_MODULES.name: SUBSYSTEM_MODULES,
    MODULE_SUBSYSTEMS.name: MODULE_SUBSYSTEMS,
    SUBSYSTEM_PROFILES.name: SUBSYSTEM_PROFILES,
    SUBSYSTEM_COVERAGE.name: SUBSYSTEM_COVERAGE,
    # Datasets
    DATASETS_LIST.name: DATASETS_LIST,
    DATASET_ROWS.name: DATASET_ROWS,
    DATASET_SCHEMA.name: DATASET_SCHEMA,
}


def get_contract(operation_name: str) -> OperationContract | None:
    """
    Look up an operation contract by name.

    Parameters
    ----------
    operation_name
        The operation identifier (e.g., "function.architecture").

    Returns
    -------
    OperationContract | None
        The contract if found, otherwise None.
    """
    return OPERATION_CONTRACTS.get(operation_name)


def contracts_for_source(source_type: DataSourceType) -> list[OperationContract]:
    """
    Return all operation contracts that use a specific data source type.

    Parameters
    ----------
    source_type
        The type of data source to filter by.

    Returns
    -------
    list[OperationContract]
        Contracts matching the source type.
    """
    return [c for c in OPERATION_CONTRACTS.values() if c.data_source == source_type]


def contracts_using_view(view_name: str) -> list[OperationContract]:
    """
    Return all operation contracts that query a specific view.

    Parameters
    ----------
    view_name
        The view name to filter by (e.g., "docs.v_function_architecture").

    Returns
    -------
    list[OperationContract]
        Contracts that query the specified view.
    """
    return [
        c
        for c in OPERATION_CONTRACTS.values()
        if c.data_source == DataSourceType.VIEW and c.source_name == view_name
    ]


__all__ = [
    "CALLGRAPH_NEIGHBORHOOD",
    "CALLGRAPH_NEIGHBORS",
    "DATASETS_LIST",
    "DATASET_ROWS",
    "DATASET_SCHEMA",
    "FILE_HINTS",
    "FILE_PROFILE",
    "FILE_SUMMARY",
    "FUNCTION_ARCHITECTURE",
    "FUNCTION_PROFILE",
    "FUNCTION_SUMMARY",
    "HIGH_RISK_FUNCTIONS",
    "IMPORT_BOUNDARY",
    "MODULE_ARCHITECTURE",
    "MODULE_PROFILE",
    "MODULE_SUBSYSTEMS",
    "OPERATION_CONTRACTS",
    "SUBSYSTEMS_LIST",
    "SUBSYSTEM_COVERAGE",
    "SUBSYSTEM_MODULES",
    "SUBSYSTEM_PROFILES",
    "TESTS_FOR_FUNCTION",
    "DataSourceType",
    "OperationContract",
    "contracts_for_source",
    "contracts_using_view",
    "get_contract",
]
