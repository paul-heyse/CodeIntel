"""Modular dataset contract definitions, schemas, row models, and SQL helpers.

This package is the canonical source for dataset configuration. All new code
should import from `codeintel.config.datasets`.

Schema registry and validation have moved to the Hamilton build layer:
`codeintel.build.hamilton.contracts.schemas`.

Submodules
----------
primitives
    Core types (Column, TableSchema, Index) and column fragments.
schemas
    TABLE_SCHEMAS (79 schemas), COMPOSITE_SCHEMAS (4 schemas).
contracts
    DatasetContract, RowBinding, and contract registries.
rows
    TypedDict row models and serializer functions.
sql
    SQL generation helpers (INSERT, DELETE statements).
dataflow
    Dataflow graph types and builders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.config.datasets.contracts import (
    DatasetContract,
    RowBinding,
    get_composite_schemas,
    get_dataset_contracts,
    get_dataset_contracts_by_table_key,
    get_row_bindings,
    get_table_schemas,
)

if TYPE_CHECKING:
    from codeintel.config.datasets.primitives import CompositeSchema, TableSchema

from codeintel.config.datasets.columns import (
    AST_METRICS_DELETE,
    AST_NODES_DELETE,
    CALL_GRAPH_EDGES_DELETE,
    CALL_GRAPH_NODES_DELETE,
    CFG_BLOCKS_DELETE,
    CFG_EDGES_DELETE,
    CST_NODES_DELETE,
    DFG_EDGES_DELETE,
    FILE_STATE_DELETE,
    GOID_CROSSWALK_UPDATE_SCIP,
    SYMBOL_USE_DELETE,
    TAGS_INDEX_DELETE,
    TEST_CATALOG_UPDATE_GOIDS,
    load_columns_by_table,
    serialize_row,
)
from codeintel.config.datasets.dataflow import (
    DataflowEdge,
    DataflowNode,
    EdgeType,
    NodeKind,
    build_contract_dataflow_graph,
    iter_composite_edges,
    iter_dataset_nodes,
    iter_dependency_edges,
    iter_docs_view_alias_edges,
    iter_docs_view_alias_nodes,
)
from codeintel.config.datasets.dependencies import (
    DependencyAggregateRow,
    DependencyCallRow,
    compute_dep_id,
    to_decimal,
)
from codeintel.config.datasets.primitives import (
    COLUMN_TYPE,
    CREATED_AT_COL,
    CREATED_AT_COL_NULLABLE,
    FUNCTION_ENTITY_COLS,
    FUNCTION_GOID_COL,
    FUNCTION_GOID_COL_NULLABLE,
    MODULE_ENTITY_COLS,
    OWNERSHIP_COLS,
    REPO_COMMIT_COLS,
    RISK_COLS,
    SOURCE_SPAN_COLS,
    SUBSYSTEM_ENTITY_COLS,
    TEST_ENTITY_COLS,
    Column,
    ColumnType,
    CompositeSchema,
    Index,
    RowDictType,
    RowToTuple,
    TableSchema,
)
from codeintel.config.datasets.row_factory import (
    row_serializer_from_pandera,
    typed_dict_from_pandera,
)
from codeintel.config.datasets.rows import (
    BEHAVIORAL_COVERAGE_COLUMNS,
    FILE_PROFILE_COLUMNS,
    FUNCTION_CONTRACTS_COLUMNS,
    FUNCTION_EFFECTS_COLUMNS,
    FUNCTION_METRICS_COLUMNS,
    FUNCTION_PROFILE_COLUMNS,
    FUNCTION_TYPES_COLUMNS,
    GRAPH_METRICS_FUNCTIONS_COLUMNS,
    GRAPH_METRICS_FUNCTIONS_EXT_COLUMNS,
    GRAPH_METRICS_MODULES_COLUMNS,
    GRAPH_METRICS_MODULES_EXT_COLUMNS,
    MODULE_PROFILE_COLUMNS,
    SUBSYSTEM_COVERAGE_COLUMNS,
    SUBSYSTEM_PROFILE_COLUMNS,
    TEST_COVERAGE_EDGE_COLUMNS,
    TEST_PROFILE_COLUMNS,
    BehavioralCoverageRowModel,
    CallGraphEdgeRow,
    CallGraphNodeRow,
    CFGBlockRow,
    CFGEdgeRow,
    ConfigValueRow,
    CoverageLineRow,
    DFGEdgeRow,
    DocstringRow,
    FileProfileRowModel,
    FunctionAstFeaturesRow,
    FunctionContractsRow,
    FunctionEffectsRow,
    FunctionMetricsRow,
    FunctionProfileRowModel,
    FunctionTypesRow,
    FunctionValidationRow,
    GoidCrosswalkRow,
    GoidRow,
    GraphMetricsFunctionsExtRow,
    GraphMetricsFunctionsRow,
    GraphMetricsModulesExtRow,
    GraphMetricsModulesRow,
    GraphValidationRow,
    HotspotRow,
    ImportEdgeRow,
    ImportModuleRow,
    IngestRunLike,
    IngestRunRow,
    ModuleProfileRowModel,
    ProfileRowModel,
    StaticDiagnosticRow,
    SubsystemCoverageCacheRow,
    SubsystemProfileCacheRow,
    SymbolUseRow,
    TestCatalogRowModel,
    TestCoverageEdgeRow,
    TypednessRow,
    behavioral_coverage_row_to_tuple,
    call_graph_edge_to_tuple,
    call_graph_node_to_tuple,
    config_value_to_tuple,
    coverage_line_to_tuple,
    dict_to_call_graph_edge,
    dict_to_call_graph_node,
    docstring_row_to_tuple,
    file_profile_row_to_tuple,
    function_ast_features_row_to_tuple,
    function_contracts_row_to_tuple,
    function_effects_row_to_tuple,
    function_metrics_row_to_tuple,
    function_profile_row_to_tuple,
    function_types_row_to_tuple,
    function_validation_row_to_tuple,
    graph_metrics_functions_ext_row_to_tuple,
    graph_metrics_functions_row_to_tuple,
    graph_metrics_modules_ext_row_to_tuple,
    graph_metrics_modules_row_to_tuple,
    graph_validation_row_to_tuple,
    hotspot_row_to_tuple,
    ingest_run_to_tuple,
    module_profile_row_to_tuple,
    serialize_test_catalog_row,
    serialize_test_coverage_edge,
    serialize_test_profile_row,
    static_diagnostic_to_tuple,
    subsystem_coverage_cache_to_tuple,
    subsystem_profile_cache_to_tuple,
    typedness_row_to_tuple,
)
from codeintel.config.datasets.semantic_roles import (
    FUNCTION_COLUMNS as SEMANTIC_ROLE_FUNCTION_COLUMNS,
)
from codeintel.config.datasets.semantic_roles import (
    MODULE_COLUMNS as SEMANTIC_ROLE_MODULE_COLUMNS,
)
from codeintel.config.datasets.semantic_roles import (
    FunctionSemanticRoleRow,
    ModuleSemanticRoleRow,
)
from codeintel.config.datasets.semantic_roles import (
    normalize_function_row as normalize_semantic_role_function_row,
)
from codeintel.config.datasets.semantic_roles import (
    normalize_module_row as normalize_semantic_role_module_row,
)
from codeintel.config.datasets.semantic_roles import (
    timestamp_str as semantic_role_timestamp_str,
)


def get_table_columns(table_key: str) -> list[str]:
    """Return ordered column names for a specific table.

    Parameters
    ----------
    table_key
        Fully qualified table key (e.g., "core.ast_nodes").

    Returns
    -------
    list[str]
        Column names in storage order.
    """
    return list(load_columns_by_table().get(table_key, []))


TABLE_SCHEMAS: dict[str, TableSchema] = get_table_schemas()
COMPOSITE_SCHEMAS: dict[str, CompositeSchema] = get_composite_schemas()
DATASET_CONTRACTS: dict[str, DatasetContract] = get_dataset_contracts()
DATASET_CONTRACTS_BY_TABLE_KEY: dict[str, DatasetContract] = get_dataset_contracts_by_table_key()
ROW_BINDINGS_BY_TABLE_KEY: dict[str, RowBinding] = get_row_bindings()

JSON_SCHEMA_BY_DATASET_NAME: dict[str, str] = {
    name: contract.json_schema_id
    for name, contract in DATASET_CONTRACTS.items()
    if contract.json_schema_id is not None
}

DEFAULT_JSONL_FILENAMES: dict[str, str] = {
    contract.table_key: contract.jsonl_filename
    for contract in DATASET_CONTRACTS.values()
    if contract.jsonl_filename is not None
}

DEFAULT_PARQUET_FILENAMES: dict[str, str] = {
    contract.table_key: contract.parquet_filename
    for contract in DATASET_CONTRACTS.values()
    if contract.parquet_filename is not None
}

DEPENDENCIES_BY_DATASET_NAME: dict[str, tuple[str, ...]] = {
    name: contract.upstream_dependencies
    for name, contract in DATASET_CONTRACTS.items()
    if contract.upstream_dependencies
}

DESCRIPTION_BY_DATASET_NAME: dict[str, str] = {
    name: contract.description
    for name, contract in DATASET_CONTRACTS.items()
    if contract.description is not None
}

OWNER_BY_DATASET_NAME: dict[str, str] = {
    name: contract.owner for name, contract in DATASET_CONTRACTS.items() if contract.owner
}

FRESHNESS_BY_DATASET_NAME: dict[str, str] = {
    name: contract.freshness_sla
    for name, contract in DATASET_CONTRACTS.items()
    if contract.freshness_sla is not None
}

RETENTION_BY_DATASET_NAME: dict[str, str] = {
    name: contract.retention_policy
    for name, contract in DATASET_CONTRACTS.items()
    if contract.retention_policy is not None
}

STABLE_ID_BY_DATASET_NAME: dict[str, str] = {
    name: contract.stable_id for name, contract in DATASET_CONTRACTS.items() if contract.stable_id
}

SCHEMA_VERSION_BY_DATASET_NAME: dict[str, str] = {
    name: contract.schema_version
    for name, contract in DATASET_CONTRACTS.items()
    if contract.schema_version is not None
}

VALIDATION_PROFILE_BY_DATASET_NAME: dict[str, str] = {
    name: contract.validation_profile
    for name, contract in DATASET_CONTRACTS.items()
    if contract.validation_profile is not None
}


__all__ = [
    "AST_METRICS_DELETE",
    "AST_NODES_DELETE",
    "BEHAVIORAL_COVERAGE_COLUMNS",
    "CALL_GRAPH_EDGES_DELETE",
    "CALL_GRAPH_NODES_DELETE",
    "CFG_BLOCKS_DELETE",
    "CFG_EDGES_DELETE",
    "COLUMN_TYPE",
    "COMPOSITE_SCHEMAS",
    "CREATED_AT_COL",
    "CREATED_AT_COL_NULLABLE",
    "CST_NODES_DELETE",
    "DATASET_CONTRACTS",
    "DATASET_CONTRACTS_BY_TABLE_KEY",
    "DEFAULT_JSONL_FILENAMES",
    "DEFAULT_PARQUET_FILENAMES",
    "DEPENDENCIES_BY_DATASET_NAME",
    "DESCRIPTION_BY_DATASET_NAME",
    "DFG_EDGES_DELETE",
    "FILE_PROFILE_COLUMNS",
    "FILE_STATE_DELETE",
    "FRESHNESS_BY_DATASET_NAME",
    "FUNCTION_CONTRACTS_COLUMNS",
    "FUNCTION_EFFECTS_COLUMNS",
    "FUNCTION_ENTITY_COLS",
    "FUNCTION_GOID_COL",
    "FUNCTION_GOID_COL_NULLABLE",
    "FUNCTION_METRICS_COLUMNS",
    "FUNCTION_PROFILE_COLUMNS",
    "FUNCTION_TYPES_COLUMNS",
    "GOID_CROSSWALK_UPDATE_SCIP",
    "GRAPH_METRICS_FUNCTIONS_COLUMNS",
    "GRAPH_METRICS_FUNCTIONS_EXT_COLUMNS",
    "GRAPH_METRICS_MODULES_COLUMNS",
    "GRAPH_METRICS_MODULES_EXT_COLUMNS",
    "JSON_SCHEMA_BY_DATASET_NAME",
    "MODULE_ENTITY_COLS",
    "MODULE_PROFILE_COLUMNS",
    "OWNERSHIP_COLS",
    "OWNER_BY_DATASET_NAME",
    "REPO_COMMIT_COLS",
    "RETENTION_BY_DATASET_NAME",
    "RISK_COLS",
    "ROW_BINDINGS_BY_TABLE_KEY",
    "SCHEMA_VERSION_BY_DATASET_NAME",
    "SEMANTIC_ROLE_FUNCTION_COLUMNS",
    "SEMANTIC_ROLE_MODULE_COLUMNS",
    "SOURCE_SPAN_COLS",
    "STABLE_ID_BY_DATASET_NAME",
    "SUBSYSTEM_COVERAGE_COLUMNS",
    "SUBSYSTEM_ENTITY_COLS",
    "SUBSYSTEM_PROFILE_COLUMNS",
    "SYMBOL_USE_DELETE",
    "TABLE_SCHEMAS",
    "TAGS_INDEX_DELETE",
    "TEST_CATALOG_UPDATE_GOIDS",
    "TEST_COVERAGE_EDGE_COLUMNS",
    "TEST_ENTITY_COLS",
    "TEST_PROFILE_COLUMNS",
    "VALIDATION_PROFILE_BY_DATASET_NAME",
    "BehavioralCoverageRowModel",
    "CFGBlockRow",
    "CFGEdgeRow",
    "CallGraphEdgeRow",
    "CallGraphNodeRow",
    "Column",
    "ColumnType",
    "CompositeSchema",
    "ConfigValueRow",
    "CoverageLineRow",
    "DFGEdgeRow",
    "DataflowEdge",
    "DataflowNode",
    "DatasetContract",
    "DependencyAggregateRow",
    "DependencyCallRow",
    "DocstringRow",
    "EdgeType",
    "FileProfileRowModel",
    "FunctionAstFeaturesRow",
    "FunctionContractsRow",
    "FunctionEffectsRow",
    "FunctionMetricsRow",
    "FunctionProfileRowModel",
    "FunctionSemanticRoleRow",
    "FunctionTypesRow",
    "FunctionValidationRow",
    "GoidCrosswalkRow",
    "GoidRow",
    "GraphMetricsFunctionsExtRow",
    "GraphMetricsFunctionsRow",
    "GraphMetricsModulesExtRow",
    "GraphMetricsModulesRow",
    "GraphValidationRow",
    "HotspotRow",
    "ImportEdgeRow",
    "ImportModuleRow",
    "Index",
    "IngestRunLike",
    "IngestRunRow",
    "ModuleProfileRowModel",
    "ModuleSemanticRoleRow",
    "NodeKind",
    "ProfileRowModel",
    "RowBinding",
    "RowDictType",
    "RowToTuple",
    "StaticDiagnosticRow",
    "SubsystemCoverageCacheRow",
    "SubsystemProfileCacheRow",
    "SymbolUseRow",
    "TableSchema",
    "TestCatalogRowModel",
    "TestCoverageEdgeRow",
    "TypednessRow",
    "behavioral_coverage_row_to_tuple",
    "build_contract_dataflow_graph",
    "call_graph_edge_to_tuple",
    "call_graph_node_to_tuple",
    "compute_dep_id",
    "config_value_to_tuple",
    "coverage_line_to_tuple",
    "dict_to_call_graph_edge",
    "dict_to_call_graph_node",
    "docstring_row_to_tuple",
    "file_profile_row_to_tuple",
    "function_ast_features_row_to_tuple",
    "function_contracts_row_to_tuple",
    "function_effects_row_to_tuple",
    "function_metrics_row_to_tuple",
    "function_profile_row_to_tuple",
    "function_types_row_to_tuple",
    "function_validation_row_to_tuple",
    "get_composite_schemas",
    "get_dataset_contracts",
    "get_dataset_contracts_by_table_key",
    "get_row_bindings",
    "get_table_columns",
    "get_table_schemas",
    "graph_metrics_functions_ext_row_to_tuple",
    "graph_metrics_functions_row_to_tuple",
    "graph_metrics_modules_ext_row_to_tuple",
    "graph_metrics_modules_row_to_tuple",
    "graph_validation_row_to_tuple",
    "hotspot_row_to_tuple",
    "ingest_run_to_tuple",
    "iter_composite_edges",
    "iter_dataset_nodes",
    "iter_dependency_edges",
    "iter_docs_view_alias_edges",
    "iter_docs_view_alias_nodes",
    "load_columns_by_table",
    "module_profile_row_to_tuple",
    "normalize_semantic_role_function_row",
    "normalize_semantic_role_module_row",
    "row_serializer_from_pandera",
    "semantic_role_timestamp_str",
    "serialize_row",
    "serialize_test_catalog_row",
    "serialize_test_coverage_edge",
    "serialize_test_profile_row",
    "static_diagnostic_to_tuple",
    "subsystem_coverage_cache_to_tuple",
    "subsystem_profile_cache_to_tuple",
    "to_decimal",
    "typed_dict_from_pandera",
    "typedness_row_to_tuple",
]
