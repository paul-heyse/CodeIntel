"""TypedDict row models and serializer functions for DuckDB tables.

This module provides TypedDict definitions for table row shapes and
serializer functions that convert row dictionaries to tuples for INSERT.

Submodules
----------
core
    Core entity rows: IngestRunRow, GoidRow, DocstringRow, etc.
analytics
    Analytics table rows: CoverageLineRow, FunctionMetricsRow, etc.
graph
    Graph table rows: CallGraphNodeRow, CFGBlockRow, etc.
profiles
    Profile table rows: FunctionProfileRowModel, FileProfileRowModel, etc.
test
    Test-related rows: TestCatalogRowModel, BehavioralCoverageRowModel, etc.
"""

from __future__ import annotations

# Analytics table rows
from codeintel.config.datasets.rows.analytics import (
    FUNCTION_METRICS_COLUMNS,
    FUNCTION_TYPES_COLUMNS,
    CoverageLineRow,
    FunctionMetricsRow,
    FunctionTypesRow,
    FunctionValidationRow,
    GraphValidationRow,
    HotspotRow,
    StaticDiagnosticRow,
    TypednessRow,
    coverage_line_to_tuple,
    function_metrics_row_to_tuple,
    function_types_row_to_tuple,
    function_validation_row_to_tuple,
    graph_validation_row_to_tuple,
    hotspot_row_to_tuple,
    static_diagnostic_to_tuple,
    typedness_row_to_tuple,
)

# Core entity rows
from codeintel.config.datasets.rows.core import (
    ConfigValueRow,
    DocstringRow,
    GoidCrosswalkRow,
    GoidRow,
    IngestRunLike,
    IngestRunRow,
    config_value_to_tuple,
    docstring_row_to_tuple,
    ingest_run_to_tuple,
)

# Graph table rows
from codeintel.config.datasets.rows.graph import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    CFGBlockRow,
    CFGEdgeRow,
    DFGEdgeRow,
    ImportEdgeRow,
    ImportModuleRow,
    SymbolUseRow,
    call_graph_edge_to_tuple,
    call_graph_node_to_tuple,
    dict_to_call_graph_edge,
    dict_to_call_graph_node,
)

# Profile table rows
from codeintel.config.datasets.rows.profiles import (
    FILE_PROFILE_COLUMNS,
    FUNCTION_PROFILE_COLUMNS,
    GRAPH_METRICS_FUNCTIONS_COLUMNS,
    GRAPH_METRICS_FUNCTIONS_EXT_COLUMNS,
    GRAPH_METRICS_MODULES_COLUMNS,
    GRAPH_METRICS_MODULES_EXT_COLUMNS,
    MODULE_PROFILE_COLUMNS,
    FileProfileRowModel,
    FunctionAstFeaturesRow,
    FunctionProfileRowModel,
    GraphMetricsFunctionsExtRow,
    GraphMetricsFunctionsRow,
    GraphMetricsModulesExtRow,
    GraphMetricsModulesRow,
    ModuleProfileRowModel,
    file_profile_row_to_tuple,
    function_ast_features_row_to_tuple,
    function_profile_row_to_tuple,
    graph_metrics_functions_ext_row_to_tuple,
    graph_metrics_functions_row_to_tuple,
    graph_metrics_modules_ext_row_to_tuple,
    graph_metrics_modules_row_to_tuple,
    module_profile_row_to_tuple,
)

# Test-related rows
from codeintel.config.datasets.rows.test import (
    BEHAVIORAL_COVERAGE_COLUMNS,
    SUBSYSTEM_COVERAGE_COLUMNS,
    SUBSYSTEM_PROFILE_COLUMNS,
    TEST_COVERAGE_EDGE_COLUMNS,
    TEST_PROFILE_COLUMNS,
    BehavioralCoverageRowModel,
    ProfileRowModel,
    SubsystemCoverageCacheRow,
    SubsystemProfileCacheRow,
    TestCatalogRowModel,
    TestCoverageEdgeRow,
    behavioral_coverage_row_to_tuple,
    serialize_test_catalog_row,
    serialize_test_coverage_edge,
    serialize_test_profile_row,
    subsystem_coverage_cache_to_tuple,
    subsystem_profile_cache_to_tuple,
)

__all__ = [
    'BEHAVIORAL_COVERAGE_COLUMNS',
    'BehavioralCoverageRowModel',
    'CFGBlockRow',
    'CFGEdgeRow',
    'CallGraphEdgeRow',
    'CallGraphNodeRow',
    'ConfigValueRow',
    'CoverageLineRow',
    'DFGEdgeRow',
    'DocstringRow',
    'FILE_PROFILE_COLUMNS',
    'FUNCTION_METRICS_COLUMNS',
    'FUNCTION_PROFILE_COLUMNS',
    'FUNCTION_TYPES_COLUMNS',
    'FunctionAstFeaturesRow',
    'FunctionMetricsRow',
    'FunctionProfileRowModel',
    'FunctionTypesRow',
    'FunctionValidationRow',
    'GoidCrosswalkRow',
    'GoidRow',
    'GraphMetricsFunctionsExtRow',
    'GraphMetricsFunctionsRow',
    'GraphMetricsModulesExtRow',
    'GraphMetricsModulesRow',
    'GraphValidationRow',
    'HotspotRow',
    'ImportEdgeRow',
    'ImportModuleRow',
    'IngestRunLike',
    'IngestRunRow',
    'MODULE_PROFILE_COLUMNS',
    'ModuleProfileRowModel',
    'ProfileRowModel',
    'STATIC_COLUMNS',
    'SUBSYSTEM_COVERAGE_COLUMNS',
    'SUBSYSTEM_PROFILE_COLUMNS',
    'StaticDiagnosticRow',
    'SubsystemCoverageCacheRow',
    'SubsystemProfileCacheRow',
    'SymbolUseRow',
    'TEST_COVERAGE_EDGE_COLUMNS',
    'TEST_PROFILE_COLUMNS',
    'TestCatalogRowModel',
    'TestCoverageEdgeRow',
    'TypednessRow',
    'behavioral_coverage_row_to_tuple',
    'call_graph_edge_to_tuple',
    'call_graph_node_to_tuple',
    'config_value_to_tuple',
    'coverage_line_to_tuple',
    'dict_to_call_graph_edge',
    'dict_to_call_graph_node',
    'docstring_row_to_tuple',
    'file_profile_row_to_tuple',
    'function_ast_features_row_to_tuple',
    'function_metrics_row_to_tuple',
    'function_profile_row_to_tuple',
    'function_types_row_to_tuple',
    'function_validation_row_to_tuple',
    'graph_metrics_functions_ext_row_to_tuple',
    'graph_metrics_functions_row_to_tuple',
    'graph_metrics_modules_ext_row_to_tuple',
    'graph_metrics_modules_row_to_tuple',
    'graph_validation_row_to_tuple',
    'hotspot_row_to_tuple',
    'ingest_run_to_tuple',
    'module_profile_row_to_tuple',
    'serialize_test_catalog_row',
    'serialize_test_coverage_edge',
    'serialize_test_profile_row',
    'static_diagnostic_to_tuple',
    'subsystem_coverage_cache_to_tuple',
    'subsystem_profile_cache_to_tuple',
    'typedness_row_to_tuple',
]
