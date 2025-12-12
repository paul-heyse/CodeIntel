"""Profile table TypedDict row models and serializers.

This module provides TypedDict definitions for profile DuckDB tables:
- FunctionProfileRowModel for analytics.function_profile
- FileProfileRowModel for analytics.file_profile
- ModuleProfileRowModel for analytics.module_profile
- FunctionAstFeaturesRow for analytics.function_ast_features
- GraphMetricsFunctionsRow for analytics.graph_metrics_functions
- GraphMetricsModulesRow for analytics.graph_metrics_modules
- GraphMetricsFunctionsExtRow for analytics.graph_metrics_functions_ext
- GraphMetricsModulesExtRow for analytics.graph_metrics_modules_ext
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Final, TypedDict, TypeVar

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_DATETIME = datetime

_Column = TypeVar("_Column", bound=str)


def _serialize_row(
    row: Mapping[_Column, object],
    columns: Sequence[_Column],
) -> tuple[object, ...]:
    """Serialize a mapping using a stable column sequence.

    Parameters
    ----------
    row
        Row data as a mapping from column name to value.
    columns
        Ordered sequence of column names.

    Returns
    -------
    tuple[object, ...]
        Values ordered according to ``columns``.
    """
    return tuple(row[column] for column in columns)


FUNCTION_PROFILE_COLUMNS: Final[tuple[str, ...]] = (
    "function_goid_h128",
    "urn",
    "repo",
    "commit",
    "rel_path",
    "module",
    "language",
    "kind",
    "qualname",
    "start_line",
    "end_line",
    "loc",
    "logical_loc",
    "cyclomatic_complexity",
    "complexity_bucket",
    "param_count",
    "positional_params",
    "keyword_params",
    "vararg",
    "kwarg",
    "max_nesting_depth",
    "stmt_count",
    "decorator_count",
    "has_docstring",
    "total_params",
    "annotated_params",
    "return_type",
    "param_types",
    "fully_typed",
    "partial_typed",
    "untyped",
    "typedness_bucket",
    "typedness_source",
    "file_typed_ratio",
    "static_error_count",
    "has_static_errors",
    "executable_lines",
    "covered_lines",
    "coverage_ratio",
    "tested",
    "untested_reason",
    "tests_touching",
    "failing_tests",
    "slow_tests",
    "flaky_tests",
    "last_test_status",
    "dominant_test_status",
    "slow_test_threshold_ms",
    "created_in_commit",
    "created_at_history",
    "last_modified_commit",
    "last_modified_at",
    "age_days",
    "commit_count",
    "author_count",
    "lines_added",
    "lines_deleted",
    "churn_score",
    "stability_bucket",
    "call_fan_in",
    "call_fan_out",
    "call_edge_in_count",
    "call_edge_out_count",
    "call_is_leaf",
    "call_is_entrypoint",
    "call_is_public",
    "risk_score",
    "risk_level",
    "risk_component_coverage",
    "risk_component_complexity",
    "risk_component_static",
    "risk_component_hotspot",
    "is_pure",
    "uses_io",
    "touches_db",
    "uses_time",
    "uses_randomness",
    "modifies_globals",
    "modifies_closure",
    "spawns_threads_or_tasks",
    "has_transitive_effects",
    "purity_confidence",
    "param_nullability_json",
    "return_nullability",
    "has_preconditions",
    "has_postconditions",
    "has_raises",
    "contract_confidence",
    "role",
    "framework",
    "role_confidence",
    "role_sources_json",
    "tags",
    "owners",
    "doc_short",
    "doc_long",
    "doc_params",
    "doc_returns",
    "created_at",
)


class FunctionProfileRowModel(TypedDict):
    """Row shape for ``analytics.function_profile`` inserts."""

    function_goid_h128: int
    urn: str | None
    repo: str
    commit: str
    rel_path: str
    module: str | None
    language: str | None
    kind: str | None
    qualname: str | None
    start_line: int | None
    end_line: int | None
    loc: int
    logical_loc: int
    cyclomatic_complexity: int
    complexity_bucket: str | None
    param_count: int
    positional_params: int
    keyword_params: int
    vararg: bool
    kwarg: bool
    max_nesting_depth: int | None
    stmt_count: int | None
    decorator_count: int | None
    has_docstring: bool
    total_params: int
    annotated_params: int
    return_type: str | None
    param_types: object
    fully_typed: bool
    partial_typed: bool
    untyped: bool
    typedness_bucket: str | None
    typedness_source: str | None
    file_typed_ratio: float | None
    static_error_count: int
    has_static_errors: bool
    executable_lines: int
    covered_lines: int
    coverage_ratio: float | None
    tested: bool
    untested_reason: str | None
    tests_touching: int
    failing_tests: int
    slow_tests: int
    flaky_tests: int
    last_test_status: str | None
    dominant_test_status: str | None
    slow_test_threshold_ms: float
    created_in_commit: str | None
    created_at_history: datetime | None
    last_modified_commit: str | None
    last_modified_at: datetime | None
    age_days: int | None
    commit_count: int
    author_count: int
    lines_added: int
    lines_deleted: int
    churn_score: float | None
    stability_bucket: str | None
    call_fan_in: int
    call_fan_out: int
    call_edge_in_count: int
    call_edge_out_count: int
    call_is_leaf: bool
    call_is_entrypoint: bool
    call_is_public: bool
    risk_score: float
    risk_level: str | None
    risk_component_coverage: float
    risk_component_complexity: float
    risk_component_static: float
    risk_component_hotspot: float
    is_pure: bool
    uses_io: bool
    touches_db: bool
    uses_time: bool
    uses_randomness: bool
    modifies_globals: bool
    modifies_closure: bool
    spawns_threads_or_tasks: bool
    has_transitive_effects: bool
    purity_confidence: float | None
    param_nullability_json: object
    return_nullability: str | None
    has_preconditions: bool
    has_postconditions: bool
    has_raises: bool
    contract_confidence: float | None
    role: str | None
    framework: str | None
    role_confidence: float | None
    role_sources_json: object
    tags: object
    owners: object
    doc_short: str | None
    doc_long: str | None
    doc_params: object
    doc_returns: object
    created_at: datetime


def function_profile_row_to_tuple(row: FunctionProfileRowModel) -> tuple[object, ...]:
    """Serialize a FunctionProfileRowModel into INSERT column order.

    Parameters
    ----------
    row
        The function profile row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by function_profile INSERTs.
    """
    return _serialize_row(row, FUNCTION_PROFILE_COLUMNS)


_FUNCTION_AST_FEATURES_COLUMNS: Final[tuple[str, ...]] = (
    "repo",
    "commit",
    "function_goid_h128",
    "rel_path",
    "qualname",
    "is_async",
    "uses_network",
    "uses_db",
    "uses_filesystem",
    "uses_subprocess",
    "uses_concurrency_lib",
    "uses_threading",
    "uses_asyncio_lib",
    "http_client_libs",
    "http_server_libs",
    "db_libs",
    "message_libs",
    "config_read_count",
    "feature_flag_count",
    "decorators",
    "libraries_used",
    "created_at",
)


class FunctionAstFeaturesRow(TypedDict):
    """Row shape for ``analytics.function_ast_features`` inserts."""

    repo: str
    commit: str
    function_goid_h128: int
    rel_path: str
    qualname: str
    is_async: bool
    uses_network: bool
    uses_db: bool
    uses_filesystem: bool
    uses_subprocess: bool
    uses_concurrency_lib: bool
    uses_threading: bool
    uses_asyncio_lib: bool
    http_client_libs: list[str]
    http_server_libs: list[str]
    db_libs: list[str]
    message_libs: list[str]
    config_read_count: int
    feature_flag_count: int
    decorators: list[str]
    libraries_used: list[str]
    created_at: datetime


def function_ast_features_row_to_tuple(row: FunctionAstFeaturesRow) -> tuple[object, ...]:
    """Serialize a FunctionAstFeaturesRow into INSERT column order.

    Parameters
    ----------
    row
        The function AST features row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values ordered per analytics.function_ast_features definition.
    """
    return _serialize_row(row, _FUNCTION_AST_FEATURES_COLUMNS)


FILE_PROFILE_COLUMNS: Final[tuple[str, ...]] = (
    "repo",
    "commit",
    "rel_path",
    "module",
    "language",
    "node_count",
    "function_count",
    "class_count",
    "avg_depth",
    "max_depth",
    "ast_complexity",
    "hotspot_score",
    "commit_count",
    "author_count",
    "lines_added",
    "lines_deleted",
    "annotation_ratio",
    "untyped_defs",
    "overlay_needed",
    "type_error_count",
    "static_error_count",
    "has_static_errors",
    "total_functions",
    "public_functions",
    "avg_loc",
    "max_loc",
    "avg_cyclomatic_complexity",
    "max_cyclomatic_complexity",
    "high_risk_function_count",
    "medium_risk_function_count",
    "max_risk_score",
    "file_coverage_ratio",
    "tested_function_count",
    "untested_function_count",
    "tests_touching",
    "tags",
    "owners",
    "created_at",
)


class FileProfileRowModel(TypedDict):
    """Row shape for ``analytics.file_profile`` inserts."""

    repo: str
    commit: str
    rel_path: str
    module: str | None
    language: str | None
    node_count: int | None
    function_count: int | None
    class_count: int | None
    avg_depth: float | None
    max_depth: int | None
    ast_complexity: float | None
    hotspot_score: float | None
    commit_count: int | None
    author_count: int | None
    lines_added: int | None
    lines_deleted: int | None
    annotation_ratio: float | None
    untyped_defs: int | None
    overlay_needed: bool | None
    type_error_count: int | None
    static_error_count: int | None
    has_static_errors: bool | None
    total_functions: int | None
    public_functions: int | None
    avg_loc: float | None
    max_loc: int | None
    avg_cyclomatic_complexity: float | None
    max_cyclomatic_complexity: int | None
    high_risk_function_count: int | None
    medium_risk_function_count: int | None
    max_risk_score: float | None
    file_coverage_ratio: float | None
    tested_function_count: int | None
    untested_function_count: int | None
    tests_touching: int | None
    tags: object
    owners: object
    created_at: datetime


def file_profile_row_to_tuple(row: FileProfileRowModel) -> tuple[object, ...]:
    """Serialize a FileProfileRowModel into INSERT column order.

    Parameters
    ----------
    row
        The file profile row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by file_profile INSERTs.
    """
    return _serialize_row(row, FILE_PROFILE_COLUMNS)


MODULE_PROFILE_COLUMNS: Final[tuple[str, ...]] = (
    "repo",
    "commit",
    "module",
    "path",
    "language",
    "file_count",
    "total_loc",
    "total_logical_loc",
    "function_count",
    "class_count",
    "avg_file_complexity",
    "max_file_complexity",
    "high_risk_function_count",
    "medium_risk_function_count",
    "low_risk_function_count",
    "max_risk_score",
    "avg_risk_score",
    "module_coverage_ratio",
    "tested_function_count",
    "untested_function_count",
    "import_fan_in",
    "import_fan_out",
    "cycle_group",
    "in_cycle",
    "role",
    "role_confidence",
    "role_sources_json",
    "tags",
    "owners",
    "created_at",
)


class ModuleProfileRowModel(TypedDict):
    """Row shape for ``analytics.module_profile`` inserts."""

    repo: str
    commit: str
    module: str
    path: str | None
    language: str | None
    file_count: int | None
    total_loc: int | None
    total_logical_loc: int | None
    function_count: int | None
    class_count: int | None
    avg_file_complexity: float | None
    max_file_complexity: float | None
    high_risk_function_count: int | None
    medium_risk_function_count: int | None
    low_risk_function_count: int | None
    max_risk_score: float | None
    avg_risk_score: float | None
    module_coverage_ratio: float | None
    tested_function_count: int | None
    untested_function_count: int | None
    import_fan_in: int | None
    import_fan_out: int | None
    cycle_group: int | None
    in_cycle: bool | None
    role: str | None
    role_confidence: float | None
    role_sources_json: object
    tags: object
    owners: object
    created_at: datetime


def module_profile_row_to_tuple(row: ModuleProfileRowModel) -> tuple[object, ...]:
    """Serialize a ModuleProfileRowModel into INSERT column order.

    Parameters
    ----------
    row
        The module profile row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by module_profile INSERTs.
    """
    return _serialize_row(row, MODULE_PROFILE_COLUMNS)


GRAPH_METRICS_FUNCTIONS_COLUMNS: Final[tuple[str, ...]] = (
    "repo",
    "commit",
    "function_goid_h128",
    "call_fan_in",
    "call_fan_out",
    "call_in_degree",
    "call_out_degree",
    "call_pagerank",
    "call_betweenness",
    "call_closeness",
    "call_cycle_member",
    "call_cycle_id",
    "call_layer",
    "created_at",
)


class GraphMetricsFunctionsRow(TypedDict):
    """Row shape for analytics.graph_metrics_functions inserts."""

    repo: str
    commit: str
    function_goid_h128: int
    call_fan_in: int
    call_fan_out: int
    call_in_degree: int
    call_out_degree: int
    call_pagerank: float | None
    call_betweenness: float | None
    call_closeness: float | None
    call_cycle_member: bool
    call_cycle_id: int | None
    call_layer: int | None
    created_at: datetime


def graph_metrics_functions_row_to_tuple(
    row: GraphMetricsFunctionsRow,
) -> tuple[object, ...]:
    """Serialize a GraphMetricsFunctionsRow into INSERT column order.

    Parameters
    ----------
    row
        The graph metrics functions row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values aligned with analytics.graph_metrics_functions columns.
    """
    return _serialize_row(row, GRAPH_METRICS_FUNCTIONS_COLUMNS)


GRAPH_METRICS_MODULES_COLUMNS: Final[tuple[str, ...]] = (
    "repo",
    "commit",
    "module",
    "import_fan_in",
    "import_fan_out",
    "import_in_degree",
    "import_out_degree",
    "import_pagerank",
    "import_betweenness",
    "import_closeness",
    "import_cycle_member",
    "import_cycle_id",
    "import_layer",
    "symbol_fan_in",
    "symbol_fan_out",
    "created_at",
)


class GraphMetricsModulesRow(TypedDict):
    """Row shape for analytics.graph_metrics_modules inserts."""

    repo: str
    commit: str
    module: str
    import_fan_in: int
    import_fan_out: int
    import_in_degree: int
    import_out_degree: int
    import_pagerank: float | None
    import_betweenness: float | None
    import_closeness: float | None
    import_cycle_member: bool
    import_cycle_id: int | None
    import_layer: int | None
    symbol_fan_in: int
    symbol_fan_out: int
    created_at: datetime


def graph_metrics_modules_row_to_tuple(
    row: GraphMetricsModulesRow,
) -> tuple[object, ...]:
    """Serialize a GraphMetricsModulesRow into INSERT column order.

    Parameters
    ----------
    row
        The graph metrics modules row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values aligned with analytics.graph_metrics_modules columns.
    """
    return _serialize_row(row, GRAPH_METRICS_MODULES_COLUMNS)


GRAPH_METRICS_FUNCTIONS_EXT_COLUMNS: Final[tuple[str, ...]] = (
    "repo",
    "commit",
    "function_goid_h128",
    "call_betweenness",
    "call_closeness",
    "call_eigenvector",
    "call_harmonic",
    "call_core_number",
    "call_clustering_coeff",
    "call_triangle_count",
    "call_is_articulation",
    "call_articulation_impact",
    "call_is_bridge_endpoint",
    "call_component_id",
    "call_component_size",
    "call_scc_id",
    "call_scc_size",
    "call_ancestor_count",
    "call_descendant_count",
    "call_community_id",
    "created_at",
)


class GraphMetricsFunctionsExtRow(TypedDict):
    """Row shape for analytics.graph_metrics_functions_ext inserts."""

    repo: str
    commit: str
    function_goid_h128: int
    call_betweenness: float | None
    call_closeness: float | None
    call_eigenvector: float | None
    call_harmonic: float | None
    call_core_number: int | None
    call_clustering_coeff: float | None
    call_triangle_count: int | None
    call_is_articulation: bool | None
    call_articulation_impact: int | None
    call_is_bridge_endpoint: bool | None
    call_component_id: int | None
    call_component_size: int | None
    call_scc_id: int | None
    call_scc_size: int | None
    call_ancestor_count: int | None
    call_descendant_count: int | None
    call_community_id: int | None
    created_at: datetime


def graph_metrics_functions_ext_row_to_tuple(
    row: GraphMetricsFunctionsExtRow,
) -> tuple[object, ...]:
    """Serialize a GraphMetricsFunctionsExtRow into INSERT column order.

    Parameters
    ----------
    row
        The graph metrics functions extended row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values aligned with analytics.graph_metrics_functions_ext columns.
    """
    return _serialize_row(row, GRAPH_METRICS_FUNCTIONS_EXT_COLUMNS)


GRAPH_METRICS_MODULES_EXT_COLUMNS: Final[tuple[str, ...]] = (
    "repo",
    "commit",
    "module",
    "import_betweenness",
    "import_closeness",
    "import_eigenvector",
    "import_harmonic",
    "import_k_core",
    "import_constraint",
    "import_effective_size",
    "import_rich_club",
    "import_shell_index",
    "import_community_id",
    "import_component_id",
    "import_component_size",
    "import_scc_id",
    "import_scc_size",
    "created_at",
)


class GraphMetricsModulesExtRow(TypedDict):
    """Row shape for analytics.graph_metrics_modules_ext inserts."""

    repo: str
    commit: str
    module: str
    import_betweenness: float | None
    import_closeness: float | None
    import_eigenvector: float | None
    import_harmonic: float | None
    import_k_core: int | None
    import_constraint: float | None
    import_effective_size: float | None
    import_rich_club: bool | None
    import_shell_index: int | None
    import_community_id: int | None
    import_component_id: int | None
    import_component_size: int | None
    import_scc_id: int | None
    import_scc_size: int | None
    created_at: datetime


def graph_metrics_modules_ext_row_to_tuple(
    row: GraphMetricsModulesExtRow,
) -> tuple[object, ...]:
    """Serialize a GraphMetricsModulesExtRow into INSERT column order.

    Parameters
    ----------
    row
        The graph metrics modules extended row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values aligned with analytics.graph_metrics_modules_ext columns.
    """
    return _serialize_row(row, GRAPH_METRICS_MODULES_EXT_COLUMNS)


__all__ = [
    "FILE_PROFILE_COLUMNS",
    "FUNCTION_PROFILE_COLUMNS",
    "GRAPH_METRICS_FUNCTIONS_COLUMNS",
    "GRAPH_METRICS_FUNCTIONS_EXT_COLUMNS",
    "GRAPH_METRICS_MODULES_COLUMNS",
    "GRAPH_METRICS_MODULES_EXT_COLUMNS",
    "MODULE_PROFILE_COLUMNS",
    "FileProfileRowModel",
    "FunctionAstFeaturesRow",
    "FunctionProfileRowModel",
    "GraphMetricsFunctionsExtRow",
    "GraphMetricsFunctionsRow",
    "GraphMetricsModulesExtRow",
    "GraphMetricsModulesRow",
    "ModuleProfileRowModel",
    "file_profile_row_to_tuple",
    "function_ast_features_row_to_tuple",
    "function_profile_row_to_tuple",
    "graph_metrics_functions_ext_row_to_tuple",
    "graph_metrics_functions_row_to_tuple",
    "graph_metrics_modules_ext_row_to_tuple",
    "graph_metrics_modules_row_to_tuple",
    "module_profile_row_to_tuple",
]
