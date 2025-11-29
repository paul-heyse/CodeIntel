"""Typed row models for DuckDB inserts with helpers to keep column order stable."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Literal, TypedDict, TypeVar

__all__ = [
    "BehavioralCoverageRowModel",
    "CFGBlockRow",
    "CFGEdgeRow",
    "CallGraphEdgeRow",
    "CallGraphNodeRow",
    "ConfigValueRow",
    "CoverageLineRow",
    "DFGEdgeRow",
    "DocstringRow",
    "FileProfileRowModel",
    "FunctionProfileRowModel",
    "FunctionValidationRow",
    "GoidCrosswalkRow",
    "GoidRow",
    "HotspotRow",
    "ImportEdgeRow",
    "ImportModuleRow",
    "ModuleProfileRowModel",
    "StaticDiagnosticRow",
    "SymbolUseRow",
    "TestCatalogRowModel",
    "TestCoverageEdgeRow",
    "TestProfileRowModel",
    "TypednessRow",
    "behavioral_coverage_row_to_tuple",
    "call_graph_edge_to_tuple",
    "call_graph_node_to_tuple",
    "cfg_block_to_tuple",
    "cfg_edge_to_tuple",
    "config_value_to_tuple",
    "coverage_line_to_tuple",
    "dfg_edge_to_tuple",
    "docstring_row_to_tuple",
    "file_profile_row_to_tuple",
    "function_profile_row_to_tuple",
    "function_validation_row_to_tuple",
    "goid_crosswalk_to_tuple",
    "goid_to_tuple",
    "hotspot_row_to_tuple",
    "import_edge_to_tuple",
    "import_module_to_tuple",
    "module_profile_row_to_tuple",
    "serialize_test_catalog_row",
    "serialize_test_coverage_edge",
    "serialize_test_profile_row",
    "static_diagnostic_to_tuple",
    "symbol_use_to_tuple",
    "typedness_row_to_tuple",
]

_Column = TypeVar("_Column", bound=str)


def _serialize_row(row: Mapping[_Column, object], columns: Sequence[_Column]) -> tuple[object, ...]:
    """
    Serialize a mapping using a stable column sequence.

    Returns
    -------
    tuple[object, ...]
        Values ordered according to ``columns``.
    """
    return tuple(row[column] for column in columns)


class CoverageLineRow(TypedDict):
    """Row shape for analytics.coverage_lines inserts."""

    repo: str
    commit: str
    rel_path: str
    line: int
    is_executable: bool
    is_covered: bool
    hits: int
    context_count: int
    created_at: datetime


def coverage_line_to_tuple(row: CoverageLineRow) -> tuple[object, ...]:
    """
    Serialize a CoverageLineRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by coverage_lines INSERTs.
    """
    return (
        row["repo"],
        row["commit"],
        row["rel_path"],
        row["line"],
        row["is_executable"],
        row["is_covered"],
        row["hits"],
        row["context_count"],
        row["created_at"],
    )


class DocstringRow(TypedDict):
    """Row shape for core.docstrings inserts."""

    repo: str
    commit: str
    rel_path: str
    module: str
    qualname: str
    kind: str
    lineno: int | None
    end_lineno: int | None
    raw_docstring: str | None
    style: str | None
    short_desc: str | None
    long_desc: str | None
    params: object
    returns: object
    raises: object
    examples: object
    created_at: datetime


def docstring_row_to_tuple(row: DocstringRow) -> tuple[object, ...]:
    """
    Serialize a DocstringRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by docstrings INSERTs.
    """
    return (
        row["repo"],
        row["commit"],
        row["rel_path"],
        row["module"],
        row["qualname"],
        row["kind"],
        row["lineno"],
        row["end_lineno"],
        row["raw_docstring"],
        row["style"],
        row["short_desc"],
        row["long_desc"],
        row["params"],
        row["returns"],
        row["raises"],
        row["examples"],
        row["created_at"],
    )


class SymbolUseRow(TypedDict):
    """Row shape for graph.symbol_use_edges inserts."""

    symbol: str
    def_path: str
    use_path: str
    same_file: bool
    same_module: bool
    def_goid_h128: int | None
    use_goid_h128: int | None


def symbol_use_to_tuple(row: SymbolUseRow) -> tuple[object, ...]:
    """
    Serialize a SymbolUseRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by symbol_use_edges INSERTs.
    """
    return (
        row["symbol"],
        row["def_path"],
        row["use_path"],
        row["same_file"],
        row["same_module"],
        row["def_goid_h128"],
        row["use_goid_h128"],
    )


class ConfigValueRow(TypedDict):
    """Row shape for analytics.config_values inserts."""

    repo: str
    commit: str
    config_path: str
    format: str
    key: str
    reference_paths: list[str]
    reference_modules: list[str]
    reference_count: int


def config_value_to_tuple(row: ConfigValueRow) -> tuple[object, ...]:
    """
    Serialize a ConfigValueRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by config_values INSERTs.
    """
    return (
        row["repo"],
        row["commit"],
        row["config_path"],
        row["format"],
        row["key"],
        row["reference_paths"],
        row["reference_modules"],
        row["reference_count"],
    )


class GoidRow(TypedDict):
    """Row shape for core.goids inserts."""

    goid_h128: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    language: str
    kind: str
    qualname: str
    start_line: int | None
    end_line: int | None
    created_at: datetime


def goid_to_tuple(row: GoidRow) -> tuple[object, ...]:
    """
    Serialize a GoidRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by goids INSERTs.
    """
    return (
        row["goid_h128"],
        row["urn"],
        row["repo"],
        row["commit"],
        row["rel_path"],
        row["language"],
        row["kind"],
        row["qualname"],
        row["start_line"],
        row["end_line"],
        row["created_at"],
    )


class GoidCrosswalkRow(TypedDict):
    """Row shape for core.goid_crosswalk inserts."""

    repo: str
    commit: str
    goid: str
    lang: str
    module_path: str
    file_path: str
    start_line: int | None
    end_line: int | None
    scip_symbol: str | None
    ast_qualname: str | None
    cst_node_id: str | None
    chunk_id: str | None
    symbol_id: str | None
    updated_at: datetime


def goid_crosswalk_to_tuple(row: GoidCrosswalkRow) -> tuple[object, ...]:
    """
    Serialize a GoidCrosswalkRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by goid_crosswalk INSERTs.
    """
    return (
        row["repo"],
        row["commit"],
        row["goid"],
        row["lang"],
        row["module_path"],
        row["file_path"],
        row["start_line"],
        row["end_line"],
        row["scip_symbol"],
        row["ast_qualname"],
        row["cst_node_id"],
        row["chunk_id"],
        row["symbol_id"],
        row["updated_at"],
    )


class TypednessRow(TypedDict):
    """Row shape for analytics.typedness inserts."""

    repo: str
    commit: str
    path: str
    type_error_count: int
    annotation_ratio: dict[str, float]
    untyped_defs: int
    overlay_needed: bool


def typedness_row_to_tuple(row: TypednessRow) -> tuple[object, ...]:
    """
    Serialize a TypednessRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by typedness INSERTs.
    """
    return (
        row["repo"],
        row["commit"],
        row["path"],
        row["type_error_count"],
        row["annotation_ratio"],
        row["untyped_defs"],
        row["overlay_needed"],
    )


class StaticDiagnosticRow(TypedDict):
    """Row shape for analytics.static_diagnostics inserts."""

    repo: str
    commit: str
    rel_path: str
    pyrefly_errors: int
    pyright_errors: int
    ruff_errors: int
    total_errors: int
    has_errors: bool


def static_diagnostic_to_tuple(row: StaticDiagnosticRow) -> tuple[object, ...]:
    """
    Serialize a StaticDiagnosticRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by static_diagnostics INSERTs.
    """
    return (
        row["repo"],
        row["commit"],
        row["rel_path"],
        row["pyrefly_errors"],
        row["pyright_errors"],
        row["ruff_errors"],
        row["total_errors"],
        row["has_errors"],
    )


class FunctionValidationRow(TypedDict):
    """Row shape for analytics.function_validation inserts."""

    repo: str
    commit: str
    function_goid_h128: int
    rel_path: str
    qualname: str
    issue: str
    detail: str
    created_at: datetime


def function_validation_row_to_tuple(row: FunctionValidationRow) -> tuple[object, ...]:
    """
    Serialize a FunctionValidationRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by function_validation INSERTs.
    """
    return (
        row["repo"],
        row["commit"],
        row["function_goid_h128"],
        row["rel_path"],
        row["qualname"],
        row["issue"],
        row["detail"],
        row["created_at"],
    )


class GraphValidationRow(TypedDict):
    """Row shape for analytics.graph_validation inserts."""

    repo: str
    commit: str
    graph_name: str
    entity_id: str
    issue: str
    severity: str | None
    rel_path: str | None
    detail: str
    metadata: object | None
    created_at: datetime


def graph_validation_row_to_tuple(row: GraphValidationRow) -> tuple[object, ...]:
    """
    Serialize a GraphValidationRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by graph_validation INSERTs.
    """
    return (
        row["repo"],
        row["commit"],
        row["graph_name"],
        row["entity_id"],
        row["issue"],
        row["severity"],
        row["rel_path"],
        row["detail"],
        row["metadata"],
        row["created_at"],
    )


class HotspotRow(TypedDict):
    """Row shape for analytics.hotspots inserts."""

    rel_path: str
    commit_count: int
    author_count: int
    lines_added: int
    lines_deleted: int
    complexity: float
    score: float


def hotspot_row_to_tuple(row: HotspotRow) -> tuple[object, ...]:
    """
    Serialize a HotspotRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by hotspots INSERTs.
    """
    return (
        row["rel_path"],
        row["commit_count"],
        row["author_count"],
        row["lines_added"],
        row["lines_deleted"],
        row["complexity"],
        row["score"],
    )


class CallGraphNodeRow(TypedDict):
    """Row shape for graph.call_graph_nodes inserts."""

    goid_h128: int
    language: str
    kind: str
    arity: int
    is_public: bool
    rel_path: str


def call_graph_node_to_tuple(row: CallGraphNodeRow) -> tuple[object, ...]:
    """Serialize a CallGraphNodeRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values aligned with call_graph_nodes INSERT order.
    """
    return (
        row["goid_h128"],
        row["language"],
        row["kind"],
        row["arity"],
        row["is_public"],
        row["rel_path"],
    )


class CallGraphEdgeRow(TypedDict):
    """Row shape for graph.call_graph_edges inserts."""

    repo: str
    commit: str
    caller_goid_h128: int
    callee_goid_h128: int | None
    callsite_path: str
    callsite_line: int
    callsite_col: int
    language: str
    kind: str
    resolved_via: str | None
    confidence: float | None
    evidence_json: object


def call_graph_edge_to_tuple(row: CallGraphEdgeRow) -> tuple[object, ...]:
    """Serialize a CallGraphEdgeRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values aligned with call_graph_edges INSERT order.
    """
    return (
        row["repo"],
        row["commit"],
        row["caller_goid_h128"],
        row["callee_goid_h128"],
        row["callsite_path"],
        row["callsite_line"],
        row["callsite_col"],
        row["language"],
        row["kind"],
        row["resolved_via"],
        row["confidence"],
        row["evidence_json"],
    )


class ImportEdgeRow(TypedDict):
    """Row shape for graph.import_graph_edges inserts."""

    repo: str
    commit: str
    src_module: str
    dst_module: str
    src_fan_out: int
    dst_fan_in: int
    cycle_group: int
    module_layer: int | None


def import_edge_to_tuple(row: ImportEdgeRow) -> tuple[object, ...]:
    """Serialize an ImportEdgeRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values aligned with import_graph_edges INSERT order.
    """
    return (
        row["repo"],
        row["commit"],
        row["src_module"],
        row["dst_module"],
        row["src_fan_out"],
        row["dst_fan_in"],
        row["cycle_group"],
        row.get("module_layer"),
    )


class ImportModuleRow(TypedDict):
    """Row shape for graph.import_modules inserts."""

    repo: str
    commit: str
    module: str
    scc_id: int
    component_size: int
    layer: int | None
    cycle_group: int


def import_module_to_tuple(row: ImportModuleRow) -> tuple[object, ...]:
    """
    Serialize an ImportModuleRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values aligned with import_modules INSERT order.
    """
    return (
        row["repo"],
        row["commit"],
        row["module"],
        row["scc_id"],
        row["component_size"],
        row.get("layer"),
        row["cycle_group"],
    )


class CFGBlockRow(TypedDict):
    """Row shape for graph.cfg_blocks inserts."""

    function_goid_h128: int
    block_idx: int
    block_id: str
    label: str
    file_path: str
    start_line: int
    end_line: int
    kind: str
    stmts_json: object
    in_degree: int
    out_degree: int


def cfg_block_to_tuple(row: CFGBlockRow) -> tuple[object, ...]:
    """Serialize a CFGBlockRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values aligned with cfg_blocks INSERT order.
    """
    return (
        row["function_goid_h128"],
        row["block_idx"],
        row["block_id"],
        row["label"],
        row["file_path"],
        row["start_line"],
        row["end_line"],
        row["kind"],
        row["stmts_json"],
        row["in_degree"],
        row["out_degree"],
    )


class CFGEdgeRow(TypedDict):
    """Row shape for graph.cfg_edges inserts."""

    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    edge_kind: str | None


def cfg_edge_to_tuple(row: CFGEdgeRow) -> tuple[object, ...]:
    """Serialize a CFGEdgeRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values aligned with cfg_edges INSERT order.
    """
    return (
        row["function_goid_h128"],
        row["src_block_id"],
        row["dst_block_id"],
        row["edge_kind"],
    )


class DFGEdgeRow(TypedDict):
    """Row shape for graph.dfg_edges inserts."""

    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    src_var: str | None
    dst_var: str | None
    edge_kind: str | None
    via_phi: bool
    use_kind: str | None


def dfg_edge_to_tuple(row: DFGEdgeRow) -> tuple[object, ...]:
    """Serialize a DFGEdgeRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values aligned with dfg_edges INSERT order.
    """
    return (
        row["function_goid_h128"],
        row["src_block_id"],
        row["dst_block_id"],
        row["src_var"],
        row["dst_var"],
        row["edge_kind"],
        row["via_phi"],
        row["use_kind"],
    )


class TestCatalogRowModel(TypedDict):
    """Row shape for analytics.test_catalog inserts."""

    test_id: str
    test_goid_h128: int | None
    urn: str | None
    repo: str
    commit: str
    rel_path: str
    qualname: str | None
    kind: str
    status: str
    duration_ms: float
    markers: list[str]
    parametrized: bool
    flaky: bool
    created_at: datetime


def serialize_test_catalog_row(row: TestCatalogRowModel) -> tuple[object, ...]:
    """
    Serialize a TestCatalogRowModel into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by test_catalog INSERTs.
    """
    return (
        row["test_id"],
        row["test_goid_h128"],
        row["urn"],
        row["repo"],
        row["commit"],
        row["rel_path"],
        row["qualname"],
        row["kind"],
        row["status"],
        row["duration_ms"],
        row["markers"],
        row["parametrized"],
        row["flaky"],
        row["created_at"],
    )


class TestCoverageEdgeRow(TypedDict):
    """Row shape for analytics.test_coverage_edges inserts."""

    test_id: str
    test_goid_h128: int | None
    function_goid_h128: int
    urn: str | None
    repo: str
    commit: str
    rel_path: str
    qualname: str | None
    covered_lines: int
    executable_lines: int
    coverage_ratio: float
    last_status: str
    created_at: datetime


_TestCoverageEdgeColumn = Literal[
    "test_id",
    "test_goid_h128",
    "function_goid_h128",
    "urn",
    "repo",
    "commit",
    "rel_path",
    "qualname",
    "covered_lines",
    "executable_lines",
    "coverage_ratio",
    "last_status",
    "created_at",
]
TEST_COVERAGE_EDGE_COLUMNS: tuple[_TestCoverageEdgeColumn, ...] = (
    "test_id",
    "test_goid_h128",
    "function_goid_h128",
    "urn",
    "repo",
    "commit",
    "rel_path",
    "qualname",
    "covered_lines",
    "executable_lines",
    "coverage_ratio",
    "last_status",
    "created_at",
)


def serialize_test_coverage_edge(row: TestCoverageEdgeRow) -> tuple[object, ...]:
    """
    Serialize a TestCoverageEdgeRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by test_coverage_edges INSERTs.
    """
    return _serialize_row(row, TEST_COVERAGE_EDGE_COLUMNS)


_FunctionProfileColumn = Literal[
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
]
FUNCTION_PROFILE_COLUMNS: tuple[_FunctionProfileColumn, ...] = (
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
    """
    Serialize a FunctionProfileRowModel into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by function_profile INSERTs.
    """
    return _serialize_row(row, FUNCTION_PROFILE_COLUMNS)


_FileProfileColumn = Literal[
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
]
FILE_PROFILE_COLUMNS: tuple[_FileProfileColumn, ...] = (
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
    """
    Serialize a FileProfileRowModel into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by file_profile INSERTs.
    """
    return _serialize_row(row, FILE_PROFILE_COLUMNS)


_ModuleProfileColumn = Literal[
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
]
MODULE_PROFILE_COLUMNS: tuple[_ModuleProfileColumn, ...] = (
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
    """
    Serialize a ModuleProfileRowModel into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by module_profile INSERTs.
    """
    return _serialize_row(row, MODULE_PROFILE_COLUMNS)


_TestProfileColumn = Literal[
    "repo",
    "commit",
    "test_id",
    "test_goid_h128",
    "urn",
    "rel_path",
    "module",
    "qualname",
    "language",
    "kind",
    "status",
    "duration_ms",
    "markers",
    "flaky",
    "last_run_at",
    "functions_covered",
    "functions_covered_count",
    "primary_function_goids",
    "subsystems_covered",
    "subsystems_covered_count",
    "primary_subsystem_id",
    "assert_count",
    "raise_count",
    "uses_parametrize",
    "uses_fixtures",
    "io_bound",
    "uses_network",
    "uses_db",
    "uses_filesystem",
    "uses_subprocess",
    "flakiness_score",
    "importance_score",
    "notes",
    "tg_degree",
    "tg_weighted_degree",
    "tg_proj_degree",
    "tg_proj_weight",
    "tg_proj_clustering",
    "tg_proj_betweenness",
    "created_at",
]
TEST_PROFILE_COLUMNS: tuple[_TestProfileColumn, ...] = (
    "repo",
    "commit",
    "test_id",
    "test_goid_h128",
    "urn",
    "rel_path",
    "module",
    "qualname",
    "language",
    "kind",
    "status",
    "duration_ms",
    "markers",
    "flaky",
    "last_run_at",
    "functions_covered",
    "functions_covered_count",
    "primary_function_goids",
    "subsystems_covered",
    "subsystems_covered_count",
    "primary_subsystem_id",
    "assert_count",
    "raise_count",
    "uses_parametrize",
    "uses_fixtures",
    "io_bound",
    "uses_network",
    "uses_db",
    "uses_filesystem",
    "uses_subprocess",
    "flakiness_score",
    "importance_score",
    "notes",
    "tg_degree",
    "tg_weighted_degree",
    "tg_proj_degree",
    "tg_proj_weight",
    "tg_proj_clustering",
    "tg_proj_betweenness",
    "created_at",
)


class TestProfileRowModel(TypedDict):
    """Row shape for ``analytics.test_profile`` inserts."""

    repo: str
    commit: str
    test_id: str
    test_goid_h128: int | None
    urn: str | None
    rel_path: str
    module: str | None
    qualname: str | None
    language: str | None
    kind: str | None
    status: str | None
    duration_ms: float | None
    markers: object
    flaky: bool | None
    last_run_at: datetime | None
    functions_covered: object
    functions_covered_count: int | None
    primary_function_goids: object
    subsystems_covered: object
    subsystems_covered_count: int | None
    primary_subsystem_id: str | None
    assert_count: int | None
    raise_count: int | None
    uses_parametrize: bool | None
    uses_fixtures: bool | None
    io_bound: bool | None
    uses_network: bool | None
    uses_db: bool | None
    uses_filesystem: bool | None
    uses_subprocess: bool | None
    flakiness_score: float | None
    importance_score: float | None
    notes: str | None
    tg_degree: int | None
    tg_weighted_degree: float | None
    tg_proj_degree: int | None
    tg_proj_weight: float | None
    tg_proj_clustering: float | None
    tg_proj_betweenness: float | None
    created_at: datetime


def serialize_test_profile_row(row: TestProfileRowModel) -> tuple[object, ...]:
    """
    Serialize a TestProfileRowModel into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by test_profile INSERTs.
    """
    return _serialize_row(row, TEST_PROFILE_COLUMNS)


_BehavioralCoverageColumn = Literal[
    "repo",
    "commit",
    "test_id",
    "test_goid_h128",
    "rel_path",
    "qualname",
    "behavior_tags",
    "tag_source",
    "heuristic_version",
    "llm_model",
    "llm_run_id",
    "created_at",
]
BEHAVIORAL_COVERAGE_COLUMNS: tuple[_BehavioralCoverageColumn, ...] = (
    "repo",
    "commit",
    "test_id",
    "test_goid_h128",
    "rel_path",
    "qualname",
    "behavior_tags",
    "tag_source",
    "heuristic_version",
    "llm_model",
    "llm_run_id",
    "created_at",
)


class BehavioralCoverageRowModel(TypedDict):
    """Row shape for ``analytics.behavioral_coverage`` inserts."""

    repo: str
    commit: str
    test_id: str
    test_goid_h128: int | None
    rel_path: str
    qualname: str | None
    behavior_tags: object
    tag_source: str
    heuristic_version: str | None
    llm_model: str | None
    llm_run_id: str | None
    created_at: datetime


def behavioral_coverage_row_to_tuple(row: BehavioralCoverageRowModel) -> tuple[object, ...]:
    """
    Serialize a BehavioralCoverageRowModel into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by behavioral_coverage INSERTs.
    """
    return _serialize_row(row, BEHAVIORAL_COVERAGE_COLUMNS)
