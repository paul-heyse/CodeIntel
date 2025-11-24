"""Typed row models for DuckDB inserts with helpers to keep column order stable."""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

__all__ = [
    "CFGBlockRow",
    "CFGEdgeRow",
    "CallGraphEdgeRow",
    "CallGraphNodeRow",
    "ConfigValueRow",
    "CoverageLineRow",
    "DFGEdgeRow",
    "DocstringRow",
    "FunctionValidationRow",
    "GoidCrosswalkRow",
    "GoidRow",
    "HotspotRow",
    "ImportEdgeRow",
    "StaticDiagnosticRow",
    "SymbolUseRow",
    "TestCatalogRowModel",
    "TestCoverageEdgeRow",
    "TypednessRow",
    "call_graph_edge_to_tuple",
    "call_graph_node_to_tuple",
    "cfg_block_to_tuple",
    "cfg_edge_to_tuple",
    "config_value_to_tuple",
    "coverage_line_to_tuple",
    "dfg_edge_to_tuple",
    "docstring_row_to_tuple",
    "function_validation_row_to_tuple",
    "goid_crosswalk_to_tuple",
    "goid_to_tuple",
    "hotspot_row_to_tuple",
    "import_edge_to_tuple",
    "static_diagnostic_to_tuple",
    "symbol_use_to_tuple",
    "test_catalog_row_to_tuple",
    "test_coverage_edge_to_tuple",
    "typedness_row_to_tuple",
]


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
    )


class ConfigValueRow(TypedDict):
    """Row shape for analytics.config_values inserts."""

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
        row["path"],
        row["type_error_count"],
        row["annotation_ratio"],
        row["untyped_defs"],
        row["overlay_needed"],
    )


class StaticDiagnosticRow(TypedDict):
    """Row shape for analytics.static_diagnostics inserts."""

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
    rel_path: str
    qualname: str
    issue: str
    detail: str | None
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
        row["rel_path"],
        row["qualname"],
        row["issue"],
        row["detail"],
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


def test_catalog_row_to_tuple(row: TestCatalogRowModel) -> tuple[object, ...]:
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
    urn: str
    repo: str
    commit: str
    rel_path: str
    qualname: str
    covered_lines: int
    executable_lines: int
    coverage_ratio: float
    last_status: str
    created_at: datetime


def test_coverage_edge_to_tuple(row: TestCoverageEdgeRow) -> tuple[object, ...]:
    """
    Serialize a TestCoverageEdgeRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by test_coverage_edges INSERTs.
    """
    return (
        row["test_id"],
        row["test_goid_h128"],
        row["function_goid_h128"],
        row["urn"],
        row["repo"],
        row["commit"],
        row["rel_path"],
        row["qualname"],
        row["covered_lines"],
        row["executable_lines"],
        row["coverage_ratio"],
        row["last_status"],
        row["created_at"],
    )
