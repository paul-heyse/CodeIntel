"""Table schema registry for ingestion and exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ColumnType = Literal[
    "BOOLEAN",
    "INTEGER",
    "BIGINT",
    "DOUBLE",
    "DECIMAL",
    "DECIMAL(38,0)",
    "VARCHAR",
    "JSON",
    "TIMESTAMP",
]
COLUMN_TYPE = ColumnType


@dataclass(frozen=True)
class Column:
    """Definition of a single table column."""

    name: str
    type: ColumnType
    nullable: bool = True
    description: str | None = None


@dataclass(frozen=True)
class TableSchema:
    """Schema definition for a DuckDB table."""

    schema: str
    name: str
    columns: list[Column]
    primary_key: tuple[str, ...] = ()
    indexes: tuple[Index, ...] = ()
    description: str | None = None

    @property
    def fq_name(self) -> str:
        """Fully qualified table name."""
        return f"{self.schema}.{self.name}"

    def column_names(self) -> list[str]:
        """
        Ordered column names.

        Returns
        -------
        list[str]
            Column names in definition order.
        """
        return [col.name for col in self.columns]


@dataclass(frozen=True)
class Index:
    """Secondary index definition."""

    name: str
    columns: tuple[str, ...]
    unique: bool = False


TABLE_SCHEMAS: dict[str, TableSchema] = {
    "core.ast_nodes": TableSchema(
        schema="core",
        name="ast_nodes",
        columns=[
            Column("path", "VARCHAR", nullable=False, description="Relative path to source file"),
            Column("node_type", "VARCHAR", nullable=False),
            Column("name", "VARCHAR"),
            Column("qualname", "VARCHAR"),
            Column("lineno", "INTEGER"),
            Column("end_lineno", "INTEGER"),
            Column("col_offset", "INTEGER"),
            Column("end_col_offset", "INTEGER"),
            Column("parent_qualname", "VARCHAR"),
            Column("decorators", "JSON"),
            Column("docstring", "VARCHAR"),
            Column("hash", "VARCHAR", nullable=False),
        ],
        primary_key=("hash",),
        description="Flattened AST nodes",
    ),
    "core.ast_metrics": TableSchema(
        schema="core",
        name="ast_metrics",
        columns=[
            Column("rel_path", "VARCHAR", nullable=False),
            Column("node_count", "INTEGER", nullable=False),
            Column("function_count", "INTEGER", nullable=False),
            Column("class_count", "INTEGER", nullable=False),
            Column("avg_depth", "DOUBLE", nullable=False),
            Column("max_depth", "INTEGER", nullable=False),
            Column("complexity", "DOUBLE", nullable=False),
            Column("generated_at", "TIMESTAMP", nullable=False),
        ],
        primary_key=("rel_path",),
        description="Per-file AST metrics",
    ),
    "core.cst_nodes": TableSchema(
        schema="core",
        name="cst_nodes",
        columns=[
            Column("path", "VARCHAR", nullable=False),
            Column("node_id", "VARCHAR", nullable=False),
            Column("kind", "VARCHAR", nullable=False),
            Column("span", "JSON", nullable=False),
            Column("text_preview", "VARCHAR"),
            Column("parents", "JSON"),
            Column("qnames", "JSON"),
        ],
        primary_key=("node_id",),
        description="Concrete syntax tree nodes",
    ),
    "core.docstrings": TableSchema(
        schema="core",
        name="docstrings",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("rel_path", "VARCHAR", nullable=False),
            Column("module", "VARCHAR", nullable=False),
            Column("qualname", "VARCHAR", nullable=False),
            Column("kind", "VARCHAR", nullable=False),
            Column("lineno", "INTEGER"),
            Column("end_lineno", "INTEGER"),
            Column("raw_docstring", "VARCHAR"),
            Column("style", "VARCHAR"),
            Column("short_desc", "VARCHAR"),
            Column("long_desc", "VARCHAR"),
            Column("params", "JSON"),
            Column("returns", "JSON"),
            Column("raises", "JSON"),
            Column("examples", "JSON"),
            Column("created_at", "TIMESTAMP", nullable=False),
        ],
        description="Structured docstring facts extracted with griffe",
    ),
    "core.modules": TableSchema(
        schema="core",
        name="modules",
        columns=[
            Column("module", "VARCHAR", nullable=False),
            Column("path", "VARCHAR", nullable=False),
            Column("repo", "VARCHAR"),
            Column("commit", "VARCHAR"),
            Column("language", "VARCHAR"),
            Column("tags", "JSON"),
            Column("owners", "JSON"),
        ],
        primary_key=("module", "path"),
        indexes=(
            Index("idx_core_modules_path", ("path",)),
            Index("idx_core_modules_module", ("module",)),
        ),
        description="Discovered modules per repo/commit",
    ),
    "core.repo_map": TableSchema(
        schema="core",
        name="repo_map",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("modules", "JSON"),
            Column("overlays", "JSON"),
            Column("generated_at", "TIMESTAMP"),
        ],
        primary_key=("repo", "commit"),
        description="Per-commit module manifest and overlays",
    ),
    "core.goids": TableSchema(
        schema="core",
        name="goids",
        columns=[
            Column("goid_h128", "DECIMAL(38,0)", nullable=False),
            Column("urn", "VARCHAR", nullable=False),
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("rel_path", "VARCHAR", nullable=False),
            Column("language", "VARCHAR", nullable=False),
            Column("kind", "VARCHAR", nullable=False),
            Column("qualname", "VARCHAR", nullable=False),
            Column("start_line", "INTEGER"),
            Column("end_line", "INTEGER"),
            Column("created_at", "TIMESTAMP", nullable=False),
        ],
        primary_key=("goid_h128",),
        indexes=(
            Index("idx_core_goids_h128", ("goid_h128",), unique=True),
            Index("idx_core_goids_urn", ("urn",), unique=True),
            Index("idx_core_goids_path", ("rel_path",)),
        ),
        description="Global object identifiers for code entities",
    ),
    "core.goid_crosswalk": TableSchema(
        schema="core",
        name="goid_crosswalk",
        columns=[
            Column("goid", "VARCHAR", nullable=False),
            Column("lang", "VARCHAR", nullable=False),
            Column("module_path", "VARCHAR", nullable=False),
            Column("file_path", "VARCHAR", nullable=False),
            Column("start_line", "INTEGER"),
            Column("end_line", "INTEGER"),
            Column("scip_symbol", "VARCHAR"),
            Column("ast_qualname", "VARCHAR"),
            Column("cst_node_id", "VARCHAR"),
            Column("chunk_id", "VARCHAR"),
            Column("symbol_id", "VARCHAR"),
            Column("updated_at", "TIMESTAMP", nullable=False),
        ],
        indexes=(Index("idx_core_gcw_goid", ("goid",)),),
        description="Crosswalk from GOIDs to language-specific symbols/paths",
    ),
    "analytics.coverage_lines": TableSchema(
        schema="analytics",
        name="coverage_lines",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("rel_path", "VARCHAR", nullable=False),
            Column("line", "INTEGER", nullable=False),
            Column("is_executable", "BOOLEAN", nullable=False),
            Column("is_covered", "BOOLEAN", nullable=False),
            Column("hits", "INTEGER", nullable=False),
            Column("context_count", "INTEGER", nullable=False),
            Column("created_at", "TIMESTAMP", nullable=False),
        ],
        indexes=(
            Index("idx_analytics_cov_lines_repo_path", ("repo", "commit", "rel_path")),
            Index("idx_analytics_cov_lines_line", ("line",)),
        ),
        description="Line-level coverage facts",
    ),
    "analytics.test_catalog": TableSchema(
        schema="analytics",
        name="test_catalog",
        columns=[
            Column("test_id", "VARCHAR", nullable=False),
            Column("test_goid_h128", "DOUBLE"),
            Column("urn", "VARCHAR"),
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("rel_path", "VARCHAR", nullable=False),
            Column("qualname", "VARCHAR"),
            Column("kind", "VARCHAR"),
            Column("status", "VARCHAR"),
            Column("duration_ms", "DOUBLE"),
            Column("markers", "JSON"),
            Column("parametrized", "BOOLEAN"),
            Column("flaky", "BOOLEAN"),
            Column("created_at", "TIMESTAMP"),
        ],
        primary_key=("test_id",),
        indexes=(Index("idx_analytics_test_catalog_id", ("test_id",), unique=True),),
        description="Pytest test catalog",
    ),
    "analytics.config_values": TableSchema(
        schema="analytics",
        name="config_values",
        columns=[
            Column("config_path", "VARCHAR", nullable=False),
            Column("format", "VARCHAR", nullable=False),
            Column("key", "VARCHAR", nullable=False),
            Column("reference_paths", "JSON"),
            Column("reference_modules", "JSON"),
            Column("reference_count", "INTEGER", nullable=False),
        ],
        description="Flattened config key/value paths",
    ),
    "analytics.tags_index": TableSchema(
        schema="analytics",
        name="tags_index",
        columns=[
            Column("tag", "VARCHAR", nullable=False),
            Column("description", "VARCHAR"),
            Column("includes", "JSON"),
            Column("excludes", "JSON"),
            Column("matches", "JSON"),
        ],
        primary_key=("tag",),
        description="Path classification rules",
    ),
    "analytics.typedness": TableSchema(
        schema="analytics",
        name="typedness",
        columns=[
            Column("path", "VARCHAR", nullable=False),
            Column("type_error_count", "INTEGER", nullable=False),
            Column("annotation_ratio", "JSON", nullable=False),
            Column("untyped_defs", "INTEGER", nullable=False),
            Column("overlay_needed", "BOOLEAN", nullable=False),
        ],
        primary_key=("path",),
        description="Per-file annotation ratios and static error counts",
    ),
    "analytics.static_diagnostics": TableSchema(
        schema="analytics",
        name="static_diagnostics",
        columns=[
            Column("rel_path", "VARCHAR", nullable=False),
            Column("pyrefly_errors", "INTEGER", nullable=False),
            Column("pyright_errors", "INTEGER", nullable=False),
            Column("total_errors", "INTEGER", nullable=False),
            Column("has_errors", "BOOLEAN", nullable=False),
        ],
        primary_key=("rel_path",),
        description="Per-file static diagnostic counts",
    ),
    "analytics.test_coverage_edges": TableSchema(
        schema="analytics",
        name="test_coverage_edges",
        columns=[
            Column("test_id", "VARCHAR"),
            Column("test_goid_h128", "DECIMAL(38,0)"),
            Column("function_goid_h128", "DECIMAL(38,0)"),
            Column("urn", "VARCHAR"),
            Column("repo", "VARCHAR"),
            Column("commit", "VARCHAR"),
            Column("rel_path", "VARCHAR"),
            Column("qualname", "VARCHAR"),
            Column("covered_lines", "INTEGER"),
            Column("executable_lines", "INTEGER"),
            Column("coverage_ratio", "DOUBLE"),
            Column("last_status", "VARCHAR"),
            Column("created_at", "TIMESTAMP"),
        ],
        indexes=(Index("idx_analytics_test_cov_edges_goid", ("function_goid_h128",)),),
        description="Per-test coverage edges between tests and functions",
    ),
    "analytics.function_metrics": TableSchema(
        schema="analytics",
        name="function_metrics",
        columns=[
            Column("function_goid_h128", "DECIMAL(38,0)"),
            Column("urn", "VARCHAR"),
            Column("repo", "VARCHAR"),
            Column("commit", "VARCHAR"),
            Column("rel_path", "VARCHAR"),
            Column("language", "VARCHAR"),
            Column("kind", "VARCHAR"),
            Column("qualname", "VARCHAR"),
            Column("start_line", "INTEGER"),
            Column("end_line", "INTEGER"),
            Column("loc", "INTEGER"),
            Column("logical_loc", "INTEGER"),
            Column("param_count", "INTEGER"),
            Column("positional_params", "INTEGER"),
            Column("keyword_only_params", "INTEGER"),
            Column("has_varargs", "BOOLEAN"),
            Column("has_varkw", "BOOLEAN"),
            Column("is_async", "BOOLEAN"),
            Column("is_generator", "BOOLEAN"),
            Column("return_count", "INTEGER"),
            Column("yield_count", "INTEGER"),
            Column("raise_count", "INTEGER"),
            Column("cyclomatic_complexity", "INTEGER"),
            Column("max_nesting_depth", "INTEGER"),
            Column("stmt_count", "INTEGER"),
            Column("decorator_count", "INTEGER"),
            Column("has_docstring", "BOOLEAN"),
            Column("complexity_bucket", "VARCHAR"),
            Column("created_at", "TIMESTAMP"),
        ],
        indexes=(Index("idx_analytics_function_metrics_goid", ("function_goid_h128",)),),
        description="Per-function structural metrics",
    ),
    "analytics.function_types": TableSchema(
        schema="analytics",
        name="function_types",
        columns=[
            Column("function_goid_h128", "DECIMAL(38,0)"),
            Column("urn", "VARCHAR"),
            Column("repo", "VARCHAR"),
            Column("commit", "VARCHAR"),
            Column("rel_path", "VARCHAR"),
            Column("language", "VARCHAR"),
            Column("kind", "VARCHAR"),
            Column("qualname", "VARCHAR"),
            Column("start_line", "INTEGER"),
            Column("end_line", "INTEGER"),
            Column("total_params", "INTEGER"),
            Column("annotated_params", "INTEGER"),
            Column("unannotated_params", "INTEGER"),
            Column("param_typed_ratio", "DOUBLE"),
            Column("has_return_annotation", "BOOLEAN"),
            Column("return_type", "VARCHAR"),
            Column("return_type_source", "VARCHAR"),
            Column("type_comment", "VARCHAR"),
            Column("param_types", "JSON"),
            Column("fully_typed", "BOOLEAN"),
            Column("partial_typed", "BOOLEAN"),
            Column("untyped", "BOOLEAN"),
            Column("typedness_bucket", "VARCHAR"),
            Column("typedness_source", "VARCHAR"),
            Column("created_at", "TIMESTAMP"),
        ],
        indexes=(Index("idx_analytics_function_types_goid", ("function_goid_h128",)),),
        description="Per-function annotation coverage",
    ),
    "analytics.coverage_functions": TableSchema(
        schema="analytics",
        name="coverage_functions",
        columns=[
            Column("function_goid_h128", "DECIMAL(38,0)"),
            Column("urn", "VARCHAR"),
            Column("repo", "VARCHAR"),
            Column("commit", "VARCHAR"),
            Column("rel_path", "VARCHAR"),
            Column("language", "VARCHAR"),
            Column("kind", "VARCHAR"),
            Column("qualname", "VARCHAR"),
            Column("start_line", "INTEGER"),
            Column("end_line", "INTEGER"),
            Column("executable_lines", "INTEGER"),
            Column("covered_lines", "INTEGER"),
            Column("coverage_ratio", "DOUBLE"),
            Column("tested", "BOOLEAN"),
            Column("untested_reason", "VARCHAR"),
            Column("created_at", "TIMESTAMP"),
        ],
        indexes=(Index("idx_analytics_coverage_functions_goid", ("function_goid_h128",)),),
        description="Line coverage aggregates per function",
    ),
    "analytics.goid_risk_factors": TableSchema(
        schema="analytics",
        name="goid_risk_factors",
        columns=[
            Column("function_goid_h128", "DECIMAL(38,0)"),
            Column("urn", "VARCHAR"),
            Column("repo", "VARCHAR"),
            Column("commit", "VARCHAR"),
            Column("rel_path", "VARCHAR"),
            Column("language", "VARCHAR"),
            Column("kind", "VARCHAR"),
            Column("qualname", "VARCHAR"),
            Column("loc", "INTEGER"),
            Column("logical_loc", "INTEGER"),
            Column("cyclomatic_complexity", "INTEGER"),
            Column("complexity_bucket", "VARCHAR"),
            Column("typedness_bucket", "VARCHAR"),
            Column("typedness_source", "VARCHAR"),
            Column("hotspot_score", "DOUBLE"),
            Column("file_typed_ratio", "DOUBLE"),
            Column("static_error_count", "INTEGER"),
            Column("has_static_errors", "BOOLEAN"),
            Column("executable_lines", "INTEGER"),
            Column("covered_lines", "INTEGER"),
            Column("coverage_ratio", "DOUBLE"),
            Column("tested", "BOOLEAN"),
            Column("test_count", "INTEGER"),
            Column("failing_test_count", "INTEGER"),
            Column("last_test_status", "VARCHAR"),
            Column("risk_score", "DOUBLE"),
            Column("risk_level", "VARCHAR"),
            Column("tags", "JSON"),
            Column("owners", "JSON"),
            Column("created_at", "TIMESTAMP"),
        ],
        indexes=(Index("idx_analytics_gorf_goid", ("function_goid_h128",)),),
        description="Composite risk factors per function",
    ),
    "analytics.hotspots": TableSchema(
        schema="analytics",
        name="hotspots",
        columns=[
            Column("rel_path", "VARCHAR"),
            Column("commit_count", "INTEGER"),
            Column("author_count", "INTEGER"),
            Column("lines_added", "INTEGER"),
            Column("lines_deleted", "INTEGER"),
            Column("complexity", "DOUBLE"),
            Column("score", "DOUBLE"),
        ],
        description="File-level hotspot scores",
    ),
    "graph.call_graph_nodes": TableSchema(
        schema="graph",
        name="call_graph_nodes",
        columns=[
            Column("goid_h128", "DECIMAL(38,0)", nullable=False),
            Column("language", "VARCHAR", nullable=False),
            Column("kind", "VARCHAR", nullable=False),
            Column("arity", "INTEGER", nullable=False),
            Column("is_public", "BOOLEAN", nullable=False),
            Column("rel_path", "VARCHAR", nullable=False),
        ],
        primary_key=("goid_h128",),
        description="Functions and methods participating in the call graph",
    ),
    "graph.call_graph_edges": TableSchema(
        schema="graph",
        name="call_graph_edges",
        columns=[
            Column("caller_goid_h128", "DECIMAL(38,0)", nullable=False),
            Column("callee_goid_h128", "DECIMAL(38,0)"),
            Column("callsite_path", "VARCHAR", nullable=False),
            Column("callsite_line", "INTEGER", nullable=False),
            Column("callsite_col", "INTEGER", nullable=False),
            Column("language", "VARCHAR", nullable=False),
            Column("kind", "VARCHAR", nullable=False),
            Column("resolved_via", "VARCHAR"),
            Column("confidence", "DOUBLE"),
            Column("evidence_json", "JSON"),
        ],
        indexes=(
            Index("idx_graph_call_edges_caller", ("caller_goid_h128",)),
            Index("idx_graph_call_edges_callee", ("callee_goid_h128",)),
        ),
        description="Caller->callee edges with callsite evidence",
    ),
    "graph.import_graph_edges": TableSchema(
        schema="graph",
        name="import_graph_edges",
        columns=[
            Column("src_module", "VARCHAR", nullable=False),
            Column("dst_module", "VARCHAR", nullable=False),
            Column("src_fan_out", "INTEGER", nullable=False),
            Column("dst_fan_in", "INTEGER", nullable=False),
            Column("cycle_group", "INTEGER", nullable=False),
        ],
        indexes=(
            Index("idx_graph_import_edges_src", ("src_module",)),
            Index("idx_graph_import_edges_dst", ("dst_module",)),
        ),
        description="Module-level import edges with fan-in/out metrics",
    ),
    "graph.cfg_blocks": TableSchema(
        schema="graph",
        name="cfg_blocks",
        columns=[
            Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),
            Column("block_idx", "INTEGER", nullable=False),
            Column("block_id", "VARCHAR", nullable=False),
            Column("label", "VARCHAR", nullable=False),
            Column("file_path", "VARCHAR", nullable=False),
            Column("start_line", "INTEGER", nullable=False),
            Column("end_line", "INTEGER", nullable=False),
            Column("kind", "VARCHAR", nullable=False),
            Column("stmts_json", "JSON", nullable=False),
            Column("in_degree", "INTEGER", nullable=False),
            Column("out_degree", "INTEGER", nullable=False),
        ],
        primary_key=("function_goid_h128", "block_idx"),
        indexes=(Index("idx_graph_cfg_blocks_fn", ("function_goid_h128",)),),
        description="Control-flow blocks per function",
    ),
    "graph.cfg_edges": TableSchema(
        schema="graph",
        name="cfg_edges",
        columns=[
            Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),
            Column("src_block_id", "VARCHAR", nullable=False),
            Column("dst_block_id", "VARCHAR", nullable=False),
            Column("edge_kind", "VARCHAR"),
        ],
        indexes=(Index("idx_graph_cfg_edges_fn", ("function_goid_h128",)),),
        description="Control-flow edges between blocks",
    ),
    "graph.dfg_edges": TableSchema(
        schema="graph",
        name="dfg_edges",
        columns=[
            Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),
            Column("src_block_id", "VARCHAR", nullable=False),
            Column("dst_block_id", "VARCHAR", nullable=False),
            Column("src_var", "VARCHAR"),
            Column("dst_var", "VARCHAR"),
            Column("edge_kind", "VARCHAR"),
        ],
        indexes=(Index("idx_graph_dfg_edges_fn", ("function_goid_h128",)),),
        description="Data-flow edges between blocks/vars",
    ),
    "graph.symbol_use_edges": TableSchema(
        schema="graph",
        name="symbol_use_edges",
        columns=[
            Column("symbol", "VARCHAR", nullable=False),
            Column("def_path", "VARCHAR", nullable=False),
            Column("use_path", "VARCHAR", nullable=False),
            Column("same_file", "BOOLEAN", nullable=False),
            Column("same_module", "BOOLEAN", nullable=False),
        ],
        primary_key=("symbol", "def_path", "use_path"),
        indexes=(Index("idx_graph_symbol_use_symbol", ("symbol",)),),
        description="Definition-to-use edges derived from SCIP",
    ),
}
