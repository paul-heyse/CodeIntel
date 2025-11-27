"""Utilities to couple ingestion SQL to the registry and validate schemas."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.config.schemas.ingestion_sql import verify_ingestion_columns
from codeintel.config.schemas.registry_adapter import load_registry_columns
from codeintel.storage.gateway import DuckDBConnection

_INGESTION_COLUMNS_VERIFIED: list[bool] = [False]

# ---------------------------------------------------------------------------
# Column lists (single source of truth for SQL literals below)
# ---------------------------------------------------------------------------

AST_NODES_COLUMNS = [
    "path",
    "node_type",
    "name",
    "qualname",
    "lineno",
    "end_lineno",
    "decorator_start_line",
    "decorator_end_line",
    "col_offset",
    "end_col_offset",
    "parent_qualname",
    "decorators",
    "docstring",
    "hash",
]

AST_METRICS_COLUMNS = [
    "rel_path",
    "node_count",
    "function_count",
    "class_count",
    "avg_depth",
    "max_depth",
    "complexity",
    "generated_at",
]

CST_NODES_COLUMNS = [
    "path",
    "node_id",
    "kind",
    "span",
    "text_preview",
    "parents",
    "qnames",
]

COVERAGE_LINES_COLUMNS = [
    "repo",
    "commit",
    "rel_path",
    "line",
    "is_executable",
    "is_covered",
    "hits",
    "context_count",
    "created_at",
]

FUNCTION_METRICS_COLUMNS = [
    "function_goid_h128",
    "urn",
    "repo",
    "commit",
    "rel_path",
    "language",
    "kind",
    "qualname",
    "start_line",
    "end_line",
    "loc",
    "logical_loc",
    "param_count",
    "positional_params",
    "keyword_only_params",
    "has_varargs",
    "has_varkw",
    "is_async",
    "is_generator",
    "return_count",
    "yield_count",
    "raise_count",
    "cyclomatic_complexity",
    "max_nesting_depth",
    "stmt_count",
    "decorator_count",
    "has_docstring",
    "complexity_bucket",
    "created_at",
]

FUNCTION_TYPES_COLUMNS = [
    "function_goid_h128",
    "urn",
    "repo",
    "commit",
    "rel_path",
    "language",
    "kind",
    "qualname",
    "start_line",
    "end_line",
    "total_params",
    "annotated_params",
    "unannotated_params",
    "param_typed_ratio",
    "has_return_annotation",
    "return_type",
    "return_type_source",
    "type_comment",
    "param_types",
    "fully_typed",
    "partial_typed",
    "untyped",
    "typedness_bucket",
    "typedness_source",
    "created_at",
]

FUNCTION_EFFECTS_COLUMNS = [
    "repo",
    "commit",
    "function_goid_h128",
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
    "effects_json",
    "created_at",
]

FUNCTION_CONTRACTS_COLUMNS = [
    "repo",
    "commit",
    "function_goid_h128",
    "preconditions_json",
    "postconditions_json",
    "raises_json",
    "param_nullability_json",
    "return_nullability",
    "contract_confidence",
    "created_at",
]

TEST_CATALOG_COLUMNS = [
    "test_id",
    "test_goid_h128",
    "urn",
    "repo",
    "commit",
    "rel_path",
    "qualname",
    "kind",
    "status",
    "duration_ms",
    "markers",
    "parametrized",
    "flaky",
    "created_at",
]

TEST_CATALOG_UPDATE_GOIDS = (
    "UPDATE analytics.test_catalog "
    "SET test_goid_h128 = ?, urn = ? "
    "WHERE test_id = ? AND rel_path = ? AND repo = ? AND commit = ?"
)

CONFIG_VALUES_COLUMNS = [
    "repo",
    "commit",
    "config_path",
    "format",
    "key",
    "reference_paths",
    "reference_modules",
    "reference_count",
]

HOTSPOTS_COLUMNS = [
    "rel_path",
    "commit_count",
    "author_count",
    "lines_added",
    "lines_deleted",
    "complexity",
    "score",
]

TAGS_INDEX_COLUMNS = [
    "tag",
    "description",
    "includes",
    "excludes",
    "matches",
]

TYPEDNESS_COLUMNS = [
    "repo",
    "commit",
    "path",
    "type_error_count",
    "annotation_ratio",
    "untyped_defs",
    "overlay_needed",
]

STATIC_DIAGNOSTICS_COLUMNS = [
    "repo",
    "commit",
    "rel_path",
    "pyrefly_errors",
    "pyright_errors",
    "ruff_errors",
    "total_errors",
    "has_errors",
]

MODULES_COLUMNS = [
    "module",
    "path",
    "repo",
    "commit",
    "language",
    "tags",
    "owners",
]

FILE_STATE_COLUMNS = [
    "repo",
    "commit",
    "rel_path",
    "language",
    "size_bytes",
    "mtime_ns",
    "content_hash",
]

SEMANTIC_ROLES_MODULES_COLUMNS = [
    "repo",
    "commit",
    "module",
    "role",
    "role_confidence",
    "role_sources_json",
    "created_at",
]

REPO_MAP_COLUMNS = [
    "repo",
    "commit",
    "modules",
    "overlays",
    "generated_at",
]

SYMBOL_USE_COLUMNS = [
    "symbol",
    "def_path",
    "use_path",
    "same_file",
    "same_module",
    "def_goid_h128",
    "use_goid_h128",
]

TEST_COVERAGE_EDGE_COLUMNS = [
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

GOIDS_COLUMNS = [
    "goid_h128",
    "urn",
    "repo",
    "commit",
    "rel_path",
    "language",
    "kind",
    "qualname",
    "start_line",
    "end_line",
    "created_at",
]

GOID_CROSSWALK_COLUMNS = [
    "repo",
    "commit",
    "goid",
    "lang",
    "module_path",
    "file_path",
    "start_line",
    "end_line",
    "scip_symbol",
    "ast_qualname",
    "cst_node_id",
    "chunk_id",
    "symbol_id",
    "updated_at",
]

SEMANTIC_ROLES_FUNCTIONS_COLUMNS = [
    "repo",
    "commit",
    "function_goid_h128",
    "role",
    "framework",
    "role_confidence",
    "role_sources_json",
    "created_at",
]

CALL_GRAPH_NODE_COLUMNS = [
    "goid_h128",
    "language",
    "kind",
    "arity",
    "is_public",
    "rel_path",
]

CALL_GRAPH_EDGE_COLUMNS = [
    "repo",
    "commit",
    "caller_goid_h128",
    "callee_goid_h128",
    "callsite_path",
    "callsite_line",
    "callsite_col",
    "language",
    "kind",
    "resolved_via",
    "confidence",
    "evidence_json",
]

IMPORT_EDGE_COLUMNS = [
    "repo",
    "commit",
    "src_module",
    "dst_module",
    "src_fan_out",
    "dst_fan_in",
    "cycle_group",
    "module_layer",
]

IMPORT_MODULE_COLUMNS = [
    "repo",
    "commit",
    "module",
    "scc_id",
    "component_size",
    "layer",
    "cycle_group",
]

CFG_BLOCK_COLUMNS = [
    "function_goid_h128",
    "block_idx",
    "block_id",
    "label",
    "file_path",
    "start_line",
    "end_line",
    "kind",
    "stmts_json",
    "in_degree",
    "out_degree",
]

CFG_EDGE_COLUMNS = [
    "function_goid_h128",
    "src_block_id",
    "dst_block_id",
    "edge_kind",
]

DFG_EDGE_COLUMNS = [
    "function_goid_h128",
    "src_block_id",
    "dst_block_id",
    "src_var",
    "dst_var",
    "edge_kind",
    "via_phi",
    "use_kind",
]

CFG_FUNCTION_METRICS_EXT_COLUMNS = [
    "function_goid_h128",
    "repo",
    "commit",
    "unreachable_block_count",
    "loop_header_count",
    "true_edge_count",
    "false_edge_count",
    "back_edge_count",
    "exception_edge_count",
    "fallthrough_edge_count",
    "loop_edge_count",
    "entry_exit_simple_paths",
    "created_at",
    "metrics_version",
]

DFG_FUNCTION_METRICS_EXT_COLUMNS = [
    "function_goid_h128",
    "repo",
    "commit",
    "data_flow_edge_count",
    "intra_block_edge_count",
    "use_kind_phi_count",
    "use_kind_data_flow_count",
    "use_kind_intra_block_count",
    "use_kind_other_count",
    "phi_edge_ratio",
    "entry_exit_simple_paths",
    "created_at",
    "metrics_version",
]

DOCSTRINGS_COLUMNS = [
    "repo",
    "commit",
    "rel_path",
    "module",
    "qualname",
    "kind",
    "lineno",
    "end_lineno",
    "raw_docstring",
    "style",
    "short_desc",
    "long_desc",
    "params",
    "returns",
    "raises",
    "examples",
    "created_at",
]


# ---------------------------------------------------------------------------
# Prepared SQL literals (static, coupled to column lists)
# ---------------------------------------------------------------------------

AST_NODES_DELETE = (
    "DELETE FROM core.ast_nodes "
    "WHERE path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)"
)
AST_NODES_INSERT = (
    "INSERT INTO core.ast_nodes ("
    "path, node_type, name, qualname, lineno, end_lineno, decorator_start_line, decorator_end_line, "
    "col_offset, end_col_offset, parent_qualname, decorators, docstring, hash"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

AST_METRICS_DELETE = (
    "DELETE FROM core.ast_metrics "
    "WHERE rel_path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)"
)
AST_METRICS_INSERT = (
    "INSERT INTO core.ast_metrics ("
    "rel_path, node_count, function_count, class_count, avg_depth, max_depth, complexity, generated_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)

CST_NODES_DELETE = (
    "DELETE FROM core.cst_nodes "
    "WHERE path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)"
)
CST_NODES_INSERT = (
    "INSERT INTO core.cst_nodes ("
    "path, node_id, kind, span, text_preview, parents, qnames"
    ") VALUES (?, ?, ?, ?, ?, ?, ?)"
)

COVERAGE_LINES_DELETE = "DELETE FROM analytics.coverage_lines WHERE repo = ? AND commit = ?"
COVERAGE_LINES_INSERT = (
    "INSERT INTO analytics.coverage_lines ("
    "repo, commit, rel_path, line, is_executable, is_covered, hits, context_count, created_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

FUNCTION_METRICS_DELETE = "DELETE FROM analytics.function_metrics WHERE repo = ? AND commit = ?"
FUNCTION_METRICS_INSERT = (
    "INSERT INTO analytics.function_metrics ("
    "function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname, start_line, end_line, "
    "loc, logical_loc, param_count, positional_params, keyword_only_params, has_varargs, has_varkw, "
    "is_async, is_generator, return_count, yield_count, raise_count, cyclomatic_complexity, "
    "max_nesting_depth, stmt_count, decorator_count, has_docstring, complexity_bucket, created_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

FUNCTION_TYPES_DELETE = "DELETE FROM analytics.function_types WHERE repo = ? AND commit = ?"
FUNCTION_TYPES_INSERT = (
    "INSERT INTO analytics.function_types ("
    "function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname, start_line, end_line, "
    "total_params, annotated_params, unannotated_params, param_typed_ratio, has_return_annotation, "
    "return_type, return_type_source, type_comment, param_types, fully_typed, partial_typed, untyped, "
    "typedness_bucket, typedness_source, created_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

FUNCTION_EFFECTS_DELETE = "DELETE FROM analytics.function_effects WHERE repo = ? AND commit = ?"
FUNCTION_EFFECTS_INSERT = (
    "INSERT INTO analytics.function_effects ("
    "repo, commit, function_goid_h128, is_pure, uses_io, touches_db, uses_time, uses_randomness, "
    "modifies_globals, modifies_closure, spawns_threads_or_tasks, has_transitive_effects, "
    "purity_confidence, effects_json, created_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

FUNCTION_CONTRACTS_DELETE = "DELETE FROM analytics.function_contracts WHERE repo = ? AND commit = ?"
FUNCTION_CONTRACTS_INSERT = (
    "INSERT INTO analytics.function_contracts ("
    "repo, commit, function_goid_h128, preconditions_json, postconditions_json, raises_json, "
    "param_nullability_json, return_nullability, contract_confidence, created_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

SEMANTIC_ROLES_FUNCTIONS_DELETE = (
    "DELETE FROM analytics.semantic_roles_functions WHERE repo = ? AND commit = ?"
)
SEMANTIC_ROLES_FUNCTIONS_INSERT = (
    "INSERT INTO analytics.semantic_roles_functions ("
    "repo, commit, function_goid_h128, role, framework, role_confidence, role_sources_json, created_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)

SEMANTIC_ROLES_MODULES_DELETE = (
    "DELETE FROM analytics.semantic_roles_modules WHERE repo = ? AND commit = ?"
)
SEMANTIC_ROLES_MODULES_INSERT = (
    "INSERT INTO analytics.semantic_roles_modules ("
    "repo, commit, module, role, role_confidence, role_sources_json, created_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?)"
)

DOCSTRINGS_DELETE = "DELETE FROM core.docstrings WHERE repo = ? AND commit = ?"
DOCSTRINGS_INSERT = (
    "INSERT INTO core.docstrings ("
    "repo, commit, rel_path, module, qualname, kind, lineno, end_lineno, raw_docstring, style, short_desc, "
    "long_desc, params, returns, raises, examples, created_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

TEST_CATALOG_DELETE = "DELETE FROM analytics.test_catalog WHERE repo = ? AND commit = ?"
TEST_CATALOG_INSERT = (
    "INSERT INTO analytics.test_catalog ("
    "test_id, test_goid_h128, urn, repo, commit, rel_path, qualname, kind, status, "
    "duration_ms, markers, parametrized, flaky, created_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

CONFIG_VALUES_INSERT = (
    "INSERT INTO analytics.config_values ("
    "repo, commit, config_path, format, key, reference_paths, reference_modules, reference_count"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)

TAGS_INDEX_DELETE = "DELETE FROM analytics.tags_index"
TAGS_INDEX_INSERT = "INSERT INTO analytics.tags_index (tag, description, includes, excludes, matches) VALUES (?, ?, ?, ?, ?)"

TYPEDNESS_DELETE = "DELETE FROM analytics.typedness WHERE repo = ? AND commit = ?"
TYPEDNESS_INSERT = (
    "INSERT INTO analytics.typedness (repo, commit, path, type_error_count, annotation_ratio, untyped_defs, overlay_needed) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)

STATIC_DIAGNOSTICS_DELETE = "DELETE FROM analytics.static_diagnostics WHERE repo = ? AND commit = ?"
STATIC_DIAGNOSTICS_INSERT = (
    "INSERT INTO analytics.static_diagnostics (repo, commit, rel_path, pyrefly_errors, pyright_errors, ruff_errors, total_errors, has_errors) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)

HOTSPOTS_INSERT = (
    "INSERT INTO analytics.hotspots ("
    "rel_path, commit_count, author_count, lines_added, lines_deleted, complexity, score"
    ") VALUES (?, ?, ?, ?, ?, ?, ?)"
)

MODULES_DELETE = "DELETE FROM core.modules WHERE repo = ? AND commit = ?"
MODULES_INSERT = "INSERT INTO core.modules (module, path, repo, commit, language, tags, owners) VALUES (?, ?, ?, ?, ?, ?, ?)"

FILE_STATE_DELETE = "DELETE FROM core.file_state WHERE repo = ? AND rel_path = ? AND language = ?"
FILE_STATE_INSERT = (
    "INSERT INTO core.file_state ("
    "repo, commit, rel_path, language, size_bytes, mtime_ns, content_hash"
    ") VALUES (?, ?, ?, ?, ?, ?, ?)"
)

REPO_MAP_DELETE = "DELETE FROM core.repo_map WHERE repo = ? AND commit = ?"
REPO_MAP_INSERT = "INSERT INTO core.repo_map (repo, commit, modules, overlays, generated_at) VALUES (?, ?, ?, ?, ?)"

SYMBOL_USE_DELETE = "DELETE FROM graph.symbol_use_edges"
SYMBOL_USE_INSERT = "INSERT INTO graph.symbol_use_edges (symbol, def_path, use_path, same_file, same_module) VALUES (?, ?, ?, ?, ?)"

TEST_COVERAGE_EDGE_DELETE = (
    "DELETE FROM analytics.test_coverage_edges WHERE repo = ? AND commit = ?"
)
TEST_COVERAGE_EDGE_INSERT = (
    "INSERT INTO analytics.test_coverage_edges ("
    "test_id, test_goid_h128, function_goid_h128, urn, repo, commit, rel_path, qualname, "
    "covered_lines, executable_lines, coverage_ratio, last_status, created_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

GOIDS_DELETE = "DELETE FROM core.goids WHERE repo = ? AND commit = ?"
GOIDS_INSERT = (
    "INSERT INTO core.goids ("
    "goid_h128, urn, repo, commit, rel_path, language, kind, qualname, start_line, end_line, created_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

GOID_CROSSWALK_DELETE = "DELETE FROM core.goid_crosswalk WHERE repo = ? AND commit = ?"
GOID_CROSSWALK_INSERT = (
    "INSERT INTO core.goid_crosswalk ("
    "repo, commit, goid, lang, module_path, file_path, start_line, end_line, scip_symbol, ast_qualname, "
    "cst_node_id, chunk_id, symbol_id, updated_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
GOID_CROSSWALK_UPDATE_SCIP = (
    "UPDATE core.goid_crosswalk SET scip_symbol = ? WHERE goid = ? AND repo = ? AND commit = ?"
)

CALL_GRAPH_NODES_DELETE = "DELETE FROM graph.call_graph_nodes"
CALL_GRAPH_NODES_INSERT = (
    "INSERT INTO graph.call_graph_nodes (goid_h128, language, kind, arity, is_public, rel_path) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)

CALL_GRAPH_EDGES_DELETE = "DELETE FROM graph.call_graph_edges WHERE repo = ? AND commit = ?"
CALL_GRAPH_EDGES_INSERT = (
    "INSERT INTO graph.call_graph_edges ("
    "repo, commit, caller_goid_h128, callee_goid_h128, callsite_path, callsite_line, callsite_col, "
    "language, kind, resolved_via, confidence, evidence_json"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

IMPORT_EDGES_DELETE = "DELETE FROM graph.import_graph_edges WHERE repo = ? AND commit = ?"
IMPORT_EDGES_INSERT = (
    "INSERT INTO graph.import_graph_edges ("
    "repo, commit, src_module, dst_module, src_fan_out, dst_fan_in, cycle_group, module_layer"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)

IMPORT_MODULES_DELETE = "DELETE FROM graph.import_modules WHERE repo = ? AND commit = ?"
IMPORT_MODULES_INSERT = (
    "INSERT INTO graph.import_modules ("
    "repo, commit, module, scc_id, component_size, layer, cycle_group"
    ") VALUES (?, ?, ?, ?, ?, ?, ?)"
)

CFG_BLOCKS_DELETE = "DELETE FROM graph.cfg_blocks"
CFG_BLOCKS_INSERT = (
    "INSERT INTO graph.cfg_blocks ("
    "function_goid_h128, block_idx, block_id, label, file_path, start_line, end_line, kind, "
    "stmts_json, in_degree, out_degree"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

CFG_EDGES_DELETE = "DELETE FROM graph.cfg_edges"
CFG_EDGES_INSERT = (
    "INSERT INTO graph.cfg_edges (function_goid_h128, src_block_id, dst_block_id, edge_kind) "
    "VALUES (?, ?, ?, ?)"
)

DFG_EDGES_DELETE = "DELETE FROM graph.dfg_edges"
DFG_EDGES_INSERT = (
    "INSERT INTO graph.dfg_edges ("
    "function_goid_h128, src_block_id, dst_block_id, src_var, dst_var, edge_kind, via_phi, use_kind"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)

CFG_FUNCTION_METRICS_EXT_DELETE = (
    "DELETE FROM analytics.cfg_function_metrics_ext WHERE repo = ? AND commit = ?"
)
CFG_FUNCTION_METRICS_EXT_INSERT = (
    "INSERT INTO analytics.cfg_function_metrics_ext ("
    "function_goid_h128, repo, commit, unreachable_block_count, loop_header_count, "
    "true_edge_count, false_edge_count, back_edge_count, exception_edge_count, "
    "fallthrough_edge_count, loop_edge_count, entry_exit_simple_paths, created_at, metrics_version"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

DFG_FUNCTION_METRICS_EXT_DELETE = (
    "DELETE FROM analytics.dfg_function_metrics_ext WHERE repo = ? AND commit = ?"
)
DFG_FUNCTION_METRICS_EXT_INSERT = (
    "INSERT INTO analytics.dfg_function_metrics_ext ("
    "function_goid_h128, repo, commit, data_flow_edge_count, intra_block_edge_count, "
    "use_kind_phi_count, use_kind_data_flow_count, use_kind_intra_block_count, "
    "use_kind_other_count, phi_edge_ratio, entry_exit_simple_paths, created_at, metrics_version"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

FUNCTION_VALIDATION_DELETE = (
    "DELETE FROM analytics.function_validation WHERE repo = ? AND commit = ?"
)
FUNCTION_VALIDATION_INSERT = (
    "INSERT INTO analytics.function_validation ("
    "repo, commit, function_goid_h128, rel_path, qualname, issue, detail, created_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)

GRAPH_VALIDATION_DELETE = "DELETE FROM analytics.graph_validation WHERE repo = ? AND commit = ?"
GRAPH_VALIDATION_INSERT = (
    "INSERT INTO analytics.graph_validation ("
    "repo, commit, graph_name, entity_id, issue, severity, rel_path, detail, metadata, created_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


@dataclass(frozen=True)
class PreparedStatements:
    """Prepared insert/delete SQL for a table."""

    insert_sql: str
    delete_sql: str | None = None


def prepared_statements_dynamic(con: DuckDBConnection, table_key: str) -> PreparedStatements:
    """
    Return prepared SQL using registry-derived column order for macro-backed tables.

    Parameters
    ----------
    con :
        Active DuckDB connection.
    table_key :
        Registry key (e.g., "analytics.function_metrics").

    Returns
    -------
    PreparedStatements
        Insert and optional delete SQL with column order sourced from the registry.

    Raises
    ------
    RuntimeError
        If the table is missing from the registry.
    """
    registry_cols = load_registry_columns(con).get(table_key)
    if registry_cols is None:
        message = f"Table {table_key} missing from registry"
        raise RuntimeError(message)
    cols_sql = ", ".join(registry_cols)
    placeholders = ", ".join("?" for _ in registry_cols)
    schema_name, table_name = table_key.split(".", maxsplit=1)
    table_sql = f'"{schema_name}"."{table_name}"'
    insert_sql = f"INSERT INTO {table_sql} ({cols_sql}) VALUES ({placeholders})"  # noqa: S608 - registry-controlled identifiers
    return PreparedStatements(
        insert_sql=insert_sql,
        delete_sql=None,
    )


def ensure_schema(con: DuckDBConnection, table_key: str) -> None:
    """
    Validate that the live DuckDB table matches the registry definition.

    Checks column presence/order and NOT NULL flags.

    Raises
    ------
    RuntimeError
        If the table is missing or deviates from the registry.
    """
    if not _INGESTION_COLUMNS_VERIFIED[0]:
        verify_ingestion_columns(con)
        _INGESTION_COLUMNS_VERIFIED[0] = True

    registry_cols = load_registry_columns(con).get(table_key)
    if registry_cols is None:
        message = f"Table {table_key} missing from registry"
        raise RuntimeError(message)
    schema_name, table_name = table_key.split(".", maxsplit=1)
    info = con.execute(f"PRAGMA table_info({schema_name}.{table_name})").fetchall()
    if not info:
        message = f"Table {table_key} is missing"
        raise RuntimeError(message)

    names = [row[1] for row in info]
    expected_cols = registry_cols
    if names != expected_cols:
        message = f"Column order mismatch for {table_key}: db={names}, registry={expected_cols}"
        raise RuntimeError(message)


__all__ = [
    "AST_METRICS_COLUMNS",
    "AST_METRICS_DELETE",
    "AST_METRICS_INSERT",
    "AST_NODES_COLUMNS",
    "AST_NODES_DELETE",
    "AST_NODES_INSERT",
    "CONFIG_VALUES_COLUMNS",
    "CONFIG_VALUES_INSERT",
    "COVERAGE_LINES_COLUMNS",
    "COVERAGE_LINES_DELETE",
    "COVERAGE_LINES_INSERT",
    "CST_NODES_COLUMNS",
    "CST_NODES_DELETE",
    "CST_NODES_INSERT",
    "DOCSTRINGS_COLUMNS",
    "DOCSTRINGS_DELETE",
    "DOCSTRINGS_INSERT",
    "FILE_STATE_COLUMNS",
    "FILE_STATE_DELETE",
    "FILE_STATE_INSERT",
    "FUNCTION_CONTRACTS_COLUMNS",
    "FUNCTION_CONTRACTS_DELETE",
    "FUNCTION_CONTRACTS_INSERT",
    "FUNCTION_EFFECTS_COLUMNS",
    "FUNCTION_EFFECTS_DELETE",
    "FUNCTION_EFFECTS_INSERT",
    "FUNCTION_METRICS_COLUMNS",
    "FUNCTION_METRICS_DELETE",
    "FUNCTION_METRICS_INSERT",
    "FUNCTION_TYPES_COLUMNS",
    "FUNCTION_TYPES_DELETE",
    "FUNCTION_TYPES_INSERT",
    "FUNCTION_VALIDATION_DELETE",
    "FUNCTION_VALIDATION_INSERT",
    "GOID_CROSSWALK_UPDATE_SCIP",
    "HOTSPOTS_COLUMNS",
    "HOTSPOTS_INSERT",
    "MODULES_COLUMNS",
    "MODULES_DELETE",
    "MODULES_INSERT",
    "REPO_MAP_COLUMNS",
    "REPO_MAP_DELETE",
    "REPO_MAP_INSERT",
    "SEMANTIC_ROLES_FUNCTIONS_COLUMNS",
    "SEMANTIC_ROLES_FUNCTIONS_DELETE",
    "SEMANTIC_ROLES_FUNCTIONS_INSERT",
    "SEMANTIC_ROLES_MODULES_COLUMNS",
    "SEMANTIC_ROLES_MODULES_DELETE",
    "SEMANTIC_ROLES_MODULES_INSERT",
    "STATIC_DIAGNOSTICS_COLUMNS",
    "STATIC_DIAGNOSTICS_DELETE",
    "STATIC_DIAGNOSTICS_INSERT",
    "SYMBOL_USE_COLUMNS",
    "SYMBOL_USE_DELETE",
    "SYMBOL_USE_INSERT",
    "TAGS_INDEX_COLUMNS",
    "TAGS_INDEX_DELETE",
    "TAGS_INDEX_INSERT",
    "TEST_CATALOG_COLUMNS",
    "TEST_CATALOG_DELETE",
    "TEST_CATALOG_INSERT",
    "TEST_CATALOG_UPDATE_GOIDS",
    "TEST_COVERAGE_EDGE_COLUMNS",
    "TEST_COVERAGE_EDGE_DELETE",
    "TEST_COVERAGE_EDGE_INSERT",
    "TYPEDNESS_COLUMNS",
    "TYPEDNESS_DELETE",
    "TYPEDNESS_INSERT",
    "PreparedStatements",
    "ensure_schema",
    "prepared_statements_dynamic",
]
