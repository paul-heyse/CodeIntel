"""Central SQL registry for ingestion steps (coupled to table schemas)."""

from __future__ import annotations

from codeintel.config.schemas.tables import TABLE_SCHEMAS

# Column lists (single source of truth for SQL literals below)
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

CONFIG_VALUES_COLUMNS = [
    "config_path",
    "format",
    "key",
    "reference_paths",
    "reference_modules",
    "reference_count",
]


def _assert_columns(table_key: str, columns: list[str]) -> None:
    registry_cols = TABLE_SCHEMAS[table_key].column_names()
    if registry_cols != columns:
        message = f"Column drift for {table_key}: registry={registry_cols} literals={columns}"
        raise RuntimeError(message)


_assert_columns("core.ast_nodes", AST_NODES_COLUMNS)
_assert_columns("core.ast_metrics", AST_METRICS_COLUMNS)
_assert_columns("core.cst_nodes", CST_NODES_COLUMNS)
_assert_columns("analytics.coverage_lines", COVERAGE_LINES_COLUMNS)
_assert_columns("analytics.test_catalog", TEST_CATALOG_COLUMNS)
_assert_columns("analytics.config_values", CONFIG_VALUES_COLUMNS)


# SQL literals derived from column lists above (no dynamic f-strings)
AST_NODES_DELETE = "DELETE FROM core.ast_nodes WHERE path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)"
AST_NODES_INSERT = (
    "INSERT INTO core.ast_nodes ("
    "path, node_type, name, qualname, lineno, end_lineno, decorator_start_line, decorator_end_line, "
    "col_offset, end_col_offset, parent_qualname, decorators, docstring, hash"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

AST_METRICS_DELETE = "DELETE FROM core.ast_metrics WHERE rel_path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)"
AST_METRICS_INSERT = (
    "INSERT INTO core.ast_metrics ("
    "rel_path, node_count, function_count, class_count, avg_depth, max_depth, complexity, generated_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)

CST_NODES_DELETE = "DELETE FROM core.cst_nodes WHERE path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)"
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

TEST_CATALOG_DELETE = "DELETE FROM analytics.test_catalog WHERE repo = ? AND commit = ?"
TEST_CATALOG_INSERT = (
    "INSERT INTO analytics.test_catalog ("
    "test_id, test_goid_h128, urn, repo, commit, rel_path, qualname, kind, status, "
    "duration_ms, markers, parametrized, flaky, created_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

CONFIG_VALUES_INSERT = (
    "INSERT INTO analytics.config_values ("
    "config_path, format, key, reference_paths, reference_modules, reference_count"
    ") VALUES (?, ?, ?, ?, ?, ?)"
)

__all__ = [
    "AST_METRICS_DELETE",
    "AST_METRICS_INSERT",
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
    "TEST_CATALOG_COLUMNS",
    "TEST_CATALOG_DELETE",
    "TEST_CATALOG_INSERT",
]
