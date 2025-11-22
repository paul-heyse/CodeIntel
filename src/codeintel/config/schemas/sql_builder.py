"""Utilities to couple ingestion SQL to the registry and validate schemas."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from codeintel.config.schemas.tables import TABLE_SCHEMAS

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
    """
    Raise if registry column order drifts from the hardcoded literals.

    Parameters
    ----------
    table_key:
        Registry key (e.g., "core.ast_nodes").
    columns:
        Expected column list for the table.

    Raises
    ------
    RuntimeError
        If the registry column order does not match literals.
    """
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

# ---------------------------------------------------------------------------
# Prepared SQL literals (static, coupled to column lists)
# ---------------------------------------------------------------------------

AST_NODES_DELETE = (
    "DELETE FROM core.ast_nodes "
    "WHERE path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)"
)
AST_NODES_INSERT = (
    "INSERT INTO core.ast_nodes ("
    "path, node_type, name, qualname, lineno, end_lineno, col_offset, end_col_offset, "
    "parent_qualname, decorators, docstring, hash"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
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


@dataclass(frozen=True)
class PreparedStatements:
    """Prepared insert/delete SQL for a table."""

    insert_sql: str
    delete_sql: str | None = None


PREPARED: dict[str, PreparedStatements] = {
    "core.ast_nodes": PreparedStatements(insert_sql=AST_NODES_INSERT, delete_sql=AST_NODES_DELETE),
    "core.ast_metrics": PreparedStatements(insert_sql=AST_METRICS_INSERT, delete_sql=AST_METRICS_DELETE),
    "core.cst_nodes": PreparedStatements(insert_sql=CST_NODES_INSERT, delete_sql=CST_NODES_DELETE),
    "analytics.coverage_lines": PreparedStatements(
        insert_sql=COVERAGE_LINES_INSERT,
        delete_sql=COVERAGE_LINES_DELETE,
    ),
    "analytics.test_catalog": PreparedStatements(
        insert_sql=TEST_CATALOG_INSERT,
        delete_sql=TEST_CATALOG_DELETE,
    ),
    "analytics.config_values": PreparedStatements(insert_sql=CONFIG_VALUES_INSERT, delete_sql="DELETE FROM analytics.config_values"),
}


def prepared_statements(table_key: str) -> PreparedStatements:
    """
    Return pre-built SQL strings for the given table key.

    Parameters
    ----------
    table_key:
        Registry key (e.g., "core.ast_nodes").

    Returns
    -------
    PreparedStatements
        Insert and optional delete SQL for the table.
    """
    return PREPARED[table_key]


def ensure_schema(con: duckdb.DuckDBPyConnection, table_key: str) -> None:
    """
    Validate that the live DuckDB table matches the registry definition.

    Checks column presence/order and NOT NULL flags.

    Raises
    ------
    RuntimeError
        If the table is missing or deviates from the registry.
    """
    table = TABLE_SCHEMAS[table_key]
    info = con.execute(f"PRAGMA table_info({table.schema}.{table.name})").fetchall()
    if not info:
        message = f"Table {table.fq_name} is missing"
        raise RuntimeError(message)

    names = [row[1] for row in info]
    expected_cols = table.column_names()
    if names != expected_cols:
        message = f"Column order mismatch for {table.fq_name}: db={names}, registry={expected_cols}"
        raise RuntimeError(message)

    notnull_flags = {row[1]: bool(row[3]) for row in info}  # column_name -> notnull (1/0)
    for col in table.columns:
        db_nullable = not bool(notnull_flags.get(col.name))
        if db_nullable != col.nullable:
            message = (
                f"Nullability mismatch for {table.fq_name}.{col.name}: "
                f"db nullable={db_nullable}, expected {col.nullable}"
            )
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
    "PREPARED",
    "PreparedStatements",
    "TEST_CATALOG_COLUMNS",
    "TEST_CATALOG_DELETE",
    "TEST_CATALOG_INSERT",
    "ensure_schema",
    "prepared_statements",
]
