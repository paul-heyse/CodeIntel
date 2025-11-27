"""Central SQL registry for ingestion steps (coupled to registry metadata)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.config.schemas.registry_adapter import load_registry_columns

if TYPE_CHECKING:
    from codeintel.storage.gateway import DuckDBConnection

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

FILE_STATE_COLUMNS = [
    "repo",
    "commit",
    "rel_path",
    "language",
    "size_bytes",
    "mtime_ns",
    "content_hash",
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
    "repo",
    "commit",
    "config_path",
    "format",
    "key",
    "reference_paths",
    "reference_modules",
    "reference_count",
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


LITERAL_COLUMNS: dict[str, list[str]] = {
    "core.ast_nodes": AST_NODES_COLUMNS,
    "core.ast_metrics": AST_METRICS_COLUMNS,
    "core.cst_nodes": CST_NODES_COLUMNS,
    "core.file_state": FILE_STATE_COLUMNS,
    "analytics.coverage_lines": COVERAGE_LINES_COLUMNS,
    "analytics.test_catalog": TEST_CATALOG_COLUMNS,
    "analytics.config_values": CONFIG_VALUES_COLUMNS,
    "analytics.typedness": TYPEDNESS_COLUMNS,
    "analytics.static_diagnostics": STATIC_DIAGNOSTICS_COLUMNS,
}


def verify_ingestion_columns(con: DuckDBConnection) -> None:
    """
    Verify that ingestion column literals match the live registry.

    Raises
    ------
    RuntimeError
        If any literal column order deviates from the registry.
    """
    registry = load_registry_columns(con)
    for table_key, literal_cols in LITERAL_COLUMNS.items():
        registry_cols = registry.get(table_key)
        if registry_cols is None:
            message = f"{table_key} missing from registry"
            raise RuntimeError(message)
        if registry_cols != literal_cols:
            message = (
                f"Column drift for {table_key}: registry={registry_cols} literals={literal_cols}"
            )
            raise RuntimeError(message)


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

FILE_STATE_DELETE = "DELETE FROM core.file_state WHERE repo = ? AND rel_path = ? AND language = ?"
FILE_STATE_INSERT = (
    "INSERT INTO core.file_state ("
    "repo, commit, rel_path, language, size_bytes, mtime_ns, content_hash"
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
    "repo, commit, config_path, format, key, reference_paths, reference_modules, reference_count"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
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
    "FILE_STATE_COLUMNS",
    "FILE_STATE_DELETE",
    "FILE_STATE_INSERT",
    "TEST_CATALOG_COLUMNS",
    "TEST_CATALOG_DELETE",
    "TEST_CATALOG_INSERT",
]
