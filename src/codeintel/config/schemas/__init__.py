"""Shared SQL snippets for ingestion steps (central registry)."""

from __future__ import annotations

from codeintel.config.schemas.ingestion_sql import (
    AST_METRICS_DELETE,
    AST_METRICS_INSERT,
    AST_NODES_DELETE,
    AST_NODES_INSERT,
    CONFIG_VALUES_INSERT,
    COVERAGE_LINES_DELETE,
    COVERAGE_LINES_INSERT,
    CST_NODES_DELETE,
    CST_NODES_INSERT,
    TEST_CATALOG_DELETE,
    TEST_CATALOG_INSERT,
)
from codeintel.config.schemas.tables import (
    COLUMN_TYPE,
    TABLE_SCHEMAS,
    Column,
    ColumnType,
    TableSchema,
)

__all__ = [
    "AST_METRICS_DELETE",
    "AST_METRICS_INSERT",
    "AST_NODES_DELETE",
    "AST_NODES_INSERT",
    "COLUMN_TYPE",
    "CONFIG_VALUES_INSERT",
    "COVERAGE_LINES_DELETE",
    "COVERAGE_LINES_INSERT",
    "CST_NODES_DELETE",
    "CST_NODES_INSERT",
    "TABLE_SCHEMAS",
    "TEST_CATALOG_DELETE",
    "TEST_CATALOG_INSERT",
    "Column",
    "ColumnType",
    "TableSchema",
]
