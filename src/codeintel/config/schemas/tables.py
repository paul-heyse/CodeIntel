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
}
