"""Single source of truth for dataset contracts, row models, and serializers.

This module consolidates all dataset configuration into a single authoritative
source. It contains:

1. TableSchema and Column dataclasses for schema definitions
2. TABLE_SCHEMAS dictionary with all table/view definitions
3. TypedDict row models for DuckDB table inserts
4. Serializer functions to convert row models to tuples
5. Column constants derived from TABLE_SCHEMAS
6. DatasetContract metadata and RowBinding definitions
7. SQL generation helpers (INSERT, DELETE statements)
8. Contract building and registry exports

All imports of schemas, row models, and SQL should come from this module.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypedDict, TypeVar, cast

from codeintel.storage.views import DERIVED_DOCS_VIEWS

if TYPE_CHECKING:
    from codeintel.ingestion.ingest_runs import IngestRunMode, IngestRunStatus


# ---------------------------------------------------------------------------
# Section 0: Schema Definition Types
# ---------------------------------------------------------------------------

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
    "TIMESTAMPTZ",
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
class Index:
    """Secondary index definition."""

    name: str
    columns: tuple[str, ...]
    unique: bool = False


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
        Return ordered column names.

        Returns
        -------
        list[str]
            Column names in definition order.
        """
        return [col.name for col in self.columns]


RowToTuple = Callable[[Mapping[str, object]], tuple[object, ...]]
RowDictType = type[object]
_Column = TypeVar("_Column", bound=str)


# ---------------------------------------------------------------------------
# Section 0.4: Reusable Column Fragments
# ---------------------------------------------------------------------------
# These fragments provide composable building blocks for TABLE_SCHEMAS.
# Use tuple unpacking (*FRAGMENT) to include them in column lists.

# Versioning context (repo + commit)
REPO_COMMIT_COLS: Final[tuple[Column, ...]] = (
    Column("repo", "VARCHAR", nullable=False),
    Column("commit", "VARCHAR", nullable=False),
)

# Function entity identification (GOID only, nullable=False)
FUNCTION_GOID_COL: Final[tuple[Column, ...]] = (
    Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),
)

# Function entity identification (GOID only, nullable)
FUNCTION_GOID_COL_NULLABLE: Final[tuple[Column, ...]] = (
    Column("function_goid_h128", "DECIMAL(38,0)"),
)

# Function entity with full context (nullable columns for analytics tables)
FUNCTION_ENTITY_COLS: Final[tuple[Column, ...]] = (
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
)

# Module entity identification
MODULE_ENTITY_COLS: Final[tuple[Column, ...]] = (
    Column("repo", "VARCHAR", nullable=False),
    Column("commit", "VARCHAR", nullable=False),
    Column("module", "VARCHAR", nullable=False),
)

# Subsystem entity identification
SUBSYSTEM_ENTITY_COLS: Final[tuple[Column, ...]] = (
    Column("repo", "VARCHAR", nullable=False),
    Column("commit", "VARCHAR", nullable=False),
    Column("subsystem_id", "VARCHAR", nullable=False),
)

# Test entity identification
TEST_ENTITY_COLS: Final[tuple[Column, ...]] = (
    Column("test_id", "VARCHAR", nullable=False),
    Column("test_goid_h128", "DECIMAL(38,0)"),
    Column("repo", "VARCHAR", nullable=False),
    Column("commit", "VARCHAR", nullable=False),
)

# Timestamp suffix (nullable=False)
CREATED_AT_COL: Final[tuple[Column, ...]] = (
    Column("created_at", "TIMESTAMP", nullable=False),
)

# Timestamp suffix (nullable)
CREATED_AT_COL_NULLABLE: Final[tuple[Column, ...]] = (
    Column("created_at", "TIMESTAMP"),
)

# Location columns (for entities with source spans)
SOURCE_SPAN_COLS: Final[tuple[Column, ...]] = (
    Column("rel_path", "VARCHAR"),
    Column("start_line", "INTEGER"),
    Column("end_line", "INTEGER"),
)

# Risk columns (used in risk factor tables)
RISK_COLS: Final[tuple[Column, ...]] = (
    Column("risk_score", "DOUBLE"),
    Column("risk_level", "VARCHAR"),
)

# Ownership columns (tags and owners)
OWNERSHIP_COLS: Final[tuple[Column, ...]] = (
    Column("tags", "JSON"),
    Column("owners", "JSON"),
)


# ---------------------------------------------------------------------------
# Section 0.5: TABLE_SCHEMAS - All table definitions
# ---------------------------------------------------------------------------

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
            Column("decorator_start_line", "INTEGER"),
            Column("decorator_end_line", "INTEGER"),
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
    "core.ingest_runs": TableSchema(
        schema="core",
        name="ingest_runs",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("step", "VARCHAR", nullable=False),
            Column("run_id", "VARCHAR", nullable=False),
            Column("mode", "VARCHAR", nullable=False),
            Column("started_at", "TIMESTAMPTZ", nullable=False),
            Column("finished_at", "TIMESTAMPTZ"),
            Column("duration_s", "DOUBLE"),
            Column("rows_inserted", "BIGINT", nullable=False),
            Column("rows_deleted", "BIGINT", nullable=False),
            Column("status", "VARCHAR", nullable=False),
            Column("error_kind", "VARCHAR"),
            Column("error_message", "VARCHAR"),
            Column("datasets", "JSON"),
            Column("modules_total", "BIGINT"),
            Column("modules_changed", "BIGINT"),
            Column("modules_deleted", "BIGINT"),
            Column("modules_changed_ratio", "DOUBLE"),
            Column("modules_deleted_ratio", "DOUBLE"),
            Column("use_full_rebuild", "BOOLEAN"),
        ],
        description="Per-step ingest run telemetry for control plane reporting.",
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
    "core.file_state": TableSchema(
        schema="core",
        name="file_state",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("rel_path", "VARCHAR", nullable=False),
            Column("language", "VARCHAR", nullable=False),
            Column("size_bytes", "BIGINT", nullable=False),
            Column("mtime_ns", "BIGINT", nullable=False),
            Column("content_hash", "VARCHAR", nullable=False),
        ],
        primary_key=("repo", "rel_path", "language"),
        indexes=(
            Index("idx_core_file_state_path", ("rel_path",)),
            Index("idx_core_file_state_repo_commit", ("repo", "commit")),
        ),
        description="Per-commit file digests used for incremental ingestion",
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
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
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
        primary_key=("repo", "commit", "goid"),
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
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("config_path", "VARCHAR", nullable=False),
            Column("format", "VARCHAR", nullable=False),
            Column("key", "VARCHAR", nullable=False),
            Column("reference_paths", "JSON"),
            Column("reference_modules", "JSON"),
            Column("reference_count", "INTEGER", nullable=False),
        ],
        primary_key=("repo", "commit", "config_path", "key"),
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
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("path", "VARCHAR", nullable=False),
            Column("type_error_count", "INTEGER", nullable=False),
            Column("annotation_ratio", "JSON", nullable=False),
            Column("untyped_defs", "INTEGER", nullable=False),
            Column("overlay_needed", "BOOLEAN", nullable=False),
        ],
        primary_key=("repo", "commit", "path"),
        description="Per-file annotation ratios and static error counts",
    ),
    "analytics.static_diagnostics": TableSchema(
        schema="analytics",
        name="static_diagnostics",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("rel_path", "VARCHAR", nullable=False),
            Column("pyrefly_errors", "INTEGER", nullable=False),
            Column("pyright_errors", "INTEGER", nullable=False),
            Column("ruff_errors", "INTEGER", nullable=False),
            Column("total_errors", "INTEGER", nullable=False),
            Column("has_errors", "BOOLEAN", nullable=False),
        ],
        primary_key=("repo", "commit", "rel_path"),
        description="Per-file static diagnostic counts",
    ),
    "analytics.entrypoints": TableSchema(
        schema="analytics",
        name="entrypoints",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("entrypoint_id", "VARCHAR", nullable=False),
            Column("kind", "VARCHAR", nullable=False),
            Column("framework", "VARCHAR"),
            Column("handler_goid_h128", "DECIMAL(38,0)", nullable=False),
            Column("handler_urn", "VARCHAR", nullable=False),
            Column("handler_rel_path", "VARCHAR", nullable=False),
            Column("handler_module", "VARCHAR", nullable=False),
            Column("handler_qualname", "VARCHAR", nullable=False),
            Column("http_method", "VARCHAR"),
            Column("route_path", "VARCHAR"),
            Column("status_codes", "JSON"),
            Column("auth_required", "BOOLEAN"),
            Column("command_name", "VARCHAR"),
            Column("arguments_schema", "JSON"),
            Column("schedule", "VARCHAR"),
            Column("trigger", "VARCHAR"),
            Column("extra", "JSON"),
            Column("subsystem_id", "VARCHAR"),
            Column("subsystem_name", "VARCHAR"),
            Column("tags", "JSON"),
            Column("owners", "JSON"),
            Column("tests_touching", "INTEGER"),
            Column("failing_tests", "INTEGER"),
            Column("slow_tests", "INTEGER"),
            Column("flaky_tests", "INTEGER"),
            Column("entrypoint_coverage_ratio", "DOUBLE"),
            Column("last_test_status", "VARCHAR"),
            Column("created_at", "TIMESTAMP", nullable=False),
        ],
        primary_key=("repo", "commit", "entrypoint_id"),
        description="External entrypoints mapped to handlers, subsystems, and tests",
    ),
    "analytics.entrypoint_tests": TableSchema(
        schema="analytics",
        name="entrypoint_tests",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("entrypoint_id", "VARCHAR", nullable=False),
            Column("test_id", "VARCHAR", nullable=False),
            Column("test_goid_h128", "DECIMAL(38,0)"),
            Column("coverage_ratio", "DOUBLE"),
            Column("status", "VARCHAR"),
            Column("duration_ms", "DOUBLE"),
            Column("created_at", "TIMESTAMP", nullable=False),
        ],
        primary_key=("repo", "commit", "entrypoint_id", "test_id"),
        description="Bipartite edges between entrypoints and tests",
    ),
    "analytics.external_dependencies": TableSchema(
        schema="analytics",
        name="external_dependencies",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("dep_id", "VARCHAR", nullable=False),
            Column("library", "VARCHAR", nullable=False),
            Column("service_name", "VARCHAR"),
            Column("category", "VARCHAR"),
            Column("language", "VARCHAR"),
            Column("severity", "VARCHAR"),
            Column("criticality", "DOUBLE"),
            Column("risk_score", "DOUBLE"),
            Column("function_count", "INTEGER", nullable=False),
            Column("callsite_count", "INTEGER", nullable=False),
            Column("modules_json", "JSON", nullable=False),
            Column("usage_modes", "JSON", nullable=False),
            Column("config_keys", "JSON"),
            Column("risk_level", "VARCHAR"),
            Column("created_at", "TIMESTAMP", nullable=False),
        ],
        primary_key=("repo", "commit", "dep_id"),
        description="Aggregated view of external libraries/services used by the repo",
    ),
    "analytics.external_dependency_calls": TableSchema(
        schema="analytics",
        name="external_dependency_calls",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("dep_id", "VARCHAR", nullable=False),
            Column("library", "VARCHAR", nullable=False),
            Column("service_name", "VARCHAR"),
            Column("language", "VARCHAR"),
            Column("severity", "VARCHAR"),
            Column("criticality", "DOUBLE"),
            Column("risk_score", "DOUBLE"),
            Column("matched_pattern", "VARCHAR"),
            Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),
            Column("function_urn", "VARCHAR", nullable=False),
            Column("rel_path", "VARCHAR", nullable=False),
            Column("module", "VARCHAR", nullable=False),
            Column("qualname", "VARCHAR", nullable=False),
            Column("callsite_count", "INTEGER", nullable=False),
            Column("modes", "JSON", nullable=False),
            Column("evidence_json", "JSON"),
            Column("created_at", "TIMESTAMP", nullable=False),
        ],
        primary_key=("repo", "commit", "dep_id", "function_goid_h128"),
        description="Function-level callsites into external dependencies with modes and evidence",
    ),
    "analytics.data_models": TableSchema(
        schema="analytics",
        name="data_models",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("model_id", "VARCHAR", nullable=False),
            Column("goid_h128", "DECIMAL(38,0)"),
            Column("model_name", "VARCHAR", nullable=False),
            Column("module", "VARCHAR", nullable=False),
            Column("rel_path", "VARCHAR", nullable=False),
            Column("model_kind", "VARCHAR", nullable=False),
            Column("base_classes_json", "JSON"),
            Column("doc_short", "VARCHAR"),
            Column("doc_long", "VARCHAR"),
            Column("created_at", "TIMESTAMP", nullable=False),
        ],
        primary_key=("repo", "commit", "model_id"),
        description="Extracted data models (dataclasses, Pydantic, TypedDicts, ORMs)",
    ),
    "analytics.data_model_fields": TableSchema(
        schema="analytics",
        name="data_model_fields",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("model_id", "VARCHAR", nullable=False),
            Column("field_name", "VARCHAR", nullable=False),
            Column("field_type", "VARCHAR"),
            Column("required", "BOOLEAN", nullable=False),
            Column("has_default", "BOOLEAN", nullable=False),
            Column("default_expr", "VARCHAR"),
            Column("constraints_json", "JSON", nullable=False),
            Column("source", "VARCHAR", nullable=False),
            Column("rel_path", "VARCHAR", nullable=False),
            Column("lineno", "INTEGER"),
            Column("created_at", "TIMESTAMP", nullable=False),
        ],
        primary_key=("repo", "commit", "model_id", "field_name"),
        description="Normalized field definitions extracted from analytics.data_models.",
    ),
    "analytics.data_model_relationships": TableSchema(
        schema="analytics",
        name="data_model_relationships",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("source_model_id", "VARCHAR", nullable=False),
            Column("target_model_id", "VARCHAR", nullable=False),
            Column("target_module", "VARCHAR"),
            Column("target_model_name", "VARCHAR"),
            Column("field_name", "VARCHAR", nullable=False),
            Column("relationship_kind", "VARCHAR", nullable=False),
            Column("multiplicity", "VARCHAR"),
            Column("via", "VARCHAR"),
            Column("evidence_json", "JSON"),
            Column("rel_path", "VARCHAR", nullable=False),
            Column("lineno", "INTEGER"),
            Column("created_at", "TIMESTAMP", nullable=False),
        ],
        primary_key=(
            "repo",
            "commit",
            "source_model_id",
            "field_name",
            "target_model_id",
            "relationship_kind",
        ),
        description="Resolved relationships between data models with evidence and provenance.",
    ),
    "analytics.data_model_usage": TableSchema(
        schema="analytics",
        name="data_model_usage",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("model_id", "VARCHAR", nullable=False),
            Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),
            Column("usage_kinds_json", "JSON", nullable=False),
            Column("evidence_json", "JSON"),
            Column("context_json", "JSON"),
            Column("created_at", "TIMESTAMP", nullable=False),
        ],
        primary_key=("repo", "commit", "model_id", "function_goid_h128"),
        description="Per function/model usage summary and context (CRUD/validate/serialize).",
    ),
    "analytics.config_data_flow": TableSchema(
        schema="analytics",
        name="config_data_flow",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("config_key", "VARCHAR", nullable=False),
            Column("config_path", "VARCHAR", nullable=False),
            Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),
            Column("usage_kind", "VARCHAR", nullable=False),
            Column("evidence_json", "JSON"),
            Column("call_chain_id", "VARCHAR", nullable=False),
            Column("call_chain_json", "JSON"),
            Column("created_at", "TIMESTAMP", nullable=False),
        ],
        primary_key=(
            "repo",
            "commit",
            "config_key",
            "config_path",
            "function_goid_h128",
            "usage_kind",
            "call_chain_id",
        ),
        description="Function-level config key usage and call-chain context from entrypoints.",
    ),
    "analytics.function_validation": TableSchema(
        schema="analytics",
        name="function_validation",
        columns=[
            *REPO_COMMIT_COLS,
            *FUNCTION_GOID_COL,
            Column("rel_path", "VARCHAR", nullable=False),
            Column("qualname", "VARCHAR", nullable=False),
            Column("issue", "VARCHAR", nullable=False),
            Column("detail", "VARCHAR", nullable=False),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "function_goid_h128", "issue"),
        indexes=(Index("idx_function_validation_repo_commit", ("repo", "commit")),),
        description="Validation findings for function analytics gaps",
    ),
    "analytics.graph_validation": TableSchema(
        schema="analytics",
        name="graph_validation",
        columns=[
            *REPO_COMMIT_COLS,
            Column("graph_name", "VARCHAR", nullable=False),
            Column("entity_id", "VARCHAR", nullable=False),
            Column("issue", "VARCHAR", nullable=False),
            Column("severity", "VARCHAR", nullable=True),
            Column("rel_path", "VARCHAR", nullable=True),
            Column("detail", "VARCHAR", nullable=False),
            Column("metadata", "JSON", nullable=True),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "graph_name", "entity_id", "issue"),
        indexes=(Index("idx_graph_validation_repo_commit", ("repo", "commit")),),
        description="Validation findings for analytics graph consistency checks",
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
            *FUNCTION_ENTITY_COLS,
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
            *CREATED_AT_COL_NULLABLE,
        ],
        indexes=(Index("idx_analytics_function_metrics_goid", ("function_goid_h128",)),),
        description="Per-function structural metrics",
    ),
    "analytics.function_types": TableSchema(
        schema="analytics",
        name="function_types",
        columns=[
            *FUNCTION_ENTITY_COLS,
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
            *CREATED_AT_COL_NULLABLE,
        ],
        indexes=(Index("idx_analytics_function_types_goid", ("function_goid_h128",)),),
        description="Per-function annotation coverage",
    ),
    "analytics.function_effects": TableSchema(
        schema="analytics",
        name="function_effects",
        columns=[
            *REPO_COMMIT_COLS,
            *FUNCTION_GOID_COL,
            Column("is_pure", "BOOLEAN", nullable=False),
            Column("uses_io", "BOOLEAN", nullable=False),
            Column("touches_db", "BOOLEAN", nullable=False),
            Column("uses_time", "BOOLEAN", nullable=False),
            Column("uses_randomness", "BOOLEAN", nullable=False),
            Column("modifies_globals", "BOOLEAN", nullable=False),
            Column("modifies_closure", "BOOLEAN", nullable=False),
            Column("spawns_threads_or_tasks", "BOOLEAN", nullable=False),
            Column("has_transitive_effects", "BOOLEAN", nullable=False),
            Column("purity_confidence", "DOUBLE"),
            Column("effects_json", "JSON"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "function_goid_h128"),
        indexes=(Index("idx_analytics_function_effects_goid", ("function_goid_h128",)),),
        description="Side-effect and purity classification per function GOID",
    ),
    "analytics.function_contracts": TableSchema(
        schema="analytics",
        name="function_contracts",
        columns=[
            *REPO_COMMIT_COLS,
            *FUNCTION_GOID_COL,
            Column("preconditions_json", "JSON"),
            Column("postconditions_json", "JSON"),
            Column("raises_json", "JSON"),
            Column("param_nullability_json", "JSON"),
            Column("return_nullability", "VARCHAR"),
            Column("contract_confidence", "DOUBLE"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "function_goid_h128"),
        indexes=(Index("idx_analytics_function_contracts_goid", ("function_goid_h128",)),),
        description="Inferred pre/postconditions and nullability per function",
    ),
    "analytics.coverage_functions": TableSchema(
        schema="analytics",
        name="coverage_functions",
        columns=[
            *FUNCTION_ENTITY_COLS,
            Column("executable_lines", "INTEGER"),
            Column("covered_lines", "INTEGER"),
            Column("coverage_ratio", "DOUBLE"),
            Column("tested", "BOOLEAN"),
            Column("untested_reason", "VARCHAR"),
            *CREATED_AT_COL_NULLABLE,
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
            *RISK_COLS,
            *OWNERSHIP_COLS,
            *CREATED_AT_COL_NULLABLE,
        ],
        indexes=(Index("idx_analytics_gorf_goid", ("function_goid_h128",)),),
        description="Composite risk factors per function",
    ),
    "analytics.graph_metrics_functions": TableSchema(
        schema="analytics",
        name="graph_metrics_functions",
        columns=[
            *REPO_COMMIT_COLS,
            *FUNCTION_GOID_COL,
            Column("call_fan_in", "INTEGER", nullable=False),
            Column("call_fan_out", "INTEGER", nullable=False),
            Column("call_in_degree", "INTEGER", nullable=False),
            Column("call_out_degree", "INTEGER", nullable=False),
            Column("call_pagerank", "DOUBLE"),
            Column("call_betweenness", "DOUBLE"),
            Column("call_closeness", "DOUBLE"),
            Column("call_cycle_member", "BOOLEAN", nullable=False),
            Column("call_cycle_id", "INTEGER"),
            Column("call_layer", "INTEGER"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "function_goid_h128"),
        indexes=(Index("idx_analytics_graph_metrics_fn_goid", ("function_goid_h128",)),),
        description="Graph metrics per function computed from the call graph",
    ),
    "analytics.graph_metrics_modules": TableSchema(
        schema="analytics",
        name="graph_metrics_modules",
        columns=[
            *MODULE_ENTITY_COLS,
            Column("import_fan_in", "INTEGER", nullable=False),
            Column("import_fan_out", "INTEGER", nullable=False),
            Column("import_in_degree", "INTEGER", nullable=False),
            Column("import_out_degree", "INTEGER", nullable=False),
            Column("import_pagerank", "DOUBLE"),
            Column("import_betweenness", "DOUBLE"),
            Column("import_closeness", "DOUBLE"),
            Column("import_cycle_member", "BOOLEAN", nullable=False),
            Column("import_cycle_id", "INTEGER"),
            Column("import_layer", "INTEGER"),
            Column("symbol_fan_in", "INTEGER", nullable=False),
            Column("symbol_fan_out", "INTEGER", nullable=False),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "module"),
        indexes=(Index("idx_analytics_graph_metrics_module", ("module",)),),
        description="Graph metrics per module computed from imports and symbol uses",
    ),
    "analytics.graph_metrics_modules_ext": TableSchema(
        schema="analytics",
        name="graph_metrics_modules_ext",
        columns=[
            *MODULE_ENTITY_COLS,
            Column("import_betweenness", "DOUBLE"),
            Column("import_closeness", "DOUBLE"),
            Column("import_eigenvector", "DOUBLE"),
            Column("import_harmonic", "DOUBLE"),
            Column("import_k_core", "INTEGER"),
            Column("import_constraint", "DOUBLE"),
            Column("import_effective_size", "DOUBLE"),
            Column("import_rich_club", "BOOLEAN"),
            Column("import_shell_index", "INTEGER"),
            Column("import_community_id", "INTEGER"),
            Column("import_component_id", "INTEGER"),
            Column("import_component_size", "INTEGER"),
            Column("import_scc_id", "INTEGER"),
            Column("import_scc_size", "INTEGER"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "module"),
        indexes=(Index("idx_analytics_graph_metrics_modules_ext_module", ("module",)),),
        description="Extended import-graph metrics per module (centralities, cores, structural holes)",
    ),
    "analytics.graph_metrics_functions_ext": TableSchema(
        schema="analytics",
        name="graph_metrics_functions_ext",
        columns=[
            *REPO_COMMIT_COLS,
            *FUNCTION_GOID_COL,
            Column("call_betweenness", "DOUBLE"),
            Column("call_closeness", "DOUBLE"),
            Column("call_eigenvector", "DOUBLE"),
            Column("call_harmonic", "DOUBLE"),
            Column("call_core_number", "INTEGER"),
            Column("call_clustering_coeff", "DOUBLE"),
            Column("call_triangle_count", "BIGINT"),
            Column("call_is_articulation", "BOOLEAN"),
            Column("call_articulation_impact", "INTEGER"),
            Column("call_is_bridge_endpoint", "BOOLEAN"),
            Column("call_component_id", "INTEGER"),
            Column("call_component_size", "INTEGER"),
            Column("call_scc_id", "INTEGER"),
            Column("call_scc_size", "INTEGER"),
            Column("call_ancestor_count", "INTEGER"),
            Column("call_descendant_count", "INTEGER"),
            Column("call_community_id", "INTEGER"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "function_goid_h128"),
        indexes=(Index("idx_analytics_graph_metrics_ext_fn_goid", ("function_goid_h128",)),),
        description="Extended call graph metrics per function (centralities, components)",
    ),
    "analytics.subsystem_graph_metrics": TableSchema(
        schema="analytics",
        name="subsystem_graph_metrics",
        columns=[
            *SUBSYSTEM_ENTITY_COLS,
            Column("import_in_degree", "DOUBLE"),
            Column("import_out_degree", "DOUBLE"),
            Column("import_pagerank", "DOUBLE"),
            Column("import_betweenness", "DOUBLE"),
            Column("import_closeness", "DOUBLE"),
            Column("import_layer", "INTEGER"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "subsystem_id"),
        description="Graph metrics on the subsystem-level condensed import graph",
    ),
    "analytics.subsystem_profile_cache": TableSchema(
        schema="analytics",
        name="subsystem_profile_cache",
        columns=[
            *SUBSYSTEM_ENTITY_COLS,
            Column("name", "VARCHAR"),
            Column("description", "VARCHAR"),
            Column("module_count", "INTEGER"),
            Column("modules_json", "JSON"),
            Column("entrypoints_json", "JSON"),
            Column("internal_edge_count", "INTEGER"),
            Column("external_edge_count", "INTEGER"),
            Column("fan_in", "INTEGER"),
            Column("fan_out", "INTEGER"),
            Column("function_count", "INTEGER"),
            Column("avg_risk_score", "DOUBLE"),
            Column("max_risk_score", "DOUBLE"),
            Column("high_risk_function_count", "INTEGER"),
            Column("risk_level", "VARCHAR"),
            Column("import_in_degree", "DOUBLE"),
            Column("import_out_degree", "DOUBLE"),
            Column("import_pagerank", "DOUBLE"),
            Column("import_betweenness", "DOUBLE"),
            Column("import_closeness", "DOUBLE"),
            Column("import_layer", "INTEGER"),
            *CREATED_AT_COL_NULLABLE,
        ],
        primary_key=("repo", "commit", "subsystem_id"),
        indexes=(Index("idx_subsystem_profile_cache_repo_commit", ("repo", "commit")),),
        description="Materialized subsystem profile rows for docs views",
    ),
    "analytics.subsystem_coverage_cache": TableSchema(
        schema="analytics",
        name="subsystem_coverage_cache",
        columns=[
            *SUBSYSTEM_ENTITY_COLS,
            Column("name", "VARCHAR"),
            Column("description", "VARCHAR"),
            Column("module_count", "INTEGER"),
            Column("function_count", "INTEGER"),
            Column("risk_level", "VARCHAR"),
            Column("avg_risk_score", "DOUBLE"),
            Column("max_risk_score", "DOUBLE"),
            Column("test_count", "INTEGER"),
            Column("passed_test_count", "INTEGER"),
            Column("failed_test_count", "INTEGER"),
            Column("skipped_test_count", "INTEGER"),
            Column("xfail_test_count", "INTEGER"),
            Column("flaky_test_count", "INTEGER"),
            Column("total_functions_covered", "INTEGER"),
            Column("avg_functions_covered", "DOUBLE"),
            Column("max_functions_covered", "DOUBLE"),
            Column("min_functions_covered", "DOUBLE"),
            Column("function_coverage_ratio", "DOUBLE"),
            *CREATED_AT_COL_NULLABLE,
        ],
        primary_key=("repo", "commit", "subsystem_id"),
        indexes=(Index("idx_subsystem_coverage_cache_repo_commit", ("repo", "commit")),),
        description="Materialized subsystem coverage aggregates for docs views",
    ),
    "analytics.graph_stats": TableSchema(
        schema="analytics",
        name="graph_stats",
        columns=[
            Column("graph_name", "VARCHAR", nullable=False),
            *REPO_COMMIT_COLS,
            Column("node_count", "BIGINT"),
            Column("edge_count", "BIGINT"),
            Column("weak_component_count", "INTEGER"),
            Column("scc_count", "INTEGER"),
            Column("component_layers", "INTEGER"),
            Column("avg_clustering", "DOUBLE"),
            Column("diameter_estimate", "DOUBLE"),
            Column("avg_shortest_path_estimate", "DOUBLE"),
            *CREATED_AT_COL,
        ],
        primary_key=("graph_name", "repo", "commit"),
        description="Global graph-level statistics for call/import graphs",
    ),
    "analytics.symbol_graph_metrics_modules": TableSchema(
        schema="analytics",
        name="symbol_graph_metrics_modules",
        columns=[
            *MODULE_ENTITY_COLS,
            Column("symbol_betweenness", "DOUBLE"),
            Column("symbol_closeness", "DOUBLE"),
            Column("symbol_eigenvector", "DOUBLE"),
            Column("symbol_harmonic", "DOUBLE"),
            Column("symbol_k_core", "INTEGER"),
            Column("symbol_constraint", "DOUBLE"),
            Column("symbol_effective_size", "DOUBLE"),
            Column("symbol_community_id", "INTEGER"),
            Column("symbol_component_id", "INTEGER"),
            Column("symbol_component_size", "INTEGER"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "module"),
        description="Symbol-coupling graph metrics per module",
    ),
    "analytics.symbol_graph_metrics_functions": TableSchema(
        schema="analytics",
        name="symbol_graph_metrics_functions",
        columns=[
            *REPO_COMMIT_COLS,
            *FUNCTION_GOID_COL,
            Column("symbol_betweenness", "DOUBLE"),
            Column("symbol_closeness", "DOUBLE"),
            Column("symbol_eigenvector", "DOUBLE"),
            Column("symbol_harmonic", "DOUBLE"),
            Column("symbol_k_core", "INTEGER"),
            Column("symbol_constraint", "DOUBLE"),
            Column("symbol_effective_size", "DOUBLE"),
            Column("symbol_community_id", "INTEGER"),
            Column("symbol_component_id", "INTEGER"),
            Column("symbol_component_size", "INTEGER"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "function_goid_h128"),
        indexes=(Index("idx_symbol_graph_metrics_fn", ("function_goid_h128",)),),
        description="Symbol-coupling graph metrics per function",
    ),
    "analytics.config_graph_metrics_keys": TableSchema(
        schema="analytics",
        name="config_graph_metrics_keys",
        columns=[
            *REPO_COMMIT_COLS,
            Column("config_key", "VARCHAR", nullable=False),
            Column("degree", "INTEGER"),
            Column("weighted_degree", "DOUBLE"),
            Column("betweenness", "DOUBLE"),
            Column("closeness", "DOUBLE"),
            Column("community_id", "INTEGER"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "config_key"),
        description="Metrics for config keys in config-module bipartite/projection graphs",
    ),
    "analytics.config_graph_metrics_modules": TableSchema(
        schema="analytics",
        name="config_graph_metrics_modules",
        columns=[
            *MODULE_ENTITY_COLS,
            Column("degree", "INTEGER"),
            Column("weighted_degree", "DOUBLE"),
            Column("betweenness", "DOUBLE"),
            Column("closeness", "DOUBLE"),
            Column("community_id", "INTEGER"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "module"),
        description="Metrics for modules in config-module bipartite/projection graphs",
    ),
    "analytics.config_projection_key_edges": TableSchema(
        schema="analytics",
        name="config_projection_key_edges",
        columns=[
            *REPO_COMMIT_COLS,
            Column("src_key", "VARCHAR", nullable=False),
            Column("dst_key", "VARCHAR", nullable=False),
            Column("weight", "DOUBLE"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "src_key", "dst_key"),
        description="Projected key-key edges from shared module usage",
    ),
    "analytics.config_projection_module_edges": TableSchema(
        schema="analytics",
        name="config_projection_module_edges",
        columns=[
            *REPO_COMMIT_COLS,
            Column("src_module", "VARCHAR", nullable=False),
            Column("dst_module", "VARCHAR", nullable=False),
            Column("weight", "DOUBLE"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "src_module", "dst_module"),
        description="Projected module-module edges from shared config keys",
    ),
    "analytics.subsystem_agreement": TableSchema(
        schema="analytics",
        name="subsystem_agreement",
        columns=[
            *MODULE_ENTITY_COLS,
            Column("subsystem_id", "VARCHAR"),
            Column("import_community_id", "INTEGER"),
            Column("agrees", "BOOLEAN"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "module"),
        description="Agreement check between subsystem labels and import communities",
    ),
    "analytics.test_graph_metrics_tests": TableSchema(
        schema="analytics",
        name="test_graph_metrics_tests",
        columns=[
            Column("test_id", "VARCHAR", nullable=False),
            *REPO_COMMIT_COLS,
            Column("degree", "INTEGER"),
            Column("weighted_degree", "DOUBLE"),
            Column("degree_centrality", "DOUBLE"),
            Column("proj_degree", "INTEGER"),
            Column("proj_weight", "DOUBLE"),
            Column("proj_clustering", "DOUBLE"),
            Column("proj_betweenness", "DOUBLE"),
            Column("risk_weighted_degree", "DOUBLE"),
            *CREATED_AT_COL,
        ],
        primary_key=("test_id", "repo", "commit"),
        description="Graph metrics for tests in the test-function bipartite graph",
    ),
    "analytics.test_graph_metrics_functions": TableSchema(
        schema="analytics",
        name="test_graph_metrics_functions",
        columns=[
            *FUNCTION_GOID_COL,
            *REPO_COMMIT_COLS,
            Column("tests_degree", "INTEGER"),
            Column("tests_weighted_degree", "DOUBLE"),
            Column("tests_degree_centrality", "DOUBLE"),
            Column("proj_degree", "INTEGER"),
            Column("proj_weight", "DOUBLE"),
            Column("proj_clustering", "DOUBLE"),
            Column("proj_betweenness", "DOUBLE"),
            Column("tests_risk_weighted_degree", "DOUBLE"),
            *CREATED_AT_COL,
        ],
        primary_key=("function_goid_h128", "repo", "commit"),
        indexes=(Index("idx_analytics_test_graph_metrics_fn_goid", ("function_goid_h128",)),),
        description="Function-side graph metrics from the test-function bipartite graph",
    ),
    "analytics.test_profile": TableSchema(
        schema="analytics",
        name="test_profile",
        columns=[
            *REPO_COMMIT_COLS,
            Column("test_id", "VARCHAR", nullable=False),
            Column("test_goid_h128", "DECIMAL(38,0)"),
            Column("urn", "VARCHAR"),
            Column("rel_path", "VARCHAR", nullable=False),
            Column("module", "VARCHAR"),
            Column("qualname", "VARCHAR"),
            Column("language", "VARCHAR"),
            Column("kind", "VARCHAR"),
            Column("status", "VARCHAR"),
            Column("duration_ms", "DOUBLE"),
            Column("markers", "JSON"),
            Column("flaky", "BOOLEAN"),
            Column("last_run_at", "TIMESTAMP"),
            Column("functions_covered", "JSON"),
            Column("functions_covered_count", "INTEGER"),
            Column("primary_function_goids", "JSON"),
            Column("subsystems_covered", "JSON"),
            Column("subsystems_covered_count", "INTEGER"),
            Column("primary_subsystem_id", "VARCHAR"),
            Column("assert_count", "INTEGER"),
            Column("raise_count", "INTEGER"),
            Column("uses_parametrize", "BOOLEAN"),
            Column("uses_fixtures", "BOOLEAN"),
            Column("io_bound", "BOOLEAN"),
            Column("uses_network", "BOOLEAN"),
            Column("uses_db", "BOOLEAN"),
            Column("uses_filesystem", "BOOLEAN"),
            Column("uses_subprocess", "BOOLEAN"),
            Column("flakiness_score", "DOUBLE"),
            Column("importance_score", "DOUBLE"),
            Column("notes", "VARCHAR"),
            Column("tg_degree", "INTEGER"),
            Column("tg_weighted_degree", "DOUBLE"),
            Column("tg_proj_degree", "INTEGER"),
            Column("tg_proj_weight", "DOUBLE"),
            Column("tg_proj_clustering", "DOUBLE"),
            Column("tg_proj_betweenness", "DOUBLE"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "test_id"),
        indexes=(
            Index("idx_analytics_test_profile_test_id", ("test_id",)),
            Index(
                "idx_analytics_test_profile_primary_subsystem",
                ("primary_subsystem_id", "repo", "commit"),
            ),
        ),
        description="Per-test profile combining execution status, coverage footprint, AST metrics, IO usage, and flakiness/importance heuristics.",
    ),
    "analytics.behavioral_coverage": TableSchema(
        schema="analytics",
        name="behavioral_coverage",
        columns=[
            *REPO_COMMIT_COLS,
            Column("test_id", "VARCHAR", nullable=False),
            Column("test_goid_h128", "DECIMAL(38,0)"),
            Column("rel_path", "VARCHAR", nullable=False),
            Column("qualname", "VARCHAR"),
            Column("behavior_tags", "JSON", nullable=False),
            Column("tag_source", "VARCHAR", nullable=False),
            Column("heuristic_version", "VARCHAR"),
            Column("llm_model", "VARCHAR"),
            Column("llm_run_id", "VARCHAR"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "test_id"),
        indexes=(Index("idx_analytics_behavioral_cov_test_id", ("test_id",)),),
        description="Behavioral coverage tags per test with heuristic/LLM provenance.",
    ),
    "analytics.cfg_block_metrics": TableSchema(
        schema="analytics",
        name="cfg_block_metrics",
        columns=[
            *FUNCTION_GOID_COL,
            *REPO_COMMIT_COLS,
            Column("block_idx", "INTEGER", nullable=False),
            Column("is_entry", "BOOLEAN"),
            Column("is_exit", "BOOLEAN"),
            Column("is_branch", "BOOLEAN"),
            Column("is_join", "BOOLEAN"),
            Column("dom_depth", "INTEGER"),
            Column("dominates_exit", "BOOLEAN"),
            Column("bc_betweenness", "DOUBLE"),
            Column("bc_closeness", "DOUBLE"),
            Column("bc_eigenvector", "DOUBLE"),
            Column("in_loop_scc", "BOOLEAN"),
            Column("loop_header", "BOOLEAN"),
            Column("loop_nesting_depth", "INTEGER"),
            *CREATED_AT_COL,
            Column("metrics_version", "INTEGER"),
        ],
        primary_key=("repo", "commit", "function_goid_h128", "block_idx"),
        indexes=(Index("idx_analytics_cfg_block_fn", ("function_goid_h128",)),),
        description="Control-flow graph block-level metrics per function",
    ),
    "analytics.cfg_function_metrics": TableSchema(
        schema="analytics",
        name="cfg_function_metrics",
        columns=[
            *FUNCTION_GOID_COL,
            *REPO_COMMIT_COLS,
            Column("rel_path", "VARCHAR", nullable=False),
            Column("module", "VARCHAR"),
            Column("qualname", "VARCHAR"),
            Column("cfg_block_count", "INTEGER"),
            Column("cfg_edge_count", "INTEGER"),
            Column("cfg_has_cycles", "BOOLEAN"),
            Column("cfg_scc_count", "INTEGER"),
            Column("cfg_longest_path_len", "INTEGER"),
            Column("cfg_avg_shortest_path_len", "DOUBLE"),
            Column("cfg_branching_factor_mean", "DOUBLE"),
            Column("cfg_branching_factor_max", "INTEGER"),
            Column("cfg_linear_block_fraction", "DOUBLE"),
            Column("cfg_dom_tree_height", "INTEGER"),
            Column("cfg_dominance_frontier_size_mean", "DOUBLE"),
            Column("cfg_dominance_frontier_size_max", "INTEGER"),
            Column("cfg_loop_count", "INTEGER"),
            Column("cfg_loop_nesting_depth_max", "INTEGER"),
            Column("cfg_bc_betweenness_max", "DOUBLE"),
            Column("cfg_bc_betweenness_mean", "DOUBLE"),
            Column("cfg_bc_closeness_mean", "DOUBLE"),
            Column("cfg_bc_eigenvector_max", "DOUBLE"),
            *CREATED_AT_COL,
            Column("metrics_version", "INTEGER"),
        ],
        primary_key=("repo", "commit", "function_goid_h128"),
        indexes=(Index("idx_analytics_cfg_fn_goid", ("function_goid_h128",)),),
        description="Control-flow graph summary metrics per function",
    ),
    "analytics.cfg_function_metrics_ext": TableSchema(
        schema="analytics",
        name="cfg_function_metrics_ext",
        columns=[
            *FUNCTION_GOID_COL,
            *REPO_COMMIT_COLS,
            Column("unreachable_block_count", "INTEGER"),
            Column("loop_header_count", "INTEGER"),
            Column("true_edge_count", "INTEGER"),
            Column("false_edge_count", "INTEGER"),
            Column("back_edge_count", "INTEGER"),
            Column("exception_edge_count", "INTEGER"),
            Column("fallthrough_edge_count", "INTEGER"),
            Column("loop_edge_count", "INTEGER"),
            Column("entry_exit_simple_paths", "INTEGER"),
            *CREATED_AT_COL,
            Column("metrics_version", "INTEGER"),
        ],
        primary_key=("repo", "commit", "function_goid_h128"),
        indexes=(Index("idx_analytics_cfg_fn_ext_goid", ("function_goid_h128",)),),
        description="Extended control-flow graph metrics per function",
    ),
    "analytics.dfg_block_metrics": TableSchema(
        schema="analytics",
        name="dfg_block_metrics",
        columns=[
            *FUNCTION_GOID_COL,
            *REPO_COMMIT_COLS,
            Column("block_idx", "INTEGER", nullable=False),
            Column("dfg_in_degree", "INTEGER"),
            Column("dfg_out_degree", "INTEGER"),
            Column("dfg_phi_in_degree", "INTEGER"),
            Column("dfg_phi_out_degree", "INTEGER"),
            Column("dfg_bc_betweenness", "DOUBLE"),
            Column("dfg_bc_closeness", "DOUBLE"),
            Column("dfg_bc_eigenvector", "DOUBLE"),
            Column("dfg_in_chain", "BOOLEAN"),
            Column("dfg_in_scc", "BOOLEAN"),
            *CREATED_AT_COL,
            Column("metrics_version", "INTEGER"),
        ],
        primary_key=("repo", "commit", "function_goid_h128", "block_idx"),
        indexes=(Index("idx_analytics_dfg_block_fn", ("function_goid_h128",)),),
        description="Data-flow graph block-level metrics per function",
    ),
    "analytics.dfg_function_metrics": TableSchema(
        schema="analytics",
        name="dfg_function_metrics",
        columns=[
            *FUNCTION_GOID_COL,
            *REPO_COMMIT_COLS,
            Column("rel_path", "VARCHAR", nullable=False),
            Column("module", "VARCHAR"),
            Column("qualname", "VARCHAR"),
            Column("dfg_block_count", "INTEGER"),
            Column("dfg_edge_count", "INTEGER"),
            Column("dfg_phi_edge_count", "INTEGER"),
            Column("dfg_symbol_count", "INTEGER"),
            Column("dfg_component_count", "INTEGER"),
            Column("dfg_scc_count", "INTEGER"),
            Column("dfg_has_cycles", "BOOLEAN"),
            Column("dfg_longest_chain_len", "INTEGER"),
            Column("dfg_avg_shortest_path_len", "DOUBLE"),
            Column("dfg_avg_in_degree", "DOUBLE"),
            Column("dfg_avg_out_degree", "DOUBLE"),
            Column("dfg_max_in_degree", "INTEGER"),
            Column("dfg_max_out_degree", "INTEGER"),
            Column("dfg_branchy_block_fraction", "DOUBLE"),
            Column("dfg_bc_betweenness_max", "DOUBLE"),
            Column("dfg_bc_betweenness_mean", "DOUBLE"),
            Column("dfg_bc_eigenvector_max", "DOUBLE"),
            *CREATED_AT_COL,
            Column("metrics_version", "INTEGER"),
        ],
        primary_key=("repo", "commit", "function_goid_h128"),
        indexes=(Index("idx_analytics_dfg_fn_goid", ("function_goid_h128",)),),
        description="Data-flow graph summary metrics per function",
    ),
    "analytics.dfg_function_metrics_ext": TableSchema(
        schema="analytics",
        name="dfg_function_metrics_ext",
        columns=[
            *FUNCTION_GOID_COL,
            *REPO_COMMIT_COLS,
            Column("data_flow_edge_count", "INTEGER"),
            Column("intra_block_edge_count", "INTEGER"),
            Column("use_kind_phi_count", "INTEGER"),
            Column("use_kind_data_flow_count", "INTEGER"),
            Column("use_kind_intra_block_count", "INTEGER"),
            Column("use_kind_other_count", "INTEGER"),
            Column("phi_edge_ratio", "DOUBLE"),
            Column("entry_exit_simple_paths", "INTEGER"),
            *CREATED_AT_COL,
            Column("metrics_version", "INTEGER"),
        ],
        primary_key=("repo", "commit", "function_goid_h128"),
        indexes=(Index("idx_analytics_dfg_fn_ext_goid", ("function_goid_h128",)),),
        description="Extended data-flow graph metrics per function",
    ),
    "analytics.function_history": TableSchema(
        schema="analytics",
        name="function_history",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),
            Column("urn", "VARCHAR", nullable=False),
            Column("rel_path", "VARCHAR", nullable=False),
            Column("module", "VARCHAR", nullable=False),
            Column("qualname", "VARCHAR", nullable=False),
            Column("created_in_commit", "VARCHAR"),
            Column("created_at", "TIMESTAMP"),
            Column("last_modified_commit", "VARCHAR"),
            Column("last_modified_at", "TIMESTAMP"),
            Column("age_days", "INTEGER"),
            Column("commit_count", "INTEGER", nullable=False),
            Column("author_count", "INTEGER", nullable=False),
            Column("lines_added", "BIGINT", nullable=False),
            Column("lines_deleted", "BIGINT", nullable=False),
            Column("churn_score", "DOUBLE", nullable=False),
            Column("stability_bucket", "VARCHAR", nullable=False),
            Column("history_window_start", "TIMESTAMP"),
            Column("history_window_end", "TIMESTAMP"),
            Column("created_at_row", "TIMESTAMP", nullable=False),
        ],
        primary_key=("repo", "commit", "function_goid_h128"),
        indexes=(
            Index("idx_analytics_function_history_goid", ("function_goid_h128",)),
            Index("idx_analytics_function_history_repo_commit", ("repo", "commit")),
        ),
        description="Per-function compressed git history & churn metrics derived from file history and GOID spans.",
    ),
    "analytics.function_profile": TableSchema(
        schema="analytics",
        name="function_profile",
        columns=[
            Column("function_goid_h128", "DECIMAL(38,0)"),
            Column("urn", "VARCHAR"),
            Column("repo", "VARCHAR"),
            Column("commit", "VARCHAR"),
            Column("rel_path", "VARCHAR"),
            Column("module", "VARCHAR"),
            Column("language", "VARCHAR"),
            Column("kind", "VARCHAR"),
            Column("qualname", "VARCHAR"),
            Column("start_line", "INTEGER"),
            Column("end_line", "INTEGER"),
            Column("loc", "INTEGER"),
            Column("logical_loc", "INTEGER"),
            Column("cyclomatic_complexity", "INTEGER"),
            Column("complexity_bucket", "VARCHAR"),
            Column("param_count", "INTEGER"),
            Column("positional_params", "INTEGER"),
            Column("keyword_params", "INTEGER"),
            Column("vararg", "BOOLEAN"),
            Column("kwarg", "BOOLEAN"),
            Column("max_nesting_depth", "INTEGER"),
            Column("stmt_count", "INTEGER"),
            Column("decorator_count", "INTEGER"),
            Column("has_docstring", "BOOLEAN"),
            Column("total_params", "INTEGER"),
            Column("annotated_params", "INTEGER"),
            Column("return_type", "VARCHAR"),
            Column("param_types", "JSON"),
            Column("fully_typed", "BOOLEAN"),
            Column("partial_typed", "BOOLEAN"),
            Column("untyped", "BOOLEAN"),
            Column("typedness_bucket", "VARCHAR"),
            Column("typedness_source", "VARCHAR"),
            Column("file_typed_ratio", "DOUBLE"),
            Column("static_error_count", "INTEGER"),
            Column("has_static_errors", "BOOLEAN"),
            Column("executable_lines", "INTEGER"),
            Column("covered_lines", "INTEGER"),
            Column("coverage_ratio", "DOUBLE"),
            Column("tested", "BOOLEAN"),
            Column("untested_reason", "VARCHAR"),
            Column("tests_touching", "INTEGER"),
            Column("failing_tests", "INTEGER"),
            Column("slow_tests", "INTEGER"),
            Column("flaky_tests", "INTEGER"),
            Column("last_test_status", "VARCHAR"),
            Column("dominant_test_status", "VARCHAR"),
            Column("slow_test_threshold_ms", "DOUBLE"),
            Column("created_in_commit", "VARCHAR"),
            Column("created_at_history", "TIMESTAMP"),
            Column("last_modified_commit", "VARCHAR"),
            Column("last_modified_at", "TIMESTAMP"),
            Column("age_days", "INTEGER"),
            Column("commit_count", "INTEGER"),
            Column("author_count", "INTEGER"),
            Column("lines_added", "BIGINT"),
            Column("lines_deleted", "BIGINT"),
            Column("churn_score", "DOUBLE"),
            Column("stability_bucket", "VARCHAR"),
            Column("call_fan_in", "INTEGER"),
            Column("call_fan_out", "INTEGER"),
            Column("call_edge_in_count", "INTEGER"),
            Column("call_edge_out_count", "INTEGER"),
            Column("call_is_leaf", "BOOLEAN"),
            Column("call_is_entrypoint", "BOOLEAN"),
            Column("call_is_public", "BOOLEAN"),
            Column("risk_score", "DOUBLE"),
            Column("risk_level", "VARCHAR"),
            Column("risk_component_coverage", "DOUBLE"),
            Column("risk_component_complexity", "DOUBLE"),
            Column("risk_component_static", "DOUBLE"),
            Column("risk_component_hotspot", "DOUBLE"),
            Column("is_pure", "BOOLEAN"),
            Column("uses_io", "BOOLEAN"),
            Column("touches_db", "BOOLEAN"),
            Column("uses_time", "BOOLEAN"),
            Column("uses_randomness", "BOOLEAN"),
            Column("modifies_globals", "BOOLEAN"),
            Column("modifies_closure", "BOOLEAN"),
            Column("spawns_threads_or_tasks", "BOOLEAN"),
            Column("has_transitive_effects", "BOOLEAN"),
            Column("purity_confidence", "DOUBLE"),
            Column("param_nullability_json", "JSON"),
            Column("return_nullability", "VARCHAR"),
            Column("has_preconditions", "BOOLEAN"),
            Column("has_postconditions", "BOOLEAN"),
            Column("has_raises", "BOOLEAN"),
            Column("contract_confidence", "DOUBLE"),
            Column("role", "VARCHAR"),
            Column("framework", "VARCHAR"),
            Column("role_confidence", "DOUBLE"),
            Column("role_sources_json", "JSON"),
            Column("tags", "JSON"),
            Column("owners", "JSON"),
            Column("doc_short", "VARCHAR"),
            Column("doc_long", "VARCHAR"),
            Column("doc_params", "JSON"),
            Column("doc_returns", "JSON"),
            Column("created_at", "TIMESTAMP"),
        ],
        indexes=(
            Index("idx_analytics_function_profile_goid", ("function_goid_h128",)),
            Index("idx_analytics_function_profile_repo_commit", ("repo", "commit")),
        ),
        description="Denormalized per-function profile combining risk, coverage, tests, docs, and graph metrics",
    ),
    "analytics.function_ast_features": TableSchema(
        schema="analytics",
        name="function_ast_features",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),
            Column("rel_path", "VARCHAR", nullable=False),
            Column("qualname", "VARCHAR", nullable=False),
            Column("is_async", "BOOLEAN", nullable=False),
            Column("uses_network", "BOOLEAN", nullable=False),
            Column("uses_db", "BOOLEAN", nullable=False),
            Column("uses_filesystem", "BOOLEAN", nullable=False),
            Column("uses_subprocess", "BOOLEAN", nullable=False),
            Column("uses_concurrency_lib", "BOOLEAN", nullable=False),
            Column("uses_threading", "BOOLEAN", nullable=False),
            Column("uses_asyncio_lib", "BOOLEAN", nullable=False),
            Column("http_client_libs", "JSON", nullable=False),
            Column("http_server_libs", "JSON", nullable=False),
            Column("db_libs", "JSON", nullable=False),
            Column("message_libs", "JSON", nullable=False),
            Column("config_read_count", "INTEGER", nullable=False),
            Column("feature_flag_count", "INTEGER", nullable=False),
            Column("decorators", "JSON", nullable=False),
            Column("libraries_used", "JSON", nullable=False),
            Column("created_at", "TIMESTAMP", nullable=False),
        ],
        primary_key=("repo", "commit", "function_goid_h128"),
        indexes=(
            Index(
                "idx_analytics_function_ast_features_repo_commit",
                ("repo", "commit"),
            ),
        ),
        description="Per-function AST-derived semantic features for explainability and classification.",
    ),
    "analytics.file_profile": TableSchema(
        schema="analytics",
        name="file_profile",
        columns=[
            Column("repo", "VARCHAR"),
            Column("commit", "VARCHAR"),
            Column("rel_path", "VARCHAR"),
            Column("module", "VARCHAR"),
            Column("language", "VARCHAR"),
            Column("node_count", "INTEGER"),
            Column("function_count", "INTEGER"),
            Column("class_count", "INTEGER"),
            Column("avg_depth", "DOUBLE"),
            Column("max_depth", "INTEGER"),
            Column("ast_complexity", "DOUBLE"),
            Column("hotspot_score", "DOUBLE"),
            Column("commit_count", "INTEGER"),
            Column("author_count", "INTEGER"),
            Column("lines_added", "INTEGER"),
            Column("lines_deleted", "INTEGER"),
            Column("annotation_ratio", "DOUBLE"),
            Column("untyped_defs", "INTEGER"),
            Column("overlay_needed", "BOOLEAN"),
            Column("type_error_count", "INTEGER"),
            Column("static_error_count", "INTEGER"),
            Column("has_static_errors", "BOOLEAN"),
            Column("total_functions", "INTEGER"),
            Column("public_functions", "INTEGER"),
            Column("avg_loc", "DOUBLE"),
            Column("max_loc", "INTEGER"),
            Column("avg_cyclomatic_complexity", "DOUBLE"),
            Column("max_cyclomatic_complexity", "INTEGER"),
            Column("high_risk_function_count", "INTEGER"),
            Column("medium_risk_function_count", "INTEGER"),
            Column("max_risk_score", "DOUBLE"),
            Column("file_coverage_ratio", "DOUBLE"),
            Column("tested_function_count", "INTEGER"),
            Column("untested_function_count", "INTEGER"),
            Column("tests_touching", "INTEGER"),
            Column("tags", "JSON"),
            Column("owners", "JSON"),
            Column("created_at", "TIMESTAMP"),
        ],
        indexes=(
            Index(
                "idx_analytics_file_profile_repo_commit_relpath",
                ("repo", "commit", "rel_path"),
            ),
        ),
        description="Per-file aggregation of structure, risk, coverage, and ownership",
    ),
    "analytics.history_timeseries": TableSchema(
        schema="analytics",
        name="history_timeseries",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("entity_kind", "VARCHAR", nullable=False),
            Column("entity_stable_id", "VARCHAR", nullable=False),
            Column("function_goid_h128", "DECIMAL(38,0)"),
            Column("module", "VARCHAR"),
            Column("rel_path", "VARCHAR", nullable=False),
            Column("language", "VARCHAR", nullable=False),
            Column("qualname", "VARCHAR"),
            Column("commit", "VARCHAR", nullable=False),
            Column("commit_ts", "TIMESTAMP", nullable=False),
            Column("loc", "INTEGER"),
            Column("cyclomatic_complexity", "INTEGER"),
            Column("coverage_ratio", "DOUBLE"),
            Column("static_error_count", "INTEGER"),
            Column("typedness_bucket", "VARCHAR"),
            Column("risk_score", "DOUBLE"),
            Column("risk_level", "VARCHAR"),
            Column("bucket_label", "VARCHAR"),
            Column("created_at_row", "TIMESTAMP", nullable=False),
        ],
        primary_key=("repo", "entity_kind", "entity_stable_id", "commit"),
        indexes=(
            Index(
                "idx_analytics_history_timeseries_entity",
                ("repo", "entity_kind", "entity_stable_id"),
            ),
        ),
        description="Per-commit metrics for selected functions/modules for temporal analysis.",
    ),
    "analytics.module_profile": TableSchema(
        schema="analytics",
        name="module_profile",
        columns=[
            Column("repo", "VARCHAR"),
            Column("commit", "VARCHAR"),
            Column("module", "VARCHAR"),
            Column("path", "VARCHAR"),
            Column("language", "VARCHAR"),
            Column("file_count", "INTEGER"),
            Column("total_loc", "INTEGER"),
            Column("total_logical_loc", "INTEGER"),
            Column("function_count", "INTEGER"),
            Column("class_count", "INTEGER"),
            Column("avg_file_complexity", "DOUBLE"),
            Column("max_file_complexity", "DOUBLE"),
            Column("high_risk_function_count", "INTEGER"),
            Column("medium_risk_function_count", "INTEGER"),
            Column("low_risk_function_count", "INTEGER"),
            Column("max_risk_score", "DOUBLE"),
            Column("avg_risk_score", "DOUBLE"),
            Column("module_coverage_ratio", "DOUBLE"),
            Column("tested_function_count", "INTEGER"),
            Column("untested_function_count", "INTEGER"),
            Column("import_fan_in", "INTEGER"),
            Column("import_fan_out", "INTEGER"),
            Column("cycle_group", "INTEGER"),
            Column("in_cycle", "BOOLEAN"),
            Column("role", "VARCHAR"),
            Column("role_confidence", "DOUBLE"),
            Column("role_sources_json", "JSON"),
            Column("tags", "JSON"),
            Column("owners", "JSON"),
            Column("created_at", "TIMESTAMP"),
        ],
        indexes=(
            Index(
                "idx_analytics_module_profile_repo_commit_module",
                ("repo", "commit", "module"),
            ),
        ),
        description="Per-module summary of size, risk, coverage, imports, and ownership",
    ),
    "analytics.semantic_roles_functions": TableSchema(
        schema="analytics",
        name="semantic_roles_functions",
        columns=[
            *REPO_COMMIT_COLS,
            *FUNCTION_GOID_COL,
            Column("role", "VARCHAR"),
            Column("framework", "VARCHAR"),
            Column("role_confidence", "DOUBLE"),
            Column("role_sources_json", "JSON"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "function_goid_h128"),
        indexes=(Index("idx_analytics_semantic_roles_fn", ("function_goid_h128",)),),
        description="Semantic role classification per function",
    ),
    "analytics.semantic_roles_modules": TableSchema(
        schema="analytics",
        name="semantic_roles_modules",
        columns=[
            *MODULE_ENTITY_COLS,
            Column("role", "VARCHAR"),
            Column("role_confidence", "DOUBLE"),
            Column("role_sources_json", "JSON"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "module"),
        indexes=(Index("idx_analytics_semantic_roles_mod", ("module",)),),
        description="Semantic role classification per module",
    ),
    "analytics.subsystems": TableSchema(
        schema="analytics",
        name="subsystems",
        columns=[
            *SUBSYSTEM_ENTITY_COLS,
            Column("name", "VARCHAR", nullable=False),
            Column("description", "VARCHAR"),
            Column("module_count", "INTEGER", nullable=False),
            Column("modules_json", "JSON", nullable=False),
            Column("entrypoints_json", "JSON"),
            Column("internal_edge_count", "INTEGER", nullable=False),
            Column("external_edge_count", "INTEGER", nullable=False),
            Column("fan_in", "INTEGER", nullable=False),
            Column("fan_out", "INTEGER", nullable=False),
            Column("function_count", "INTEGER", nullable=False),
            Column("avg_risk_score", "DOUBLE"),
            Column("max_risk_score", "DOUBLE"),
            Column("high_risk_function_count", "INTEGER", nullable=False),
            Column("risk_level", "VARCHAR"),
            *CREATED_AT_COL,
        ],
        primary_key=("repo", "commit", "subsystem_id"),
        indexes=(
            Index(
                "idx_analytics_subsystems_repo_commit_id",
                ("repo", "commit", "subsystem_id"),
            ),
        ),
        description="Inferred architectural subsystems with summary metrics",
    ),
    "analytics.subsystem_modules": TableSchema(
        schema="analytics",
        name="subsystem_modules",
        columns=[
            *SUBSYSTEM_ENTITY_COLS,
            Column("module", "VARCHAR", nullable=False),
            Column("role", "VARCHAR"),
        ],
        primary_key=("repo", "commit", "subsystem_id", "module"),
        indexes=(Index("idx_analytics_subsystem_modules_module", ("module",)),),
        description="Mapping of subsystems to member modules",
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
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
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
            Index("idx_graph_call_edges_repo_commit", ("repo", "commit")),
            Index("idx_graph_call_edges_caller", ("caller_goid_h128",)),
            Index("idx_graph_call_edges_callee", ("callee_goid_h128",)),
        ),
        description="Caller->callee edges with callsite evidence",
    ),
    "graph.import_graph_edges": TableSchema(
        schema="graph",
        name="import_graph_edges",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("src_module", "VARCHAR", nullable=False),
            Column("dst_module", "VARCHAR", nullable=False),
            Column("src_fan_out", "INTEGER", nullable=False),
            Column("dst_fan_in", "INTEGER", nullable=False),
            Column("cycle_group", "INTEGER", nullable=False),
            Column("module_layer", "INTEGER"),
        ],
        indexes=(
            Index("idx_graph_import_edges_repo_commit", ("repo", "commit")),
            Index("idx_graph_import_edges_src", ("src_module",)),
            Index("idx_graph_import_edges_dst", ("dst_module",)),
        ),
        description="Module-level import edges with fan-in/out metrics",
    ),
    "graph.import_modules": TableSchema(
        schema="graph",
        name="import_modules",
        columns=[
            *MODULE_ENTITY_COLS,
            Column("scc_id", "INTEGER", nullable=False),
            Column("component_size", "INTEGER", nullable=False),
            Column("layer", "INTEGER"),
            Column("cycle_group", "INTEGER", nullable=False),
        ],
        indexes=(
            Index("idx_graph_import_modules_repo_commit", ("repo", "commit")),
            Index("idx_graph_import_modules_module", ("module",)),
        ),
        description="Per-module import graph condensation metadata keyed by repo/commit",
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
            Column("via_phi", "BOOLEAN"),
            Column("use_kind", "VARCHAR"),
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
            Column("def_goid_h128", "DECIMAL(38,0)"),
            Column("use_goid_h128", "DECIMAL(38,0)"),
        ],
        primary_key=("symbol", "def_path", "use_path"),
        indexes=(Index("idx_graph_symbol_use_symbol", ("symbol",)),),
        description="Definition-to-use edges derived from SCIP",
    ),
    "docs.v_validation_summary": TableSchema(
        schema="docs",
        name="v_validation_summary",
        columns=[
            Column("domain", "VARCHAR"),
            Column("repo", "VARCHAR"),
            Column("commit", "VARCHAR"),
            Column("entity_id", "VARCHAR"),
            Column("issue", "VARCHAR"),
            Column("detail", "VARCHAR"),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Section 0.7: Named Column Constants (derived from TABLE_SCHEMAS)
# ---------------------------------------------------------------------------
# These constants provide backward-compatible access to column names for
# commonly used tables. Use get_table_columns(table_key) for dynamic access.

AST_NODES_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["core.ast_nodes"].column_names()
AST_METRICS_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["core.ast_metrics"].column_names()
CST_NODES_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["core.cst_nodes"].column_names()
DOCSTRINGS_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["core.docstrings"].column_names()
MODULES_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["core.modules"].column_names()
FILE_STATE_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["core.file_state"].column_names()
REPO_MAP_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["core.repo_map"].column_names()
GOIDS_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["core.goids"].column_names()
GOID_CROSSWALK_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["core.goid_crosswalk"].column_names()

COVERAGE_LINES_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["analytics.coverage_lines"].column_names()
TEST_CATALOG_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["analytics.test_catalog"].column_names()
CONFIG_VALUES_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["analytics.config_values"].column_names()
TAGS_INDEX_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["analytics.tags_index"].column_names()
TYPEDNESS_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["analytics.typedness"].column_names()
STATIC_DIAGNOSTICS_COLUMNS: Final[list[str]] = (
    TABLE_SCHEMAS["analytics.static_diagnostics"].column_names()
)
HOTSPOTS_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["analytics.hotspots"].column_names()
# Note: FUNCTION_METRICS_COLUMNS, FUNCTION_TYPES_COLUMNS, and TEST_COVERAGE_EDGE_COLUMNS
# are defined below in the serialization section as tuple[str, ...] for use with _serialize_row
FUNCTION_EFFECTS_COLUMNS: Final[list[str]] = (
    TABLE_SCHEMAS["analytics.function_effects"].column_names()
)
FUNCTION_CONTRACTS_COLUMNS: Final[list[str]] = (
    TABLE_SCHEMAS["analytics.function_contracts"].column_names()
)
SEMANTIC_ROLES_FUNCTIONS_COLUMNS: Final[list[str]] = (
    TABLE_SCHEMAS["analytics.semantic_roles_functions"].column_names()
)
SEMANTIC_ROLES_MODULES_COLUMNS: Final[list[str]] = (
    TABLE_SCHEMAS["analytics.semantic_roles_modules"].column_names()
)

CALL_GRAPH_NODE_COLUMNS: Final[list[str]] = (
    TABLE_SCHEMAS["graph.call_graph_nodes"].column_names()
)
CALL_GRAPH_EDGE_COLUMNS: Final[list[str]] = (
    TABLE_SCHEMAS["graph.call_graph_edges"].column_names()
)
IMPORT_EDGE_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["graph.import_graph_edges"].column_names()
IMPORT_MODULE_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["graph.import_modules"].column_names()
CFG_BLOCK_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["graph.cfg_blocks"].column_names()
CFG_EDGE_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["graph.cfg_edges"].column_names()
DFG_EDGE_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["graph.dfg_edges"].column_names()
SYMBOL_USE_COLUMNS: Final[list[str]] = TABLE_SCHEMAS["graph.symbol_use_edges"].column_names()

CFG_FUNCTION_METRICS_EXT_COLUMNS: Final[list[str]] = (
    TABLE_SCHEMAS["analytics.cfg_function_metrics_ext"].column_names()
)
DFG_FUNCTION_METRICS_EXT_COLUMNS: Final[list[str]] = (
    TABLE_SCHEMAS["analytics.dfg_function_metrics_ext"].column_names()
)


# ---------------------------------------------------------------------------
# Section 0.75: SQL Generation from TABLE_SCHEMAS
# ---------------------------------------------------------------------------


def _build_insert_sql(table_key: str) -> str:
    """
    Generate an INSERT SQL statement from the TableSchema.

    Parameters
    ----------
    table_key
        Fully qualified DuckDB table identifier, e.g. "analytics.function_metrics".

    Returns
    -------
    str
        INSERT INTO statement with placeholders.

    Raises
    ------
    ValueError
        If no schema is defined for the given table key.
    """
    schema = TABLE_SCHEMAS.get(table_key)
    if schema is None:
        message = f"No schema defined for table key: {table_key}"
        raise ValueError(message)
    col_names = [col.name for col in schema.columns]
    cols_str = ", ".join(col_names)
    placeholders = ", ".join("?" * len(col_names))
    return f"INSERT INTO {table_key} ({cols_str}) VALUES ({placeholders})"  # noqa: S608


def _build_insert_sql_by_table() -> dict[str, str]:
    """
    Generate INSERT SQL statements for all non-view tables.

    Returns
    -------
    dict[str, str]
        Mapping from table key to INSERT SQL statement.
    """
    result: dict[str, str] = {}
    for table_key in TABLE_SCHEMAS:
        if table_key.startswith("docs."):
            continue
        result[table_key] = _build_insert_sql(table_key)
    return result


def _build_delete_sql(table_key: str) -> str | None:
    """
    Generate a DELETE SQL statement from the TableSchema.

    Parameters
    ----------
    table_key
        Fully qualified DuckDB table identifier.

    Returns
    -------
    str | None
        DELETE FROM statement with placeholders, or None if not applicable.
    """
    schema = TABLE_SCHEMAS.get(table_key)
    if schema is None:
        return None
    col_names = [col.name for col in schema.columns]
    if "repo" in col_names and "commit" in col_names:
        return f"DELETE FROM {table_key} WHERE repo = ? AND commit = ?"  # noqa: S608
    return None


def _build_delete_sql_by_table() -> dict[str, str]:
    """
    Generate DELETE SQL statements for all tables with repo+commit columns.

    Returns
    -------
    dict[str, str]
        Mapping from table key to DELETE SQL statement.
    """
    result: dict[str, str] = {}
    for table_key in TABLE_SCHEMAS:
        if table_key.startswith("docs."):
            continue
        sql = _build_delete_sql(table_key)
        if sql is not None:
            result[table_key] = sql
    return result


INSERT_SQL_BY_TABLE: Final[dict[str, str]] = _build_insert_sql_by_table()
DELETE_SQL_BY_TABLE: Final[dict[str, str]] = _build_delete_sql_by_table()


# ---------------------------------------------------------------------------
# Special SQL Constants (non-generatable patterns)
# ---------------------------------------------------------------------------

# AST/CST tables use path-based subqueries for deletion (no direct repo/commit)
AST_NODES_DELETE = (
    "DELETE FROM core.ast_nodes "
    "WHERE path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)"
)
AST_METRICS_DELETE = (
    "DELETE FROM core.ast_metrics "
    "WHERE rel_path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)"
)
CST_NODES_DELETE = (
    "DELETE FROM core.cst_nodes "
    "WHERE path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)"
)

# File state uses 3 parameters for deletion
FILE_STATE_DELETE = "DELETE FROM core.file_state WHERE repo = ? AND rel_path = ? AND language = ?"

# Global tables without repo/commit
TAGS_INDEX_DELETE = "DELETE FROM analytics.tags_index"
SYMBOL_USE_DELETE = "DELETE FROM graph.symbol_use_edges"
CALL_GRAPH_NODES_DELETE = "DELETE FROM graph.call_graph_nodes"
CFG_BLOCKS_DELETE = "DELETE FROM graph.cfg_blocks"
CFG_EDGES_DELETE = "DELETE FROM graph.cfg_edges"
DFG_EDGES_DELETE = "DELETE FROM graph.dfg_edges"

# UPDATE statements (can't be auto-generated)
TEST_CATALOG_UPDATE_GOIDS = (
    "UPDATE analytics.test_catalog "
    "SET test_goid_h128 = ?, urn = ? "
    "WHERE test_id = ? AND rel_path = ? AND repo = ? AND commit = ?"
)
GOID_CROSSWALK_UPDATE_SCIP = (
    "UPDATE core.goid_crosswalk SET scip_symbol = ? WHERE goid = ? AND repo = ? AND commit = ?"
)


# ---------------------------------------------------------------------------
# Registry Adapter Functions (absorbed from registry_adapter.py)
# ---------------------------------------------------------------------------


def load_columns_by_table() -> dict[str, list[str]]:
    """
    Return column-name lists for all tables tracked in TABLE_SCHEMAS.

    This is the no-DB-connection alternative to load_registry_columns().

    Returns
    -------
    dict[str, list[str]]
        Mapping of table key -> ordered column names.
    """
    return {table_key: [col.name for col in schema.columns] for table_key, schema in TABLE_SCHEMAS.items()}


def get_table_columns(table_key: str) -> list[str]:
    """
    Return ordered column names for a specific table.

    Parameters
    ----------
    table_key
        Fully qualified DuckDB table identifier, e.g. "analytics.function_metrics".

    Returns
    -------
    list[str]
        Column names in definition order.

    Raises
    ------
    KeyError
        If no schema is defined for the given table key.
    """
    schema = TABLE_SCHEMAS.get(table_key)
    if schema is None:
        message = f"No schema defined for table key: {table_key}"
        raise KeyError(message)
    return [col.name for col in schema.columns]


# ---------------------------------------------------------------------------
# Section 1: Row Serialization Helpers
# ---------------------------------------------------------------------------


def _serialize_row(row: Mapping[_Column, object], columns: Sequence[_Column]) -> tuple[object, ...]:
    """
    Serialize a mapping using a stable column sequence.

    Returns
    -------
    tuple[object, ...]
        Values ordered according to ``columns``.
    """
    return tuple(row[column] for column in columns)


def _get_contract_columns(table_key: str) -> tuple[str, ...]:
    """
    Retrieve column names from the TableSchema for a given table key.

    Parameters
    ----------
    table_key
        Fully qualified DuckDB table identifier, e.g. "analytics.function_metrics".

    Returns
    -------
    tuple[str, ...]
        Column names in schema definition order.

    Raises
    ------
    ValueError
        If no schema is defined for the given table key.
    """
    schema = TABLE_SCHEMAS.get(table_key)
    if schema is None:
        message = f"No schema defined for table key: {table_key}"
        raise ValueError(message)
    return tuple(schema.column_names())


# ---------------------------------------------------------------------------
# Section 2: TypedDict Row Models and Serializers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestRunRow:
    """Row shape for control-plane ingest runs persisted to DuckDB."""

    repo: str
    commit: str
    step: str
    run_id: str
    mode: str
    started_at: datetime
    finished_at: datetime | None
    duration_s: float | None
    rows_inserted: int
    rows_deleted: int
    status: str
    error_kind: str | None
    error_message: str | None
    datasets: str
    modules_total: int | None
    modules_changed: int | None
    modules_deleted: int | None
    modules_changed_ratio: float | None
    modules_deleted_ratio: float | None
    use_full_rebuild: bool | None


class IngestRunLike(Protocol):
    """Structural contract for ingest run serialization."""

    repo: str
    commit: str
    step: str
    run_id: str
    mode: IngestRunMode
    started_at: datetime
    finished_at: datetime | None
    duration_s: float | None
    rows_inserted: int
    rows_deleted: int
    status: IngestRunStatus
    error_kind: str | None
    error_message: str | None
    datasets: tuple[str, ...]
    modules_total: int | None
    modules_changed: int | None
    modules_deleted: int | None
    modules_changed_ratio: float | None
    modules_deleted_ratio: float | None
    use_full_rebuild: bool | None


def ingest_run_to_tuple(run: IngestRunLike) -> tuple[object, ...]:
    """
    Serialize an IngestRun into the INSERT column order for core.ingest_runs.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by ingest_runs INSERTs.
    """
    return (
        run.repo,
        run.commit,
        run.step,
        run.run_id,
        run.mode.value,
        run.started_at,
        run.finished_at,
        run.duration_s,
        run.rows_inserted,
        run.rows_deleted,
        run.status.value,
        run.error_kind,
        run.error_message,
        json.dumps(list(run.datasets)),
        run.modules_total,
        run.modules_changed,
        run.modules_deleted,
        run.modules_changed_ratio,
        run.modules_deleted_ratio,
        run.use_full_rebuild,
    )


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


class FunctionMetricsRow(TypedDict):
    """Row shape for analytics.function_metrics inserts."""

    function_goid_h128: int
    urn: str | None
    repo: str
    commit: str
    rel_path: str
    language: str | None
    kind: str | None
    qualname: str | None
    start_line: int | None
    end_line: int | None
    loc: int | None
    logical_loc: int | None
    param_count: int | None
    positional_params: int | None
    keyword_only_params: int | None
    has_varargs: bool
    has_varkw: bool
    is_async: bool
    is_generator: bool
    return_count: int | None
    yield_count: int | None
    raise_count: int | None
    cyclomatic_complexity: int | None
    max_nesting_depth: int | None
    stmt_count: int | None
    decorator_count: int | None
    has_docstring: bool
    complexity_bucket: str | None
    created_at: datetime


FUNCTION_METRICS_COLUMNS: tuple[str, ...] = _get_contract_columns("analytics.function_metrics")


def function_metrics_row_to_tuple(row: FunctionMetricsRow) -> tuple[object, ...]:
    """
    Serialize a FunctionMetricsRow into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values aligned with analytics.function_metrics columns.
    """
    return _serialize_row(row, FUNCTION_METRICS_COLUMNS)


class FunctionTypesRow(TypedDict):
    """Row shape for analytics.function_types inserts."""

    function_goid_h128: int
    urn: str | None
    repo: str
    commit: str
    rel_path: str
    language: str | None
    kind: str | None
    qualname: str | None
    start_line: int | None
    end_line: int | None
    total_params: int | None
    annotated_params: int | None
    unannotated_params: int | None
    param_typed_ratio: float | None
    has_return_annotation: bool
    return_type: str | None
    return_type_source: str | None
    type_comment: str | None
    param_types: object
    fully_typed: bool
    partial_typed: bool
    untyped: bool
    typedness_bucket: str | None
    typedness_source: str | None
    created_at: datetime


FUNCTION_TYPES_COLUMNS: tuple[str, ...] = _get_contract_columns("analytics.function_types")


def function_types_row_to_tuple(row: FunctionTypesRow) -> tuple[object, ...]:
    """
    Serialize a FunctionTypesRow into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values aligned with analytics.function_types columns.
    """
    return _serialize_row(row, FUNCTION_TYPES_COLUMNS)


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


GRAPH_METRICS_FUNCTIONS_COLUMNS: tuple[str, ...] = _get_contract_columns(
    "analytics.graph_metrics_functions"
)


def graph_metrics_functions_row_to_tuple(
    row: GraphMetricsFunctionsRow,
) -> tuple[object, ...]:
    """
    Serialize a GraphMetricsFunctionsRow into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values aligned with analytics.graph_metrics_functions columns.
    """
    return _serialize_row(row, GRAPH_METRICS_FUNCTIONS_COLUMNS)


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


GRAPH_METRICS_MODULES_COLUMNS: tuple[str, ...] = _get_contract_columns(
    "analytics.graph_metrics_modules"
)


def graph_metrics_modules_row_to_tuple(
    row: GraphMetricsModulesRow,
) -> tuple[object, ...]:
    """
    Serialize a GraphMetricsModulesRow into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values aligned with analytics.graph_metrics_modules columns.
    """
    return _serialize_row(row, GRAPH_METRICS_MODULES_COLUMNS)


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


GRAPH_METRICS_FUNCTIONS_EXT_COLUMNS: tuple[str, ...] = _get_contract_columns(
    "analytics.graph_metrics_functions_ext"
)


def graph_metrics_functions_ext_row_to_tuple(
    row: GraphMetricsFunctionsExtRow,
) -> tuple[object, ...]:
    """
    Serialize a GraphMetricsFunctionsExtRow into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values aligned with analytics.graph_metrics_functions_ext columns.
    """
    return _serialize_row(row, GRAPH_METRICS_FUNCTIONS_EXT_COLUMNS)


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


GRAPH_METRICS_MODULES_EXT_COLUMNS: tuple[str, ...] = _get_contract_columns(
    "analytics.graph_metrics_modules_ext"
)


def graph_metrics_modules_ext_row_to_tuple(
    row: GraphMetricsModulesExtRow,
) -> tuple[object, ...]:
    """
    Serialize a GraphMetricsModulesExtRow into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values aligned with analytics.graph_metrics_modules_ext columns.
    """
    return _serialize_row(row, GRAPH_METRICS_MODULES_EXT_COLUMNS)


class CallGraphNodeRow(TypedDict):
    """Row shape for graph.call_graph_nodes inserts."""

    goid_h128: int
    language: str
    kind: str
    arity: int
    is_public: bool
    rel_path: str


def call_graph_node_to_tuple(row: CallGraphNodeRow) -> tuple[object, ...]:
    """
    Serialize a CallGraphNodeRow into the INSERT column order.

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
    """
    Serialize a CallGraphEdgeRow into the INSERT column order.

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
    """
    Serialize an ImportEdgeRow into the INSERT column order.

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
    """
    Serialize a CFGBlockRow into the INSERT column order.

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
    """
    Serialize a CFGEdgeRow into the INSERT column order.

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
    """
    Serialize a DFGEdgeRow into the INSERT column order.

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


TEST_COVERAGE_EDGE_COLUMNS: tuple[str, ...] = _get_contract_columns(
    "analytics.test_coverage_edges"
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


FUNCTION_PROFILE_COLUMNS: tuple[str, ...] = _get_contract_columns("analytics.function_profile")


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


_FUNCTION_AST_FEATURES_COLUMNS: tuple[str, ...] = _get_contract_columns(
    "analytics.function_ast_features"
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
    """
    Serialize a FunctionAstFeaturesRow into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values ordered per analytics.function_ast_features definition.
    """
    return _serialize_row(row, _FUNCTION_AST_FEATURES_COLUMNS)


FILE_PROFILE_COLUMNS: tuple[str, ...] = _get_contract_columns("analytics.file_profile")


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


MODULE_PROFILE_COLUMNS: tuple[str, ...] = _get_contract_columns("analytics.module_profile")


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


TEST_PROFILE_COLUMNS: tuple[str, ...] = _get_contract_columns("analytics.test_profile")


class ProfileRowModel(TypedDict):
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


TestProfileRowModel = ProfileRowModel


def serialize_test_profile_row(row: ProfileRowModel) -> tuple[object, ...]:
    """
    Serialize a ProfileRowModel into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by test_profile INSERTs.
    """
    return _serialize_row(row, TEST_PROFILE_COLUMNS)


BEHAVIORAL_COVERAGE_COLUMNS: tuple[str, ...] = _get_contract_columns(
    "analytics.behavioral_coverage"
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


SUBSYSTEM_PROFILE_COLUMNS: tuple[str, ...] = _get_contract_columns(
    "analytics.subsystem_profile_cache"
)


class SubsystemProfileCacheRow(TypedDict):
    """Row shape for ``analytics.subsystem_profile_cache`` inserts."""

    repo: str
    commit: str
    subsystem_id: str
    name: str | None
    description: str | None
    module_count: int | None
    modules_json: object | None
    entrypoints_json: list[object] | dict[str, object] | None
    internal_edge_count: int | None
    external_edge_count: int | None
    fan_in: int | None
    fan_out: int | None
    function_count: int | None
    avg_risk_score: float | None
    max_risk_score: float | None
    high_risk_function_count: int | None
    risk_level: str | None
    import_in_degree: float | None
    import_out_degree: float | None
    import_pagerank: float | None
    import_betweenness: float | None
    import_closeness: float | None
    import_layer: int | None
    created_at: datetime | None


def subsystem_profile_cache_to_tuple(row: SubsystemProfileCacheRow) -> tuple[object, ...]:
    """
    Serialize a SubsystemProfileCacheRow into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by subsystem_profile_cache INSERTs.
    """
    return _serialize_row(row, SUBSYSTEM_PROFILE_COLUMNS)


SUBSYSTEM_COVERAGE_COLUMNS: tuple[str, ...] = _get_contract_columns(
    "analytics.subsystem_coverage_cache"
)


class SubsystemCoverageCacheRow(TypedDict):
    """Row shape for ``analytics.subsystem_coverage_cache`` inserts."""

    repo: str
    commit: str
    subsystem_id: str
    name: str | None
    description: str | None
    module_count: int | None
    function_count: int | None
    risk_level: str | None
    avg_risk_score: float | None
    max_risk_score: float | None
    test_count: int | None
    passed_test_count: int | None
    failed_test_count: int | None
    skipped_test_count: int | None
    xfail_test_count: int | None
    flaky_test_count: int | None
    total_functions_covered: int | None
    avg_functions_covered: float | None
    max_functions_covered: float | None
    min_functions_covered: float | None
    function_coverage_ratio: float | None
    created_at: datetime | None


def subsystem_coverage_cache_to_tuple(row: SubsystemCoverageCacheRow) -> tuple[object, ...]:
    """
    Serialize a SubsystemCoverageCacheRow into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by subsystem_coverage_cache INSERTs.
    """
    return _serialize_row(row, SUBSYSTEM_COVERAGE_COLUMNS)


# ---------------------------------------------------------------------------
# Section 3: DatasetContract and RowBinding
# ---------------------------------------------------------------------------


# Datasets that should bypass normalized macros and use dataset_rows-only access.
_DATASET_ROWS_ONLY: Final[set[str]] = {
    "analytics.config_graph_metrics_keys",
    "analytics.config_graph_metrics_modules",
    "analytics.config_projection_key_edges",
    "analytics.config_projection_module_edges",
    "analytics.config_values",
    "analytics.coverage_lines",
    "analytics.data_model_fields",
    "analytics.data_model_relationships",
    "analytics.data_models",
    "analytics.external_dependencies",
    "analytics.file_profile",
    "analytics.graph_metrics_modules",
    "analytics.graph_metrics_modules_ext",
    "analytics.graph_stats",
    "analytics.hotspots",
    "analytics.module_profile",
    "analytics.subsystem_profile_cache",
    "analytics.subsystem_coverage_cache",
    "analytics.semantic_roles_modules",
    "analytics.static_diagnostics",
    "analytics.subsystem_agreement",
    "analytics.subsystem_graph_metrics",
    "analytics.subsystem_modules",
    "analytics.subsystems",
    "analytics.symbol_graph_metrics_modules",
    "analytics.tags_index",
    "analytics.test_graph_metrics_tests",
    "analytics.typedness",
    "core.ast_metrics",
    "core.ast_nodes",
    "core.cst_nodes",
    "core.docstrings",
    "core.file_state",
    "core.goid_crosswalk",
    "core.goids",
    "core.modules",
    "core.repo_map",
    "graph.call_graph_nodes",
    "graph.import_graph_edges",
    "graph.import_modules",
}


@dataclass(frozen=True)
class RowBinding:
    """Connect a DuckDB table key to a TypedDict row model and serializer."""

    row_type: RowDictType
    to_tuple: RowToTuple


@dataclass(frozen=True)
class DatasetContract:
    """Metadata describing a logical dataset backed by a DuckDB table or view.

    Attributes
    ----------
    table_key
        Fully qualified DuckDB identifier, e.g. "analytics.function_profile".
    name
        Logical dataset name, e.g. "function_profile".
    schema
        Statically defined TableSchema when the dataset is backed by a table;
        None when the dataset is a view.
    row_binding
        Optional binding to a TypedDict row model and serializer.
    json_schema_id
        Optional JSON Schema identifier (without .json) used for export validation.
    jsonl_filename
        Default filename for JSONL exports (may be None when not exported).
    parquet_filename
        Default filename for Parquet exports (may be None when not exported).
    is_view
        True when this dataset is a docs.* view instead of a base table.
    owner_package
        Optional package ownership derived from schema prefix (core, analytics, graphs, qa, docs).
    tags
        Classification tags applied to the dataset (e.g., base_table, docs_view, read_only).
    description
        Optional human-readable description of the dataset's purpose.
    family
        Optional dataset family inferred from the schema prefix (e.g., "core",
        "analytics", "docs").
    owner
        Optional team or individual owner for stewardship and escalation.
    freshness_sla
        Optional freshness expectation (e.g., "daily", "hourly").
    retention_policy
        Optional retention policy descriptor (e.g., "90d").
    stable_id
        Optional stable identifier for comparing contracts across versions.
    schema_version
        Optional schema version string for change tracking.
    upstream_dependencies
        Optional tuple of other dataset names this dataset depends on.
    """

    table_key: str
    name: str
    schema: TableSchema | None
    row_binding: RowBinding | None = None
    json_schema_id: str | None = None
    jsonl_filename: str | None = None
    parquet_filename: str | None = None
    is_view: bool = False
    owner_package: Literal["core", "analytics", "graphs", "qa", "docs"] | None = None
    tags: frozenset[str] = frozenset()
    description: str | None = None
    family: str | None = None
    owner: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    stable_id: str | None = None
    schema_version: str | None = None
    upstream_dependencies: tuple[str, ...] = ()
    validation_profile: Literal["strict", "lenient"] = "strict"

    def has_row_binding(self) -> bool:
        """
        Return True when this dataset has a TypedDict row binding.

        Returns
        -------
        bool
            True when a row binding is configured.
        """
        return self.row_binding is not None

    def require_row_binding(self) -> RowBinding:
        """
        Return the row binding or raise a clear error if missing.

        Returns
        -------
        RowBinding
            Configured row binding for this dataset.

        Raises
        ------
        KeyError
            If no row binding is configured for this dataset.
        """
        if self.row_binding is None:
            message = f"Dataset {self.name} ({self.table_key}) has no row binding"
            raise KeyError(message)
        return self.row_binding

    def capabilities(self) -> dict[str, bool]:
        """
        Return capability flags derived from attached metadata.

        Returns
        -------
        dict[str, bool]
            Flags for validation and export support.
        """
        docs_view = self.table_key.startswith("docs.")
        read_only = self.is_view or docs_view or "read_only" in self.tags
        requires_normalized_macro = self.schema is not None and "dataset_rows_only" not in self.tags
        return {
            "can_validate": self.json_schema_id is not None,
            "can_export_jsonl": self.jsonl_filename is not None,
            "can_export_parquet": self.parquet_filename is not None,
            "has_row_binding": self.row_binding is not None,
            "is_view": self.is_view,
            "docs_view": docs_view,
            "read_only": read_only,
            "dataset_rows_only": "dataset_rows_only" in self.tags,
            "requires_normalized_macro": requires_normalized_macro,
        }

    def column_names(self) -> tuple[str, ...]:
        """
        Return column names in schema definition order.

        Returns
        -------
        tuple[str, ...]
            Ordered column names from the underlying TableSchema, or empty
            tuple when this dataset has no schema (e.g., views).
        """
        if self.schema is None:
            return ()
        return tuple(self.schema.column_names())


def _row_binding(
    row_type: RowDictType,
    to_tuple: Callable[..., tuple[object, ...]],
) -> RowBinding:
    return RowBinding(row_type=row_type, to_tuple=cast("RowToTuple", to_tuple))


def _metadata_for_name(name: str) -> dict[str, object]:
    return {
        "description": _DESCRIPTION_BY_DATASET_NAME.get(name),
        "owner": _OWNER_BY_DATASET_NAME.get(name),
        "freshness_sla": _FRESHNESS_BY_DATASET_NAME.get(name),
        "retention_policy": _RETENTION_BY_DATASET_NAME.get(name),
        "upstream_dependencies": _DEPENDENCIES_BY_DATASET_NAME.get(name, ()),
        "stable_id": _STABLE_ID_BY_DATASET_NAME.get(name, name),
        "schema_version": _SCHEMA_VERSION_BY_DATASET_NAME.get(name, "1"),
        "validation_profile": _VALIDATION_PROFILE_BY_DATASET_NAME.get(name, "strict"),
    }


ROW_BINDINGS_BY_TABLE_KEY: Final[dict[str, RowBinding]] = {
    "analytics.coverage_lines": _row_binding(
        row_type=CoverageLineRow,
        to_tuple=coverage_line_to_tuple,
    ),
    "analytics.config_values": _row_binding(
        row_type=ConfigValueRow,
        to_tuple=config_value_to_tuple,
    ),
    "analytics.typedness": _row_binding(
        row_type=TypednessRow,
        to_tuple=typedness_row_to_tuple,
    ),
    "analytics.static_diagnostics": _row_binding(
        row_type=StaticDiagnosticRow,
        to_tuple=static_diagnostic_to_tuple,
    ),
    "analytics.function_validation": _row_binding(
        row_type=FunctionValidationRow,
        to_tuple=function_validation_row_to_tuple,
    ),
    "analytics.function_metrics": _row_binding(
        row_type=FunctionMetricsRow,
        to_tuple=function_metrics_row_to_tuple,
    ),
    "analytics.function_types": _row_binding(
        row_type=FunctionTypesRow,
        to_tuple=function_types_row_to_tuple,
    ),
    "analytics.graph_validation": _row_binding(
        row_type=GraphValidationRow,
        to_tuple=graph_validation_row_to_tuple,
    ),
    "analytics.hotspots": _row_binding(
        row_type=HotspotRow,
        to_tuple=hotspot_row_to_tuple,
    ),
    "analytics.test_catalog": _row_binding(
        row_type=TestCatalogRowModel,
        to_tuple=serialize_test_catalog_row,
    ),
    "analytics.test_coverage_edges": _row_binding(
        row_type=TestCoverageEdgeRow,
        to_tuple=serialize_test_coverage_edge,
    ),
    "core.docstrings": _row_binding(
        row_type=DocstringRow,
        to_tuple=docstring_row_to_tuple,
    ),
    "core.goids": _row_binding(
        row_type=GoidRow,
        to_tuple=goid_to_tuple,
    ),
    "core.goid_crosswalk": _row_binding(
        row_type=GoidCrosswalkRow,
        to_tuple=goid_crosswalk_to_tuple,
    ),
    "analytics.function_profile": _row_binding(
        row_type=FunctionProfileRowModel,
        to_tuple=function_profile_row_to_tuple,
    ),
    "analytics.function_ast_features": _row_binding(
        row_type=FunctionAstFeaturesRow,
        to_tuple=function_ast_features_row_to_tuple,
    ),
    "analytics.file_profile": _row_binding(
        row_type=FileProfileRowModel,
        to_tuple=file_profile_row_to_tuple,
    ),
    "analytics.module_profile": _row_binding(
        row_type=ModuleProfileRowModel,
        to_tuple=module_profile_row_to_tuple,
    ),
    "graph.call_graph_nodes": _row_binding(
        row_type=CallGraphNodeRow,
        to_tuple=call_graph_node_to_tuple,
    ),
    "graph.call_graph_edges": _row_binding(
        row_type=CallGraphEdgeRow,
        to_tuple=call_graph_edge_to_tuple,
    ),
    "graph.import_graph_edges": _row_binding(
        row_type=ImportEdgeRow,
        to_tuple=import_edge_to_tuple,
    ),
    "graph.import_modules": _row_binding(
        row_type=ImportModuleRow,
        to_tuple=import_module_to_tuple,
    ),
    "graph.cfg_blocks": _row_binding(
        row_type=CFGBlockRow,
        to_tuple=cfg_block_to_tuple,
    ),
    "graph.cfg_edges": _row_binding(
        row_type=CFGEdgeRow,
        to_tuple=cfg_edge_to_tuple,
    ),
    "graph.dfg_edges": _row_binding(
        row_type=DFGEdgeRow,
        to_tuple=dfg_edge_to_tuple,
    ),
    "graph.symbol_use_edges": _row_binding(
        row_type=SymbolUseRow,
        to_tuple=symbol_use_to_tuple,
    ),
    "analytics.graph_metrics_functions": _row_binding(
        row_type=GraphMetricsFunctionsRow,
        to_tuple=graph_metrics_functions_row_to_tuple,
    ),
    "analytics.graph_metrics_modules": _row_binding(
        row_type=GraphMetricsModulesRow,
        to_tuple=graph_metrics_modules_row_to_tuple,
    ),
    "analytics.graph_metrics_functions_ext": _row_binding(
        row_type=GraphMetricsFunctionsExtRow,
        to_tuple=graph_metrics_functions_ext_row_to_tuple,
    ),
    "analytics.graph_metrics_modules_ext": _row_binding(
        row_type=GraphMetricsModulesExtRow,
        to_tuple=graph_metrics_modules_ext_row_to_tuple,
    ),
    "analytics.test_profile": _row_binding(
        row_type=ProfileRowModel,
        to_tuple=serialize_test_profile_row,
    ),
    "analytics.behavioral_coverage": _row_binding(
        row_type=BehavioralCoverageRowModel,
        to_tuple=behavioral_coverage_row_to_tuple,
    ),
    "analytics.subsystem_profile_cache": _row_binding(
        row_type=SubsystemProfileCacheRow,
        to_tuple=subsystem_profile_cache_to_tuple,
    ),
    "analytics.subsystem_coverage_cache": _row_binding(
        row_type=SubsystemCoverageCacheRow,
        to_tuple=subsystem_coverage_cache_to_tuple,
    ),
    "docs.v_subsystem_profile": _row_binding(
        row_type=SubsystemProfileCacheRow,
        to_tuple=subsystem_profile_cache_to_tuple,
    ),
    "docs.v_subsystem_coverage": _row_binding(
        row_type=SubsystemCoverageCacheRow,
        to_tuple=subsystem_coverage_cache_to_tuple,
    ),
}

# Dataset-level JSON Schema metadata.
# Keys: dataset logical names (Dataset.name).
# Values: JSON Schema identifiers (filenames without .json) under
# src/codeintel/config/schemas/export/.
_JSON_SCHEMA_BY_DATASET_NAME: Final[dict[str, str]] = {
    # Profiles
    "function_profile": "function_profile",
    "file_profile": "file_profile",
    "module_profile": "module_profile",
    # Graph edges
    "call_graph_edges": "call_graph_edges",
    "symbol_use_edges": "symbol_use_edges",
    "test_coverage_edges": "test_coverage_edges",
    # Tests
    "test_profile": "test_profile",
    "behavioral_coverage": "behavioral_coverage",
    "v_subsystem_profile": "v_subsystem_profile",
    "v_subsystem_coverage": "v_subsystem_coverage",
    "subsystem_profile_cache": "subsystem_profile_cache",
    "subsystem_coverage_cache": "subsystem_coverage_cache",
    # Data models
    "data_model_fields": "data_model_fields",
    "data_model_relationships": "data_model_relationships",
}

_DESCRIPTION_BY_DATASET_NAME: Final[dict[str, str]] = {
    "function_profile": "Function-level profile combining metrics, risk, and topology.",
    "file_profile": "File-level profile with coverage, hotspots, and ownership signals.",
    "module_profile": "Module-level profile aggregating functions, imports, and risk.",
    "v_subsystem_profile": "Subsystem-level profile combining risk, connectivity, and metadata.",
    "v_subsystem_coverage": "Subsystem coverage rollup derived from test profiles.",
    "subsystem_profile_cache": "Materialized subsystem profile cache for docs views.",
    "subsystem_coverage_cache": "Materialized subsystem coverage cache for docs views.",
    "call_graph_edges": "Directed call graph edges across the codebase.",
    "symbol_use_edges": "Symbol use edges linking definitions to references.",
    "test_coverage_edges": "Test-to-target coverage edges for tracing impacts.",
    "test_profile": "Test-level profile including outcomes and runtime metadata.",
    "behavioral_coverage": "Behavioral coverage findings captured during scenario runs.",
    "data_model_fields": "Normalized data model field definitions for analytics export.",
    "data_model_relationships": "Normalized data model relationships for analytics export.",
}

_OWNER_BY_DATASET_NAME: Final[dict[str, str]] = {
    "function_profile": "analytics",
    "file_profile": "analytics",
    "module_profile": "analytics",
    "call_graph_edges": "graphs",
    "symbol_use_edges": "graphs",
    "test_coverage_edges": "analytics",
    "test_profile": "qa",
    "behavioral_coverage": "qa",
    "v_subsystem_profile": "docs",
    "v_subsystem_coverage": "docs",
    "subsystem_profile_cache": "analytics",
    "subsystem_coverage_cache": "analytics",
    "data_model_fields": "analytics",
    "data_model_relationships": "analytics",
}

_FRESHNESS_BY_DATASET_NAME: Final[dict[str, str]] = {
    "function_profile": "daily",
    "file_profile": "daily",
    "module_profile": "daily",
    "call_graph_edges": "daily",
    "symbol_use_edges": "daily",
    "test_coverage_edges": "daily",
    "test_profile": "daily",
    "behavioral_coverage": "daily",
    "v_subsystem_profile": "daily",
    "v_subsystem_coverage": "daily",
    "subsystem_profile_cache": "daily",
    "subsystem_coverage_cache": "daily",
    "data_model_fields": "daily",
    "data_model_relationships": "daily",
}

_RETENTION_BY_DATASET_NAME: Final[dict[str, str]] = {
    "function_profile": "90d",
    "file_profile": "90d",
    "module_profile": "90d",
    "call_graph_edges": "90d",
    "symbol_use_edges": "90d",
    "test_coverage_edges": "90d",
    "test_profile": "90d",
    "behavioral_coverage": "90d",
    "v_subsystem_profile": "90d",
    "v_subsystem_coverage": "90d",
    "subsystem_profile_cache": "90d",
    "subsystem_coverage_cache": "90d",
    "data_model_fields": "90d",
    "data_model_relationships": "90d",
}

_STABLE_ID_BY_DATASET_NAME: Final[dict[str, str]] = {}
_SCHEMA_VERSION_BY_DATASET_NAME: Final[dict[str, str]] = {}
_VALIDATION_PROFILE_BY_DATASET_NAME: Final[dict[str, Literal["strict", "lenient"]]] = {}

_DEPENDENCIES_BY_DATASET_NAME: Final[dict[str, tuple[str, ...]]] = {
    "function_profile": ("call_graph_edges", "symbol_use_edges"),
    "file_profile": ("call_graph_edges",),
    "module_profile": ("call_graph_edges", "symbol_use_edges"),
    "test_profile": ("test_coverage_edges",),
    "behavioral_coverage": ("test_profile",),
    "data_model_relationships": ("data_model_fields",),
}

_DEFAULT_JSONL_FILENAMES: Final[dict[str, str]] = {
    # GOIDs / crosswalk
    "core.goids": "goids.jsonl",
    "core.goid_crosswalk": "goid_crosswalk.jsonl",
    # Call graph
    "graph.call_graph_nodes": "call_graph_nodes.jsonl",
    "graph.call_graph_edges": "call_graph_edges.jsonl",
    # CFG / DFG
    "graph.cfg_blocks": "cfg_blocks.jsonl",
    "graph.cfg_edges": "cfg_edges.jsonl",
    "graph.dfg_edges": "dfg_edges.jsonl",
    # Import / symbol uses
    "graph.import_graph_edges": "import_graph_edges.jsonl",
    "graph.symbol_use_edges": "symbol_use_edges.jsonl",
    # AST / CST
    "core.ast_nodes": "ast_nodes.jsonl",
    "core.ast_metrics": "ast_metrics.jsonl",
    "core.cst_nodes": "cst_nodes.jsonl",
    "core.docstrings": "docstrings.jsonl",
    # Modules / config / diagnostics
    "core.modules": "modules.jsonl",
    "analytics.config_values": "config_values.jsonl",
    "analytics.data_models": "data_models.jsonl",
    "analytics.data_model_fields": "data_model_fields.jsonl",
    "analytics.data_model_relationships": "data_model_relationships.jsonl",
    "analytics.data_model_usage": "data_model_usage.jsonl",
    "analytics.config_data_flow": "config_data_flow.jsonl",
    "analytics.static_diagnostics": "static_diagnostics.jsonl",
    # AST analytics / typing
    "analytics.hotspots": "hotspots.jsonl",
    "analytics.typedness": "typedness.jsonl",
    # Function analytics
    "analytics.function_metrics": "function_metrics.jsonl",
    "analytics.function_types": "function_types.jsonl",
    "analytics.function_effects": "function_effects.jsonl",
    "analytics.function_contracts": "function_contracts.jsonl",
    "analytics.function_ast_features": "function_ast_features.jsonl",
    "analytics.semantic_roles_functions": "semantic_roles_functions.jsonl",
    "analytics.semantic_roles_modules": "semantic_roles_modules.jsonl",
    # Coverage + tests
    "analytics.coverage_lines": "coverage_lines.jsonl",
    "analytics.coverage_functions": "coverage_functions.jsonl",
    "analytics.test_catalog": "test_catalog.jsonl",
    "analytics.test_coverage_edges": "test_coverage_edges.jsonl",
    "analytics.entrypoints": "entrypoints.jsonl",
    "analytics.entrypoint_tests": "entrypoint_tests.jsonl",
    "analytics.external_dependencies": "external_dependencies.jsonl",
    "analytics.external_dependency_calls": "external_dependency_calls.jsonl",
    "analytics.graph_validation": "graph_validation.jsonl",
    "analytics.function_validation": "function_validation.jsonl",
    # Risk factors
    "analytics.goid_risk_factors": "goid_risk_factors.jsonl",
    "analytics.function_profile": "function_profile.jsonl",
    "analytics.function_history": "function_history.jsonl",
    "analytics.history_timeseries": "history_timeseries.jsonl",
    "analytics.file_profile": "file_profile.jsonl",
    "analytics.module_profile": "module_profile.jsonl",
    "analytics.graph_metrics_functions": "graph_metrics_functions.jsonl",
    "analytics.graph_metrics_functions_ext": "graph_metrics_functions_ext.jsonl",
    "analytics.graph_metrics_modules": "graph_metrics_modules.jsonl",
    "analytics.graph_metrics_modules_ext": "graph_metrics_modules_ext.jsonl",
    "analytics.subsystem_graph_metrics": "subsystem_graph_metrics.jsonl",
    "analytics.symbol_graph_metrics_modules": "symbol_graph_metrics_modules.jsonl",
    "analytics.symbol_graph_metrics_functions": "symbol_graph_metrics_functions.jsonl",
    "analytics.config_graph_metrics_keys": "config_graph_metrics_keys.jsonl",
    "analytics.config_graph_metrics_modules": "config_graph_metrics_modules.jsonl",
    "analytics.config_projection_key_edges": "config_projection_key_edges.jsonl",
    "analytics.config_projection_module_edges": "config_projection_module_edges.jsonl",
    "analytics.subsystem_agreement": "subsystem_agreement.jsonl",
    "analytics.graph_stats": "graph_stats.jsonl",
    "analytics.test_graph_metrics_tests": "test_graph_metrics_tests.jsonl",
    "analytics.test_graph_metrics_functions": "test_graph_metrics_functions.jsonl",
    "analytics.test_profile": "test_profile.jsonl",
    "analytics.behavioral_coverage": "behavioral_coverage.jsonl",
    "analytics.cfg_block_metrics": "cfg_block_metrics.jsonl",
    "analytics.cfg_function_metrics": "cfg_function_metrics.jsonl",
    "analytics.dfg_block_metrics": "dfg_block_metrics.jsonl",
    "analytics.dfg_function_metrics": "dfg_function_metrics.jsonl",
    "analytics.subsystems": "subsystems.jsonl",
    "analytics.subsystem_modules": "subsystem_modules.jsonl",
    # Docs views
    "docs.v_validation_summary": "validation_summary.jsonl",
}


_DEFAULT_PARQUET_FILENAMES: Final[dict[str, str]] = {
    # GOIDs / crosswalk
    "core.goids": "goids.parquet",
    "core.goid_crosswalk": "goid_crosswalk.parquet",
    # Call graph
    "graph.call_graph_nodes": "call_graph_nodes.parquet",
    "graph.call_graph_edges": "call_graph_edges.parquet",
    # CFG / DFG
    "graph.cfg_blocks": "cfg_blocks.parquet",
    "graph.cfg_edges": "cfg_edges.parquet",
    "graph.dfg_edges": "dfg_edges.parquet",
    # Import / symbol uses
    "graph.import_graph_edges": "import_graph_edges.parquet",
    "graph.symbol_use_edges": "symbol_use_edges.parquet",
    # AST / CST
    "core.ast_nodes": "ast_nodes.parquet",
    "core.ast_metrics": "ast_metrics.parquet",
    "core.cst_nodes": "cst_nodes.parquet",
    "core.docstrings": "docstrings.parquet",
    # Modules / config / diagnostics
    "core.modules": "modules.parquet",
    "analytics.config_values": "config_values.parquet",
    "analytics.data_models": "data_models.parquet",
    "analytics.data_model_fields": "data_model_fields.parquet",
    "analytics.data_model_relationships": "data_model_relationships.parquet",
    "analytics.data_model_usage": "data_model_usage.parquet",
    "analytics.config_data_flow": "config_data_flow.parquet",
    "analytics.static_diagnostics": "static_diagnostics.parquet",
    # AST analytics / typing
    "analytics.hotspots": "hotspots.parquet",
    "analytics.typedness": "typedness.parquet",
    # Function analytics
    "analytics.function_metrics": "function_metrics.parquet",
    "analytics.function_types": "function_types.parquet",
    "analytics.function_effects": "function_effects.parquet",
    "analytics.function_contracts": "function_contracts.parquet",
    "analytics.function_ast_features": "function_ast_features.parquet",
    "analytics.semantic_roles_functions": "semantic_roles_functions.parquet",
    "analytics.semantic_roles_modules": "semantic_roles_modules.parquet",
    # Coverage + tests
    "analytics.coverage_lines": "coverage_lines.parquet",
    "analytics.coverage_functions": "coverage_functions.parquet",
    "analytics.test_catalog": "test_catalog.parquet",
    "analytics.test_coverage_edges": "test_coverage_edges.parquet",
    "analytics.entrypoints": "entrypoints.parquet",
    "analytics.entrypoint_tests": "entrypoint_tests.parquet",
    "analytics.external_dependencies": "external_dependencies.parquet",
    "analytics.external_dependency_calls": "external_dependency_calls.parquet",
    "analytics.graph_validation": "graph_validation.parquet",
    "analytics.function_validation": "function_validation.parquet",
    # Risk factors
    "analytics.goid_risk_factors": "goid_risk_factors.parquet",
    "analytics.function_profile": "function_profile.parquet",
    "analytics.function_history": "function_history.parquet",
    "analytics.history_timeseries": "history_timeseries.parquet",
    "analytics.file_profile": "file_profile.parquet",
    "analytics.module_profile": "module_profile.parquet",
    "analytics.graph_metrics_functions": "graph_metrics_functions.parquet",
    "analytics.graph_metrics_functions_ext": "graph_metrics_functions_ext.parquet",
    "analytics.graph_metrics_modules": "graph_metrics_modules.parquet",
    "analytics.graph_metrics_modules_ext": "graph_metrics_modules_ext.parquet",
    "analytics.subsystem_graph_metrics": "subsystem_graph_metrics.parquet",
    "analytics.symbol_graph_metrics_modules": "symbol_graph_metrics_modules.parquet",
    "analytics.symbol_graph_metrics_functions": "symbol_graph_metrics_functions.parquet",
    "analytics.config_graph_metrics_keys": "config_graph_metrics_keys.parquet",
    "analytics.config_graph_metrics_modules": "config_graph_metrics_modules.parquet",
    "analytics.config_projection_key_edges": "config_projection_key_edges.parquet",
    "analytics.config_projection_module_edges": "config_projection_module_edges.parquet",
    "analytics.subsystem_agreement": "subsystem_agreement.parquet",
    "analytics.graph_stats": "graph_stats.parquet",
    "analytics.test_graph_metrics_tests": "test_graph_metrics_tests.parquet",
    "analytics.test_graph_metrics_functions": "test_graph_metrics_functions.parquet",
    "analytics.test_profile": "test_profile.parquet",
    "analytics.behavioral_coverage": "behavioral_coverage.parquet",
    "analytics.cfg_block_metrics": "cfg_block_metrics.parquet",
    "analytics.cfg_function_metrics": "cfg_function_metrics.parquet",
    "analytics.dfg_block_metrics": "dfg_block_metrics.parquet",
    "analytics.dfg_function_metrics": "dfg_function_metrics.parquet",
    "analytics.subsystems": "subsystems.parquet",
    "analytics.subsystem_modules": "subsystem_modules.parquet",
    # Docs views
    "docs.v_validation_summary": "validation_summary.parquet",
}


def _owner_package_for_prefix(
    prefix: str,
) -> Literal["core", "analytics", "graphs", "qa", "docs"] | None:
    if prefix == "core":
        return "core"
    if prefix == "analytics":
        return "analytics"
    if prefix in {"graph", "cfg"}:
        return "graphs"
    if prefix == "docs":
        return "docs"
    if prefix == "qa":
        return "qa"
    return None


def _build_contracts() -> dict[str, DatasetContract]:
    contracts: dict[str, DatasetContract] = {}

    for table_key, schema in TABLE_SCHEMAS.items():
        if table_key.startswith("tmp_"):
            continue
        schema_prefix, name = table_key.split(".", maxsplit=1)
        meta = _metadata_for_name(name)
        row_binding = ROW_BINDINGS_BY_TABLE_KEY.get(table_key)
        json_schema_id = _JSON_SCHEMA_BY_DATASET_NAME.get(name)
        jsonl_filename = _DEFAULT_JSONL_FILENAMES.get(table_key)
        parquet_filename = _DEFAULT_PARQUET_FILENAMES.get(table_key)
        owner_package = _owner_package_for_prefix(schema_prefix)
        family = schema_prefix

        tags = {"base_table"}
        if table_key in _DATASET_ROWS_ONLY:
            tags.add("dataset_rows_only")
        contracts[name] = DatasetContract(
            name=name,
            table_key=table_key,
            schema=schema,
            row_binding=row_binding,
            json_schema_id=json_schema_id,
            jsonl_filename=jsonl_filename,
            parquet_filename=parquet_filename,
            is_view=False,
            owner_package=owner_package,
            tags=frozenset(tags),
            description=cast("str | None", meta["description"]),
            family=family,
            owner=cast("str | None", meta["owner"]),
            freshness_sla=cast("str | None", meta["freshness_sla"]),
            retention_policy=cast("str | None", meta["retention_policy"]),
            stable_id=cast("str | None", meta["stable_id"]),
            schema_version=cast("str | None", meta["schema_version"]),
            upstream_dependencies=cast("tuple[str, ...]", meta["upstream_dependencies"]),
            validation_profile=cast("Literal['strict', 'lenient']", meta["validation_profile"]),
        )

    for view_key in DERIVED_DOCS_VIEWS:
        schema_prefix, view_name = view_key.split(".", maxsplit=1)
        meta = _metadata_for_name(view_name)
        row_binding = ROW_BINDINGS_BY_TABLE_KEY.get(view_key)
        json_schema_id = _JSON_SCHEMA_BY_DATASET_NAME.get(view_name)
        jsonl_filename = _DEFAULT_JSONL_FILENAMES.get(view_key)
        parquet_filename = _DEFAULT_PARQUET_FILENAMES.get(view_key)
        owner_package = _owner_package_for_prefix(schema_prefix)
        family = schema_prefix

        view_schema = TABLE_SCHEMAS.get(view_key)
        contracts[view_name] = DatasetContract(
            name=view_name,
            table_key=view_key,
            schema=view_schema,
            row_binding=row_binding,
            json_schema_id=json_schema_id,
            jsonl_filename=jsonl_filename,
            parquet_filename=parquet_filename,
            is_view=True,
            owner_package=owner_package,
            tags=frozenset({"docs_view", "read_only"}),
            description=cast("str | None", meta["description"]),
            family=family,
            owner=cast("str | None", meta["owner"]),
            freshness_sla=cast("str | None", meta["freshness_sla"]),
            retention_policy=cast("str | None", meta["retention_policy"]),
            stable_id=cast("str | None", meta["stable_id"]),
            schema_version=cast("str | None", meta["schema_version"]),
            upstream_dependencies=cast("tuple[str, ...]", meta["upstream_dependencies"]),
            validation_profile=cast("Literal['strict', 'lenient']", meta["validation_profile"]),
        )

    return contracts


DATASET_CONTRACTS: Final[dict[str, DatasetContract]] = _build_contracts()
DATASET_CONTRACTS_BY_TABLE_KEY: Final[dict[str, DatasetContract]] = {
    contract.table_key: contract for contract in DATASET_CONTRACTS.values()
}

JSON_SCHEMA_BY_DATASET_NAME: Final[dict[str, str]] = {
    name: contract.json_schema_id
    for name, contract in DATASET_CONTRACTS.items()
    if contract.json_schema_id is not None
}

DEFAULT_JSONL_FILENAMES: Final[dict[str, str]] = {
    contract.table_key: contract.jsonl_filename
    for contract in DATASET_CONTRACTS.values()
    if contract.jsonl_filename is not None
}

DEFAULT_PARQUET_FILENAMES: Final[dict[str, str]] = {
    contract.table_key: contract.parquet_filename
    for contract in DATASET_CONTRACTS.values()
    if contract.parquet_filename is not None
}

DEPENDENCIES_BY_DATASET_NAME: Final[dict[str, tuple[str, ...]]] = {
    name: contract.upstream_dependencies
    for name, contract in DATASET_CONTRACTS.items()
    if contract.upstream_dependencies
}

DESCRIPTION_BY_DATASET_NAME: Final[dict[str, str]] = {
    name: contract.description
    for name, contract in DATASET_CONTRACTS.items()
    if contract.description is not None
}

OWNER_BY_DATASET_NAME: Final[dict[str, str]] = {
    name: contract.owner for name, contract in DATASET_CONTRACTS.items() if contract.owner
}

FRESHNESS_BY_DATASET_NAME: Final[dict[str, str]] = {
    name: contract.freshness_sla
    for name, contract in DATASET_CONTRACTS.items()
    if contract.freshness_sla is not None
}

RETENTION_BY_DATASET_NAME: Final[dict[str, str]] = {
    name: contract.retention_policy
    for name, contract in DATASET_CONTRACTS.items()
    if contract.retention_policy is not None
}

STABLE_ID_BY_DATASET_NAME: Final[dict[str, str]] = {
    name: contract.stable_id for name, contract in DATASET_CONTRACTS.items() if contract.stable_id
}

SCHEMA_VERSION_BY_DATASET_NAME: Final[dict[str, str]] = {
    name: contract.schema_version
    for name, contract in DATASET_CONTRACTS.items()
    if contract.schema_version is not None
}

VALIDATION_PROFILE_BY_DATASET_NAME: Final[dict[str, Literal["strict", "lenient"]]] = {
    name: contract.validation_profile
    for name, contract in DATASET_CONTRACTS.items()
    if contract.validation_profile is not None
}
