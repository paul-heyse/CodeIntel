"""Metadata schema contracts for the `metadata.*` DuckDB schema.

These table contracts are used for DDL creation and alignment in the storage
layer. They intentionally use the same `TableSchema` language as dataset
contracts so that DDL generation is consistent and testable.
"""

from __future__ import annotations

from codeintel.core.schemas.primitives import Column, Index, TableSchema

__all__ = [
    "BUILD_OUTPUT_CATALOG_TABLE",
    "BUILD_RUN_INDEX_TABLE",
    "BUILD_RUN_METADATA_TABLE",
    "BUILD_RUN_TAG_SUMMARY_TABLE",
    "CANONICAL_CATALOGS_TABLE",
    "EXPORT_AUDIT_TABLE",
    "METADATA_TABLES",
    "SCHEMA_MANIFEST_RUNS_TABLE",
    "SCHEMA_OBSERVATIONS_TABLE",
    "SCHEMA_VALIDATION_RUNS_TABLE",
    "SCHEMA_VERSIONS_TABLE",
    "TABLE_SCHEMA_OVERRIDE_REGISTRY_TABLE",
    "TABLE_SCHEMA_OVERRIDE_VERSIONS_TABLE",
    "TABLE_SCHEMA_REGISTRY_TABLE",
]

EXPORT_AUDIT_TABLE = TableSchema(
    schema="metadata",
    name="export_audit",
    columns=[
        Column("dataset", "VARCHAR", nullable=False),
        Column("macro", "VARCHAR", nullable=False),
        Column("rows", "BIGINT"),
        Column("duration_s", "DOUBLE", nullable=False),
        Column("output_path", "VARCHAR", nullable=False),
        Column("sql", "VARCHAR"),
        Column("plan", "VARCHAR"),
        Column("created_at", "TIMESTAMPTZ", nullable=False),
    ],
)

BUILD_RUN_INDEX_TABLE = TableSchema(
    schema="metadata",
    name="build_run_index",
    columns=[
        Column("run_id", "VARCHAR", nullable=False),
        Column("repo", "VARCHAR"),
        Column("commit", "VARCHAR"),
        Column("started_at", "TIMESTAMPTZ"),
        Column("duration_ms", "DOUBLE"),
        Column("success", "BOOLEAN"),
        Column("report_path", "VARCHAR"),
        Column("computed_targets_count", "BIGINT"),
        Column("skipped_targets_count", "BIGINT"),
        Column("failed_targets_count", "BIGINT"),
    ],
    primary_key=("run_id",),
    indexes=(Index("idx_build_run_index_repo_commit", ("repo", "commit", "started_at")),),
)

BUILD_RUN_METADATA_TABLE = TableSchema(
    schema="metadata",
    name="build_run_metadata",
    columns=[
        Column("run_id", "VARCHAR", nullable=False),
        Column("repo", "VARCHAR"),
        Column("commit", "VARCHAR"),
        Column("snapshot_id", "VARCHAR"),
        Column("started_at", "TIMESTAMPTZ"),
        Column("duration_ms", "DOUBLE"),
        Column("success", "BOOLEAN"),
        Column("computed_targets", "JSON"),
        Column("skipped_targets", "JSON"),
        Column("failed_targets", "JSON"),
        Column("error_summary", "VARCHAR"),
    ],
    primary_key=("run_id",),
    indexes=(Index("idx_build_run_metadata_repo_commit", ("repo", "commit", "started_at")),),
)

BUILD_RUN_TAG_SUMMARY_TABLE = TableSchema(
    schema="metadata",
    name="build_run_tag_summary",
    columns=[
        Column("run_id", "VARCHAR", nullable=False),
        Column("repo", "VARCHAR"),
        Column("commit", "VARCHAR"),
        Column("snapshot_id", "VARCHAR"),
        Column("summary", "JSON", nullable=False),
    ],
    primary_key=("run_id",),
    indexes=(Index("idx_build_run_tag_summary_repo_commit", ("repo", "commit")),),
)

BUILD_OUTPUT_CATALOG_TABLE = TableSchema(
    schema="metadata",
    name="build_output_catalog",
    columns=[
        Column("run_id", "VARCHAR", nullable=False),
        Column("output_kind", "VARCHAR", nullable=False),
        Column("output_key", "VARCHAR", nullable=False),
        Column("table_key", "VARCHAR"),
        Column("artifact_name", "VARCHAR"),
        Column("artifact_type", "VARCHAR"),
        Column("artifact_path", "VARCHAR"),
        Column("target", "VARCHAR", nullable=False),
        Column("status", "VARCHAR", nullable=False),
        Column("row_count", "BIGINT"),
        Column("manifest_row_count", "BIGINT"),
        Column("schema_hash", "VARCHAR"),
        Column("dataset_manifest_path", "VARCHAR"),
        Column("output_role", "VARCHAR"),
        Column("saver_node", "VARCHAR"),
        Column("sink", "VARCHAR"),
        Column("tags", "JSON"),
        Column("repo", "VARCHAR"),
        Column("commit", "VARCHAR"),
        Column("snapshot_id", "VARCHAR"),
    ],
    primary_key=("run_id", "output_kind", "output_key", "target"),
    indexes=(
        Index("idx_build_output_catalog_target", ("target",)),
        Index("idx_build_output_catalog_repo_commit", ("repo", "commit")),
    ),
)

CANONICAL_CATALOGS_TABLE = TableSchema(
    schema="metadata",
    name="canonical_catalogs",
    columns=[
        Column("catalog_kind", "VARCHAR", nullable=False),
        Column("catalog_hash", "VARCHAR", nullable=False),
        Column("payload", "JSON", nullable=False),
        Column("inputs", "JSON"),
        Column("created_at", "TIMESTAMPTZ", nullable=False),
    ],
    primary_key=("catalog_kind", "catalog_hash"),
)

SCHEMA_VERSIONS_TABLE = TableSchema(
    schema="metadata",
    name="schema_versions",
    columns=[
        Column("schema_digest", "VARCHAR", nullable=False),
        Column("schema_hash", "VARCHAR", nullable=False),
        Column("schema_json", "JSON", nullable=False),
        Column("renderer_cache", "JSON"),
        Column("created_at", "TIMESTAMPTZ", nullable=False),
    ],
    primary_key=("schema_digest",),
    indexes=(Index("idx_schema_versions_schema_hash", ("schema_hash",)),),
)

TABLE_SCHEMA_REGISTRY_TABLE = TableSchema(
    schema="metadata",
    name="table_schema_registry",
    columns=[
        Column("table_key", "VARCHAR", nullable=False),
        Column("schema_digest", "VARCHAR", nullable=False),
        Column("schema_hash", "VARCHAR", nullable=False),
        Column("derivation_kind", "VARCHAR", nullable=False),
        Column("derivation_source", "VARCHAR", nullable=False),
        Column("inference_status", "VARCHAR"),
        Column("inference_error", "VARCHAR"),
        Column("catalog_hash", "VARCHAR"),
        Column("updated_at", "TIMESTAMPTZ", nullable=False),
    ],
    primary_key=("table_key",),
    indexes=(
        Index("idx_table_schema_registry_schema_digest", ("schema_digest",)),
        Index("idx_table_schema_registry_schema_hash", ("schema_hash",)),
        Index("idx_table_schema_registry_derivation_kind", ("derivation_kind",)),
        Index("idx_table_schema_registry_inference_status", ("inference_status",)),
        Index("idx_table_schema_registry_catalog_hash", ("catalog_hash",)),
    ),
)

TABLE_SCHEMA_OVERRIDE_VERSIONS_TABLE = TableSchema(
    schema="metadata",
    name="table_schema_override_versions",
    columns=[
        Column("version_id", "VARCHAR", nullable=False),
        Column("table_key", "VARCHAR", nullable=False),
        Column("schema_digest", "VARCHAR", nullable=False),
        Column("schema_hash", "VARCHAR", nullable=False),
        Column("catalog_hash", "VARCHAR"),
        Column("created_at", "TIMESTAMPTZ", nullable=False),
    ],
    primary_key=("version_id", "table_key"),
    indexes=(
        Index("idx_override_versions_table_key", ("table_key", "created_at")),
        Index("idx_override_versions_schema_digest", ("schema_digest",)),
        Index("idx_override_versions_version_id", ("version_id",)),
    ),
)

TABLE_SCHEMA_OVERRIDE_REGISTRY_TABLE = TableSchema(
    schema="metadata",
    name="table_schema_override_registry",
    columns=[
        Column("table_key", "VARCHAR", nullable=False),
        Column("schema_digest", "VARCHAR", nullable=False),
        Column("schema_hash", "VARCHAR", nullable=False),
        Column("version_id", "VARCHAR", nullable=False),
        Column("updated_at", "TIMESTAMPTZ", nullable=False),
    ],
    primary_key=("table_key",),
    indexes=(
        Index("idx_override_registry_schema_digest", ("schema_digest",)),
        Index("idx_override_registry_version_id", ("version_id",)),
    ),
)

SCHEMA_MANIFEST_RUNS_TABLE = TableSchema(
    schema="metadata",
    name="schema_manifest_runs",
    columns=[
        Column("run_id", "VARCHAR", nullable=False),
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("manifest_kind", "VARCHAR", nullable=False),
        Column("catalog_hash", "VARCHAR", nullable=False),
        Column("created_at", "TIMESTAMPTZ", nullable=False),
    ],
    primary_key=("run_id",),
    indexes=(
        Index("idx_schema_manifest_runs_repo_commit", ("repo", "commit", "created_at")),
        Index("idx_schema_manifest_runs_manifest_kind", ("manifest_kind", "created_at")),
        Index("idx_schema_manifest_runs_catalog_hash", ("catalog_hash",)),
    ),
)

SCHEMA_VALIDATION_RUNS_TABLE = TableSchema(
    schema="metadata",
    name="schema_validation_runs",
    columns=[
        Column("validation_id", "VARCHAR", nullable=False),
        Column("repo", "VARCHAR"),
        Column("commit", "VARCHAR"),
        Column("validation_mode", "VARCHAR", nullable=False),
        Column("include_views", "BOOLEAN", nullable=False),
        Column("issue_count", "BIGINT", nullable=False),
        Column("status", "VARCHAR", nullable=False),
        Column("issues", "JSON"),
        Column("created_at", "TIMESTAMPTZ", nullable=False),
    ],
    primary_key=("validation_id",),
    indexes=(
        Index("idx_schema_validation_runs_repo_commit", ("repo", "commit", "created_at")),
        Index("idx_schema_validation_runs_status", ("status", "created_at")),
    ),
)

SCHEMA_OBSERVATIONS_TABLE = TableSchema(
    schema="metadata",
    name="schema_observations",
    columns=[
        Column("observation_id", "VARCHAR", nullable=False),
        Column("table_key", "VARCHAR", nullable=False),
        Column("repo", "VARCHAR"),
        Column("commit", "VARCHAR"),
        Column("target_name", "VARCHAR"),
        Column("schema_digest", "VARCHAR", nullable=False),
        Column("schema_hash", "VARCHAR", nullable=False),
        Column("arrow_schema_ipc_b64", "VARCHAR", nullable=False),
        Column("column_stats", "JSON"),
        Column("dataset_stats", "JSON"),
        Column("derived_settings", "JSON"),
        Column("drift_summary", "JSON"),
        Column("observed_at", "TIMESTAMPTZ", nullable=False),
    ],
    primary_key=("observation_id",),
    indexes=(
        Index("idx_schema_observations_table_key", ("table_key", "observed_at")),
        Index("idx_schema_observations_schema_digest", ("schema_digest",)),
        Index("idx_schema_observations_schema_hash", ("schema_hash",)),
        Index("idx_schema_observations_repo_commit", ("repo", "commit", "observed_at")),
    ),
)

METADATA_TABLES: tuple[TableSchema, ...] = (
    EXPORT_AUDIT_TABLE,
    BUILD_RUN_INDEX_TABLE,
    BUILD_RUN_METADATA_TABLE,
    BUILD_RUN_TAG_SUMMARY_TABLE,
    BUILD_OUTPUT_CATALOG_TABLE,
    CANONICAL_CATALOGS_TABLE,
    SCHEMA_VERSIONS_TABLE,
    TABLE_SCHEMA_REGISTRY_TABLE,
    TABLE_SCHEMA_OVERRIDE_VERSIONS_TABLE,
    TABLE_SCHEMA_OVERRIDE_REGISTRY_TABLE,
    SCHEMA_OBSERVATIONS_TABLE,
    SCHEMA_MANIFEST_RUNS_TABLE,
    SCHEMA_VALIDATION_RUNS_TABLE,
    TableSchema(
        schema="metadata",
        name="datasets",
        columns=[
            Column("table_key", "VARCHAR", nullable=False),
            Column("name", "VARCHAR", nullable=False),
            Column("is_view", "BOOLEAN", nullable=False),
            Column("jsonl_filename", "VARCHAR"),
            Column("parquet_filename", "VARCHAR"),
            Column("family", "VARCHAR"),
            Column("description", "VARCHAR"),
            Column("schema_version", "VARCHAR"),
        ],
        primary_key=("table_key",),
    ),
    TableSchema(
        schema="metadata",
        name="dataset_dataflow_nodes",
        columns=[
            Column("id", "VARCHAR", nullable=False),
            Column("kind", "VARCHAR", nullable=False),
            Column("family", "VARCHAR"),
            Column("owner_package", "VARCHAR"),
            Column("description", "VARCHAR"),
        ],
        primary_key=("id",),
    ),
    TableSchema(
        schema="metadata",
        name="dataset_dataflow_edges",
        columns=[
            Column("src", "VARCHAR", nullable=False),
            Column("dst", "VARCHAR", nullable=False),
            Column("edge_type", "VARCHAR", nullable=False),
        ],
        primary_key=("src", "dst", "edge_type"),
        indexes=(
            Index("idx_dataset_dataflow_edges_src", ("src",)),
            Index("idx_dataset_dataflow_edges_dst", ("dst",)),
        ),
    ),
    TableSchema(
        schema="metadata",
        name="derived_lineage_edges",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("downstream", "VARCHAR", nullable=False),
            Column("upstream", "VARCHAR", nullable=False),
            Column("edge_type", "VARCHAR", nullable=False),
        ],
        primary_key=("repo", "commit", "downstream", "upstream", "edge_type"),
        indexes=(
            Index("idx_derived_lineage_edges_downstream", ("repo", "commit", "downstream")),
            Index("idx_derived_lineage_edges_upstream", ("repo", "commit", "upstream")),
        ),
    ),
    TableSchema(
        schema="metadata",
        name="derived_lineage_columns",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("downstream_table", "VARCHAR", nullable=False),
            Column("downstream_column", "VARCHAR", nullable=False),
            Column("upstream_table", "VARCHAR", nullable=False),
            Column("upstream_column", "VARCHAR", nullable=False),
            Column("edge_type", "VARCHAR", nullable=False),
        ],
        primary_key=(
            "repo",
            "commit",
            "downstream_table",
            "downstream_column",
            "upstream_table",
            "upstream_column",
            "edge_type",
        ),
        indexes=(
            Index(
                "idx_derived_lineage_columns_downstream",
                ("repo", "commit", "downstream_table", "downstream_column"),
            ),
            Index(
                "idx_derived_lineage_columns_upstream",
                ("repo", "commit", "upstream_table", "upstream_column"),
            ),
        ),
    ),
    TableSchema(
        schema="metadata",
        name="pipeline_runs",
        columns=[
            Column("run_id", "VARCHAR", nullable=False),
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("kind", "VARCHAR", nullable=False),
            Column("trigger", "VARCHAR", nullable=False),
            Column("requested_operation", "VARCHAR"),
            Column("requested_datasets", "JSON"),
            Column("started_at", "TIMESTAMPTZ", nullable=False),
            Column("completed_at", "TIMESTAMPTZ"),
            Column("status", "VARCHAR", nullable=False),
            Column("error_summary", "VARCHAR"),
            Column("pipeline_name", "VARCHAR"),
        ],
        primary_key=("run_id",),
        indexes=(
            Index("idx_pipeline_runs_repo_commit", ("repo", "commit", "started_at")),
            Index("idx_pipeline_runs_status", ("status", "repo", "commit")),
        ),
    ),
    TableSchema(
        schema="metadata",
        name="pipeline_steps",
        columns=[
            Column("run_id", "VARCHAR", nullable=False),
            Column("module", "VARCHAR", nullable=False),
            Column("stage", "VARCHAR", nullable=False),
            Column("name", "VARCHAR", nullable=False),
            Column("started_at", "TIMESTAMPTZ", nullable=False),
            Column("completed_at", "TIMESTAMPTZ"),
            Column("status", "VARCHAR", nullable=False),
            Column("row_counts", "JSON"),
            Column("extra", "JSON"),
        ],
        primary_key=("run_id", "module", "name"),
        indexes=(Index("idx_pipeline_steps_run", ("run_id", "module", "stage")),),
    ),
)
