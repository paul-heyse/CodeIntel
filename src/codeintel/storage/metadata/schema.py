"""Metadata schema contracts for the `metadata.*` DuckDB schema.

These table contracts are used for DDL creation and alignment in the storage
layer. They intentionally use the same `TableSchema` language as dataset
contracts so that DDL generation is consistent and testable.
"""

from __future__ import annotations

from codeintel.core.schemas.primitives import Column, Index, TableSchema

__all__ = [
    "METADATA_TABLES",
]


METADATA_TABLES: tuple[TableSchema, ...] = (
    TableSchema(
        schema="metadata",
        name="dataset_schema_registry",
        columns=[
            Column("table_key", "VARCHAR", nullable=False),
            Column("schema_hash", "VARCHAR", nullable=False),
        ],
        primary_key=("table_key",),
    ),
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
            Column("deprecated", "BOOLEAN"),
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
