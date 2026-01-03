"""Canonical table and composite schema registry.

This module contains:
- TABLE_SCHEMAS: Table/view schema definitions.
- COMPOSITE_SCHEMAS: Profile composition schema definitions.

The registry is the canonical source of explicit table schemas used by
metadata tables, caches, docs views, and non-inferable output overrides.
Inferable outputs are persisted in metadata and rendered on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from codeintel.config.datasets.primitives import (
    CREATED_AT_COL_NULLABLE,
    SUBSYSTEM_ENTITY_COLS,
    Column,
    Index,
    TableSchema,
)
from codeintel.core.schemas.output_registry import OUTPUT_TABLE_SCHEMAS
from codeintel.core.schemas.view_registry import build_view_schema_overrides

if TYPE_CHECKING:
    from codeintel.config.datasets.primitives import CompositeSchema

TABLE_SCHEMAS: dict[str, TableSchema] = {
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
            Column("datasets", "LIST(VARCHAR)"),
            Column("modules_total", "BIGINT"),
            Column("modules_changed", "BIGINT"),
            Column("modules_deleted", "BIGINT"),
            Column("modules_changed_ratio", "DOUBLE"),
            Column("modules_deleted_ratio", "DOUBLE"),
            Column("use_full_rebuild", "BOOLEAN"),
        ],
        description="Per-step ingest run telemetry for control plane reporting.",
    ),
    "analytics.tags_index": TableSchema(
        schema="analytics",
        name="tags_index",
        columns=[
            Column("tag", "VARCHAR", nullable=False),
            Column("description", "VARCHAR"),
            Column("includes", "BLOB"),
            Column("excludes", "BLOB"),
            Column("matches", "BLOB"),
        ],
        primary_key=("tag",),
        description="Path classification rules",
    ),
    "analytics.subsystem_profile_cache": TableSchema(
        schema="analytics",
        name="subsystem_profile_cache",
        columns=[
            *SUBSYSTEM_ENTITY_COLS,
            Column("name", "VARCHAR"),
            Column("description", "VARCHAR"),
            Column("module_count", "INTEGER"),
            Column("modules_json", "BLOB"),
            Column("entrypoints_json", "BLOB"),
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
    "docs.v_validation_summary": TableSchema(
        schema="docs",
        name="v_validation_summary",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("validation_type", "VARCHAR", nullable=False),
            Column("issue_count", "INTEGER"),
            Column("affected_files", "INTEGER"),
            Column("affected_functions", "INTEGER"),
        ],
    ),
    "build.output_manifests": TableSchema(
        schema="build",
        name="output_manifests",
        columns=[
            Column(
                "target",
                "VARCHAR",
                nullable=False,
                description="Target name (e.g., 'function_types')",
            ),
            Column("repo", "VARCHAR", nullable=False, description="Repository slug"),
            Column("commit", "VARCHAR", nullable=False, description="Commit SHA"),
            Column(
                "impl_kind",
                "VARCHAR",
                nullable=False,
                description="Implementation kind that produced this target",
            ),
            Column(
                "computed_at",
                "TIMESTAMPTZ",
                nullable=False,
                description="When the target was computed",
            ),
            Column("duration_ms", "DOUBLE", description="Computation duration in milliseconds"),
            Column(
                "input_hash",
                "VARCHAR",
                nullable=False,
                description="Hash of all inputs (deps + options)",
            ),
            Column("output_hash", "VARCHAR", description="Hash of output data for integrity"),
            Column("row_count", "INTEGER", description="Number of rows written"),
            Column(
                "options_hash",
                "VARCHAR",
                description="Hash of implementation configuration options",
            ),
            Column(
                "change_delta",
                "BLOB",
                description="JSON change delta from ingestion change detection",
            ),
        ],
        primary_key=("target", "repo", "commit"),
        indexes=(
            Index("idx_build_output_manifests_repo_commit", ("repo", "commit")),
            Index("idx_build_output_manifests_computed_at", ("computed_at",)),
        ),
        description="Manifest of computed build targets for incremental computation",
    ),
    "build.runs": TableSchema(
        schema="build",
        name="runs",
        columns=[
            Column("run_id", "VARCHAR", nullable=False, description="Unique run identifier"),
            Column("repo", "VARCHAR", nullable=False, description="Repository slug"),
            Column("commit", "VARCHAR", nullable=False, description="Commit SHA"),
            Column(
                "requested_targets",
                "BLOB",
                nullable=False,
                description="JSON array of targets requested",
            ),
            Column(
                "computed_targets",
                "BLOB",
                nullable=False,
                description="JSON array of targets computed",
            ),
            Column(
                "skipped_targets",
                "BLOB",
                nullable=False,
                description="JSON array of targets skipped",
            ),
            Column("started_at", "TIMESTAMPTZ", nullable=False, description="Run start time"),
            Column("completed_at", "TIMESTAMPTZ", description="Run completion time"),
            Column(
                "status",
                "VARCHAR",
                nullable=False,
                description="Run status: running/succeeded/failed",
            ),
            Column("error_summary", "VARCHAR", description="Error summary if failed"),
            Column("duration_ms", "DOUBLE", description="Total run duration in milliseconds"),
        ],
        primary_key=("run_id",),
        indexes=(
            Index("idx_build_runs_repo_commit", ("repo", "commit")),
            Index("idx_build_runs_started_at", ("started_at",)),
        ),
        description="Build system run tracking for debugging and observability",
    ),
    "build.run_targets": TableSchema(
        schema="build",
        name="run_targets",
        columns=[
            Column("run_id", "VARCHAR", nullable=False, description="Parent run identifier"),
            Column("repo", "VARCHAR", nullable=False, description="Repository slug"),
            Column("commit", "VARCHAR", nullable=False, description="Commit SHA"),
            Column("target", "VARCHAR", nullable=False, description="Target name"),
            Column("impl_kind", "VARCHAR", nullable=False, description="Implementation kind"),
            Column(
                "status",
                "VARCHAR",
                nullable=False,
                description="Target status: succeeded/failed/skipped",
            ),
            Column("input_hash", "VARCHAR", description="Input hash for cache validation"),
            Column("options_hash", "VARCHAR", description="Target options hash"),
            Column(
                "duration_ms",
                "DOUBLE",
                nullable=False,
                description="Target execution duration in milliseconds",
            ),
            Column(
                "row_counts",
                "BLOB",
                nullable=False,
                description="JSON object mapping table keys to row counts",
            ),
            Column(
                "drift_summaries",
                "BLOB",
                nullable=False,
                description="JSON object mapping table keys to drift summaries",
            ),
            Column("error", "VARCHAR", description="Error message if failed"),
            Column(
                "dep_hashes",
                "BLOB",
                description="JSON mapping of dependency names to their input hashes",
            ),
            Column(
                "recorded_at",
                "TIMESTAMPTZ",
                nullable=False,
                description="When the record was persisted",
            ),
        ],
        primary_key=("run_id", "target"),
        indexes=(Index("idx_build_run_targets_repo_commit", ("repo", "commit")),),
        description="Per-target execution records for build observability",
    ),
    "build.run_environments": TableSchema(
        schema="build",
        name="run_environments",
        columns=[
            Column("run_id", "VARCHAR", nullable=False, description="Parent run identifier"),
            Column(
                "python_version", "VARCHAR", nullable=False, description="Python version string"
            ),
            Column("os_name", "VARCHAR", nullable=False, description="Operating system name"),
            Column("os_version", "VARCHAR", nullable=False, description="Operating system release"),
            Column(
                "tool_versions",
                "BLOB",
                nullable=False,
                description="JSON mapping of tool names to versions",
            ),
            Column("config_hash", "VARCHAR", description="Hash of build configuration"),
            Column(
                "git_dirty",
                "BOOLEAN",
                nullable=False,
                description="Whether git working tree had uncommitted changes",
            ),
            Column("captured_at", "TIMESTAMPTZ", nullable=False, description="When captured"),
        ],
        primary_key=("run_id",),
        indexes=(Index("idx_build_run_environments_captured_at", ("captured_at",)),),
        description="Captured tool versions and runtime environment per build run",
    ),
    "build.asset_versions": TableSchema(
        schema="build",
        name="asset_versions",
        columns=[
            Column("asset_kind", "VARCHAR", nullable=False, description="table|view|artifact"),
            Column("asset_key", "VARCHAR", nullable=False, description="Logical asset identifier"),
            Column(
                "version_hash",
                "VARCHAR",
                nullable=False,
                description="Content-addressed version hash",
            ),
            Column("schema_hash", "VARCHAR", description="Schema fingerprint (datasets)"),
            Column("row_count", "BIGINT", description="Row count (datasets)"),
            Column("bytes", "BIGINT", description="Size in bytes (artifacts)"),
            Column(
                "created_at",
                "TIMESTAMPTZ",
                nullable=False,
                description="When this version was first recorded",
            ),
            Column("meta", "BLOB", description="Extra metadata for versioning/fingerprinting"),
        ],
        primary_key=("asset_kind", "asset_key", "version_hash"),
        indexes=(
            Index("idx_build_asset_versions_asset", ("asset_kind", "asset_key")),
            Index("idx_build_asset_versions_created_at", ("created_at",)),
        ),
        description="Content-addressed versions of assets recorded by builds",
    ),
    "build.asset_version_events": TableSchema(
        schema="build",
        name="asset_version_events",
        columns=[
            Column("run_id", "VARCHAR", nullable=False, description="Run identifier"),
            Column("repo", "VARCHAR", nullable=False, description="Repository slug"),
            Column("commit", "VARCHAR", nullable=False, description="Commit SHA"),
            Column("asset_kind", "VARCHAR", nullable=False, description="table|view|artifact"),
            Column("asset_key", "VARCHAR", nullable=False, description="Logical asset identifier"),
            Column(
                "version_hash", "VARCHAR", nullable=False, description="Resolved asset version hash"
            ),
            Column("target", "VARCHAR", description="Target that produced the asset"),
            Column("impl_kind", "VARCHAR", description="Implementation kind"),
            Column("status", "VARCHAR", nullable=False, description="materialized|reused|failed"),
            Column("location", "VARCHAR", description="Table name/view name/path/URI"),
            Column("input_hash", "VARCHAR", description="Target input hash at event time"),
            Column("options_hash", "VARCHAR", description="Target options hash at event time"),
            Column(
                "recorded_at",
                "TIMESTAMPTZ",
                nullable=False,
                description="When the event was recorded",
            ),
            Column("meta", "BLOB", description="Extra metadata for version events"),
        ],
        primary_key=("run_id", "asset_kind", "asset_key"),
        indexes=(
            Index("idx_build_asset_version_events_repo_commit", ("repo", "commit")),
            Index("idx_build_asset_version_events_asset", ("asset_kind", "asset_key")),
            Index("idx_build_asset_version_events_run_id", ("run_id",)),
        ),
        description="Run-scoped event records for asset version usage and creation",
    ),
    "build.run_asset_versions": TableSchema(
        schema="build",
        name="run_asset_versions",
        columns=[
            Column("run_id", "VARCHAR", nullable=False, description="Run identifier"),
            Column("repo", "VARCHAR", nullable=False, description="Repository slug"),
            Column("commit", "VARCHAR", nullable=False, description="Commit SHA"),
            Column("asset_kind", "VARCHAR", nullable=False, description="table|view|artifact"),
            Column("asset_key", "VARCHAR", nullable=False, description="Logical asset identifier"),
            Column(
                "version_hash", "VARCHAR", nullable=False, description="Resolved asset version hash"
            ),
            Column("target", "VARCHAR", description="Target that produced or reused the asset"),
            Column("resolution_kind", "VARCHAR", nullable=False, description="materialized|reused"),
            Column(
                "recorded_at",
                "TIMESTAMPTZ",
                nullable=False,
                description="When the mapping was recorded",
            ),
            Column("meta", "BLOB", description="Extra metadata for run-to-asset resolution"),
        ],
        primary_key=("run_id", "asset_kind", "asset_key"),
        indexes=(
            Index("idx_build_run_asset_versions_repo_commit", ("repo", "commit")),
            Index("idx_build_run_asset_versions_asset", ("asset_kind", "asset_key")),
            Index("idx_build_run_asset_versions_run_id", ("run_id",)),
        ),
        description="Mapping of build runs to the exact asset versions used/produced",
    ),
    "build.asset_lineage": TableSchema(
        schema="build",
        name="asset_lineage",
        columns=[
            Column("downstream_kind", "VARCHAR", nullable=False),
            Column("downstream_key", "VARCHAR", nullable=False),
            Column("downstream_version", "VARCHAR", nullable=False),
            Column("upstream_kind", "VARCHAR", nullable=False),
            Column("upstream_key", "VARCHAR", nullable=False),
            Column("upstream_version", "VARCHAR", nullable=False),
            Column(
                "edge_kind", "VARCHAR", nullable=False, description="depends_on|reads_from|reuses"
            ),
            Column("created_at", "TIMESTAMPTZ", nullable=False),
            Column("meta", "BLOB"),
        ],
        primary_key=(
            "downstream_kind",
            "downstream_key",
            "downstream_version",
            "upstream_kind",
            "upstream_key",
            "upstream_version",
            "edge_kind",
        ),
        indexes=(
            Index(
                "idx_build_asset_lineage_downstream",
                ("downstream_kind", "downstream_key", "downstream_version"),
            ),
            Index(
                "idx_build_asset_lineage_upstream",
                ("upstream_kind", "upstream_key", "upstream_version"),
            ),
        ),
        description="Lineage edges between specific asset versions",
    ),
    "build.asset_aliases": TableSchema(
        schema="build",
        name="asset_aliases",
        columns=[
            Column("alias", "VARCHAR", nullable=False),
            Column("asset_kind", "VARCHAR", nullable=False),
            Column("asset_key", "VARCHAR", nullable=False),
            Column("version_hash", "VARCHAR", nullable=False),
            Column("set_by_run_id", "VARCHAR"),
            Column("set_at", "TIMESTAMPTZ", nullable=False),
            Column("note", "VARCHAR"),
        ],
        primary_key=("alias", "asset_kind", "asset_key"),
        indexes=(
            Index("idx_build_asset_aliases_asset", ("asset_kind", "asset_key")),
            Index("idx_build_asset_aliases_alias", ("alias",)),
        ),
        description="Human-friendly aliases for asset versions (promotion/labels)",
    ),
    "build.asset_diffs": TableSchema(
        schema="build",
        name="asset_diffs",
        columns=[
            Column("asset_kind", "VARCHAR", nullable=False),
            Column("asset_key", "VARCHAR", nullable=False),
            Column("from_version_hash", "VARCHAR", nullable=False),
            Column("to_version_hash", "VARCHAR", nullable=False),
            Column("diff_kind", "VARCHAR", nullable=False, description="schema|rowcount|profile"),
            Column("summary", "BLOB", description="Diff summary JSON"),
            Column("computed_at", "TIMESTAMPTZ", nullable=False),
            Column("computed_by_run_id", "VARCHAR"),
        ],
        primary_key=(
            "asset_kind",
            "asset_key",
            "from_version_hash",
            "to_version_hash",
            "diff_kind",
        ),
        indexes=(Index("idx_build_asset_diffs_asset", ("asset_kind", "asset_key")),),
        description="Cached diffs between asset versions",
    ),
    "build.run_nodes": TableSchema(
        schema="build",
        name="run_nodes",
        columns=[
            Column("run_id", "VARCHAR", nullable=False, description="Parent run identifier"),
            Column("node_name", "VARCHAR", nullable=False, description="Hamilton node name"),
            Column("target", "VARCHAR", description="Parent target if applicable"),
            Column(
                "node_type", "VARCHAR", description="Node type: compute, materialize, tool, etc."
            ),
            Column("status", "VARCHAR", nullable=False, description="succeeded, failed, skipped"),
            Column("started_at", "TIMESTAMPTZ", nullable=False),
            Column("completed_at", "TIMESTAMPTZ"),
            Column("duration_ms", "DOUBLE"),
            Column("error", "VARCHAR"),
            Column("tags", "BLOB", description="Hamilton tags from node"),
        ],
        primary_key=("run_id", "node_name"),
        indexes=(
            Index("idx_build_run_nodes_run_id", ("run_id",)),
            Index("idx_build_run_nodes_target", ("target",)),
            Index("idx_build_run_nodes_status", ("status",)),
        ),
        description="Node-level execution telemetry for fine-grained profiling",
    ),
    "build.scip_runs": TableSchema(
        schema="build",
        name="scip_runs",
        columns=[
            Column("run_id", "VARCHAR", nullable=False),
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("mode", "VARCHAR", nullable=False),
            Column("options_hash", "VARCHAR"),
            Column("tool_version", "VARCHAR"),
            Column("total_modules", "INTEGER"),
            Column("changed_modules", "INTEGER"),
            Column("deleted_modules", "INTEGER"),
            Column("changed_ratio", "DOUBLE"),
            Column("batch_size", "INTEGER"),
            Column("batch_count", "INTEGER"),
            Column("decision", "VARCHAR"),
            Column("ratio_gate_applied", "BOOLEAN"),
            Column("ratio_gate_min_modules", "INTEGER"),
            Column("ratio_gate_min_changed", "INTEGER"),
            Column("hash_source", "VARCHAR"),
            Column("hash_source_breakdown", "VARCHAR"),
            Column("hash_reused", "INTEGER"),
            Column("hash_computed", "INTEGER"),
            Column("plan_ms", "DOUBLE"),
            Column("hash_ms", "DOUBLE"),
            Column("tool_ms", "DOUBLE"),
            Column("parse_ms", "DOUBLE"),
            Column("merge_ms", "DOUBLE"),
            Column("write_ms", "DOUBLE"),
            Column("total_ms", "DOUBLE"),
            Column("status", "VARCHAR"),
            Column("error_summary", "VARCHAR"),
            Column("output_scip", "VARCHAR"),
            Column("recorded_at", "TIMESTAMPTZ", nullable=False),
        ],
        primary_key=("run_id",),
        indexes=(
            Index("idx_build_scip_runs_repo_commit", ("repo", "commit")),
            Index("idx_build_scip_runs_status", ("status",)),
            Index("idx_build_scip_runs_recorded_at", ("recorded_at",)),
        ),
        description="SCIP indexing telemetry and performance metadata",
    ),
}

TABLE_SCHEMAS.update(OUTPUT_TABLE_SCHEMAS)
_VIEW_SCHEMA_OVERRIDES = build_view_schema_overrides(TABLE_SCHEMAS)
TABLE_SCHEMAS.update(_VIEW_SCHEMA_OVERRIDES)

COMPOSITE_SCHEMAS: Final[dict[str, CompositeSchema]] = {}


def get_table_schema(table_key: str) -> TableSchema | None:
    """Get a table schema by key.

    Looks up schemas in TABLE_SCHEMAS and (in future) derived schemas
    from build targets.

    Parameters
    ----------
    table_key
        Fully-qualified table name (e.g., "core.ast_nodes").

    Returns
    -------
    TableSchema | None
        Schema if found, None otherwise.
    """
    return TABLE_SCHEMAS.get(table_key)


def merge_with_derived_schemas(derived: dict[str, TableSchema]) -> dict[str, TableSchema]:
    """Merge derived schemas with static TABLE_SCHEMAS.

    Derived schemas take precedence over static schemas.

    Parameters
    ----------
    derived
        Schemas derived from build target contracts.

    Returns
    -------
    dict[str, TableSchema]
        Merged schemas.
    """
    result = dict(TABLE_SCHEMAS)
    result.update(derived)
    return result


__all__ = [
    "COMPOSITE_SCHEMAS",
    "TABLE_SCHEMAS",
    "get_table_schema",
    "merge_with_derived_schemas",
]
