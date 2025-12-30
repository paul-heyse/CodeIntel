"""Canonical table and composite schema registry.

This module contains:
- TABLE_SCHEMAS: Table/view schema definitions.
- COMPOSITE_SCHEMAS: Profile composition schema definitions.

The registry is the canonical source of explicit table schemas used by
metadata tables, caches, docs views, and non-inferable output overrides.
Inferable outputs are persisted in metadata and rendered on demand.
"""

from __future__ import annotations

from typing import Final

from codeintel.config.datasets.primitives import (
    CREATED_AT_COL_NULLABLE,
    FUNCTION_ENTITY_COLS,
    MODULE_ENTITY_COLS,
    REPO_COMMIT_COLS,
    SUBSYSTEM_ENTITY_COLS,
    Column,
    CompositeSchema,
    Index,
    TableSchema,
)
from codeintel.core.schemas.output_registry import OUTPUT_TABLE_SCHEMAS

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
    "build.output_manifests": TableSchema(
        schema="build",
        name="output_manifests",
        columns=[
            Column(
                "target",
                "VARCHAR",
                nullable=False,
                description="Target name (e.g., 'risk_factors')",
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
                "JSON",
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
                "JSON",
                nullable=False,
                description="JSON array of targets requested",
            ),
            Column(
                "computed_targets",
                "JSON",
                nullable=False,
                description="JSON array of targets computed",
            ),
            Column(
                "skipped_targets",
                "JSON",
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
                "JSON",
                nullable=False,
                description="JSON object mapping table keys to row counts",
            ),
            Column("error", "VARCHAR", description="Error message if failed"),
            Column(
                "dep_hashes",
                "JSON",
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
                "JSON",
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
            Column("meta", "JSON", description="Extra metadata for versioning/fingerprinting"),
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
            Column("meta", "JSON", description="Extra metadata for version events"),
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
            Column("meta", "JSON", description="Extra metadata for run-to-asset resolution"),
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
            Column("meta", "JSON"),
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
            Column("summary", "JSON", description="Diff summary JSON"),
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
            Column("tags", "JSON", description="Hamilton tags from node"),
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

COMPOSITE_SCHEMAS: Final[dict[str, CompositeSchema]] = {
    "analytics.function_profile": CompositeSchema(
        composed_of=(
            "analytics.function_metrics",
            "analytics.function_types",
            "analytics.function_effects",
            "analytics.function_contracts",
            "analytics.coverage_functions",
            "analytics.semantic_roles_functions",
            "analytics.function_history",
            "analytics.goid_risk_factors",
        ),
        shared_fragments=(FUNCTION_ENTITY_COLS,),
        additional_columns=(
            Column("module", "VARCHAR"),
            Column("file_typed_ratio", "DOUBLE"),
            Column("static_error_count", "INTEGER"),
            Column("has_static_errors", "BOOLEAN"),
            Column("tests_touching", "INTEGER"),
            Column("failing_tests", "INTEGER"),
            Column("slow_tests", "INTEGER"),
            Column("flaky_tests", "INTEGER"),
            Column("last_test_status", "VARCHAR"),
            Column("dominant_test_status", "VARCHAR"),
            Column("slow_test_threshold_ms", "DOUBLE"),
            Column("created_at_history", "TIMESTAMP"),
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
            Column("has_preconditions", "BOOLEAN"),
            Column("has_postconditions", "BOOLEAN"),
            Column("has_raises", "BOOLEAN"),
            Column("doc_short", "VARCHAR"),
            Column("doc_long", "VARCHAR"),
            Column("doc_params", "JSON"),
            Column("doc_returns", "JSON"),
            Column("tags", "JSON"),
            Column("owners", "JSON"),
            Column("created_at", "TIMESTAMP"),
        ),
        column_mappings={
            "keyword_only_params": "keyword_params",
            "has_varargs": "vararg",
            "has_varkw": "kwarg",
            "fan_in_count": "call_fan_in",
            "fan_out_count": "call_fan_out",
            "has_tests": "tested",
        },
        excluded_columns=frozenset(
            {
                "created_at",
                "effects_json",
                "preconditions_json",
                "postconditions_json",
                "raises_json",
                "unannotated_params",
                "param_typed_ratio",
                "has_return_annotation",
                "return_type_source",
                "type_comment",
                "is_async",
                "is_generator",
                "return_count",
                "yield_count",
                "raise_count",
                "history_window_start",
                "history_window_end",
                "created_at_row",
            }
        ),
    ),
    "analytics.file_profile": CompositeSchema(
        composed_of=(
            "analytics.typedness",
            "analytics.static_diagnostics",
            "analytics.hotspots",
            "analytics.tags_index",
        ),
        shared_fragments=(REPO_COMMIT_COLS,),
        additional_columns=(
            Column("rel_path", "VARCHAR"),
            Column("module", "VARCHAR"),
            Column("language", "VARCHAR"),
            Column("node_count", "INTEGER"),
            Column("function_count", "INTEGER"),
            Column("class_count", "INTEGER"),
            Column("avg_depth", "DOUBLE"),
            Column("max_depth", "INTEGER"),
            Column("ast_complexity", "DOUBLE"),
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
        ),
        column_mappings={},
        excluded_columns=frozenset(
            {
                "created_at",
                "rel_path",
                "path",
                "has_errors",
                "pyrefly_errors",
                "pyright_errors",
                "ruff_errors",
                "total_errors",
                "complexity",
                "score",
                "tag",
                "description",
                "includes",
                "excludes",
                "matches",
            }
        ),
    ),
    "analytics.module_profile": CompositeSchema(
        composed_of=(
            "analytics.graph_metrics_modules",
            "analytics.semantic_roles_modules",
            "analytics.tags_index",
        ),
        shared_fragments=(MODULE_ENTITY_COLS,),
        additional_columns=(
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
            Column("cycle_group", "INTEGER"),
            Column("in_cycle", "BOOLEAN"),
            Column("tags", "JSON"),
            Column("owners", "JSON"),
            Column("created_at", "TIMESTAMP"),
        ),
        column_mappings={
            "import_cycle_member": "in_cycle",
        },
        excluded_columns=frozenset(
            {
                "created_at",
                "import_in_degree",
                "import_out_degree",
                "import_pagerank",
                "import_betweenness",
                "import_closeness",
                "import_cycle_id",
                "import_layer",
                "symbol_fan_in",
                "symbol_fan_out",
                "framework",
                "tag",
                "description",
                "includes",
                "excludes",
                "matches",
            }
        ),
    ),
    "analytics.test_profile": CompositeSchema(
        composed_of=(
            "analytics.test_catalog",
            "analytics.test_graph_metrics_tests",
            "analytics.behavioral_coverage",
        ),
        shared_fragments=(REPO_COMMIT_COLS,),
        additional_columns=(
            Column("test_id", "VARCHAR"),
            Column("test_goid_h128", "DECIMAL(38,0)"),
            Column("urn", "VARCHAR"),
            Column("rel_path", "VARCHAR"),
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
            Column("created_at", "TIMESTAMP"),
        ),
        column_mappings={
            "degree": "tg_degree",
            "weighted_degree": "tg_weighted_degree",
            "proj_degree": "tg_proj_degree",
            "proj_weight": "tg_proj_weight",
            "proj_clustering": "tg_proj_clustering",
            "proj_betweenness": "tg_proj_betweenness",
            "parametrized": "uses_parametrize",
        },
        excluded_columns=frozenset(
            {
                "created_at",
                "degree_centrality",
                "risk_weighted_degree",
                "behavior_tags",
                "tag_source",
                "heuristic_version",
                "llm_model",
                "llm_run_id",
            }
        ),
    ),
}


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
