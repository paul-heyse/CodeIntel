"""Schema definition for planning entries."""

from __future__ import annotations

from codeintel.core.schemas.primitives import Column, Index, TableSchema

CI_PLAN_ENTRIES_TABLE_KEY = "ci.plan_entries"

CI_PLAN_ENTRIES_TABLE_SCHEMA = TableSchema(
    schema="ci",
    name="plan_entries",
    columns=[
        Column("run_id", "VARCHAR", nullable=False),
        Column("created_at_utc", "TIMESTAMPTZ", nullable=False),
        Column("requested_targets", "BLOB", nullable=False),
        Column("target", "VARCHAR", nullable=False),
        Column("domain", "VARCHAR", nullable=False),
        Column("action", "VARCHAR", nullable=False),
        Column("cache_hit_ratio", "DOUBLE"),
        Column("block_reasons", "BLOB"),
        Column("miss_nodes", "BLOB"),
        Column("reads", "BLOB"),
        Column("writes_tables", "BLOB"),
        Column("writes_artifacts", "BLOB"),
        Column("build_fingerprint", "VARCHAR", nullable=False),
        Column("plan_schema_version", "VARCHAR", nullable=False),
    ],
    primary_key=("run_id", "target"),
    indexes=(Index("idx_ci_plan_entries_target", ("target",)),),
    description="Planning entries emitted by the plan DAG output.",
)

__all__ = [
    "CI_PLAN_ENTRIES_TABLE_KEY",
    "CI_PLAN_ENTRIES_TABLE_SCHEMA",
]
