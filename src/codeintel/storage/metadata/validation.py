"""Schema validation persistence helpers for meta catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlglot import exp

from codeintel.core.serialization.payload import encode_payload
from codeintel.core.sqlglot_tools import render_sql_duckdb, table_expr_from_ref
from codeintel.core.time import utc_now
from codeintel.storage.metadata.meta_catalog import meta_table_ref

if TYPE_CHECKING:
    from datetime import datetime

    from duckdb import DuckDBPyConnection

__all__ = ["SchemaValidationRun", "record_schema_validation_run"]


@dataclass(frozen=True, slots=True)
class SchemaValidationRun:
    """Schema validation run payload for metadata persistence."""

    repo: str | None
    commit: str | None
    validation_mode: str
    include_views: bool
    issues: list[str]
    created_at: datetime | None = None


def record_schema_validation_run(
    con: DuckDBPyConnection,
    run: SchemaValidationRun,
) -> str:
    """Persist a schema validation run to metadata.

    Returns
    -------
    str
        Validation run identifier.
    """
    validation_id = uuid4().hex
    issue_count = len(run.issues)
    status = "passed" if issue_count == 0 else "failed"
    issues_payload = encode_payload(run.issues) if run.issues else None
    run_ref = meta_table_ref("metadata.schema_validation_runs")
    columns = [
        "validation_id",
        "repo",
        "commit",
        "validation_mode",
        "include_views",
        "issue_count",
        "status",
        "issues",
        "created_at",
    ]
    insert = exp.Insert(
        this=exp.Schema(
            this=table_expr_from_ref(run_ref),
            expressions=[exp.to_identifier(column) for column in columns],
        ),
        expression=exp.Values(
            expressions=[exp.Tuple(expressions=[exp.Placeholder() for _ in columns])]
        ),
    )
    con.execute(
        render_sql_duckdb(insert),
        [
            validation_id,
            run.repo,
            run.commit,
            run.validation_mode,
            run.include_views,
            issue_count,
            status,
            issues_payload,
            run.created_at or utc_now(),
        ],
    )
    return validation_id
