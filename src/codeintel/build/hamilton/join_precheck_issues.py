"""Persist join precheck diagnostics as build datasets."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.tabular.finalize_ops import JoinPrecheckReport, drain_join_precheck_reports
from codeintel.core.columnar.iter import iter_rows
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.datasets.arrow_store import ArrowDatasetWriteOptions, write_dataset
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import TableSchema, resolve_stable_sort_keys
from codeintel.core.schemas.service import get_schema_service

log = logging.getLogger(__name__)

JOIN_PRECHECK_ISSUES_TABLE_KEY = "build.join_precheck_issues"


def persist_join_precheck_issues(
    *,
    env: BuildEnv,
    run_id: str,
) -> bool:
    """Persist join precheck diagnostics as a dataset.

    Returns
    -------
    bool
        True when rows were written, otherwise False.
    """
    reports = drain_join_precheck_reports()
    dataset_root = env.paths.dataset_root_dir
    if dataset_root is None:
        if reports:
            log.info("Join precheck diagnostics skipped; dataset_root_dir unavailable.")
        return False
    snapshot_id = env.commit.strip()
    if not snapshot_id:
        log.warning("Join precheck diagnostics skipped; snapshot_id missing.")
        return False
    if not run_id:
        log.warning("Join precheck diagnostics skipped; run_id missing.")
        return False

    rows = _issue_rows(env=env, run_id=run_id, reports=reports)
    if rows:
        table, _ = table_for_rows(JOIN_PRECHECK_ISSUES_TABLE_KEY, rows)
    else:
        table = empty_table_for_table(JOIN_PRECHECK_ISSUES_TABLE_KEY)

    schema_service = get_schema_service()
    table_schema = schema_service.require_table_schema(JOIN_PRECHECK_ISSUES_TABLE_KEY)
    schema_hash_value = schema_hash(table_schema)
    partition_columns = _partition_columns_for_schema(table_schema)
    options = ArrowDatasetWriteOptions(
        partition_columns=partition_columns,
        schema_hash=schema_hash_value,
        manifest_extras={"table_schema": table_schema.to_json_obj()},
        stable_sort_keys=resolve_stable_sort_keys(table_schema),
    )
    write_dataset(
        dataset_root=dataset_root,
        table_key=JOIN_PRECHECK_ISSUES_TABLE_KEY,
        snapshot_id=snapshot_id,
        data=table,
        options=options,
    )
    return bool(rows)


def _issue_rows(
    *,
    env: BuildEnv,
    run_id: str,
    reports: tuple[JoinPrecheckReport, ...],
) -> list[dict[str, object]]:
    recorded_at = datetime.now(tz=UTC)
    rows: list[dict[str, object]] = []
    for report in reports:
        target_name = report.target_name or "unknown"
        table_key = report.table_key or "derived"
        join_keys = list(report.join_keys)
        join_key_signature = ",".join(report.join_keys)
        rows.extend(
            [
                {
                    "run_id": run_id,
                    "repo": env.repo,
                    "commit": env.commit,
                    "target_name": target_name,
                    "table_key": table_key,
                    "join_keys": join_keys,
                    "join_key_signature": join_key_signature,
                    "error_code": _string_or_fallback(row.get("error_code"), "unknown"),
                    "stage": _string_or_fallback(row.get("stage"), "unknown"),
                    "column": _string_or_fallback(row.get("column"), "unknown"),
                    "detail": _string_or_fallback(row.get("detail"), "unknown"),
                    "row_id": _int_or_fallback(row.get("row_id")),
                    "key_payload": _key_payload(row, report.join_keys),
                    "recorded_at": recorded_at,
                }
                for row in iter_rows(report.errors)
            ]
        )
    return rows


def _string_or_fallback(value: object, fallback: str) -> str:
    return value if isinstance(value, str) else fallback


def _int_or_fallback(value: object) -> int:
    if isinstance(value, bool):
        return -1
    return value if isinstance(value, int) else -1


def _key_payload(row: dict[str, object], join_keys: tuple[str, ...]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key in join_keys:
        if key in row:
            payload[key] = row[key]
    return payload


def _partition_columns_for_schema(table_schema: TableSchema) -> tuple[str, ...]:
    names = set(table_schema.column_names())
    if "repo" in names and "commit" in names:
        return ("repo", "commit")
    return ()


__all__ = [
    "JOIN_PRECHECK_ISSUES_TABLE_KEY",
    "persist_join_precheck_issues",
]
