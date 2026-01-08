"""Persist contract alignment diagnostics as build datasets."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from codeintel.build.contracts.registry import contract_descriptor_for_table_key
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.tabular.arrow_ops import AlignmentReport, drain_alignment_diagnostics
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.datasets.arrow_store import ArrowDatasetWriteOptions, write_dataset
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.service import get_schema_service

log = logging.getLogger(__name__)

CONTRACT_ALIGNMENT_ISSUES_TABLE_KEY = "build.contract_alignment_issues"


def persist_contract_alignment_issues(
    *,
    env: BuildEnv,
    run_id: str,
) -> bool:
    """Persist contract alignment diagnostics as a dataset.

    Returns
    -------
    bool
        True when rows were written, otherwise False.
    """
    reports = drain_alignment_diagnostics()
    dataset_root = env.paths.dataset_root_dir
    if dataset_root is None:
        if reports:
            log.info("Contract alignment diagnostics skipped; dataset_root_dir unavailable.")
        return False
    snapshot_id = env.commit.strip()
    if not snapshot_id:
        log.warning("Contract alignment diagnostics skipped; snapshot_id missing.")
        return False
    if not run_id:
        log.warning("Contract alignment diagnostics skipped; run_id missing.")
        return False

    rows = _alignment_issue_rows(env=env, run_id=run_id, reports=reports)
    if rows:
        table, _ = table_for_rows(CONTRACT_ALIGNMENT_ISSUES_TABLE_KEY, rows)
    else:
        table = empty_table_for_table(CONTRACT_ALIGNMENT_ISSUES_TABLE_KEY)

    schema_service = get_schema_service()
    table_schema = schema_service.require_table_schema(CONTRACT_ALIGNMENT_ISSUES_TABLE_KEY)
    schema_hash_value = schema_hash(table_schema)
    partition_columns = _partition_columns_for_schema(table_schema)
    options = ArrowDatasetWriteOptions(
        partition_columns=partition_columns,
        schema_hash=schema_hash_value,
        manifest_extras={"table_schema": table_schema.to_json_obj()},
    )
    write_dataset(
        dataset_root=dataset_root,
        table_key=CONTRACT_ALIGNMENT_ISSUES_TABLE_KEY,
        snapshot_id=snapshot_id,
        data=table,
        options=options,
    )
    return bool(rows)


def _alignment_issue_rows(
    *,
    env: BuildEnv,
    run_id: str,
    reports: tuple[AlignmentReport, ...],
) -> list[dict[str, object]]:
    recorded_at = datetime.now(tz=UTC)
    rows: list[dict[str, object]] = []
    for report in reports:
        descriptor = contract_descriptor_for_table_key(report.table_key)
        target_name = report.target_name or "unknown"
        rows.append(
            {
                "run_id": run_id,
                "repo": env.repo,
                "commit": env.commit,
                "target_name": target_name,
                "table_key": report.table_key,
                "missing_columns": list(report.missing_columns),
                "extra_columns": list(report.extra_columns),
                "coerced_columns": list(report.coerced_columns),
                "missing_count": len(report.missing_columns),
                "extra_count": len(report.extra_columns),
                "coerced_count": len(report.coerced_columns),
                "row_count": report.row_count,
                "contract_hash": None if descriptor is None else descriptor.contract_hash,
                "contract_version": None if descriptor is None else descriptor.contract_version,
                "recorded_at": recorded_at,
            }
        )
    return rows


def _partition_columns_for_schema(table_schema: TableSchema) -> tuple[str, ...]:
    names = set(table_schema.column_names())
    if "repo" in names and "commit" in names:
        return ("repo", "commit")
    return ()


__all__ = [
    "CONTRACT_ALIGNMENT_ISSUES_TABLE_KEY",
    "persist_contract_alignment_issues",
]
