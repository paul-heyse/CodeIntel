"""Shared utilities for export operations.

This module contains common dataclasses and functions used by both
JSONL and Parquet export implementations.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import pandas as pd

from codeintel.build.schemas import iter_contracts
from codeintel.core.errors import ProblemDetailBuilder
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.validation import validate_contract_or_raise

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.storage.gateway import DuckDBConnection, DuckDBRelation, StorageGateway

log = logging.getLogger(__name__)

MAX_EXPORT_LIMIT = 9_223_372_036_854_775_807
AUDIT_LOG_PATH = os.getenv("CODEINTEL_EXPORT_AUDIT_LOG")
AUDIT_TABLE_ENABLED = os.getenv("CODEINTEL_EXPORT_AUDIT_TABLE") is not None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class ExportError(Exception):
    """Export operation failure."""


def export_problem(code: str, title: str, detail: str, **extras: object) -> Exception:
    """Create an export error with structured problem details.

    Parameters
    ----------
    code
        Problem code (e.g., "export.validation_failed").
    title
        Short problem title.
    detail
        Detailed error message.
    **extras
        Additional context fields.

    Returns
    -------
    Exception
        ExportError with structured details.
    """
    builder = ProblemDetailBuilder(code=code, title=title, status=500)
    problem_detail = builder.build(detail).with_extensions(**extras)
    return ExportError(problem_detail.detail or title)


def log_export_error(code: str, title: str, detail: str, **extras: object) -> None:
    """Log an export error with structured problem details.

    Parameters
    ----------
    code
        Problem code.
    title
        Short problem title.
    detail
        Detailed error message.
    **extras
        Additional context fields.
    """
    builder = ProblemDetailBuilder(code=code, title=title, status=500)
    problem_detail = builder.build(detail).with_extensions(**extras)
    log.error(json.dumps(problem_detail.to_dict()))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditRecord:
    """Metadata about a completed export for optional audit logging."""

    table_name: str
    macro: str
    rows: int | None
    duration_s: float
    output_path: Path


@dataclass(frozen=True)
class ExportTarget:
    """Inputs describing a dataset export request."""

    dataset_name: str
    table_name: str
    output_path: Path
    dataset: DatasetContract | None


@dataclass(frozen=True)
class ExportCallOptions:
    """Options controlling dataset selection, validation, and macro enforcement."""

    validate_exports: bool = True
    schemas: list[str] | None = None
    datasets: list[str] | None = None
    validation_profile: Literal["strict", "lenient"] | None = None
    force_full_export: bool = False


# ---------------------------------------------------------------------------
# Registry validation
# ---------------------------------------------------------------------------


def validate_registry_or_raise(gateway: StorageGateway) -> None:
    """Validate dataset registry and normalize error type for schema mismatches.

    Parameters
    ----------
    gateway
        StorageGateway providing access to dataset registry.

    Raises
    ------
    ValueError
        If required tables or views are missing from the registry.
    ExportError
        If tables exist but their schemas do not match expectations.
    """
    missing_tables: list[str] = []
    for dataset_name, table_key in gateway.datasets.mapping.items():
        schema_name, table_name = split_table_key(table_key)
        exists = gateway.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = ? AND table_name = ?
            LIMIT 1
            """,
            [schema_name, table_name],
        ).fetchone()
        if exists is None:
            missing_tables.append(f"{dataset_name} -> {table_key}")

    if missing_tables:
        message = "Dataset registry missing tables/views: " + ", ".join(sorted(missing_tables))
        raise ValueError(message)

    try:
        validate_contract_or_raise(gateway.con)
    except ValueError as exc:
        detail = str(exc)
        log_export_error(
            code="export.validation_failed",
            title="Export validation failed",
            detail=detail,
            stage="dataset_registry",
        )
        raise ExportError(detail) from exc


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------


def resolve_dataset_table(dataset_name: str, dataset_mapping: Mapping[str, str]) -> str:
    """Resolve a dataset name to its table key.

    Parameters
    ----------
    dataset_name
        Logical dataset name.
    dataset_mapping
        Mapping of dataset name to table key.

    Returns
    -------
    str
        Fully qualified table key.

    Raises
    ------
    ValueError
        If dataset name is not in the mapping.
    """
    table = dataset_mapping.get(dataset_name)
    if table is None:
        message = f"Unknown dataset: {dataset_name}"
        raise ValueError(message)
    return table


def select_dataset_tables(
    dataset_mapping: Mapping[str, str],
    format_mapping: Mapping[str, str],
    datasets: list[str] | None,
) -> dict[str, str]:
    """Determine which dataset names and tables to export.

    Parameters
    ----------
    dataset_mapping
        Mapping of dataset name to table/view key from the gateway registry.
    format_mapping
        Mapping of table/view key to filename from the gateway registry.
    datasets
        Optional list of dataset names requested by the caller.

    Returns
    -------
    dict[str, str]
        Selected dataset name to table/view key mapping.
    """
    if datasets is None:
        return {name: table for name, table in dataset_mapping.items() if table in format_mapping}
    selected: dict[str, str] = {}
    for dataset_name in datasets:
        selected[dataset_name] = resolve_dataset_table(dataset_name, dataset_mapping)
    return selected


# ---------------------------------------------------------------------------
# Validation profile and schema digest
# ---------------------------------------------------------------------------


def resolve_validation_profile(
    options: ExportCallOptions,
    dataset: DatasetContract | None,
) -> str:
    """Resolve the validation profile for an export.

    Parameters
    ----------
    options
        Export options that may override the profile.
    dataset
        Dataset contract with default profile.

    Returns
    -------
    str
        Validation profile ("strict" or "lenient").
    """
    if options.validation_profile is not None:
        return options.validation_profile
    if dataset is not None:
        return dataset.validation_profile
    return "strict"


def compute_schema_digest(dataset: DatasetContract | None) -> str | None:
    """Compute digest of the generated JSON schema for a dataset.

    Parameters
    ----------
    dataset
        Dataset contract to compute digest for.

    Returns
    -------
    str | None
        SHA-256 hex digest, or None if unavailable.
    """
    if dataset is None or dataset.json_schema_id is None:
        return None
    try:
        from codeintel.build.schemas.json_schema_registry import (  # noqa: PLC0415
            compute_json_schema_digest,
        )

        return compute_json_schema_digest(dataset.table_key)
    except Exception:  # noqa: BLE001
        log.debug("Generated schema digest unavailable for %s", dataset.table_key, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Row count and relation building
# ---------------------------------------------------------------------------


def get_row_count(gateway: StorageGateway, table_name: str) -> int | None:
    """Get the row count for a table.

    Parameters
    ----------
    gateway
        StorageGateway providing Ibis access.
    table_name
        Fully qualified table name.

    Returns
    -------
    int | None
        Row count, or None if unavailable.
    """
    try:
        table = gateway.ibis.table(table_name)
        row = table.count().execute()
    except DuckDBError:
        log.debug("Row count unavailable for %s", table_name, exc_info=True)
        return None
    if row is None:
        return None
    if isinstance(row, pd.DataFrame):
        if row.empty:
            return None
        return int(row.iloc[0, 0])
    if isinstance(row, (list, tuple)):
        return int(row[0]) if row else None
    return int(cast("int", row))


def build_export_relation(
    gateway: StorageGateway,
    table_key: str,
    row_limit: int,
    row_offset: int,
) -> DuckDBRelation:
    """Build a DuckDB relation for export.

    Parameters
    ----------
    gateway
        StorageGateway providing connection.
    table_key
        Fully qualified table key.
    row_limit
        Maximum rows to export.
    row_offset
        Offset for pagination.

    Returns
    -------
    DuckDBRelation
        Relation ready for export.
    """
    from codeintel.build.exports.exprs import (  # noqa: PLC0415
        build_export_expr,
        compile_export_sql,
    )

    expr = build_export_expr(gateway, table_key, limit=row_limit, offset=row_offset)
    sql = compile_export_sql(expr)
    return gateway.con.sql(sql)


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


def write_audit_entry(
    record: AuditRecord,
    *,
    con: DuckDBConnection,
) -> None:
    """Write an audit entry for an export operation.

    Parameters
    ----------
    record
        Audit record to write.
    con
        DuckDB connection for table logging.
    """
    if AUDIT_LOG_PATH is None and not AUDIT_TABLE_ENABLED:
        return
    json_record = {
        "table": record.table_name,
        "macro": record.macro,
        "rows": record.rows,
        "duration_s": record.duration_s,
        "output": str(record.output_path),
    }
    if AUDIT_LOG_PATH is not None:
        with Path(AUDIT_LOG_PATH).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_record))
            handle.write("\n")

    if AUDIT_TABLE_ENABLED:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata.export_audit (
                dataset TEXT,
                macro TEXT,
                rows BIGINT,
                duration_s DOUBLE,
                output_path TEXT,
                sql TEXT,
                plan TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )
        con.execute(
            """
            INSERT INTO metadata.export_audit
                (dataset, macro, rows, duration_s, output_path, sql, plan)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.table_name,
                record.macro,
                record.rows,
                record.duration_s,
                str(record.output_path),
                None,
                None,
            ],
        )


# ---------------------------------------------------------------------------
# Default validation schemas
# ---------------------------------------------------------------------------


def default_validation_schemas() -> list[str]:
    """Return the set of dataset names that should be validated by default.

    Derived from contracts in the build.schemas contract provider.

    Returns
    -------
    list[str]
        Sorted dataset names with JSON Schema validation configured.
    """
    return sorted(c.name for c in iter_contracts() if c.json_schema_id is not None)


__all__ = [
    "AUDIT_LOG_PATH",
    "AUDIT_TABLE_ENABLED",
    "MAX_EXPORT_LIMIT",
    "AuditRecord",
    "ExportCallOptions",
    "ExportError",
    "ExportTarget",
    "build_export_relation",
    "compute_schema_digest",
    "default_validation_schemas",
    "export_problem",
    "get_row_count",
    "log_export_error",
    "resolve_dataset_table",
    "resolve_validation_profile",
    "select_dataset_tables",
    "validate_registry_or_raise",
    "write_audit_entry",
]
