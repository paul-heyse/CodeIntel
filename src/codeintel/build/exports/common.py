"""Shared utilities for export operations.

This module contains common dataclasses and functions used by both
JSONL and Parquet export implementations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from codeintel.build.errors import BuildProblemError
from codeintel.build.exports.exprs import build_export_expr, compile_export_sql
from codeintel.build.schemas import iter_contracts
from codeintel.build.schemas.json_schema_registry import compute_json_schema_digest
from codeintel.core.config.settings import ExportAuditSettings
from codeintel.core.errors.schema import SchemaError
from codeintel.core.errors.taxonomy import SCHEMA_MISMATCH, ErrorCode
from codeintel.storage.exports import ExportAuditRecord as AuditRecord
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.protocols import ExportRelation
from codeintel.storage.validation import validate_contract_or_raise

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

MAX_EXPORT_LIMIT = 9_223_372_036_854_775_807


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def export_problem(
    error_code: ErrorCode,
    detail: str,
    **extras: object,
) -> BuildProblemError:
    """Create an export error with structured problem details.

    Parameters
    ----------
    error_code
        Canonical error taxonomy entry.
    detail
        Detailed error message.
    **extras
        Additional context fields.

    Returns
    -------
    BuildProblemError
        BuildProblemError with structured details.
    """
    return BuildProblemError.from_error_code(error_code=error_code, detail=detail, **extras)


def log_export_error(
    error_code: ErrorCode,
    detail: str,
    **extras: object,
) -> None:
    """Log an export error with structured problem details.

    Parameters
    ----------
    error_code
        Canonical error taxonomy entry.
    detail
        Detailed error message.
    **extras
        Additional context fields.
    """
    error = BuildProblemError.from_error_code(error_code=error_code, detail=detail, **extras)
    log.error(json.dumps(error.problem_detail.to_dict()))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportTarget:
    """Inputs describing a dataset export request."""

    dataset_name: str
    table_name: str
    output_path: Path
    dataset: DatasetContract | None


@dataclass(frozen=True)
class ExportCallOptions:
    """Options controlling dataset selection, validation, and macro enforcement.

    Attributes
    ----------
    schemas
        Optional list of table keys to validate.
    datasets
        Optional list of dataset names to export.
    """

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
    BuildProblemError
        If tables exist but their schemas do not match expectations.
    """
    missing_tables: list[str] = []
    for dataset_name, contract in gateway.datasets.by_name.items():
        schema_name, table_name = split_table_key(contract.table_key)
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
            missing_tables.append(f"{dataset_name} -> {contract.table_key}")

    if missing_tables:
        message = "Dataset registry missing tables/views: " + ", ".join(sorted(missing_tables))
        raise ValueError(message)

    try:
        validate_contract_or_raise(gateway.con)
    except ValueError as exc:
        detail = str(exc)
        log_export_error(
            SCHEMA_MISMATCH,
            detail,
            stage="dataset_registry",
        )
        problem = BuildProblemError.from_error_code(
            error_code=SCHEMA_MISMATCH,
            detail=detail,
        ).problem_detail
        raise BuildProblemError(problem) from exc


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
        return compute_json_schema_digest(dataset.table_key)
    except SchemaError as e:
        log.debug("Generated schema digest unavailable for %s: %s", dataset.table_key, e)
        return None


# ---------------------------------------------------------------------------
def build_export_relation(
    gateway: StorageGateway,
    table_key: str,
    row_limit: int,
    row_offset: int,
) -> ExportRelation:
    """Build an export relation for a dataset table.

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
    ExportRelation
        Export relation adapter.
    """
    expr = build_export_expr(gateway, table_key, limit=row_limit, offset=row_offset)
    sql = compile_export_sql(expr)
    return gateway.exports.build_export_relation(sql=sql)


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


def write_audit_entry(
    record: AuditRecord,
    *,
    gateway: StorageGateway,
    settings: ExportAuditSettings,
) -> None:
    """Write an audit entry for an export operation.

    Parameters
    ----------
    record
        Audit record to write.
    gateway
        Storage gateway providing audit logging access.
    settings
        Export audit settings for the write.
    """
    if not gateway.exports.audit_enabled(settings):
        return
    gateway.exports.write_export_audit(record, settings=settings)


# ---------------------------------------------------------------------------
# Default validation schemas
# ---------------------------------------------------------------------------


def default_validation_schemas() -> list[str]:
    """Return the set of table keys that should be validated by default.

    Derived from contracts in the build.schemas contract provider.

    Returns
    -------
    list[str]
        Sorted table keys with JSON Schema validation configured.
    """
    return sorted(c.table_key for c in iter_contracts() if c.json_schema_id is not None)


__all__ = [
    "MAX_EXPORT_LIMIT",
    "AuditRecord",
    "ExportCallOptions",
    "ExportTarget",
    "build_export_relation",
    "compute_schema_digest",
    "default_validation_schemas",
    "export_problem",
    "log_export_error",
    "resolve_dataset_table",
    "resolve_validation_profile",
    "select_dataset_tables",
    "validate_registry_or_raise",
    "write_audit_entry",
]
