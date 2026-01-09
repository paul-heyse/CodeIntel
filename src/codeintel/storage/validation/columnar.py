"""Arrow-first validation helpers for columnar data.

Deprecated: use ``codeintel.core.validation.engine``.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.columnar.finalize_ops import FinalizeDedupe, FinalizeSpec, finalize_table
from codeintel.core.schemas.resolution import resolve_table_schema
from codeintel.core.validation.engine import (
    ColumnarValidationContext,
    TableValidationError,
    ValidationMode,
)
from codeintel.core.validation.engine import (
    validate_parquet_path as _validate_parquet_path,
)
from codeintel.core.validation.engine import (
    validate_record_batch_reader as _validate_record_batch_reader,
)
from codeintel.core.validation.engine import (
    validate_table as _validate_table,
)
from codeintel.core.validation.schema_constraints import list_alignment_specs_for_table_key
from codeintel.storage.contracts.schema_provider import get_schema_provider
from codeintel.storage.query_results import records_from_arrow_table

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.core.columnar.finalize_ops import FinalizeResult
    from codeintel.core.schemas.primitives import TableSchema


__all__ = [
    "ColumnarValidationContext",
    "TableValidationError",
    "ValidationMode",
    "validate_parquet_path",
    "validate_record_batch_reader",
    "validate_table",
]


LOG = logging.getLogger(__name__)
_DEFAULT_PROVENANCE_FIELDS = ("__filename", "__fragment_index", "__batch_index")


def _coerce_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _alignment_error_messages(
    *,
    missing: list[str],
    extra: list[str],
    coerced: list[str],
) -> list[str]:
    if not missing and not extra and not coerced:
        return []
    return [
        (
            "Alignment report: "
            f"missing_columns={missing}, "
            f"extra_columns={extra}, "
            f"coerced_columns={coerced}"
        )
    ]


def _alignment_errors_from_artifacts(alignment: pa.Table) -> list[str]:
    if alignment.num_rows == 0:
        return []
    records = records_from_arrow_table(alignment)
    if not records:
        return []
    row = records[0]
    missing = _coerce_str_list(row.get("missing_columns"))
    extra = _coerce_str_list(row.get("extra_columns"))
    coerced = _coerce_str_list(row.get("coerced_columns"))
    return _alignment_error_messages(missing=missing, extra=extra, coerced=coerced)


def _finalize_artifact_errors(result: FinalizeResult) -> list[str]:
    errors: list[str] = []
    if result.stats.num_rows:
        for row in records_from_arrow_table(result.stats):
            code = row.get("error_code")
            count = row.get("count")
            if isinstance(code, str):
                errors.append(f"Finalize error {code}: {count} rows")
            else:
                errors.append(f"Finalize error: {row}")
    errors.extend(_alignment_errors_from_artifacts(result.alignment))
    return errors


def _finalize_table_for_validation(
    table_key: str,
    table: pa.Table,
    *,
    table_schema: TableSchema | None,
    mode: ValidationMode,
) -> tuple[pa.Table, list[str]]:
    if mode == "skip":
        return table, []
    required_non_null: tuple[str, ...] = ()
    if table_schema is not None:
        required_non_null = tuple(
            column.name for column in table_schema.columns if not column.nullable
        )
    spec = FinalizeSpec(
        table_key=table_key,
        mode="tolerant",
        required_non_null=required_non_null,
        dedupe=FinalizeDedupe(enabled=False),
        key_fields=table_schema.primary_key if table_schema is not None else (),
        context_fields=_DEFAULT_PROVENANCE_FIELDS,
        emit_artifacts=True,
    )
    try:
        result = finalize_table(table, spec=spec)
    except ValueError as exc:
        return table, [str(exc)]
    return result.good, _finalize_artifact_errors(result)


def _handle_finalize_errors(
    table_key: str,
    errors: list[str],
    mode: ValidationMode,
) -> None:
    if not errors or mode == "skip":
        return
    if mode == "warn":
        for error in errors:
            LOG.warning("Finalize warning for %s: %s", table_key, error)
        return
    raise TableValidationError(table_key, errors)


def validate_table(
    table_key: str,
    table: pa.Table,
    *,
    context: ColumnarValidationContext | None = None,
    mode: ValidationMode = "strict",
) -> pa.Table:
    """Validate an Arrow table against the registered TableSchema.

    Parameters
    ----------
    table_key
        Table key used to resolve schema metadata.
    table
        Arrow table to validate.
    context
        Optional validation context overrides.
    mode
        Validation mode ("strict", "warn", or "skip").

    Returns
    -------
    pyarrow.Table
        The input table when validation passes or is skipped.
    """
    resolved_context = _resolve_context(table_key, context)
    finalized, finalize_errors = _finalize_table_for_validation(
        table_key,
        table,
        table_schema=resolved_context.table_schema,
        mode=mode,
    )
    _handle_finalize_errors(table_key, finalize_errors, mode)
    resolved_context = replace(resolved_context, finalize=False)
    return _validate_table(table_key, finalized, context=resolved_context, mode=mode)


def validate_record_batch_reader(
    table_key: str,
    reader: pa.RecordBatchReader,
    *,
    context: ColumnarValidationContext | None = None,
    mode: ValidationMode = "strict",
) -> pa.RecordBatchReader:
    """Validate a RecordBatchReader stream against the registered TableSchema.

    Parameters
    ----------
    table_key
        Table key used to resolve schema metadata.
    reader
        Record batch reader to validate.
    context
        Optional validation context overrides.
    mode
        Validation mode ("strict", "warn", or "skip").

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader that yields validated record batches.
    """
    resolved_context = _resolve_context(table_key, context)
    return _validate_record_batch_reader(table_key, reader, context=resolved_context, mode=mode)


def validate_parquet_path(
    table_key: str,
    path: Path,
    *,
    context: ColumnarValidationContext | None = None,
    mode: ValidationMode = "strict",
) -> None:
    """Validate a Parquet file against the registered TableSchema.

    Parameters
    ----------
    table_key
        Table key used to resolve schema metadata.
    path
        Parquet file path to validate.
    context
        Optional validation context overrides.
    mode
        Validation mode ("strict", "warn", or "skip").
    """
    resolved_context = _resolve_context(table_key, context)
    _validate_parquet_path(table_key, path, context=resolved_context, mode=mode)


def _resolve_context(
    table_key: str,
    context: ColumnarValidationContext | None,
) -> ColumnarValidationContext:
    resolved = context or ColumnarValidationContext()
    if resolved.table_schema is None:
        schema = _lookup_table_schema(table_key)
        resolved = replace(resolved, table_schema=schema)
    if not resolved.list_alignments:
        resolved = replace(
            resolved,
            list_alignments=list_alignment_specs_for_table_key(table_key),
        )
    return resolved


def _lookup_table_schema(table_key: str) -> TableSchema | None:
    try:
        provider = get_schema_provider()
    except RuntimeError:
        provider = None
    resolution = resolve_table_schema(table_key, schema_provider=provider)
    return resolution.table_schema
