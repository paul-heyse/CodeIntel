"""Arrow-first validation helpers for columnar data.

Deprecated: use ``codeintel.core.validation.engine``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

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
from codeintel.storage.contracts.schema_provider import get_schema_provider

if TYPE_CHECKING:
    from pathlib import Path

    import pyarrow as pa

    from codeintel.core.schemas.primitives import TableSchema


__all__ = [
    "ColumnarValidationContext",
    "TableValidationError",
    "ValidationMode",
    "validate_parquet_path",
    "validate_record_batch_reader",
    "validate_table",
]


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
    return _validate_table(table_key, table, context=resolved_context, mode=mode)


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
        return replace(resolved, table_schema=schema)
    return resolved


def _lookup_table_schema(table_key: str) -> TableSchema | None:
    try:
        provider = get_schema_provider()
    except RuntimeError:
        provider = None
    resolution = resolve_table_schema(table_key, schema_provider=provider)
    return resolution.table_schema
