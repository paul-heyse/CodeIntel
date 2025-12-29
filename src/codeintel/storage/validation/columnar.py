"""Arrow-first validation helpers for columnar data."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.schemas.arrow_gen import DEFAULT_EXTRAS_COLUMN
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.storage.contracts.schema_provider import get_schema_provider

ValidationMode = Literal["strict", "warn", "skip"]

LOG = logging.getLogger(__name__)

__all__ = [
    "TableValidationError",
    "ValidationMode",
    "validate_record_batch_reader",
    "validate_table",
]


class TableValidationError(ValueError):
    """Raised when a table fails columnar validation."""

    def __init__(self, table_key: str, errors: list[str]) -> None:
        message = f"Validation failed for {table_key}: " + "; ".join(errors)
        super().__init__(message)
        self.table_key = table_key
        self.errors = errors


def _lookup_table_schema(table_key: str) -> TableSchema | None:
    try:
        provider = get_schema_provider()
    except RuntimeError:
        return None
    return provider.get_table_schema(table_key)


def _unwrap_dictionary_type(data_type: pa.DataType) -> pa.DataType:
    if pa.types.is_dictionary(data_type):
        return data_type.value_type
    return data_type


def _is_compatible_type(column: Column, actual_type: pa.DataType) -> bool:
    normalized = _unwrap_dictionary_type(actual_type)
    if pa.types.is_null(normalized):
        return column.nullable
    if column.type == "JSON":
        return True

    def _is_decimal_or_int(data_type: pa.DataType) -> bool:
        return pa.types.is_integer(data_type) or pa.types.is_decimal(data_type)

    def _is_decimal_or_float(data_type: pa.DataType) -> bool:
        return (
            pa.types.is_floating(data_type)
            or pa.types.is_decimal(data_type)
            or pa.types.is_integer(data_type)
        )

    def _is_string_like(data_type: pa.DataType) -> bool:
        return pa.types.is_string(data_type) or pa.types.is_large_string(data_type)

    def _is_temporal(data_type: pa.DataType) -> bool:
        return pa.types.is_timestamp(data_type) or pa.types.is_date(data_type)

    predicates: dict[str, Callable[[pa.DataType], bool]] = {
        "INTEGER": pa.types.is_integer,
        "BIGINT": pa.types.is_integer,
        "DECIMAL(38,0)": _is_decimal_or_int,
        "DOUBLE": _is_decimal_or_float,
        "DECIMAL": _is_decimal_or_float,
        "BOOLEAN": pa.types.is_boolean,
        "VARCHAR": _is_string_like,
        "TIMESTAMP": _is_temporal,
        "TIMESTAMPTZ": _is_temporal,
    }
    predicate = predicates.get(column.type)
    if predicate is None:
        return True
    return predicate(normalized)


def _schema_errors(table_schema: TableSchema, actual_schema: pa.Schema) -> list[str]:
    errors: list[str] = []
    expected_names = [column.name for column in table_schema.columns]
    actual_names = [name for name in actual_schema.names if name != DEFAULT_EXTRAS_COLUMN]

    missing = [name for name in expected_names if name not in actual_names]
    extra = [name for name in actual_names if name not in expected_names]
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
    if extra:
        errors.append(f"Unexpected columns: {', '.join(extra)}")
    if not missing and not extra and expected_names != actual_names:
        errors.append(
            "Column order mismatch: expected "
            f"{', '.join(expected_names)} but got {', '.join(actual_names)}"
        )

    by_name = {column.name: column for column in table_schema.columns}
    for name in expected_names:
        if name not in actual_schema.names:
            continue
        column = by_name[name]
        actual_field = actual_schema.field(name)
        if not _is_compatible_type(column, actual_field.type):
            errors.append(
                f"Column {name} type mismatch: expected {column.type}, got {actual_field.type}"
            )
    return errors


def _has_nulls(values: pa.Array | pa.ChunkedArray) -> bool:
    null_count = values.null_count
    if null_count is not None and null_count >= 0:
        return null_count > 0
    if isinstance(values, pa.ChunkedArray):
        return any((chunk.null_count or 0) > 0 for chunk in values.chunks)
    return False


def _nullability_errors_for_table(table_schema: TableSchema, table: pa.Table) -> list[str]:
    errors: list[str] = []
    for column in table_schema.columns:
        if column.nullable:
            continue
        if column.name not in table.column_names:
            continue
        if not _all_valid(table.column(column.name)):
            errors.append(f"Column {column.name} contains nulls but is non-nullable")
    return errors


def _nullability_errors_for_batch(
    table_schema: TableSchema,
    batch: pa.RecordBatch,
) -> list[str]:
    errors: list[str] = []
    names = list(batch.schema.names)
    for column in table_schema.columns:
        if column.nullable:
            continue
        if column.name not in names:
            continue
        index = names.index(column.name)
        if not _all_valid(batch.column(index)):
            errors.append(f"Column {column.name} contains nulls but is non-nullable")
    return errors


def _all_valid(values: pa.Array | pa.ChunkedArray) -> bool:
    is_valid = getattr(pc, "is_valid", None)
    if not callable(is_valid):
        return not _has_nulls(values)
    try:
        mask = is_valid(values)
        all_fn = getattr(pc, "all", None)
        if callable(all_fn):
            result = all_fn(mask)
            as_py = getattr(result, "as_py", None)
            if callable(as_py):
                value = as_py()
                return bool(value) if value is not None else False
        return not _has_nulls(values)
    except (TypeError, pa.ArrowInvalid):
        return not _has_nulls(values)


def _arrow_validation_errors(table: pa.Table) -> list[str]:
    try:
        table.validate()
    except pa.ArrowInvalid as exc:
        return [f"Arrow validation failed: {exc}"]
    return []


def _arrow_batch_validation_errors(batch: pa.RecordBatch) -> list[str]:
    try:
        batch.validate()
    except pa.ArrowInvalid as exc:
        return [f"Arrow validation failed: {exc}"]
    return []


def _handle_errors(table_key: str, errors: list[str], mode: ValidationMode) -> None:
    if not errors or mode == "skip":
        return
    if mode == "warn":
        for error in errors:
            LOG.warning("Validation warning for %s: %s", table_key, error)
        return
    raise TableValidationError(table_key, errors)


def validate_table(
    table_key: str,
    table: pa.Table,
    *,
    table_schema: TableSchema | None = None,
    mode: ValidationMode = "strict",
) -> pa.Table:
    """Validate an Arrow table against the registered TableSchema.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    table
        Arrow table to validate.
    table_schema
        Optional schema override to use instead of the registry lookup.
    mode
        Validation behavior: ``"strict"`` raises, ``"warn"`` logs, ``"skip"`` ignores.

    Returns
    -------
    pa.Table
        The original table when validation succeeds or is skipped.
    """
    schema = table_schema or _lookup_table_schema(table_key)
    if schema is None or mode == "skip":
        return table

    errors = []
    errors.extend(_schema_errors(schema, table.schema))
    errors.extend(_arrow_validation_errors(table))
    errors.extend(_nullability_errors_for_table(schema, table))
    _handle_errors(table_key, errors, mode)
    return table


def validate_record_batch_reader(
    table_key: str,
    reader: pa.RecordBatchReader,
    *,
    table_schema: TableSchema | None = None,
    mode: ValidationMode = "strict",
) -> pa.RecordBatchReader:
    """Validate a RecordBatchReader stream against the registered TableSchema.

    Returns
    -------
    pa.RecordBatchReader
        RecordBatchReader that yields validated batches.
    """
    resolved_schema = table_schema or _lookup_table_schema(table_key)
    if resolved_schema is None or mode == "skip":
        return reader

    schema = resolved_schema
    errors = _schema_errors(schema, reader.schema)
    _handle_errors(table_key, errors, mode)

    def _iter_batches() -> Iterator[pa.RecordBatch]:
        for batch_index, batch in enumerate(reader):
            batch_errors = []
            batch_errors.extend(_arrow_batch_validation_errors(batch))
            batch_errors.extend(_nullability_errors_for_batch(schema, batch))
            if batch_errors:
                batch_errors = [f"batch {batch_index}: {error}" for error in batch_errors]
                _handle_errors(table_key, batch_errors, mode)
            yield batch

    return pa.RecordBatchReader.from_batches(reader.schema, _iter_batches())
